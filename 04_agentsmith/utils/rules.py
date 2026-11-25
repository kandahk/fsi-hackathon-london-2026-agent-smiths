"""Learned fraud rules: declarative, versioned, and safe to load (no code is ever executed from the file).

`evals/improve.py` proposes rules with an LLM, validates them against the labelled LangSmith dataset, and writes the
accepted ones to `data/learned_rules.json`. `score_risk` applies them on top of the built-in rules.

Rule types (all work on the legs of one payment chain):
  fan_out        one account sends `min_legs` transfers within `window_seconds`
  fan_in         one account receives `min_legs` transfers within `window_seconds`
  split_forward  one account receives money, then within `window_seconds` forwards it in `min_legs`+ transfers that
                 add up to at least `min_ratio` of what came in

Any rule can also set `requires_flags`: it then only fires if the bank analysts found at least one of those risk
flags in the chain. This lets a movement pattern (fast fan-out) be told apart from legitimate lookalikes (a property
settlement between established accounts) by who is involved, not by finely tuned numbers.
"""

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from utils import PROJECT_ROOT

RULES_PATH = PROJECT_ROOT / "data" / "learned_rules.json"


RiskFlag = Literal["new_account", "foreign_national", "unusual_amount"]


class LearnedRule(BaseModel):
    rule_type: Literal["fan_out", "fan_in", "split_forward"]
    min_legs: int = Field(ge=2, le=10, description="Minimum number of transfers that must match")
    window_seconds: int = Field(ge=10, le=3600, description="All matching transfers must fall inside this window")
    min_ratio: float = Field(
        default=0.9, ge=0.5, le=1.2, description="split_forward only: forwarded total / received amount"
    )
    requires_flags: list[RiskFlag] = Field(
        default_factory=list,
        description=(
            "Optional. If set, the rule only fires when the bank analysts found at least one of these flags: "
            "new_account (account < 90 days old), foreign_national (home country not GB), "
            "unusual_amount (a transfer > 3x the holder's usual maximum). Empty = no condition."
        ),
    )
    points: int = Field(ge=5, le=60, description="Risk points added when the rule fires")
    rationale: str = Field(description="One sentence: the fraud behaviour this rule captures")


class RuleSet(BaseModel):
    rules: list[LearnedRule] = Field(max_length=3)


def load_learned_rules(path=RULES_PATH) -> list[dict]:
    """The active learned rules, or [] if the file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file).get("rules", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_learned_rules(rules: list[dict], metadata: dict, path=RULES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "rules": rules}, file, indent=2)
        file.write("\n")


def _ts(leg: dict) -> datetime:
    return datetime.fromisoformat(leg["timestamp"])


def _burst(legs: list[dict], min_legs: int, window_seconds: int) -> list[dict] | None:
    """The first `min_legs` consecutive legs (by time) that fit inside the window, else None."""
    legs = sorted(legs, key=_ts)
    for i in range(len(legs) - min_legs + 1):
        if (_ts(legs[i + min_legs - 1]) - _ts(legs[i])).total_seconds() <= window_seconds:
            return legs[i : i + min_legs]
    return None


def _by_account(legs: list[dict], leg_type: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for leg in legs:
        if leg.get("type") == leg_type and leg.get("timestamp"):
            grouped.setdefault(leg["account_id"], []).append(leg)
    return grouped


def _match(legs: list[dict], rule: dict) -> str | None:
    """A human-readable reason if the rule fires on these legs, else None."""
    kind, n, window = rule["rule_type"], rule["min_legs"], rule["window_seconds"]

    if kind in ("fan_out", "fan_in"):
        leg_type = "transfer_out" if kind == "fan_out" else "transfer_in"
        verb = "sent" if kind == "fan_out" else "received"
        for account, account_legs in _by_account(legs, leg_type).items():
            group = _burst(account_legs, n, window)
            if group:
                span = int((_ts(group[-1]) - _ts(group[0])).total_seconds())
                total = sum(l["amount"] for l in group)
                return f"{kind.replace('_', '-')}: {account} {verb} {n} transfers (£{total:,.0f}) within {span}s"

    if kind == "split_forward":
        inbound, outbound = _by_account(legs, "transfer_in"), _by_account(legs, "transfer_out")
        for account, in_legs in inbound.items():
            for received in in_legs:
                forwarded = [
                    l
                    for l in outbound.get(account, [])
                    if 0 <= (_ts(l) - _ts(received)).total_seconds() <= window
                ]
                total = sum(l["amount"] for l in forwarded)
                if len(forwarded) >= n and total >= rule.get("min_ratio", 0.9) * received["amount"]:
                    return (
                        f"split-forward: {account} received £{received['amount']:,.0f} and forwarded "
                        f"£{total:,.0f} in {len(forwarded)} transfers within {window}s"
                    )
    return None


def apply_learned_rules(
    legs: list[dict], rules: list[dict], flags: set[str] | frozenset[str] = frozenset()
) -> tuple[int, list[str]]:
    """Total points and reasons for every learned rule that fires on this chain.

    `flags` are the risk flags the bank analysts found in the chain; a rule with `requires_flags` needs one of them.
    """
    points, reasons = 0, []
    for rule in rules:
        required = rule.get("requires_flags") or []
        if required and not set(required) & set(flags):
            continue
        reason = _match(legs, rule)
        if reason:
            points += rule["points"]
            condition = f", needs {'/'.join(required)}" if required else ""
            reasons.append(f"{reason} (learned rule, +{rule['points']}{condition})")
    return points, reasons
