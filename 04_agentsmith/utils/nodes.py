"""LangGraph node functions for the AgentSmith fraud detection graph."""

import json
from datetime import datetime
from typing import Literal

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from sub_agents import BankEnrichment, bank_a_subagent, bank_b_subagent, bank_c_subagent
from tools.network import bank_dir_for, find_chain, find_transaction
from utils import db
from utils.models import model
from utils.rules import apply_learned_rules, load_learned_rules
from utils.state import AgentContext, AgentState

_ENRICHMENT_ORCHESTRATOR_PROMPT = """\
You are a fraud investigation coordinator.
You have three bank analyst sub-agents: bank-a-analyst, bank-b-analyst, bank-c-analyst.

For each bank that has relevant data, dispatch a task to that bank's analyst sub-agent.
In each task, include the bank's name, the relevant account profiles, and the transaction history provided to you.

After all sub-agents respond, output ONLY a JSON array of BankEnrichment objects — one per bank that had data.
No other text. Example format:
[{"bank_id":"BANK-A","bank_name":"Albion Bank","risk_flags":["new_account"],"new_account_refs":["SUBJECT-A-001"],"assessment":"..."}]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _context(runtime) -> dict:
    """Run-time options. Tolerates a missing runtime so nodes can also be called directly (evals, notebooks)."""
    return dict(getattr(runtime, "context", None) or {})


def _fetch_bank_data(legs: list[dict]) -> dict:
    """Collect raw profiles and transaction history for each bank that appears in the chain."""
    bank_map = {"BANK-A": "bank_a", "BANK-B": "bank_b", "BANK-C": "bank_c"}
    bank_names = {"BANK-A": "Albion Bank", "BANK-B": "Bridgewater Bank", "BANK-C": "Crown Financial"}
    result = {}
    for bank_id, bank_dir in bank_map.items():
        relevant_legs = [l for l in legs if l.get("bank_id") == bank_id]
        if not relevant_legs:
            continue
        account_ids = list({l["account_id"] for l in relevant_legs})
        profiles = [db.get("customer_profiles", account_id=aid) for aid in account_ids]
        all_txns = db.table(f"{bank_dir}/transactions")
        history = [t for t in all_txns if t.get("account_id") in account_ids]
        result[bank_id] = {
            "bank_id": bank_id,
            "bank_name": bank_names[bank_id],
            "profiles": [p for p in profiles if p],
            "transaction_history": history,
        }
    return result


def _extract_enrichments(messages: list, bank_data: dict) -> list[dict]:
    """Try to extract BankEnrichment objects from agent messages; fall back to basic extraction."""
    # Try structured tool message results first
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(data, dict) and "bank_id" in data:
                    return [BankEnrichment.model_validate(data).model_dump()]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # Try parsing final AI message as JSON array
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip().startswith("["):
            try:
                items = json.loads(content.strip())
                if isinstance(items, list) and items:
                    return [BankEnrichment.model_validate(item).model_dump() for item in items]
            except (json.JSONDecodeError, ValueError):
                pass

    # Basic fallback: build minimal enrichments from raw data
    return _basic_enrichments(bank_data)


def _basic_enrichments(bank_data: dict, legs: list[dict] | None = None) -> list[dict]:
    """Deterministic enrichment: the fallback when deep agent extraction fails, and the `enrichment="basic"` mode.

    Applies the same checks the bank analyst sub-agents are prompted with (new account, foreign national, unusual
    amount). `legs` is the chain being investigated, used for the unusual-amount check.
    """
    enrichments = []
    for bank_id, data in bank_data.items():
        flags = []
        new_refs = []
        for profile in data.get("profiles", []):
            if profile.get("account_age_days", 999) < 90:
                flags.append("new_account")
                new_refs.append(profile["account_id"])
            if profile.get("home_country", "GB") != "GB":
                flags.append("foreign_national")
            usual_max = profile.get("usual_amount_range", [0, 999999])[1]
            if any(
                l.get("account_id") == profile["account_id"] and l.get("amount", 0) > 3 * usual_max
                for l in legs or []
            ):
                flags.append("unusual_amount")
        enrichments.append(BankEnrichment(
            bank_id=bank_id,
            bank_name=data["bank_name"],
            risk_flags=list(set(flags)),
            new_account_refs=new_refs,
            assessment=f"{len(data.get('profiles', []))} account(s) reviewed. Flags: {', '.join(set(flags)) or 'none'}.",
        ).model_dump())
    return enrichments


def _merge_enrichments(llm: list[dict], basic: list[dict]) -> list[dict]:
    """Per bank: keep the analyst's assessment, but union its flags and new-account names with the deterministic ones."""
    by_bank = {e["bank_id"]: dict(e) for e in llm}
    for floor in basic:
        merged = by_bank.setdefault(floor["bank_id"], dict(floor))
        merged["risk_flags"] = sorted(set(merged.get("risk_flags", [])) | set(floor["risk_flags"]))
        merged["new_account_names"] = sorted(
            set(merged.get("new_account_names", [])) | set(floor["new_account_names"])
        )
    return list(by_bank.values())


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def load_trigger(state: AgentState) -> dict:
    txn = find_transaction(state["alert_transaction_id"])
    if txn is None:
        raise ValueError(f"Transaction {state['alert_transaction_id']} not found in any bank.")
    return {"trigger_transaction": txn}


def detect_chain(state: AgentState, runtime: Runtime[AgentContext] = None) -> dict:
    """Find the legs of the payment chain. visibility="single_bank" limits the view to the alert's own bank."""
    chain_id = state["trigger_transaction"].get("chain_id")
    if not chain_id:
        return {"chain_id": None, "chain_legs": [state["trigger_transaction"]]}
    if _context(runtime).get("visibility") == "single_bank":
        legs = find_chain(chain_id, [bank_dir_for(state["trigger_transaction"]["bank_id"])])
    else:
        legs = find_chain(chain_id)
    return {"chain_id": chain_id, "chain_legs": legs}


def enrich_banks(state: AgentState, runtime: Runtime[AgentContext] = None) -> dict:
    """Dispatch one bank analyst sub-agent per involved bank, in parallel, via a deep agent."""
    legs = state["chain_legs"]
    chain_id = state.get("chain_id", "unknown")
    bank_data = _fetch_bank_data(legs)

    if not bank_data:
        return {"bank_enrichments": []}

    if _context(runtime).get("enrichment") == "basic":
        return {"bank_enrichments": _basic_enrichments(bank_data, legs)}

    prompt = (
        f"Investigate chain {chain_id}. "
        f"Here is the bank data:\n\n{json.dumps(bank_data, indent=2)}\n\n"
        "Dispatch tasks to each relevant bank analyst sub-agent and collect their findings."
    )

    try:
        enrichment_agent = create_deep_agent(
            model=init_chat_model(model),
            tools=[],
            system_prompt=_ENRICHMENT_ORCHESTRATOR_PROMPT,
            subagents=[bank_a_subagent, bank_b_subagent, bank_c_subagent],
        )
        result = enrichment_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        enrichments = _extract_enrichments(result.get("messages", []), bank_data)
    except Exception:
        enrichments = []

    # The analysts add judgement, but profile facts (new account, foreign national, unusual amount) are
    # deterministic, so they are a floor the model cannot drop. LangSmith evals showed the LLM analysts
    # sometimes returned no flags for chains whose profiles clearly had them.
    return {"bank_enrichments": _merge_enrichments(enrichments, _basic_enrichments(bank_data, legs))}


def score_risk(
    state: AgentState, runtime: Runtime[AgentContext] = None, store: BaseStore | None = None
) -> dict:
    """Rule-based fraud scorer, boosted by institutional memory of past confirmed fraud."""
    rules = load_learned_rules() if _context(runtime).get("use_learned_rules", True) else []
    score, reasons = compute_risk(state["chain_legs"], state.get("bank_enrichments", []), state.get("chain_id"), rules)

    # Memory boost: accounts in this chain previously linked to confirmed fraud
    if store and state.get("chain_id"):
        try:
            past_items = store.search(("fraud_decisions", "global"), limit=200)
            chain_accounts = {l.get("account_id") for l in state["chain_legs"]}
            for item in past_items:
                v = item.value
                if v.get("decision") == "investigate":
                    if chain_accounts & set(v.get("accounts", [])):
                        score += 10
                        reasons.append("accounts previously linked to confirmed fraud (memory)")
                        break
        except Exception:
            pass

    return {"risk_score": min(score, 100), "risk_reasons": reasons}


def compute_risk(
    legs: list[dict], enrichments: list[dict], chain_id: str | None, learned_rules: list[dict]
) -> tuple[int, list[str]]:
    """Pure scoring function: the built-in rules plus any learned rules. Returns (score, reasons).

    Kept separate from the node so the improvement loop can score candidate rules without running the graph.
    """
    state = {"chain_id": chain_id}
    score = 0
    reasons = []

    # Rule 1: circular pattern
    if state.get("chain_id") and len(legs) >= 3:
        origin_account = legs[0].get("account_id")
        final_dest = legs[-1].get("counterparty_account") or legs[-1].get("account_id")
        timestamps = [_parse_ts(l["timestamp"]) for l in legs if l.get("timestamp")]
        if timestamps:
            span = (max(timestamps) - min(timestamps)).total_seconds()
            if origin_account == final_dest and span <= 600:
                score += 50
                reasons.append(f"circular fund movement: {len(legs)} legs returning to origin in {int(span)}s")

    # Rule 2: amount decrement pattern
    amounts = [l["amount"] for l in legs if l.get("amount")]
    if len(amounts) >= 2:
        decrements = [amounts[i] / amounts[i - 1] for i in range(1, len(amounts))]
        if all(0.95 <= d < 1.0 for d in decrements):
            score += 20
            reasons.append("systematic amount decrement across legs (fee-extraction pattern)")

    # Rule 3: new accounts flagged by bank analysts
    all_new_refs = [ref for e in enrichments for ref in e.get("new_account_refs", [])]
    if all_new_refs:
        score += 15
        reasons.append(f"new accounts (< 90 days old): {', '.join(all_new_refs)}")

    # Rule 4: foreign nationals in chain
    all_flags = [f for e in enrichments for f in e.get("risk_flags", [])]
    if "foreign_national" in all_flags:
        score += 10
        reasons.append("foreign-national account holder(s) in chain")

    # Rule 5: unusual amounts flagged by bank analysts
    if "unusual_amount" in all_flags:
        score += 10
        reasons.append("transaction amounts exceed account holders' usual spending range")

    # Learned rules (data/learned_rules.json), produced by the evaluation-driven improvement loop (evals/improve.py)
    points, learned_reasons = apply_learned_rules(legs, learned_rules)
    score += points
    reasons.extend(learned_reasons)

    return min(score, 100), reasons


def route_decision(state: AgentState) -> Literal["human_review", "auto_clear"]:
    if state["risk_score"] >= 70:
        return "human_review"
    return "auto_clear"


def human_review(state: AgentState) -> dict:
    answer = interrupt(
        {
            "chain_id": state.get("chain_id"),
            "risk_score": state["risk_score"],
            "risk_reasons": state["risk_reasons"],
            "bank_assessments": [e.get("assessment") for e in state.get("bank_enrichments", [])],
            "chain_legs": state["chain_legs"],
            "question": "Investigate or clear? Reply with {'decision': 'investigate' or 'clear', 'notes': '...'}",
        }
    )
    return {
        "decision": answer["decision"],
        "decided_by": "human",
        "reviewer_notes": answer.get("notes", ""),
    }


def auto_clear(state: AgentState) -> dict:
    return {"decision": "clear", "decided_by": "agent", "reviewer_notes": ""}


def write_audit(state: AgentState, store: BaseStore | None = None) -> dict:
    """Write audit memo and persist the decision to long-term memory."""
    legs = state["chain_legs"]
    banks = sorted({l.get("bank_id", "?") for l in legs})
    total_in = legs[0].get("amount", 0) if legs else 0
    total_out = legs[-1].get("amount", 0) if legs else 0

    lines = [
        f"Alert: {state['alert_transaction_id']} | Chain: {state.get('chain_id') or 'none'} | Banks: {', '.join(banks)}",
        f"Legs: {len(legs)} | Entry: £{total_in:,.2f} | Exit: £{total_out:,.2f}",
        f"Risk score: {state['risk_score']}/100",
    ]
    if state["risk_reasons"]:
        lines.append("Risk factors: " + "; ".join(state["risk_reasons"]))
    for e in state.get("bank_enrichments", []):
        lines.append(f"  {e.get('bank_id')}: {e.get('assessment', '')}")
    lines.append(f"Decision: {state['decision'].upper()} (by {state['decided_by']})")
    if state.get("reviewer_notes"):
        lines.append(f"Reviewer notes: {state['reviewer_notes']}")

    # Persist to long-term memory so future runs can learn from this decision
    if store and state.get("chain_id"):
        try:
            store.put(
                ("fraud_decisions", "global"),
                state["chain_id"],
                {
                    "chain_id": state["chain_id"],
                    "decision": state["decision"],
                    "decided_by": state["decided_by"],
                    "risk_score": state["risk_score"],
                    "accounts": list({l.get("account_id") for l in legs}),
                    "pattern": "circular" if state.get("chain_id") else None,
                },
            )
        except Exception:
            pass

    return {"audit_memo": "\n".join(lines)}
