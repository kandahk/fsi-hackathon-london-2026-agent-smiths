"""LangGraph node functions for the AgentSmith fraud detection graph."""

from datetime import datetime
from typing import Literal

from langgraph.types import interrupt

from tools.bank import enrich_accounts
from tools.network import find_chain, find_transaction
from utils.state import AgentState


def load_trigger(state: AgentState) -> dict:
    """Load the transaction that triggered this investigation."""
    txn = find_transaction(state["alert_transaction_id"])
    if txn is None:
        raise ValueError(f"Transaction {state['alert_transaction_id']} not found in any bank.")
    return {"trigger_transaction": txn}


def detect_chain(state: AgentState) -> dict:
    """Find all legs of the payment chain the trigger transaction belongs to."""
    chain_id = state["trigger_transaction"].get("chain_id")
    if not chain_id:
        return {"chain_id": None, "chain_legs": [state["trigger_transaction"]]}

    legs = find_chain(chain_id)
    return {"chain_id": chain_id, "chain_legs": legs}


def _accounts_for_bank(legs: list[dict], bank_id: str) -> list[str]:
    return list({
        leg["account_id"]
        for leg in legs
        if leg.get("bank_id") == bank_id
    })


def enrich_bank_a(state: AgentState) -> dict:
    accounts = _accounts_for_bank(state["chain_legs"], "BANK-A")
    if not accounts:
        return {"bank_enrichments": state.get("bank_enrichments", [])}
    enrichment = enrich_accounts("bank_a", accounts)
    return {"bank_enrichments": [*state.get("bank_enrichments", []), enrichment]}


def enrich_bank_b(state: AgentState) -> dict:
    accounts = _accounts_for_bank(state["chain_legs"], "BANK-B")
    if not accounts:
        return {"bank_enrichments": state.get("bank_enrichments", [])}
    enrichment = enrich_accounts("bank_b", accounts)
    return {"bank_enrichments": [*state.get("bank_enrichments", []), enrichment]}


def enrich_bank_c(state: AgentState) -> dict:
    accounts = _accounts_for_bank(state["chain_legs"], "BANK-C")
    if not accounts:
        return {"bank_enrichments": state.get("bank_enrichments", [])}
    enrichment = enrich_accounts("bank_c", accounts)
    return {"bank_enrichments": [*state.get("bank_enrichments", []), enrichment]}


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def score_risk(state: AgentState) -> dict:
    """Rule-based fraud risk scorer. Returns risk_score (0-100) and risk_reasons."""
    legs = state["chain_legs"]
    enrichments = state.get("bank_enrichments", [])
    score = 0
    reasons = []

    # Rule 1: circular pattern — chain returns to the originating account
    if state.get("chain_id") and len(legs) >= 3:
        origin_account = legs[0].get("account_id")
        final_account = legs[-1].get("counterparty_account") or legs[-1].get("account_id")
        timestamps = [_parse_ts(l["timestamp"]) for l in legs if l.get("timestamp")]
        if timestamps:
            span_seconds = (max(timestamps) - min(timestamps)).total_seconds()
            if origin_account == final_account and span_seconds <= 600:
                score += 50
                reasons.append(
                    f"circular fund movement: {len(legs)} legs returning to origin in {int(span_seconds)}s"
                )

    # Rule 2: amount decrement pattern (each leg 95-99% of previous)
    amounts = [l["amount"] for l in legs if l.get("amount")]
    if len(amounts) >= 2:
        decrements = [amounts[i] / amounts[i - 1] for i in range(1, len(amounts))]
        if all(0.95 <= d < 1.0 for d in decrements):
            score += 20
            reasons.append("systematic amount decrement across legs (fee-extraction pattern)")

    # Rule 3: new accounts (opened < 90 days ago)
    all_profiles = [p for e in enrichments for p in e.get("accounts", [])]
    new_accounts = [p for p in all_profiles if p.get("account_age_days", 999) < 90]
    if new_accounts:
        score += 15
        names = ", ".join(p.get("name", p["account_id"]) for p in new_accounts)
        reasons.append(f"new accounts (< 90 days old) involved: {names}")

    # Rule 4: foreign counterparty
    countries = {p.get("home_country") for p in all_profiles if p.get("home_country")}
    if len(countries) > 1:
        score += 10
        reasons.append(f"cross-country accounts: {', '.join(sorted(countries))}")

    # Rule 5: amount far exceeds usual range for any account
    for profile in all_profiles:
        usual_max = profile.get("usual_amount_range", [0, 999999])[1]
        account_legs = [l for l in legs if l.get("account_id") == profile["account_id"]]
        for leg in account_legs:
            if leg.get("amount", 0) > 3 * usual_max:
                score += 10
                reasons.append(
                    f"{profile['account_id']} amount £{leg['amount']:,.0f} is "
                    f">{3}× their usual maximum £{usual_max:,.0f}"
                )
                break

    return {"risk_score": min(score, 100), "risk_reasons": reasons}


def route_decision(state: AgentState) -> Literal["human_review", "auto_clear"]:
    if state["risk_score"] >= 70:
        return "human_review"
    return "auto_clear"


def human_review(state: AgentState) -> dict:
    """Pause and present evidence to a fraud analyst."""
    answer = interrupt(
        {
            "chain_id": state.get("chain_id"),
            "risk_score": state["risk_score"],
            "risk_reasons": state["risk_reasons"],
            "chain_legs": state["chain_legs"],
            "bank_enrichments": state.get("bank_enrichments", []),
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


def write_audit(state: AgentState) -> dict:
    """Compose a plain-English audit memo."""
    chain_id = state.get("chain_id") or "none"
    legs = state["chain_legs"]
    banks = sorted({l.get("bank_id", "?") for l in legs})
    total_in = legs[0].get("amount", 0) if legs else 0
    total_out = legs[-1].get("amount", 0) if legs else 0

    memo_lines = [
        f"Alert: {state['alert_transaction_id']} | Chain: {chain_id} | Banks: {', '.join(banks)}",
        f"Legs: {len(legs)} | Entry amount: £{total_in:,.2f} | Exit amount: £{total_out:,.2f}",
        f"Risk score: {state['risk_score']}/100",
    ]
    if state["risk_reasons"]:
        memo_lines.append("Risk factors: " + "; ".join(state["risk_reasons"]))
    memo_lines.append(
        f"Decision: {state['decision'].upper()} (by {state['decided_by']})"
    )
    if state.get("reviewer_notes"):
        memo_lines.append(f"Reviewer notes: {state['reviewer_notes']}")

    return {"audit_memo": "\n".join(memo_lines)}
