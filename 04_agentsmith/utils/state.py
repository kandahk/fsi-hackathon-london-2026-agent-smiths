from typing import Literal, TypedDict


class AgentInput(TypedDict):
    alert_transaction_id: str


class AgentState(AgentInput):
    trigger_transaction: dict
    chain_id: str | None
    chain_legs: list[dict]
    bank_enrichments: list[dict]
    risk_score: int
    risk_reasons: list[str]
    decision: Literal["investigate", "clear"]
    decided_by: Literal["agent", "human"]
    reviewer_notes: str
    audit_memo: str
    notification: dict  # outcome of the analyst Slack alert: {status: sent|skipped|failed, detail}


class AgentContext(TypedDict, total=False):
    confidence_threshold: float
    circular_time_window_seconds: int
    # "collective" (default): the agent sees all three banks. "single_bank": it only sees the alert's own bank,
    # which is how each bank works today. This is the baseline for the collective-intelligence comparison.
    visibility: Literal["collective", "single_bank"]
    # True (default): apply the rules in data/learned_rules.json. False reproduces the original rule set.
    use_learned_rules: bool
    # "llm" (default): bank analyst sub-agents (Deep Agents). "basic": deterministic profile checks, used for fast,
    # repeatable evaluation runs without model calls.
    enrichment: Literal["llm", "basic"]
    # False: never alert the analyst for this run (evals and notebooks set this so they don't post to Slack).
    notify: bool
