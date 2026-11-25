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


class AgentContext(TypedDict):
    confidence_threshold: float
    circular_time_window_seconds: int
