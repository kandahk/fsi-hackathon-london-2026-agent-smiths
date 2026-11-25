"""Bank analyst sub-agents — one per bank, run in parallel by the enrichment deep agent."""
from deepagents import SubAgent
from pydantic import BaseModel, Field


class BankEnrichment(BaseModel):
    """Risk assessment returned by a single bank analyst sub-agent."""
    bank_id: str = Field(description="The bank's ID, e.g. BANK-A")
    bank_name: str = Field(description="The bank's display name")
    risk_flags: list[str] = Field(description="Risk indicators found, e.g. ['new_account','unusual_amount','foreign_national']")
    new_account_refs: list[str] = Field(description="Pseudonymous subject references for accounts less than 90 days old")
    assessment: str = Field(description="1-2 sentence plain-English assessment of risk indicators at this bank")


_BANK_ANALYST_INSTRUCTIONS = """\
You are a fraud analyst. You receive raw account profiles and transaction history for one bank.
Identify risk indicators:
- New accounts: account_age_days < 90
- Unusual amounts: transaction amount > 3× the account's usual_amount_range[1]
- Foreign nationals: home_country differs from GB
- High velocity: multiple transactions within seconds

Return your findings as a BankEnrichment structured response.
Do not include names or other direct identifiers; refer to accounts only by their pseudonymous subject references.
If the data provided is empty or the bank has no involvement, return an empty risk_flags list and a brief assessment saying so.
"""

bank_a_subagent: SubAgent = {
    "name": "bank-a-analyst",
    "description": (
        "Fraud analyst at Albion Bank (BANK-A). "
        "Pass BANK-A account profiles and transaction history in the task. "
        "Returns a BankEnrichment with risk flags and assessment."
    ),
    "system_prompt": _BANK_ANALYST_INSTRUCTIONS,
    "tools": [],
    "response_format": BankEnrichment,
}

bank_b_subagent: SubAgent = {
    "name": "bank-b-analyst",
    "description": (
        "Fraud analyst at Bridgewater Bank (BANK-B). "
        "Pass BANK-B account profiles and transaction history in the task. "
        "Returns a BankEnrichment with risk flags and assessment."
    ),
    "system_prompt": _BANK_ANALYST_INSTRUCTIONS,
    "tools": [],
    "response_format": BankEnrichment,
}

bank_c_subagent: SubAgent = {
    "name": "bank-c-analyst",
    "description": (
        "Fraud analyst at Crown Financial (BANK-C). "
        "Pass BANK-C account profiles and transaction history in the task. "
        "Returns a BankEnrichment with risk flags and assessment."
    ),
    "system_prompt": _BANK_ANALYST_INSTRUCTIONS,
    "tools": [],
    "response_format": BankEnrichment,
}
