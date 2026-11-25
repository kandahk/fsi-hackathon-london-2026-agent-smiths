"""Cross-bank network tools: find a transaction across all bank data files."""

from utils import db


BANK_DIRS = ["bank_a", "bank_b", "bank_c"]


def find_transaction(transaction_id: str) -> dict | None:
    """Search all bank transaction files for a transaction by ID."""
    for bank_dir in BANK_DIRS:
        row = db.get(f"{bank_dir}/transactions", transaction_id=transaction_id)
        if row:
            return row
    return None


def find_chain(chain_id: str) -> list[dict]:
    """Collect all legs of a chain across all banks, ordered by chain_position."""
    legs = []
    for bank_dir in BANK_DIRS:
        legs.extend(db.query(f"{bank_dir}/transactions", chain_id=chain_id))
    return sorted(legs, key=lambda t: (t.get("chain_position") or 0))
