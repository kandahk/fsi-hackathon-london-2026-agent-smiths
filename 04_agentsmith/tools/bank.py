"""Per-bank enrichment tools: account history and customer profile lookup."""

from utils import db


def enrich_accounts(bank_dir: str, account_ids: list[str]) -> dict:
    """Return customer profiles and recent transaction history for the given accounts at one bank."""
    profiles = []
    for account_id in account_ids:
        profile = db.get("customer_profiles", account_id=account_id)
        if profile:
            profiles.append(profile)

    all_txns = db.table(f"{bank_dir}/transactions")
    history = [t for t in all_txns if t.get("account_id") in account_ids]

    bank = db.get("banks", data_dir=bank_dir)
    return {
        "bank_id": bank["bank_id"] if bank else bank_dir,
        "bank_name": bank["name"] if bank else bank_dir,
        "accounts": profiles,
        "recent_transactions": history,
    }
