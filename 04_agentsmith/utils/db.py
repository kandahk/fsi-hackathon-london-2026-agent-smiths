"""A tiny read-only database over the JSONL files in `data/`.

Each `.jsonl` file is a table and each line in it is one row. Per-bank tables live under
`data/bank_a/`, `data/bank_b/`, `data/bank_c/` and are accessed with a `bank_id` prefix,
e.g. `table("bank_a/transactions")`. Cross-bank tables sit directly in `data/`.
"""

import json

from utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"


def table(name: str) -> list[dict]:
    """Every row in a table, in file order."""
    with open(DATA_DIR / f"{name}.jsonl", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def query(name: str, **where) -> list[dict]:
    """The rows whose fields equal the given values, e.g. `query("bank_a/transactions", chain_id="CHAIN-001")`."""
    return [row for row in table(name) if all(row.get(field) == value for field, value in where.items())]


def get(name: str, **where) -> dict | None:
    """The first row that matches, or None if there isn't one."""
    rows = query(name, **where)
    return rows[0] if rows else None
