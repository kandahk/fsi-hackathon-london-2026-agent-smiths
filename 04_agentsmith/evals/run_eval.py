"""Build the AgentSmith dataset in LangSmith and evaluate the deployed agent against it.

One example per transaction across bank_a / bank_b / bank_c, labelled from data/fraud_labels.jsonl.
Each example becomes an alert; the deployed agent is called remotely. Runs that score >= 70 pause at
human review (interrupt) and are NOT resumed, so "flagged" == the agent stopped for a human.

Usage (from 04_agentsmith):
    uv run --with langsmith --with langgraph-sdk --with python-dotenv python evals/run_eval.py \
        --url https://<deployment>.us.langgraph.app [--dataset-only]
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph_sdk import get_sync_client
from langsmith import Client, evaluate

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT.parent / ".env")

DATASET = "agentsmith-fraud-alerts"
BANK_DIRS = ["bank_a", "bank_b", "bank_c"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_examples() -> list[dict]:
    labels = read_jsonl(ROOT / "data" / "fraud_labels.jsonl")
    by_chain = {l["chain_id"]: l for l in labels if l.get("chain_id")}
    by_txn = {l["transaction_id"]: l for l in labels if l.get("transaction_id")}
    examples = []
    for bank_dir in BANK_DIRS:
        for txn in read_jsonl(ROOT / "data" / bank_dir / "transactions.jsonl"):
            label = by_txn.get(txn["transaction_id"]) or by_chain.get(txn.get("chain_id"))
            examples.append(
                {
                    "inputs": {"alert_transaction_id": txn["transaction_id"]},
                    "outputs": {
                        "chain_id": txn.get("chain_id"),
                        "label": label["label"],
                        "pattern": label.get("pattern"),
                        "expected_decision": "investigate" if label["label"] == "fraud" else "clear",
                    },
                    "metadata": {"bank_id": txn["bank_id"], "notes": label["notes"]},
                    "split": bank_dir,
                }
            )
    return examples


def ensure_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET):
        print(f"Dataset '{DATASET}' already exists, reusing it.")
        return client.read_dataset(dataset_name=DATASET)
    ds = client.create_dataset(
        DATASET, description="AgentSmith: every transaction across Banks A/B/C with fraud labels."
    )
    examples = build_examples()
    client.create_examples(
        dataset_id=ds.id,
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        metadata=[e["metadata"] for e in examples],
        splits=[e["split"] for e in examples],
    )
    print(f"Created dataset '{DATASET}' with {len(examples)} examples.")
    return ds


def make_target(url: str):
    sdk = get_sync_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])

    def target(inputs: dict) -> dict:
        thread = sdk.threads.create()
        try:
            sdk.runs.wait(thread["thread_id"], "agentsmith", input=inputs)
        except Exception as exc:  # e.g. transaction not found
            return {"error": str(exc), "flagged": False}
        state = sdk.threads.get_state(thread["thread_id"])
        values = state.get("values", {})
        return {
            "chain_id": values.get("chain_id"),
            "n_legs": len(values.get("chain_legs", [])),
            "risk_score": values.get("risk_score"),
            "risk_reasons": values.get("risk_reasons", []),
            "flagged": "human_review" in (state.get("next") or []),
            "decision": values.get("decision"),
            "thread_id": thread["thread_id"],
        }

    return target


# ---- evaluators -------------------------------------------------------------------------------
def flag_matches_label(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["label"] == "fraud"
    return {"key": "flag_matches_label", "score": int(outputs.get("flagged", False) == expected)}


def false_negative(outputs: dict, reference_outputs: dict) -> dict:
    """1 when a fraud case was NOT flagged (the costly miss)."""
    miss = reference_outputs["label"] == "fraud" and not outputs.get("flagged", False)
    return {"key": "false_negative", "score": int(miss)}


def false_positive(outputs: dict, reference_outputs: dict) -> dict:
    fp = reference_outputs["label"] != "fraud" and outputs.get("flagged", False)
    return {"key": "false_positive", "score": int(fp)}


def chain_detected(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "chain_detected", "score": int(outputs.get("chain_id") == reference_outputs["chain_id"])}


def reasons_for_flagged(outputs: dict) -> dict:
    if not outputs.get("flagged"):
        return {"key": "reasons_for_flagged", "score": None}
    return {"key": "reasons_for_flagged", "score": int(bool(outputs.get("risk_reasons")))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Deployment URL")
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()

    client = Client()
    ensure_dataset(client)
    if args.dataset_only:
        return

    results = evaluate(
        make_target(args.url),
        data=DATASET,
        evaluators=[flag_matches_label, false_negative, false_positive, chain_detected, reasons_for_flagged],
        experiment_prefix="agentsmith-deployed",
        max_concurrency=4,
        client=client,
        metadata={"deployment_url": args.url},
    )
    df = results.to_pandas()
    cols = [c for c in df.columns if c.startswith("feedback.")]
    print(df[cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
