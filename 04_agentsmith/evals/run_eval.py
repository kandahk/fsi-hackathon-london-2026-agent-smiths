"""Build the AgentSmith dataset in LangSmith and evaluate the agent against it.

One example per transaction across bank_a / bank_b / bank_c, labelled from data/fraud_labels.jsonl. Each example
is an alert. The agent runs either in-process (`--target local`) or on the LangSmith deployment (`--target remote`).
Runs that score >= 70 pause at human review and are NOT resumed, so "flagged" means the agent asked for a human.

Run configuration (applied through the agent's runtime context):
  --visibility  collective | single_bank   what the agent can see (all three banks, or only the alert's own bank)
  --rules       learned | baseline         with or without data/learned_rules.json
  --enrichment  llm | basic                Deep Agent bank analysts, or deterministic checks (fast, no model calls)

Examples (from 04_agentsmith):
    uv run python -m evals.run_eval --dataset-only
    uv run python -m evals.run_eval --target local --enrichment basic --visibility single_bank --rules baseline
    uv run python -m evals.run_eval --target remote --url https://<deployment>.us.langgraph.app
"""

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
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


# ---- targets ----------------------------------------------------------------------------------------
def make_local_target(context: dict):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from agent import build_agent

    graph = build_agent(checkpointer=InMemorySaver(), store=InMemoryStore())

    def target(inputs: dict) -> dict:
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
        try:
            out = graph.invoke(inputs, thread, context=context)
        except Exception as exc:  # e.g. transaction not found
            return {"error": str(exc), "flagged": False}
        return {
            "chain_id": out.get("chain_id"),
            "n_legs": len(out.get("chain_legs", [])),
            "risk_score": out.get("risk_score"),
            "risk_reasons": out.get("risk_reasons", []),
            "flagged": "__interrupt__" in out,
        }

    return target


def make_remote_target(url: str, context: dict):
    from langgraph_sdk import get_sync_client

    sdk = get_sync_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])

    def target(inputs: dict) -> dict:
        thread = sdk.threads.create()
        try:
            sdk.runs.wait(thread["thread_id"], "agentsmith", input=inputs, context=context)
        except Exception as exc:
            return {"error": str(exc), "flagged": False}
        state = sdk.threads.get_state(thread["thread_id"])
        values = state.get("values", {})
        return {
            "chain_id": values.get("chain_id"),
            "n_legs": len(values.get("chain_legs", [])),
            "risk_score": values.get("risk_score"),
            "risk_reasons": values.get("risk_reasons", []),
            "flagged": "human_review" in (state.get("next") or []),
            "thread_id": thread["thread_id"],
        }

    return target


# ---- per-example evaluators -------------------------------------------------------------------------
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
    """Was the right chain found? In single_bank mode the chain is still identified, just with fewer legs."""
    return {"key": "chain_detected", "score": int(outputs.get("chain_id") == reference_outputs["chain_id"])}


def reasons_for_flagged(outputs: dict) -> dict:
    if not outputs.get("flagged"):
        return {"key": "reasons_for_flagged", "score": None}
    return {"key": "reasons_for_flagged", "score": int(bool(outputs.get("risk_reasons")))}


# ---- headline metrics -------------------------------------------------------------------------------
def compute_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs = [(agent_outputs, reference_outputs)]. Alert-level recall/precision plus fraud chains caught."""
    tp = fp = fn = 0
    fraud_chains: dict[str, bool] = {}
    for out, ref in pairs:
        flagged, is_fraud = bool(out.get("flagged")), ref["label"] == "fraud"
        tp += flagged and is_fraud
        fp += flagged and not is_fraud
        fn += (not flagged) and is_fraud
        if is_fraud and ref.get("chain_id"):
            fraud_chains[ref["chain_id"]] = fraud_chains.get(ref["chain_id"], False) or flagged
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "false_negatives": fn,
        "false_positives": fp,
        "chains_caught": sum(fraud_chains.values()),
        "fraud_chains": len(fraud_chains),
    }


def summary_evaluator(runs, examples) -> dict:
    m = compute_metrics([(r.outputs or {}, e.outputs) for r, e in zip(runs, examples)])
    return {
        "results": [
            {"key": "recall", "score": m["recall"]},
            {"key": "precision", "score": m["precision"]},
            {"key": "fraud_chains_caught", "score": m["chains_caught"] / max(m["fraud_chains"], 1)},
        ]
    }


def run_experiment(target, prefix: str, metadata: dict, client: Client | None = None, max_concurrency: int = 4) -> dict:
    """Run one experiment in LangSmith and return its headline metrics."""
    client = client or Client()
    pairs: list[tuple[dict, dict]] = []

    def recording_target(inputs: dict) -> dict:
        return target(inputs)

    results = evaluate(
        recording_target,
        data=DATASET,
        evaluators=[flag_matches_label, false_negative, false_positive, chain_detected, reasons_for_flagged],
        summary_evaluators=[summary_evaluator],
        experiment_prefix=prefix,
        max_concurrency=max_concurrency,
        client=client,
        metadata=metadata,
    )
    for row in results:
        pairs.append((row["run"].outputs or {}, row["example"].outputs))
    metrics = compute_metrics(pairs)
    metrics["experiment"] = results.experiment_name
    return metrics


def context_from(args) -> dict:
    return {
        "visibility": args.visibility,
        "use_learned_rules": args.rules == "learned",
        "enrichment": args.enrichment,
        "notify": False,  # evals must never post to Slack
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["local", "remote"], default="local")
    parser.add_argument("--url", help="Deployment URL (required for --target remote)")
    parser.add_argument("--visibility", choices=["collective", "single_bank"], default="collective")
    parser.add_argument("--rules", choices=["learned", "baseline"], default="learned")
    parser.add_argument("--enrichment", choices=["llm", "basic"], default="llm")
    parser.add_argument("--prefix", default="agentsmith")
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()

    client = Client()
    ensure_dataset(client)
    if args.dataset_only:
        return

    context = context_from(args)
    target = make_remote_target(args.url, context) if args.target == "remote" else make_local_target(context)
    metrics = run_experiment(
        target,
        f"{args.prefix}-{args.target}-{args.visibility}-{args.rules}-{args.enrichment}",
        {"target": args.target, "deployment_url": args.url, **context},
        client,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
