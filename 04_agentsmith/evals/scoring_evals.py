"""LangSmith evaluation suite for AgentSmith.

Run with:
    cd 04_agentsmith && uv run python -m evals.scoring_evals

Each scenario is one alert_transaction_id. The agent runs to completion (auto-answering any
human_review interrupts) and its output is scored against the ground-truth fraud_labels.jsonl.

Three evaluators:
  detection_accuracy   — did the agent reach the right investigate/clear decision?
  risk_calibration     — is the risk score in the right band (≥70 for fraud, ≤30 for legitimate)?
  human_routing        — did the agent route to human review exactly when it should?

The headline number on screen is the detection accuracy across all scenarios.
"""

import uuid

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from utils import PROJECT_ROOT
from utils import db

load_dotenv(PROJECT_ROOT.parent / ".env")

DATASET_NAME = "agentsmith-fraud-scenarios-v1"
EXPERIMENT_PREFIX = "agentsmith"

# One entry per scenario: alert_transaction_id → triggering transaction for that chain
SCENARIOS = [
    {"alert_transaction_id": "TXN-A-001", "chain_id": "CHAIN-001", "expected_label": "fraud"},
    {"alert_transaction_id": "TXN-A-003", "chain_id": "CHAIN-002", "expected_label": "fraud"},
    {"alert_transaction_id": "TXN-A-005", "chain_id": "CHAIN-003", "expected_label": "fraud"},
    {"alert_transaction_id": "TXN-A-007", "chain_id": "CHAIN-004", "expected_label": "fraud"},
    {"alert_transaction_id": "TXN-A-014", "chain_id": "CHAIN-005", "expected_label": "fraud"},
    {"alert_transaction_id": "TXN-A-011", "chain_id": "CHAIN-006", "expected_label": "legitimate"},
    {"alert_transaction_id": "TXN-A-013", "chain_id": "CHAIN-007", "expected_label": "legitimate"},
    {"alert_transaction_id": "TXN-B-013", "chain_id": "CHAIN-008", "expected_label": "legitimate"},
    {"alert_transaction_id": "TXN-A-010", "chain_id": None,        "expected_label": "legitimate"},
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_examples() -> list[dict]:
    examples = []
    labels_by_chain = {row["chain_id"]: row for row in db.table("fraud_labels") if row.get("chain_id")}
    for s in SCENARIOS:
        label_row = labels_by_chain.get(s["chain_id"]) if s["chain_id"] else None
        examples.append({
            "inputs": {"alert_transaction_id": s["alert_transaction_id"]},
            "outputs": {
                "expected_label": s["expected_label"],
                "chain_id": s["chain_id"],
                "notes": label_row["notes"] if label_row else "single-bank payment",
            },
        })
    return examples


def create_dataset(client: Client, recreate: bool = False) -> object:
    if client.has_dataset(dataset_name=DATASET_NAME):
        if not recreate:
            print(f"Using existing dataset '{DATASET_NAME}' (pass recreate=True to rebuild)")
            return client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(
        DATASET_NAME,
        description="AgentSmith: fraud and legitimate scenarios with ground-truth labels.",
    )
    examples = _build_examples()
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples")
    return dataset


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def run_scenario(inputs: dict) -> dict:
    """Run the agent on one scenario and return its final state."""
    from agent import build_agent  # local import avoids circular issues at module level

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = build_agent(checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = agent.invoke({"alert_transaction_id": inputs["alert_transaction_id"]}, config)

    # Auto-answer any human_review interrupts (for eval purposes)
    while "__interrupt__" in result:
        interrupt_val = result["__interrupt__"][0].value
        agent_decision = interrupt_val.get("agent_decision", "investigate")
        result = agent.invoke(Command(resume={"decision": agent_decision, "notes": "automated eval"}), config)

    return {
        "decision": result.get("decision", "unknown"),
        "risk_score": result.get("risk_score", 0),
        "decided_by": result.get("decided_by", "agent"),
        "risk_reasons": result.get("risk_reasons", []),
    }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def detection_accuracy(outputs: dict, reference_outputs: dict) -> float:
    """1.0 if agent reached the correct investigate/clear decision, 0.0 otherwise."""
    expected = reference_outputs["expected_label"]
    decision = outputs.get("decision", "unknown")
    correct = (expected == "fraud" and decision == "investigate") or \
              (expected == "legitimate" and decision == "clear")
    return 1.0 if correct else 0.0


def risk_calibration(outputs: dict, reference_outputs: dict) -> float:
    """1.0 if risk score is in the right band: fraud ≥70, legitimate ≤30. Partial credit in between."""
    score = outputs.get("risk_score", 0)
    expected = reference_outputs["expected_label"]
    if expected == "fraud":
        return 1.0 if score >= 70 else score / 70
    else:
        return 1.0 if score <= 30 else max(0.0, (100 - score) / 70)


def human_routing(outputs: dict, reference_outputs: dict) -> float:
    """1.0 if human review was triggered exactly when it should be (fraud=human, legitimate=agent)."""
    expected = reference_outputs["expected_label"]
    decided_by = outputs.get("decided_by", "agent")
    correct = (expected == "fraud" and decided_by == "human") or \
              (expected == "legitimate" and decided_by == "agent")
    return 1.0 if correct else 0.0


EVALUATORS = [detection_accuracy, risk_calibration, human_routing]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def average_scores(results) -> dict:
    totals: dict[str, list[float]] = {}
    for r in results:
        for key, val in r.get("feedback", {}).items():
            totals.setdefault(key, []).append(val)
    return {k: sum(v) / len(v) for k, v in totals.items()}


def run_experiment(recreate_dataset: bool = False) -> dict:
    client = Client()
    create_dataset(client, recreate=recreate_dataset)

    results = evaluate(
        run_scenario,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,  # sequential to avoid checkpointer conflicts
    )

    scores = average_scores(results)
    detection = scores.get("detection_accuracy", 0.0)
    n = len(SCENARIOS)
    n_fraud = sum(1 for s in SCENARIOS if s["expected_label"] == "fraud")
    n_legit = n - n_fraud

    print(f"\n{'='*55}")
    print(f"  AgentSmith Eval Results ({n} scenarios: {n_fraud} fraud, {n_legit} legitimate)")
    print(f"{'='*55}")
    print(f"  Detection accuracy : {detection:.0%}  ({detection * n:.0f}/{n} correct)")
    print(f"  Risk calibration   : {scores.get('risk_calibration', 0):.0%}")
    print(f"  Human routing      : {scores.get('human_routing', 0):.0%}")
    print(f"{'='*55}")
    print(f"  View full results in LangSmith: https://smith.langchain.com")
    return scores


if __name__ == "__main__":
    import sys
    recreate = "--recreate" in sys.argv
    run_experiment(recreate_dataset=recreate)
