"""Evaluation-driven improvement loop for AgentSmith.

  1. Score every labelled alert with the current rules and collect the misses (false negatives) and the legitimate
     chains that must stay clear.
  2. Ask an LLM to propose up to 3 declarative rules (see utils/rules.py) that would catch the misses.
  3. Re-score the whole dataset with the candidate rules, as a LangSmith experiment.
  4. Accept the candidate only if it catches more fraud chains AND introduces no false positives. Otherwise feed
     the result back to the LLM and try again (up to --max-iterations).
  5. Write accepted rules to data/learned_rules.json. Redeploy to ship them.

The LLM never writes code: it fills in a typed rule schema, and every proposal is checked by the evaluators before
it can change the agent. Scoring here uses enrichment="basic" so each iteration is deterministic and quick.

    uv run python -m evals.improve [--max-iterations 3] [--dry-run]
"""

import argparse
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client

from evals.run_eval import DATASET, build_examples, compute_metrics, ensure_dataset, run_experiment
from tools.network import find_chain, find_transaction
from utils import PROJECT_ROOT
from utils.models import model
from utils.nodes import _basic_enrichments, _fetch_bank_data, compute_risk
from utils.rules import RULES_PATH, RuleSet, save_learned_rules

load_dotenv(PROJECT_ROOT.parent / ".env")
THRESHOLD = 70  # score at which the agent routes an alert to a human analyst


def score_alert(txn_id: str, rules: list[dict]) -> dict:
    """What the agent decides for one alert (collective view, deterministic enrichment)."""
    txn = find_transaction(txn_id)
    legs = find_chain(txn["chain_id"]) if txn.get("chain_id") else [txn]
    enrichments = _basic_enrichments(_fetch_bank_data(legs), legs)
    score, reasons = compute_risk(legs, enrichments, txn.get("chain_id"), rules)
    return {"chain_id": txn.get("chain_id"), "risk_score": score, "risk_reasons": reasons, "flagged": score >= THRESHOLD}


def make_target(rules: list[dict]):
    return lambda inputs: score_alert(inputs["alert_transaction_id"], rules)


def evaluate_rules(rules: list[dict]) -> dict:
    pairs = [(score_alert(e["inputs"]["alert_transaction_id"], rules), e["outputs"]) for e in build_examples()]
    return compute_metrics(pairs)


def chain_summary(chain_id: str | None, txn_id: str) -> dict:
    """A compact, LLM-readable view of one chain: its legs relative to the first one."""
    legs = find_chain(chain_id) if chain_id else [find_transaction(txn_id)]
    legs = sorted(legs, key=lambda l: l["timestamp"])
    t0 = datetime.fromisoformat(legs[0]["timestamp"])
    return {
        "chain_id": chain_id,
        "legs": [
            f"t+{int((datetime.fromisoformat(l['timestamp']) - t0).total_seconds())}s {l['account_id']} "
            f"{l['type']} £{l['amount']:,.0f} (counterparty {l.get('counterparty_account')})"
            for l in legs
        ],
    }


def failure_report(rules: list[dict]) -> dict:
    """Missed fraud chains and the legitimate chains that must not be flagged, with current scores."""
    missed, legit, seen = [], [], set()
    for ex in build_examples():
        txn_id, ref = ex["inputs"]["alert_transaction_id"], ex["outputs"]
        key = ref["chain_id"] or txn_id
        if key in seen:
            continue
        seen.add(key)
        result = score_alert(txn_id, rules)
        entry = {**chain_summary(ref["chain_id"], txn_id), "score": result["risk_score"], "pattern": ref["pattern"]}
        if ref["label"] == "fraud" and not result["flagged"]:
            missed.append(entry)
        elif ref["label"] != "fraud":
            legit.append(entry)
    return {"missed_fraud_chains": missed, "legitimate_chains_must_stay_below_threshold": legit}


PROMPT = """\
You improve a fraud-detection scorer for a multi-bank payment network. An alert is routed to a human analyst when
its risk score reaches {threshold}; below that it is auto-cleared. The current rules miss the fraud chains below.

Propose up to 3 rules using ONLY this schema (rule_type, min_legs, window_seconds, min_ratio, points, rationale):
- fan_out: one account sends `min_legs` transfers within `window_seconds`
- fan_in: one account receives `min_legs` transfers within `window_seconds`
- split_forward: one account receives money, then within `window_seconds` forwards it in `min_legs`+ transfers
  totalling at least `min_ratio` of the amount received
`points` is added to the chain's current score (shown below), so choose it to lift missed fraud chains to at least
{threshold} while keeping every legitimate chain below {threshold}. Prefer the fewest, most specific rules. Legitimate
payroll, supplier and property payments also move money to several accounts, so thresholds must separate them.

Current state:
{report}
{feedback}"""


def propose_rules(report: dict, feedback: str) -> list[dict]:
    llm = init_chat_model(model).with_structured_output(RuleSet)
    proposal = llm.invoke(
        PROMPT.format(threshold=THRESHOLD, report=json.dumps(report, indent=2), feedback=feedback)
    )
    return [r.model_dump() for r in proposal.rules]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Do not write data/learned_rules.json")
    args = parser.parse_args()

    client = Client()
    ensure_dataset(client)

    best_rules: list[dict] = []  # start from the built-in rules only
    best = evaluate_rules(best_rules)
    print(f"Baseline: {best['chains_caught']}/{best['fraud_chains']} fraud chains caught, "
          f"{best['false_negatives']} false negatives, {best['false_positives']} false positives")

    feedback = ""
    for i in range(1, args.max_iterations + 1):
        candidate = propose_rules(failure_report(best_rules), feedback)
        metrics = run_experiment(
            make_target(candidate),
            f"agentsmith-improve-iter{i}",
            {"iteration": i, "candidate_rules": candidate, "dataset": DATASET},
            client,
        )
        print(f"Iteration {i}: {metrics['chains_caught']}/{metrics['fraud_chains']} chains, "
              f"FN={metrics['false_negatives']} FP={metrics['false_positives']}  [{metrics['experiment']}]")
        for rule in candidate:
            print("   -", rule["rule_type"], f"min_legs={rule['min_legs']} window={rule['window_seconds']}s "
                  f"min_ratio={rule['min_ratio']} points={rule['points']}: {rule['rationale']}")

        if metrics["false_positives"] == 0 and metrics["chains_caught"] > best["chains_caught"]:
            best, best_rules = metrics, candidate
            if best["chains_caught"] == best["fraud_chains"]:
                break
            feedback = "Accepted so far; propose rules that ALSO catch the remaining missed chains."
        else:
            feedback = (
                f"Your previous proposal was rejected: it caught {metrics['chains_caught']} of "
                f"{metrics['fraud_chains']} fraud chains with {metrics['false_positives']} false positives "
                f"(it must have 0 false positives and catch more than the current {best['chains_caught']}). "
                f"Proposal was: {json.dumps(candidate)}"
            )

    print(f"Result: {best['chains_caught']}/{best['fraud_chains']} fraud chains caught, "
          f"{best['false_negatives']} false negatives, {best['false_positives']} false positives, "
          f"{len(best_rules)} learned rule(s)")
    if args.dry_run or not best_rules:
        print("Not writing rules." if best_rules == [] else "Dry run: rules not written.")
        return
    save_learned_rules(
        best_rules,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": DATASET,
            "chains_caught": f"{best['chains_caught']}/{best['fraud_chains']}",
            "false_positives": best["false_positives"],
        },
    )
    print(f"Wrote {RULES_PATH}")


if __name__ == "__main__":
    main()
