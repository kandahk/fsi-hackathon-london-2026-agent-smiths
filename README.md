# AgentSmith: collective fraud intelligence across banks

**Team: Agent Smiths** · FSI Hackathon London 2026 · built with LangChain, LangGraph, Deep Agents and LangSmith

Money launderers move funds in circles, or split them into pieces, across several banks, so that each bank only sees
a harmless-looking fragment. AgentSmith looks at all the banks at once, scores the whole payment chain, and puts
the suspicious ones in front of a human analyst, who decides.

## The use case

**Problem.** Circular payments (A to B to C and back to A) and layering (a sum split up and moved on quickly) are
only visible if you can see every leg. Each bank, working alone, sees one or two legs and nothing looks wrong.

**User.** The fraud investigator at the receiving bank who gets the alert and decides *investigate* or *clear*.

**What AgentSmith does for them.** Given one alert transaction, it finds the full chain across Banks A, B and C,
asks one analyst sub-agent per bank to assess the accounts, scores the risk, and for anything at or above 70 posts
a Slack alert and pauses until the investigator decides. Everything else is cleared automatically with an audit
memo. Every analyst decision is kept, so the system can learn from it.

## How it works

```
START -> load_trigger -> detect_chain -> enrich_banks -> score_risk
              (Deep Agent: bank-a / bank-b / bank-c analysts)  |
                    score >= 70 -> notify_analyst -> human_review --+
                    score <  70 -> auto_clear ----------------------+-> write_audit -> END
```

| Piece | What it does |
|---|---|
| **Collective view** | `detect_chain` follows a chain across all three banks. A `single_bank` mode reproduces what one bank sees today, as the baseline. |
| **Deep Agent analysts** | `enrich_banks` runs one analyst sub-agent per bank in parallel and returns risk flags (`new_account`, `foreign_national`, `unusual_amount`). Deterministic profile checks are merged in as a floor the model cannot drop. |
| **Transparent scoring** | Built-in rules (circular movement, amount decrement, new accounts, foreign holders, unusual amounts) plus **learned rules**. A rule never executes generated code; it is a typed, versioned entry in `data/learned_rules.json`. |
| **Evaluation-driven improvement** | `evals/improve.py` finds the fraud the agent missed, has an LLM propose rules in a fixed schema, scores each candidate as a LangSmith experiment, and accepts it only if it catches more fraud with zero false positives and still holds when its thresholds are loosened. |
| **Human in the loop** | Flagged chains pause at `human_review` (LangGraph `interrupt`). The analyst resumes with a decision and notes. |
| **Slack alert** | `notify_analyst` posts the evidence, the thread ID and how to resume. De-duplicated per chain, and a Slack failure never blocks the investigation. |
| **Memory and flywheel** | Decisions are stored so accounts tied to past confirmed fraud score higher, and confirmed labels are appended for the next `improve.py` run. |
| **Privacy** | All data is synthetic and pseudonymous (`SUBJECT-*` references, no names). |

## Results

All numbers are from the 58-alert synthetic dataset (7 fraud chains, 43 fraud alerts, 15 legitimate alerts). A fraud
chain counts as caught if any of its alerts is routed to a human. Recall is over fraud alerts.

| Configuration | Fraud chains caught | Recall | False positives |
|---|---|---|---|
| One bank only, original rules | 0/7 | 0% | 0 |
| One bank only, learned rules | 2/7 | 12% | 0 |
| All three banks, original rules | 3/7 | 42% | 0 |
| **All three banks, learned rules** | **5/7** | **72%** | **0** |

These are from the **deployed agent with the real LLM bank analysts**, as LangSmith experiments named
`agentsmith-v4-remote-*` on the `agentsmith-fraud-alerts` dataset. A faster, reproducible version runs the local graph
with deterministic enrichment (`--enrichment basic`); it gives the same result except one bank with learned rules, which
catches 1/7 chains there, because the LLM analysts occasionally flag more.

**What the numbers say.** Seeing all three banks takes recall from 0% to 42% with the original rules, and the learned
rules take it to 72%. Two fraud chains (010 and 011, structuring and pass-through between established domestic
accounts) are still missed: they carry no risk flag and look just like a legitimate property completion (chain 009),
so no rule over the current features separates them. The evaluation reports that miss rather than hiding it.

**Caveats.** There are only seven fraud chains and the improvement loop saw all of them, so treat the figures as a
demonstration of the method, not a measured detection rate. Add held-out chains before relying on them.

## Run it

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --package agentsmith          # install into .venv
cp .env.example .env                  # then fill in your LangSmith key, workspace and (optionally) Slack
cd 04_agentsmith
uv run langgraph dev                  # run the agent locally in LangGraph Studio
```

| Task | Command (from `04_agentsmith`) |
|---|---|
| Walk through everything | open `modules/01_agentsmith.ipynb` |
| Publish the dataset to LangSmith | `uv run python -m evals.run_eval --dataset-only` |
| Evaluate (local, fast) | `uv run python -m evals.run_eval --target local --enrichment basic` |
| Evaluate the deployed agent | `uv run python -m evals.run_eval --target remote --url <deployment-url>` |
| Compare configurations | add `--visibility single_bank\|collective` and `--rules baseline\|learned` |
| Run the improvement loop | `uv run python -m evals.improve` (add `--dry-run` to preview) |
| Deploy to LangSmith | `uvx --from langgraph-cli langgraph deploy --name agentsmith --remote` |

Evaluations pass `notify: false`, so they never post to Slack.

### Slack alerts (optional)

Set **one** of these in `.env`. With neither, the alert step is skipped.

| Variable | Meaning |
|---|---|
| `SLACK_BOT_TOKEN` + `SLACK_CHANNEL` | bot token (scope `chat:write`) and channel ID; the bot must be invited to the channel |
| `SLACK_WEBHOOK_URL` | alternatively, an incoming-webhook URL |
| `ALERT_DEDUPE_HOURS` | do not alert the same chain again within this window (default 24, `0` = off) |

The deployment reads `.env` at deploy time, so redeploy after changing it. Never commit `.env`.

## Layout

```
04_agentsmith/
  agent.py            the LangGraph graph
  sub_agents.py       the three bank analyst sub-agents
  utils/nodes.py      graph nodes and the scoring rules
  utils/rules.py      learned-rule schema and matcher
  utils/notify.py     Slack alert
  tools/              cross-bank lookups
  data/               synthetic transactions, profiles, labels, learned_rules.json
  evals/run_eval.py   dataset, evaluators and experiments
  evals/improve.py    the improvement loop
  modules/            walkthrough notebook
```

## Limitations and next steps

- The flag-free fraud chains (010, 011) need a new signal, for example several transfers just under the £10,000
  reporting threshold, added to the rule schema.
- The flywheel appends labels to a file inside the deployment container, which is lost on restart; deployed, it needs
  a persistent store.
- Held-out chains are needed to measure generalisation.
- Each transaction in a chain can raise its own alert; the Slack alert is de-duplicated, but each alert still creates
  its own paused thread.

## Team: Agent Smiths

| Member | Organisation | Email |
|---|---|---|
| Abhishek Kandivalikar | NatWest | abhishek.kandivalikar@natwest.com |
| Reena Clifton | NatWest Boxed | reena.clifton@nwboxed.com |
| Olamide Fakorede | NatWest | abdulafeez.fakorede@natwest.com | 
| Kaarthick Krishnamurthy | NatWest | Kaarthick.Krishnamurthy@natwest.com |
