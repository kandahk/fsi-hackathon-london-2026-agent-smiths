# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Three independent LangGraph/LangChain agent workshops for a financial services hackathon. Each agent lives in its own subdirectory with its own `pyproject.toml`, but all share a single `.venv` via a UV workspace at the root.

| Agent | Directory | Pattern |
|-------|-----------|---------|
| KYC Onboarding | `01_kyc_agent/` | LangGraph StateGraph with human-in-the-loop |
| Transaction Monitoring | `02_transaction_monitoring_agent/` | LangChain agent with memory, confidence scoring, LangSmith evals |
| Document Summary | `03_document_summary_agent/` | LangChain deep agent with map-reduce sub-agents |

## Setup

```bash
# Install all dependencies (creates single .venv for all three agents)
uv sync --all-packages

# Copy and fill in environment variables
cp .env.example .env
```

## Common Commands

```bash
# Run any agent in LangGraph Studio
cd 01_kyc_agent && uv run langgraph dev
cd 02_transaction_monitoring_agent && uv run langgraph dev
cd 03_document_summary_agent && uv run langgraph dev

# Run transaction monitoring agent directly
cd 02_transaction_monitoring_agent && uv run python agent.py

# Run transaction monitoring evals (Module 2)
cd 02_transaction_monitoring_agent && uv run python -m evals.scoring_evals

# Launch notebooks
cd 01_kyc_agent && uv run jupyter notebook
```

## Architecture

### Shared Infrastructure

- **Model access**: All agents route through the LangSmith LLM Gateway. The `MODEL_NAME` env var controls which model (`provider:model` format, default `openai:gpt-6-luna`).
- **Data layer**: All data is local JSONL files under each agent's `data/` directory. Access via `utils/db.py` — `db.table()`, `db.query()`, `db.get()`.
- **Environment**: Single `.env` at root; sets `LANGSMITH_API_KEY`, `LC_GATEWAY_KEY`, `OPENAI_BASE_URL` (gateway URL), `ATTENDEE_NAME`, and `MODEL_NAME`.

### Agent 1: KYC Agent

LangGraph `StateGraph` with `AgentState` dataclass. Flow: `extract_profile → screen_customer → assess_risk → route_on_risk → (human_review | auto_approve) → write_decision`. Uses `interrupt()` for high-risk cases. Tools read from JSONL: `get_application`, `check_sanctions`, `check_pep`.

### Agent 2: Transaction Monitoring Agent

LangChain agent with checkpointer-based memory and a `MyAgentContext` (user_id, confidence thresholds, grey-zone scoring). Confidence scores determine whether to auto-decide or interrupt for human review. Module 2 adds LangSmith scoring evals against analyst labels; Module 3 adds an annotation queue. Tools: `get_customer_profile`, `get_transaction`, `record_decision`.

### Agent 3: Document Summary Agent

Main deep agent orchestrates a `section_summariser` sub-agent in parallel (map-reduce). Input is document ID + firm ID. Output is `SectionSummary` Pydantic objects per section plus a compliance brief. Tools: `get_document`, `get_firm_profile`, `get_third_party_provider`. Structured outputs enforce quote and deadline validation against source text.

### Directory Layout (per agent)

```
<agent>/
├── agent.py              # Entry point / graph definition
├── tools/                # LangChain tools (data access)
├── utils/                # db.py, shared models, helpers
├── modules/              # Jupyter notebooks (teaching material)
├── data/                 # JSONL data files
├── instructions.md       # System prompt for the main agent
├── sub_agent_instructions.md  # System prompt for sub-agents (agent 3)
├── langgraph.json        # LangGraph Studio registration (Python 3.13)
└── pyproject.toml        # Per-agent dependencies
```

## Key Dependencies

- `langgraph>=1.0,<2` — graph workflows
- `langchain>=1.0,<2` — agent framework
- `deepagents / managed-deepagents` — sub-agent orchestration (agent 3)
- `langsmith>=0.4,<1` — tracing and evals
- `pydantic>=2,<3` — structured outputs
- `python-dotenv` — `.env` loading
