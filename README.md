# MarketPulse

**A self-evolving stock & crypto intelligence agent.** Ingests live market data, runs a custom Temporal Fusion Transformer for short-horizon price forecasts, has an LLM-powered Critic Agent grade each forecast's confidence against historically similar setups (hybrid RAG), publishes auto-generated intelligence briefs, and retriggers its own fine-tuning when prediction error drifts above threshold.

> It's not "I trained a model". It's "I built a system that trains, critiques, and retrains itself, with observability over every decision and a backtest that proves it on out-of-sample data."

[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Stack](https://img.shields.io/badge/stack-FastAPI%20%7C%20PyTorch%20%7C%20LangGraph%20%7C%20Next.js-orange)]()
[![License](https://img.shields.io/badge/license-personal--portfolio-lightgrey)]()

---

## Why I'm building this

Most "ML portfolio projects" stop at `model.fit() → 75% accuracy → Streamlit demo`. That's good for an internship; it doesn't move a senior-leaning interview. The interesting problems in production ML are **system problems**: model decay, calibration, debugging multi-agent pipelines, proving you didn't overfit on past data, and shipping the whole thing on a free-tier budget.

MarketPulse exists to engage with all of those at once.

## What it does (concretely)

Every hour, an AWS EventBridge cron fires a Lambda. The Lambda runs an orchestrator built on **LangGraph** that walks five nodes in sequence:

1. **Data Agent** pulls OHLCV from yFinance + financial news from Alpaca, joins them, computes features, lands Parquet on S3.
2. **Forecast Agent** runs a **Temporal Fusion Transformer** to predict the next 4 hours of price movement with a 10th/90th-percentile confidence interval.
3. **Critic Agent** uses **hybrid RAG** (Qdrant semantic search + BM25 + cross-encoder reranker) to find the three most historically similar setups in the last 5 years, then a fine-tuned **Phi-3-mini** regime classifier + Groq's Llama-3.3-70B grade the forecast's confidence as HIGH / MEDIUM / LOW.
4. **Report Agent** composes a markdown intelligence brief and writes it to S3.
5. **Drift Check** computes a rolling 7-day MAE; if it crosses threshold, fires off a retraining job — closing the loop.

A Next.js 15 dashboard on Vercel shows the latest forecast, regime, confidence, and a full agent trace (via self-hosted **Langfuse**). A walk-forward **backtesting module** replays the entire pipeline on the last 6 months of unseen data and reports Sharpe ratio + max drawdown.

## Architecture

```
                              ┌───────────────────────┐
                              │  AWS EventBridge      │  every hour
                              │  (cron scheduler)     │
                              └────────────┬──────────┘
                                           ▼
                              ┌───────────────────────┐
                              │   AWS Lambda          │
                              │   handler.py (Mangum) │
                              └────────────┬──────────┘
                                           ▼
                              ┌───────────────────────┐
                              │  Orchestrator         │  LangGraph state machine
                              │  (FastAPI)            │
                              └────────────┬──────────┘
                                           │
       ┌───────────────────────┬───────────┼──────────────────┬──────────────────────┐
       ▼                       ▼           ▼                  ▼                      ▼
┌─────────────┐         ┌───────────┐ ┌─────────┐      ┌──────────┐         ┌──────────────┐
│ data_ingest │         │ forecast  │ │ critic  │      │  report  │         │ retrigger    │
│  /fetch     │ ──────► │ /predict  │►│/critique│ ────►│/generate │ ──────► │ (drift check)│
│             │         │ TFT       │ │ Phi-3 + │      │ Groq +   │         │              │
│ yFinance +  │         │ (PyTorch) │ │ RAG +   │      │ Jinja    │         │ Retrain Y/N? │
│ Alpaca News │         │           │ │ Groq    │      │          │         │              │
└─────────────┘         └───────────┘ └─────────┘      └──────────┘         └──────────────┘
       │                      │            │                │                      │
       ▼                      ▼            ▼                ▼                      ▼
   Parquet on             TFT ckpt    Qdrant +        Markdown brief        MLflow drift
   S3 (Hive-              on S3       BM25 +          on S3                 metrics
   partitioned)                       reranker

                                           │
                                           ▼
                              ┌───────────────────────┐
                              │  Next.js 15 dashboard │  (Vercel)
                              │  Live forecast,       │
                              │  confidence, traces   │
                              └───────────────────────┘

Throughout:  Langfuse traces every LLM call.  MLflow on DagsHub tracks every model.
```

## Tech stack

| Layer | Tool | Why this and not the obvious alternative |
|---|---|---|
| Package mgmt | **uv 0.11** + workspaces | 10–100× faster than pip; one lockfile across 7 services; per-service install isolation for Docker |
| API framework | **FastAPI** | Type-safe HTTP from Pydantic; OpenAPI for free; async |
| Time-series model | **Temporal Fusion Transformer** (PyTorch Lightning) | Interpretable attention weights — can defend "why this prediction" in a way Chronos/TimesFM can't |
| LLM (judgment + report) | **Groq** Llama-3.3-70B | Free tier 30 RPM / 14.4K req/day; sub-second latency |
| LLM fine-tune | **Phi-3-mini + QLoRA** on Colab T4 | 3.8B params fits a T4; matches Llama-3-8B on classification |
| Retrieval | **Qdrant** (semantic) + **rank-bm25** (lexical) + **BAAI/bge-reranker-base** | Hybrid beats pure-semantic on ticker/event-heavy queries |
| Observability | **Langfuse** (self-hosted) | Free; debug multi-agent LLM chains |
| Experiment tracking | **MLflow on DagsHub** | 10 GB free; remote tracking server |
| Storage | **Parquet on S3** (Hive-partitioned) | Columnar, cheap, scans by date partition |
| Frontend | **Next.js 15** App Router | 2026 React standard; free on Vercel Hobby |
| Compute | **AWS Lambda** + **EventBridge** | 1M req/mo free; perfect for hourly cron |
| CI/CD | **GitHub Actions** | Auto test → ECR → Lambda on push |

Every choice has a one-sentence justification I can defend in an interview. See [`docs/adr/`](docs/) for the longer Architecture Decision Records.

## Local development

You need **uv ≥ 0.11**, **Docker ≥ 24**, **Python 3.12**, **Node ≥ 20** (frontend, later).

```bash
# Clone
git clone https://github.com/AyushTripathi1010/MarketPulse.git && cd MarketPulse

# Install all Python deps into a project-local .venv (uv workspace)
uv sync --all-packages

# Run the test suite
uv run pytest -q
# expect: 37 passed in ~13s

# Spin up the full local stack (services + Qdrant + Redis)
docker build -f infra/docker/python-base.Dockerfile -t marketplus/python-base:0.1 .
docker compose up -d

# Health check every service
make health
```

For the live data pipeline you need an Alpaca paper-trading account (free) for news; OHLCV via yFinance needs no key.

```bash
cp .env.example .env
# edit .env — add ALPACA_API_KEY, ALPACA_API_SECRET, GROQ_API_KEY, HF_TOKEN

# Trigger an ingest manually:
curl -X POST http://localhost:8001/fetch \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "lookback_days": 30, "interval": "1h"}'
```

## Repository layout

```
marketplus/
├── pyproject.toml           # uv workspace root
├── docker-compose.yml       # local dev: 7 services + qdrant + redis
├── services/
│   ├── data_ingest/         # yFinance + Alpaca → pandas → Parquet
│   ├── forecast/            # TFT inference + training trigger
│   ├── critic/              # Phi-3 regime + hybrid RAG + Groq judgment
│   ├── report/              # markdown intelligence brief generator
│   ├── orchestrator/        # LangGraph state machine — the brain
│   ├── backtest/            # walk-forward replay + Sharpe / drawdown
│   └── api_gateway/         # public FastAPI for the Next.js frontend
├── shared/                  # Pydantic models used by every service
├── frontend/                # Next.js 15 dashboard (Phase 8)
├── notebooks/               # EDA + Colab QLoRA fine-tune notebook
├── infra/
│   ├── docker/              # python-base image for fast service builds
│   ├── lambda/              # SAM template + handler
│   └── langfuse/            # self-host docker-compose for observability
├── .github/workflows/       # CI/CD (Phase 9)
├── docs/                    # ADRs, architecture, command log
└── tests/                   # cross-service integration tests
```

## Status & roadmap

This is built **incrementally over 8–10 weeks**, one phase per checkpoint. Commit history reflects that — each phase ends with a feature commit and a tagged release once tests are green.

| Phase | Focus | Status |
|---|---|---|
| 0 | Monorepo scaffolding (7 FastAPI services + Docker + tests) | ✅ done |
| 1 | Data layer (yFinance + Alpaca + pandas transforms + Parquet) | ✅ done |
| 2 | TFT forecast model + MLflow tracking | ✅ done |
| 3 | LangGraph orchestrator + Langfuse tracing | ✅ done |
| 4 | Phi-3 QLoRA fine-tune for regime classification | ✅ done |
| 5 | Hybrid RAG (Qdrant + BM25 + RRF + reranker) for the Critic agent | ✅ done |
| 6 | Walk-forward backtesting module (Sharpe + drawdown + hit rate) | ✅ done — **181 tests passing** |
| 7 | AWS Lambda + EventBridge deployment | ⏳ next |
| 8 | Next.js 15 dashboard on Vercel | ⏳ |
| 9 | GitHub Actions CI/CD | ⏳ |

The git log is the receipt — daily commits show this is hand-built, phase by phase, not lifted from a template.

## Design principles I'm holding myself to

- **No tech for the sake of tech.** Every dependency answers "what problem in this project does it solve that we can't solve with what we have?" — see [docs/adr/](docs/) for the receipts.
- **Storage by access pattern, not by reflex.** Time-series → Parquet on S3 (columnar, cheap). Embeddings → Qdrant. Cache → Redis (only if needed). No Postgres, no SQL — none of this is transactional.
- **Layered I/O.** Network calls live ONLY in `fetchers/`. Pure transforms in `transforms/`. Persistence in `storage/`. Each layer is independently testable.
- **Fail-soft on optional inputs, fail-loud on required ones.** No Alpaca key → news is empty, pipeline runs. yFinance down → 502 propagated, orchestrator can degrade gracefully.
- **Tests are mocked at API boundaries.** Real-network calls are reserved for explicit smoke tests, never CI.

## License

Personal portfolio project. Not for commercial use.
