"""
orchestrator — FastAPI entry point.

Phase 3: real /run endpoint that drives the full LangGraph pipeline.
The compiled graph is built once at lifespan startup and reused across
every request — graph compilation is cheap but not free, and a long-lived
service should pay it once.

Langfuse tracing is opt-in via env vars. Without it the graph still runs;
you just don't get a pretty UI showing the timeline.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse

from orchestrator.config import settings
from orchestrator.graph import build_graph
from orchestrator.schemas import RunRequest, RunResponse
from orchestrator.tracing import init_langfuse, is_enabled, new_trace_id, trace_run

logger = logging.getLogger(__name__)

# Module-level compiled graph — built once in lifespan.
_graph: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the graph once + configure Langfuse if creds are available."""
    global _graph
    init_langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    _graph = build_graph()
    logger.info(
        "orchestrator ready. langfuse=%s graph_nodes=%s",
        is_enabled(),
        list(_graph.nodes.keys()),
    )
    yield


app = FastAPI(
    title="MarketPulse — Orchestrator",
    version="0.1.0",
    description="LangGraph state machine: data → forecast → critic → report → drift-check.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="orchestrator",
        extras={
            "environment": settings.environment,
            "graph_ready": _graph is not None,
            "langfuse_enabled": is_enabled(),
        },
    )


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    """Execute the full pipeline for one ticker.

    Returns the trace id (which is also the run id) so the caller can
    look up the run in Langfuse or stream the markdown report later.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized.")

    trace_id = new_trace_id()
    initial_state = {
        "ticker": req.ticker,
        "horizon_hours": req.horizon_hours,
        "trace_id": trace_id,
    }

    with trace_run(trace_id, req.ticker, req.horizon_hours):
        # LangGraph's invoke runs every node synchronously. For long-running
        # branches we'd use ainvoke, but our nodes are sub-second HTTP calls.
        final_state = _graph.invoke(initial_state)

    return RunResponse(
        ticker=req.ticker,
        trace_id=trace_id,
        report_markdown=final_state.get("report_markdown", ""),
        drift_detected=bool(final_state.get("drift_detected", False)),
    )
