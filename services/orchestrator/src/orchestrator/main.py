"""
orchestrator — FastAPI entry point.

Phase 0: skeleton with /health only. Real /run endpoint arrives in Phase 3
when we wire up the LangGraph state machine and the downstream HTTP calls.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from orchestrator.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Phase 3: build the LangGraph once and reuse it across requests."""
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
        extras={"environment": settings.environment},
    )
