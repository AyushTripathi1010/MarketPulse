"""
backtest — FastAPI entry point.

Phase 0: skeleton with /health only. Real /run endpoint arrives in Phase 6
when we wire up the walk-forward replay engine.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from backtest.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Phase 6: warm up the historical-data S3 client here."""
    yield


app = FastAPI(
    title="MarketPulse — Backtest",
    version="0.1.0",
    description="Walk-forward backtester (Sharpe / max drawdown / hit rate).",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="backtest",
        extras={"environment": settings.environment},
    )
