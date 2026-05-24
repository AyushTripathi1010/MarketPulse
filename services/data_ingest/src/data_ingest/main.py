"""
data_ingest — FastAPI entry point.

Phase 0: skeleton with /health only. Real /fetch and /backfill endpoints
arrive in Phase 1, when we wire in yFinance + Alpaca + PySpark + S3.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from data_ingest.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks.

    Phase 1 will use this to warm up the boto3 client and validate credentials.
    """
    yield


app = FastAPI(
    title="MarketPulse — Data Ingest",
    version="0.1.0",
    description="Pulls OHLCV (yFinance) + news (Alpaca), lands Parquet on S3.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe used by docker-compose, Lambda, and Make."""
    return HealthResponse(
        service="data_ingest",
        extras={"environment": settings.environment},
    )
