"""
report — FastAPI entry point.

Phase 0: skeleton with /health only. Real /generate endpoint arrives in
Phase 3 alongside Jinja templates + Groq integration.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from report.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Phase 3: instantiate Groq client and load Jinja env here."""
    yield


app = FastAPI(
    title="MarketPulse — Report",
    version="0.1.0",
    description="Generates markdown intelligence briefs from Forecast + Critique.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="report",
        extras={"environment": settings.environment},
    )
