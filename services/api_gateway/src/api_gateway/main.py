"""
api_gateway — FastAPI entry point.

Phase 0: skeleton with /health only. Real proxy endpoints (forecast, report,
backtest, traces) arrive in Phase 7 when the frontend needs them.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from marketplus_shared.models import HealthResponse

from api_gateway.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Phase 7: create an httpx.AsyncClient pool here."""
    yield


app = FastAPI(
    title="MarketPulse — API Gateway",
    version="0.1.0",
    description="Public API gateway; proxies authenticated requests to internal services.",
    lifespan=lifespan,
)

# CORS so the Next.js frontend (port 3000) can hit us from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="api_gateway",
        extras={"environment": settings.environment},
    )
