"""
report — FastAPI entry point.

Phase 3: /generate endpoint that renders a Jinja markdown brief from a
Forecast + Critique, optionally polishes the prose with Groq, and returns
the markdown inline (Phase 7 will add S3-write).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from report.config import settings
from report.generator.render import generate_brief
from report.schemas import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log polish status — visibility into degraded-mode runs."""
    if not settings.groq_api_key:
        logger.warning(
            "GROQ_API_KEY not set; /generate will return deterministic briefs only."
        )
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
        extras={
            "environment": settings.environment,
            "llm_configured": bool(settings.groq_api_key),
        },
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Compose a markdown brief.

    Always returns 200 with markdown — graceful degradation through the
    whole stack means we never fail the orchestrator on Groq issues.
    """
    markdown = generate_brief(
        req.forecast,
        req.critique,
        groq_api_key=settings.groq_api_key or None,
        groq_model=settings.groq_model,
    )
    return GenerateResponse(
        ticker=req.forecast.ticker,
        generated_at=datetime.now(UTC),
        markdown=markdown,
        polished=bool(settings.groq_api_key),
    )
