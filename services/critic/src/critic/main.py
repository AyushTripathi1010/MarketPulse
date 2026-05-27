"""
critic — FastAPI entry point.

Phase 3: /critique endpoint that grades a Forecast's confidence using a
Groq LLM call. Falls back to a stub response when no API key is configured,
so the orchestrator pipeline runs end-to-end on a fresh checkout.

Phase 4 will add a fine-tuned Phi-3 regime classifier as a separate input
to the prompt. Phase 5 will add hybrid-RAG retrieval of historical analogues.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse

from critic.config import settings
from critic.judge.groq_critic import CriticLLMError, judge, stub_critique
from critic.schemas import CritiqueRequest, CritiqueResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log whether the LLM is configured at startup — fail-fast visibility."""
    if not settings.groq_api_key:
        logger.warning(
            "GROQ_API_KEY not set; /critique will return stub responses."
        )
    yield


app = FastAPI(
    title="MarketPulse — Critic",
    version="0.1.0",
    description="Critic Agent: regime classifier + hybrid RAG + Groq judgment.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="critic",
        extras={
            "environment": settings.environment,
            "llm_configured": bool(settings.groq_api_key),
        },
    )


@app.post("/critique", response_model=CritiqueResponse)
def critique(req: CritiqueRequest) -> CritiqueResponse:
    """Grade a Forecast. Returns Critique with regime + confidence + reasoning."""
    if not settings.groq_api_key:
        # Stub mode — orchestrator can still complete the pipeline.
        return stub_critique(req.forecast)

    try:
        return judge(
            req.forecast,
            req.recent_news,
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
    except CriticLLMError as e:
        # Surface as 502 — same as data_ingest does for yFinance. The
        # orchestrator catches it and degrades gracefully.
        raise HTTPException(
            status_code=502,
            detail=f"Critic LLM error: {e}",
        ) from e
