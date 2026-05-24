"""
critic — FastAPI entry point.

Phase 0: skeleton with /health only. Real /critique endpoint arrives in
Phase 4 (Phi-3 fine-tune) and Phase 5 (hybrid RAG).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from critic.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Phase 4: warm up the Phi-3 model + Qdrant client here."""
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
        extras={"environment": settings.environment},
    )
