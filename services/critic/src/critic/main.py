"""
critic — FastAPI entry point.

Phase 3: /critique endpoint that grades a Forecast's confidence using a
Groq LLM call. Phase 4: optionally calls the fine-tuned Phi-3 regime
classifier on HF Inference API and feeds its label into the Groq prompt.

Either credential is optional:
  - No GROQ_API_KEY → stub Critique with confidence=UNKNOWN
  - No HF_TOKEN → skip the Phi-3 step; Groq judges on its own (Phase 3 behavior)
  - Both present → Phase 4 flow: Phi-3 classifies → Groq judges with hint

Phase 5 will add hybrid-RAG retrieval of historical analogues.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse, Regime

from critic.classifier.phi3_regime import (
    MarketSnapshot,
    RegimeClassifierError,
    classify as classify_regime,
)
from critic.config import settings
from critic.judge.groq_critic import CriticLLMError, judge, stub_critique
from critic.schemas import CritiqueRequest, CritiqueResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Visibility into which features are active."""
    logger.info(
        "critic starting. groq=%s phi3=%s phi3_model=%s",
        bool(settings.groq_api_key),
        bool(settings.hf_token),
        settings.phi3_model_id,
    )
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set; /critique will return stub responses.")
    if not settings.hf_token:
        logger.warning("HF_TOKEN not set; Phi-3 regime classifier disabled.")
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
            "phi3_classifier_configured": bool(settings.hf_token),
        },
    )


def _maybe_classify_regime(snapshot_dict: dict[str, float] | None) -> Regime | None:
    """Run the Phi-3 classifier when both creds and snapshot are present.

    Returns None when the classifier is disabled — the Groq judge will then
    operate on forecast + news alone (the Phase 3 path).
    """
    if not settings.hf_token or snapshot_dict is None:
        return None

    try:
        snapshot = MarketSnapshot(
            ticker=str(snapshot_dict.get("ticker", "")),
            return_5d=float(snapshot_dict.get("return_5d", 0.0)),
            return_20d=float(snapshot_dict.get("return_20d", 0.0)),
            vol_20d=float(snapshot_dict.get("vol_20d", 0.0)),
            rsi_14=float(snapshot_dict.get("rsi_14", 50.0)),
            sentiment_24h=float(snapshot_dict.get("sentiment_24h", 0.0)),
        )
    except (TypeError, ValueError) as e:
        logger.warning("Phi-3: bad snapshot fields, skipping classifier: %s", e)
        return None

    try:
        return classify_regime(
            snapshot,
            hf_api_key=settings.hf_token,
            model_id=settings.phi3_model_id,
        )
    except RegimeClassifierError as e:
        # Classifier failure shouldn't block the critic — log loudly and
        # continue with Groq alone. The judge will still produce a
        # Critique; just without the Phi-3 hint.
        logger.warning("Phi-3 classifier unavailable, continuing without: %s", e)
        return None


@app.post("/critique", response_model=CritiqueResponse)
def critique(req: CritiqueRequest) -> CritiqueResponse:
    """Grade a Forecast.

    Pipeline:
      1. (Phase 4) optional Phi-3 regime classification on the snapshot.
      2. Groq judge LLM consumes forecast + news + optional regime hint.
      3. Return a Critique. Returns the stub critique when Groq is
         unconfigured so the orchestrator can complete the graph.
    """
    if not settings.groq_api_key:
        return stub_critique(req.forecast)

    regime_hint = _maybe_classify_regime(req.snapshot)

    try:
        return judge(
            req.forecast,
            req.recent_news,
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            regime_hint=regime_hint,
        )
    except CriticLLMError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Critic LLM error: {e}",
        ) from e
