"""
Pydantic models passed between services over HTTP.

These are the "vocabulary" of MarketPulse — every service speaks them.
If you change a model here, every service that consumes it must be updated.
That's a feature: a workspace-wide grep tells you exactly who's affected.

Why Pydantic v2: runtime validation + JSON serialization + JSON Schema
generation, all from type hints. FastAPI uses the schemas for /docs.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------
# Health probe — every service exposes /health that returns this.
# Used by docker-compose, Lambda container health checks, and uptime alerts.
# ----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Standard response from any service's GET /health endpoint."""

    service: str = Field(..., description="Service name, e.g. 'forecast'")
    status: Literal["ok", "degraded", "starting"] = "ok"
    version: str = "0.1.0"
    # Optional service-specific extras (e.g. forecast adds 'model_loaded': bool).
    extras: dict[str, object] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Market regime — the Critic agent's classification of current conditions.
# ----------------------------------------------------------------------------
class RegimeLabel(StrEnum):
    """Three discrete market regimes. Mutually exclusive."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class Regime(BaseModel):
    """Output of the Phi-3 regime classifier."""

    label: RegimeLabel
    # Confidence is the classifier's softmax for the chosen label, 0-1.
    confidence: float = Field(..., ge=0.0, le=1.0)
    classified_at: datetime


# ----------------------------------------------------------------------------
# Forecast — output of the TFT model.
# ----------------------------------------------------------------------------
class Forecast(BaseModel):
    """A single price-movement prediction for one ticker, one horizon."""

    ticker: str = Field(..., min_length=1, max_length=10)
    predicted_at: datetime
    horizon_hours: int = Field(..., ge=1, le=24)
    # Point estimate of price at predicted_at + horizon_hours.
    predicted_price: float
    # 10th and 90th percentile from the TFT's quantile head.
    # The width (p90 - p10) is the model's uncertainty in absolute price.
    confidence_low: float
    confidence_high: float
    # Attention weights tell us WHICH input features drove the prediction.
    # Key: feature name (e.g. "close_lag_1", "sentiment_24h"); value: weight.
    # Empty dict if the model wasn't asked to return attention.
    attention_weights: dict[str, float] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Critique — output of the Critic agent (regime + RAG + LLM judgment).
# ----------------------------------------------------------------------------
class HistoricalAnalogue(BaseModel):
    """One historically similar setup retrieved by hybrid RAG."""

    ticker: str
    occurred_at: datetime
    regime_then: RegimeLabel
    outcome_24h_return: float  # what happened in the 24h AFTER this setup
    news_summary: str
    # The retriever's similarity score (post-rerank). Higher = more similar.
    similarity_score: float


class Critique(BaseModel):
    """Critic Agent's grade of a Forecast."""

    regime: Regime
    analogues: list[HistoricalAnalogue]
    # Confidence the critic assigns to the forecast (NOT the same as the
    # forecast's own quantile width — this is the LLM's calibrated judgment).
    confidence: Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    reasoning: str  # plain-English explanation, 2-4 sentences


# ----------------------------------------------------------------------------
# Report — the final markdown brief produced by the Report agent.
# ----------------------------------------------------------------------------
class Report(BaseModel):
    """Pointer to the generated intelligence brief stored on S3."""

    ticker: str
    generated_at: datetime
    s3_path: str  # s3://marketplus-yourname/reports/AAPL/2026-05-24T09-00.md
    forecast: Forecast
    critique: Critique
    # The full markdown content is in S3; we don't ship it through HTTP
    # unless requested. Tradeoff: smaller responses, one extra fetch to display.
