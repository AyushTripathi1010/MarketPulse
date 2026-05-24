"""
forecast — FastAPI entry point.

Phase 0: skeleton with /health only. Real model loading and /predict arrive
in Phase 2 alongside the TFT implementation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from marketplus_shared.models import HealthResponse

from forecast.config import settings

# Module-level handle for the trained TFT model. Loaded once at startup,
# reused on every request. None until lifespan loads it.
_model: object | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the TFT checkpoint from S3 on startup (Phase 2 work)."""
    global _model
    # Phase 0: nothing to load yet.
    # Phase 2: _model = load_tft_from_s3(settings.s3_bucket, settings.tft_checkpoint_key)
    yield


app = FastAPI(
    title="MarketPulse — Forecast",
    version="0.1.0",
    description="Temporal Fusion Transformer prediction service.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — also reports whether the TFT model has been loaded."""
    return HealthResponse(
        service="forecast",
        extras={
            "environment": settings.environment,
            "model_loaded": _model is not None,
        },
    )
