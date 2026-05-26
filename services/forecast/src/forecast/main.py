"""
forecast — FastAPI entry point.

Phase 2: real /predict endpoint that loads the trained TFT on startup and
returns quantile forecasts (point + 10th/90th percentile) for a ticker.

The model is loaded ONCE at lifespan startup and held in module-level state
for every request. That's the right pattern for an LLM/ML model under
modest concurrency — loading on every request would cost ~1s and add zero
benefit. For Lambda we'll re-load on cold start only.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse

from forecast.config import settings
from forecast.data.loader import load_features
from forecast.model.predict import load_artifacts, predict_one
from forecast.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# Module-level handles populated by lifespan().
_model = None
_train_ds = None


def _checkpoint_exists() -> bool:
    return Path(settings.local_checkpoint_path).exists() and Path(
        settings.local_dataset_path
    ).exists()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the trained TFT once at startup."""
    global _model, _train_ds
    if _checkpoint_exists():
        try:
            _model, _train_ds = load_artifacts(
                settings.local_checkpoint_path,
                settings.local_dataset_path,
            )
            logger.info(
                "Loaded TFT checkpoint from %s",
                settings.local_checkpoint_path,
            )
        except Exception as e:  # noqa: BLE001 — startup MUST log, not crash
            logger.error("Failed to load TFT checkpoint: %s", e, exc_info=True)
    else:
        logger.warning(
            "No checkpoint at %s; /predict will return 503 until one exists.",
            settings.local_checkpoint_path,
        )
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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Produce a single forecast for the given ticker and horizon.

    The model and feature loader are coupled by design: the model expects
    features in the exact shape `data_ingest` writes. If features aren't
    available yet, we return 404 — the orchestrator should trigger
    `data_ingest /fetch` first.
    """
    if _model is None or _train_ds is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "TFT model not loaded. Train it first "
                "(see services/forecast/src/forecast/model/train.py)."
            ),
        )

    try:
        features = load_features(settings.local_data_dir, ticker=req.ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if features.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No features for ticker={req.ticker!r}. "
                "Call data_ingest /fetch first."
            ),
        )

    forecast = predict_one(
        _model,
        _train_ds,
        features,
        ticker=req.ticker,
        horizon_hours=req.horizon_hours,
    )
    return forecast
