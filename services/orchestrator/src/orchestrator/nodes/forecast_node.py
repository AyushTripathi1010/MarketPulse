"""
forecast_node — second node.

Calls the forecast service's /predict endpoint. If the model isn't loaded
on that side (503), we let the exception propagate — there's no useful
forecast without a trained model and we shouldn't fabricate one.
"""

from __future__ import annotations

import logging

import httpx
from marketplus_shared.models import Forecast

from orchestrator.config import settings
from orchestrator.state import MarketState

logger = logging.getLogger(__name__)


def run(state: MarketState) -> MarketState:
    """Get a price forecast from the forecast service."""
    response = httpx.post(
        f"{settings.forecast_url}/predict",
        json={"ticker": state["ticker"], "horizon_hours": state["horizon_hours"]},
        timeout=30.0,
    )
    response.raise_for_status()
    forecast = Forecast.model_validate(response.json())

    logger.info(
        "forecast_node: ticker=%s predicted=%.2f band=[%.2f, %.2f]",
        forecast.ticker,
        forecast.predicted_price,
        forecast.confidence_low,
        forecast.confidence_high,
    )
    return {"forecast": forecast}
