"""Pydantic request/response schemas for the forecast service."""

from __future__ import annotations

from pydantic import BaseModel, Field

from marketplus_shared.models import Forecast


class PredictRequest(BaseModel):
    """Ask for a single price-movement forecast."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    horizon_hours: int = Field(default=4, ge=1, le=24)


# The response is the shared Forecast model — every service speaks it.
PredictResponse = Forecast
