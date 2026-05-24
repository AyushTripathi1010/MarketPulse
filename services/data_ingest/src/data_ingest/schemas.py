"""
Pydantic request/response schemas for data_ingest's HTTP API.

Service-specific schemas live here. Cross-service models (Forecast, Regime,
Report) live in marketplus_shared.models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    """Ask data_ingest to pull fresh data for one ticker."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    lookback_days: int = Field(default=90, ge=1, le=365)
    interval: str = Field(default="1h", pattern=r"^(1m|5m|15m|30m|1h|1d)$")


class FetchResponse(BaseModel):
    """Confirmation that data has been written to storage."""

    ticker: str
    rows_written: int
    news_items: int
    storage_uri: str  # file:// in local dev, s3:// in production


class LatestRequest(BaseModel):
    """Load the most recently stored feature frame for a ticker."""

    ticker: str = Field(..., min_length=1, max_length=10)


class LatestResponse(BaseModel):
    """Minimal summary of the latest stored feature frame.

    We deliberately DO NOT return the entire frame in the HTTP body —
    other services that need the full data read it from the storage URI
    directly (S3 or local file). Keeps responses small and predictable.
    """

    ticker: str
    rows: int
    last_close: float
    storage_uri: str
