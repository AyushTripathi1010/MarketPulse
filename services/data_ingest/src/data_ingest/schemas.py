"""
Pydantic request/response schemas for data_ingest's HTTP API.

Service-specific schemas live here. Cross-service models (Forecast, Regime,
Report) live in marketplus_shared.models.
"""

from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    """Ask data_ingest to pull fresh data for one ticker."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    lookback_days: int = Field(default=90, ge=1, le=365)


class FetchResponse(BaseModel):
    """Confirmation that data has been written to S3."""

    ticker: str
    rows_written: int
    s3_path: str
