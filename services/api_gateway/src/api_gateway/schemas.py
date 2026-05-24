"""Pydantic schemas for the api_gateway service.

The gateway re-uses shared models for most responses. Anything that's purely
about the public API contract lives here.
"""

from pydantic import BaseModel, Field


class LatestForecastRequest(BaseModel):
    """Query for the most recent forecast for a given ticker."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
