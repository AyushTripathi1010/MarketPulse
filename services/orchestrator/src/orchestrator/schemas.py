"""Pydantic request/response schemas for the orchestrator service."""

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Kick off one full forecast → critic → report cycle for a ticker."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    horizon_hours: int = Field(default=4, ge=1, le=24)


class RunResponse(BaseModel):
    """Result of a full graph execution."""

    ticker: str
    report_s3_path: str
    trace_id: str  # Langfuse trace ID for debugging
