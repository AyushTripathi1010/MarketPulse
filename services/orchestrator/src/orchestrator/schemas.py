"""Pydantic request/response schemas for the orchestrator service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Kick off one full forecast → critic → report cycle for a ticker."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    horizon_hours: int = Field(default=4, ge=1, le=24)


class RunResponse(BaseModel):
    """Result of a full graph execution.

    We return the markdown inline in Phase 3 (no S3 yet). Phase 7 will
    write it to S3 and return a pointer.
    """

    ticker: str
    trace_id: str
    report_markdown: str
    drift_detected: bool
