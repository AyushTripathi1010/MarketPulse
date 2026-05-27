"""Pydantic request/response schemas for the report service."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel

from marketplus_shared.models import Critique, Forecast


class GenerateRequest(BaseModel):
    """Ask the report service to compose a markdown brief from a forecast + critique."""

    forecast: Forecast
    critique: Critique


class GenerateResponse(BaseModel):
    """The generated brief.

    Phase 3: we return the markdown inline (no S3 yet). Phase 7 will switch
    to writing to S3 and returning a pointer.
    """

    ticker: str
    generated_at: datetime
    markdown: str
    polished: bool  # True if LLM polish pass ran successfully
