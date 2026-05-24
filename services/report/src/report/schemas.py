"""Pydantic request/response schemas for the report service."""

from pydantic import BaseModel

from marketplus_shared.models import Critique, Forecast, Report


class GenerateRequest(BaseModel):
    """Ask the report service to compose a markdown brief from a forecast + critique."""

    forecast: Forecast
    critique: Critique


# The response is the shared Report model (includes the S3 path to the markdown).
GenerateResponse = Report
