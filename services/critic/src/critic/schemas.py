"""Pydantic request/response schemas for the critic service."""

from pydantic import BaseModel

from marketplus_shared.models import Critique, Forecast


class CritiqueRequest(BaseModel):
    """Ask the critic to grade a forecast."""

    forecast: Forecast
    # News headlines from the last 24h, fed in by the orchestrator.
    recent_news: list[str]


# The response is the shared Critique model.
CritiqueResponse = Critique
