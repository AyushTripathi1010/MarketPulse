"""Pydantic request/response schemas for the critic service."""

from __future__ import annotations

from pydantic import BaseModel

from marketplus_shared.models import Critique, Forecast


class CritiqueRequest(BaseModel):
    """Ask the critic to grade a forecast."""

    forecast: Forecast
    # News headlines from the last 24h, fed in by the orchestrator.
    recent_news: list[str] = []
    # Optional market snapshot for the Phi-3 regime classifier. When None
    # (Phase 3 path), the critic skips the classifier and uses Groq alone.
    # When provided (Phase 4+), the classifier runs first and its label
    # gets fed into the Groq judge prompt as additional context.
    #
    # Value type is `str | float` because `ticker` is a string while the
    # other fields are numeric. We DON'T model this as a strict typed
    # object because the snapshot field set is expected to evolve as we
    # tune the classifier — keeping it loose at the HTTP boundary means
    # orchestrator updates don't require lock-step critic-schema changes.
    snapshot: dict[str, str | float] | None = None


# The response is the shared Critique model.
CritiqueResponse = Critique
