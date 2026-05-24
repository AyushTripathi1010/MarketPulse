"""
marketplus_shared — common Pydantic models for every MarketPulse service.

Importable as `from marketplus_shared.models import Forecast`.
"""

from marketplus_shared.models import (
    Forecast,
    Regime,
    RegimeLabel,
    Critique,
    Report,
    HealthResponse,
)

__all__ = [
    "Forecast",
    "Regime",
    "RegimeLabel",
    "Critique",
    "Report",
    "HealthResponse",
]
