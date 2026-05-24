"""
Sanity tests for the shared Pydantic models.

These tests exist to catch breakage when someone "innocently" tweaks a model
that's imported by every service. Run them before every PR.
"""

from datetime import datetime, UTC

from marketplus_shared.models import (
    Forecast,
    HealthResponse,
    Regime,
    RegimeLabel,
)


def test_health_response_defaults() -> None:
    """HealthResponse should default status='ok' and empty extras."""
    h = HealthResponse(service="forecast")
    assert h.status == "ok"
    assert h.version == "0.1.0"
    assert h.extras == {}


def test_regime_label_is_str_enum() -> None:
    """RegimeLabel values must serialize as their string form."""
    assert RegimeLabel.BULL == "bull"
    r = Regime(
        label=RegimeLabel.BULL,
        confidence=0.83,
        classified_at=datetime.now(UTC),
    )
    assert r.label == "bull"


def test_forecast_horizon_bounds() -> None:
    """Forecast horizon must be 1-24 hours; outside raises ValidationError."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Forecast(
            ticker="AAPL",
            predicted_at=datetime.now(UTC),
            horizon_hours=99,  # too high
            predicted_price=100.0,
            confidence_low=99.0,
            confidence_high=101.0,
        )
