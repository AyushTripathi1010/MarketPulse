"""Shared fixtures for report tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from marketplus_shared.models import Critique, Forecast, Regime, RegimeLabel


@pytest.fixture
def sample_forecast() -> Forecast:
    return Forecast(
        ticker="AAPL",
        predicted_at=datetime.now(UTC),
        horizon_hours=4,
        predicted_price=187.32,
        confidence_low=185.10,
        confidence_high=189.50,
    )


@pytest.fixture
def sample_critique() -> Critique:
    return Critique(
        regime=Regime(
            label=RegimeLabel.BULL, confidence=0.82, classified_at=datetime.now(UTC)
        ),
        analogues=[],
        confidence="MEDIUM",
        reasoning="The band is tight relative to recent vol; sentiment positive.",
    )
