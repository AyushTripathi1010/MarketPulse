"""Shared fixtures for critic tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from marketplus_shared.models import Forecast


@pytest.fixture
def sample_forecast() -> Forecast:
    """A plausible Forecast the critic might grade."""
    return Forecast(
        ticker="AAPL",
        predicted_at=datetime.now(UTC),
        horizon_hours=4,
        predicted_price=187.32,
        confidence_low=185.10,
        confidence_high=189.50,
        attention_weights={},
    )
