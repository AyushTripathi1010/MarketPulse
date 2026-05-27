"""
Per-node unit tests using the fake_httpx_post fixture.

Each test patches `httpx.post` inside the NODE'S module (per pytest's
monkeypatch scoping) so we don't accidentally make real network calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from orchestrator.nodes import (
    critic_node,
    data_node,
    drift_node,
    forecast_node,
    report_node,
)


def test_data_node_returns_storage_uri(fake_httpx_post: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_node.httpx, "post", fake_httpx_post)
    out = data_node.run({"ticker": "AAPL", "horizon_hours": 4})
    assert out["ohlcv_uri"].startswith("file://")
    assert out["recent_news"] == []


def test_forecast_node_returns_forecast(
    fake_httpx_post: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forecast_node.httpx, "post", fake_httpx_post)
    out = forecast_node.run({"ticker": "AAPL", "horizon_hours": 4})
    assert out["forecast"].ticker == "AAPL"
    assert out["forecast"].predicted_price == 187.32


def test_critic_node_returns_critique(
    fake_httpx_post: Any,
    sample_forecast_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(critic_node.httpx, "post", fake_httpx_post)
    from marketplus_shared.models import Forecast

    state = {
        "ticker": "AAPL",
        "horizon_hours": 4,
        "forecast": Forecast.model_validate(sample_forecast_payload),
        "recent_news": [],
    }
    out = critic_node.run(state)
    assert out["critique"].confidence == "MEDIUM"


def test_critic_node_degrades_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If httpx raises, the critic returns an UNKNOWN-confidence stub, not a crash."""
    import httpx

    def _explode(*_a: Any, **_kw: Any) -> Any:
        raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(critic_node.httpx, "post", _explode)

    from datetime import UTC, datetime

    from marketplus_shared.models import Forecast

    state = {
        "ticker": "AAPL",
        "horizon_hours": 4,
        "forecast": Forecast(
            ticker="AAPL",
            predicted_at=datetime.now(UTC),
            horizon_hours=4,
            predicted_price=100.0,
            confidence_low=99.0,
            confidence_high=101.0,
        ),
        "recent_news": [],
    }
    out = critic_node.run(state)
    assert out["critique"].confidence == "UNKNOWN"


def test_report_node_returns_markdown(
    fake_httpx_post: Any,
    sample_forecast_payload: dict,
    sample_critique_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_node.httpx, "post", fake_httpx_post)
    from marketplus_shared.models import Critique, Forecast

    state = {
        "ticker": "AAPL",
        "horizon_hours": 4,
        "forecast": Forecast.model_validate(sample_forecast_payload),
        "critique": Critique.model_validate(sample_critique_payload),
    }
    out = report_node.run(state)
    assert "AAPL" in out["report_markdown"]


def test_drift_node_detects_wide_band(sample_forecast_payload: dict) -> None:
    """A forecast with a wide confidence band should trigger drift_detected=True."""
    from marketplus_shared.models import Forecast

    # Inflate the band to 10% of the price — well above DRIFT_MAE_THRESHOLD (2.5%).
    wide = Forecast.model_validate(
        {**sample_forecast_payload, "confidence_low": 150.0, "confidence_high": 220.0}
    )
    state = {"ticker": "AAPL", "horizon_hours": 4, "forecast": wide}
    out = drift_node.run(state)
    assert out["drift_detected"] is True
    assert out["retrain_triggered"] is False  # Phase 3 doesn't actually retrigger


def test_drift_node_does_not_detect_tight_band(sample_forecast_payload: dict) -> None:
    """A tight band (<2.5%) → no drift."""
    from marketplus_shared.models import Forecast

    forecast = Forecast.model_validate(sample_forecast_payload)  # band ~2.3%
    state = {"ticker": "AAPL", "horizon_hours": 4, "forecast": forecast}
    out = drift_node.run(state)
    assert out["drift_detected"] is False
