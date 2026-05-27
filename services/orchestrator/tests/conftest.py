"""
Shared fixtures + httpx mocking for orchestrator tests.

We use a thin _FakeHttpxPost helper to mock the four downstream service
calls. This avoids depending on respx or any extra dep, and the mock
shape is easy to inspect when a test fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from marketplus_shared.models import (
    Critique,
    Forecast,
    Regime,
    RegimeLabel,
)


@pytest.fixture
def sample_forecast_payload() -> dict[str, Any]:
    """JSON-serializable Forecast (what /predict returns)."""
    return Forecast(
        ticker="AAPL",
        predicted_at=datetime.now(UTC),
        horizon_hours=4,
        predicted_price=187.32,
        confidence_low=185.10,
        confidence_high=189.50,
    ).model_dump(mode="json")


@pytest.fixture
def sample_critique_payload() -> dict[str, Any]:
    """JSON-serializable Critique (what /critique returns)."""
    return Critique(
        regime=Regime(
            label=RegimeLabel.BULL, confidence=0.82, classified_at=datetime.now(UTC)
        ),
        analogues=[],
        confidence="MEDIUM",
        reasoning="band tight, sentiment positive",
    ).model_dump(mode="json")


class _FakeResponse:
    """Minimal stand-in for httpx.Response — only what our nodes use."""

    def __init__(self, status_code: int, json_body: dict) -> None:
        self.status_code = status_code
        self._json = json_body

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=None  # type: ignore[arg-type]
            )


@pytest.fixture
def fake_httpx_post(
    sample_forecast_payload: dict[str, Any],
    sample_critique_payload: dict[str, Any],
) -> Any:
    """
    Router-style fake httpx.post that returns canned responses by URL.

    Tests pass `monkeypatch.setattr("httpx.post", fake_httpx_post)` (per-module
    via the node modules' import path).
    """

    def _post(url: str, **_kwargs: Any) -> _FakeResponse:
        if "/fetch" in url:
            return _FakeResponse(
                200,
                {
                    "ticker": "AAPL",
                    "rows_written": 120,
                    "news_items": 0,
                    "storage_uri": "file:///tmp/data/features",
                },
            )
        if "/predict" in url:
            return _FakeResponse(200, sample_forecast_payload)
        if "/critique" in url:
            return _FakeResponse(200, sample_critique_payload)
        if "/generate" in url:
            return _FakeResponse(
                200,
                {
                    "ticker": "AAPL",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "markdown": "# AAPL brief\n",
                    "polished": False,
                },
            )
        raise AssertionError(f"unexpected URL in fake_httpx_post: {url}")

    return _post
