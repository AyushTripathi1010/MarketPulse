"""
HTTP-level tests for the backtest FastAPI app.

We mock the forecast service (httpx) AND the data path (write Parquet to a
temp dir + override config) so tests don't need any external service.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backtest import main as main_module
from backtest.engine import ForecastCall
from backtest.main import app


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parquet_dataset: tuple[Path, pd.DataFrame],
) -> TestClient:
    """Backtest service with data + output dirs pointed at temp paths."""
    data_base, _ = parquet_dataset
    monkeypatch.setattr(main_module.settings, "local_data_dir", str(data_base))
    monkeypatch.setattr(main_module.settings, "local_output_dir", str(tmp_path / "out"))
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "backtest"
    assert "forecast_url" in body["extras"]


def test_run_returns_404_when_no_features(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty data dir → 404, not 500.

    Note: `tmp_path` is shared with the `parquet_dataset` fixture inside
    `client`, so we point at a SIBLING empty directory instead. Otherwise
    we'd find the fixture's Parquet and the test would pass for the
    wrong reason (then fail when it hits the httpx call).
    """
    empty_dir = tmp_path / "definitely_empty"
    empty_dir.mkdir()
    monkeypatch.setattr(main_module.settings, "local_data_dir", str(empty_dir))
    r = client.post(
        "/run",
        json={"ticker": "AAPL", "start_date": "2026-01-01", "end_date": "2026-01-31"},
    )
    assert r.status_code == 404


def test_run_returns_404_for_unknown_ticker(client: TestClient) -> None:
    """Window has data for AAPL but not MSFT → 404."""
    r = client.post(
        "/run",
        json={"ticker": "MSFT", "start_date": "2026-01-01", "end_date": "2026-01-31"},
    )
    assert r.status_code == 404


def test_run_happy_path_with_mocked_forecaster(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock the httpx_forecaster builder; verify /run produces a full response."""

    def _fake_forecaster(_url: str, ticker: str) -> Any:
        def _call(encoder: pd.DataFrame, horizon: int) -> ForecastCall:
            last = float(encoder["Close"].iloc[-1])
            return ForecastCall(
                timestamp=datetime.now(UTC),
                predicted_price=last * 1.005,
                confidence_low=last * 1.002,
                confidence_high=last * 1.008,
            )
        return _call

    monkeypatch.setattr(main_module, "httpx_forecaster", _fake_forecaster)

    r = client.post(
        "/run",
        json={"ticker": "AAPL", "start_date": "2026-01-01", "end_date": "2026-01-31"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["num_bars"] > 0
    assert body["report_uri"].startswith("file://")
    assert body["json_uri"].startswith("file://")
    # Equity curve PNG should also be present.
    assert body["equity_curve_uri"].startswith("file://")
    # The metrics must validate against the Pydantic schema.
    m = body["metrics"]
    assert m["max_drawdown"] <= 0.0
    assert 0.0 <= m["hit_rate"] <= 1.0
