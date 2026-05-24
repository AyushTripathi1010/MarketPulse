"""
HTTP-level tests for the data_ingest FastAPI app.

Uses fastapi.testclient.TestClient (a real ASGI client wrapping the app).
External fetchers are monkey-patched so we don't hit yFinance/Alpaca.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_ingest import main as main_module
from data_ingest.main import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient that writes parquet to a per-test temp dir."""
    # Override the data dir setting so each test gets a fresh local FS.
    monkeypatch.setattr(main_module.settings, "local_data_dir", str(tmp_path))
    monkeypatch.setattr(main_module.settings, "s3_bucket", "")  # force local FS
    return TestClient(app)


def test_health_extras_include_environment(client: TestClient) -> None:
    """/health should report environment + s3 mode in extras."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "data_ingest"
    assert "environment" in body["extras"]
    assert "use_s3" in body["extras"]


def test_fetch_happy_path(
    client: TestClient,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /fetch with mocked yfinance + no news returns 200 and writes a file."""

    # Patch the yfinance fetcher to return our synthetic frame.
    def _fake_fetch_ohlcv(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return synthetic_ohlcv.copy()

    monkeypatch.setattr(main_module, "fetch_ohlcv", _fake_fetch_ohlcv)

    # Make news a no-op (zero credentials → empty list anyway, but be explicit).
    monkeypatch.setattr(main_module, "fetch_news", lambda *_a, **_kw: [])

    r = client.post(
        "/fetch",
        json={"ticker": "AAPL", "lookback_days": 4, "interval": "1h"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["rows_written"] == 100  # matches synthetic_ohlcv fixture
    assert body["news_items"] == 0
    assert body["storage_uri"].startswith("file://")


def test_fetch_propagates_yfinance_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yFinance failures should become HTTP 502, not 500."""
    from data_ingest.fetchers.yfinance_fetcher import YFinanceError

    def _boom(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise YFinanceError("simulated yahoo outage")

    monkeypatch.setattr(main_module, "fetch_ohlcv", _boom)

    r = client.post("/fetch", json={"ticker": "AAPL", "lookback_days": 1})
    assert r.status_code == 502
    assert "yFinance" in r.json()["detail"]


def test_latest_returns_404_when_nothing_written(client: TestClient) -> None:
    """POST /latest before any /fetch must return 404, not a crash."""
    r = client.post("/latest", json={"ticker": "AAPL"})
    assert r.status_code == 404


def test_fetch_then_latest_roundtrip(
    client: TestClient,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: fetch writes a file, latest finds it."""

    def _fake_fetch_ohlcv(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return synthetic_ohlcv.copy()

    monkeypatch.setattr(main_module, "fetch_ohlcv", _fake_fetch_ohlcv)
    monkeypatch.setattr(main_module, "fetch_news", lambda *_a, **_kw: [])

    fetch_resp = client.post("/fetch", json={"ticker": "AAPL"})
    assert fetch_resp.status_code == 200

    latest_resp = client.post("/latest", json={"ticker": "AAPL"})
    assert latest_resp.status_code == 200
    assert latest_resp.json()["rows"] == 100
    assert latest_resp.json()["last_close"] > 0
