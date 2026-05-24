"""
Tests for the fetchers — all external API calls are MONKEY-PATCHED.

Why mock instead of hitting the real API?
  - Tests run in CI without network access (or with rate-limited access).
  - The real yFinance endpoint is occasionally broken; we don't want our
    CI to go red because Yahoo had a hiccup.
  - We control the test data, so we can simulate weird edge cases (empty
    responses, missing columns) that real APIs rarely produce in dev.

We use pytest's `monkeypatch` fixture to patch module attributes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from data_ingest.fetchers import alpaca_fetcher, yfinance_fetcher


# ---------------------------------------------------------------------------
# yfinance_fetcher
# ---------------------------------------------------------------------------
class _FakeYfTicker:
    """Tiny stand-in for yf.Ticker that returns a fixed DataFrame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def history(self, **_kwargs: Any) -> pd.DataFrame:
        return self._frame.copy()


def test_fetch_ohlcv_happy_path(
    synthetic_ohlcv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When yfinance returns a clean frame, we add tz + ticker column."""
    # yfinance's real Ticker.history doesn't include the ticker column, so
    # drop it from our fixture before patching so we exercise the add path.
    stub_frame = synthetic_ohlcv.drop(columns=["ticker"])
    monkeypatch.setattr(
        yfinance_fetcher.yf, "Ticker", lambda _sym: _FakeYfTicker(stub_frame)
    )

    out = yfinance_fetcher.fetch_ohlcv("AAPL", lookback_days=4)

    assert "ticker" in out.columns
    assert (out["ticker"] == "AAPL").all()
    assert str(out.index.tz) == "UTC"


def test_fetch_ohlcv_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty dataframe → YFinanceError (after retries exhausted)."""
    monkeypatch.setattr(
        yfinance_fetcher.yf,
        "Ticker",
        lambda _sym: _FakeYfTicker(pd.DataFrame()),
    )
    with pytest.raises(yfinance_fetcher.YFinanceError):
        yfinance_fetcher.fetch_ohlcv("AAPL", lookback_days=1)


def test_fetch_ohlcv_raises_on_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame missing OHLC columns → YFinanceError."""
    bad = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.DatetimeIndex(["2026-05-01"], tz="UTC"),
    )
    monkeypatch.setattr(
        yfinance_fetcher.yf, "Ticker", lambda _sym: _FakeYfTicker(bad)
    )
    with pytest.raises(yfinance_fetcher.YFinanceError):
        yfinance_fetcher.fetch_ohlcv("AAPL", lookback_days=1)


# ---------------------------------------------------------------------------
# alpaca_fetcher
# ---------------------------------------------------------------------------
def test_fetch_news_returns_empty_without_creds() -> None:
    """No key → empty list. We never crash for missing optional creds."""
    items = alpaca_fetcher.fetch_news(
        "AAPL",
        api_key="",
        api_secret="",
        hours=24,
    )
    assert items == []


def test_fetch_news_returns_empty_when_secret_missing() -> None:
    """Only key, no secret → still empty (need both)."""
    items = alpaca_fetcher.fetch_news(
        "AAPL",
        api_key="some_key",
        api_secret="",
        hours=24,
    )
    assert items == []
