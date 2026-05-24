"""
Shared pytest fixtures for data_ingest tests.

conftest.py is pytest's magic file: anything defined here is auto-discovered
by tests in the same directory tree. We use it to define the small
synthetic-data fixtures every test needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import numpy as np
import pandas as pd
import pytest

from data_ingest.fetchers.alpaca_fetcher import NewsItem


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """A clean 100-bar hourly OHLCV frame indexed UTC.

    Geometric Brownian motion so the prices look plausible (slight drift +
    random walk). Seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=42)
    n = 100
    idx = pd.date_range("2026-05-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    log_returns = rng.normal(loc=0.0, scale=0.005, size=n)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "Open": closes * (1 + rng.normal(0, 0.0005, n)),
            "High": closes * (1 + np.abs(rng.normal(0, 0.001, n))),
            "Low": closes * (1 - np.abs(rng.normal(0, 0.001, n))),
            "Close": closes,
            "Volume": rng.integers(1e5, 1e7, n).astype(float),
            "ticker": "AAPL",
        },
        index=idx,
    )
    return df


@pytest.fixture
def synthetic_news() -> list[NewsItem]:
    """Three plausible news items spread over the last 12h."""
    base = datetime.now(UTC) - timedelta(hours=12)
    return [
        NewsItem(
            headline="Apple unveils new chip",
            summary="Performance bump expected.",
            created_at=base,
            symbols=("AAPL",),
            source="benzinga",
            url="https://example.com/1",
        ),
        NewsItem(
            headline="Supply chain easing",
            summary="Component costs trend down.",
            created_at=base + timedelta(hours=4),
            symbols=("AAPL",),
            source="reuters",
            url="https://example.com/2",
        ),
        NewsItem(
            headline="Analyst raises target",
            summary="Upgrade from $200 to $210.",
            created_at=base + timedelta(hours=10),
            symbols=("AAPL",),
            source="bloomberg",
            url="https://example.com/3",
        ),
    ]
