"""
Shared fixtures for forecast tests.

Generates synthetic OHLCV + features with enough data for the TFT to train
on (the model has minimums for encoder + prediction length).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _synthetic_ticker(ticker: str, n: int, seed: int) -> pd.DataFrame:
    """Geometric-Brownian-motion closes + lightly correlated features."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    log_returns = rng.normal(0.0, 0.005, n)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "Open": closes * (1 + rng.normal(0, 0.0005, n)),
            "High": closes * (1 + np.abs(rng.normal(0, 0.001, n))),
            "Low": closes * (1 - np.abs(rng.normal(0, 0.001, n))),
            "Close": closes,
            "Volume": rng.integers(1e5, 1e7, n).astype(float),
            "ticker": ticker,
            "return_1": pd.Series(log_returns).fillna(0).values,
            "return_24": pd.Series(log_returns)
            .rolling(24, min_periods=1)
            .sum()
            .fillna(0)
            .values,
            "vol_24": pd.Series(log_returns)
            .rolling(24, min_periods=2)
            .std()
            .fillna(0.005)
            .values,
            "rsi_14": rng.uniform(20, 80, n),
            "sentiment_24h": rng.uniform(0, 5, n),
        },
        index=idx,
    )
    return df


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Two tickers, 500 hourly bars each.

    Each split must be larger than encoder_length + prediction_length (100)
    for a TimeSeriesDataSet to retain ANY series after window construction.
    500 rows with an 80/20 split gives ~400 train + ~100 val per ticker —
    exactly at the minimum. Smaller datasets silently produce empty val
    sets with only a UserWarning.
    """
    return pd.concat(
        [
            _synthetic_ticker("AAPL", n=500, seed=1),
            _synthetic_ticker("MSFT", n=500, seed=2),
        ]
    )


@pytest.fixture
def parquet_dataset(synthetic_features: pd.DataFrame, tmp_path: Path) -> Path:
    """Write the synthetic features to a Hive-partitioned tree."""
    for tk in synthetic_features["ticker"].unique():
        sub = synthetic_features[synthetic_features["ticker"] == tk]
        part = tmp_path / "dt=2026-01-01" / f"ticker={tk}"
        part.mkdir(parents=True)
        sub.to_parquet(part / "00-00-00.parquet")
    return tmp_path
