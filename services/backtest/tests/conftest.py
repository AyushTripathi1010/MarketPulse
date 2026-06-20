"""
Shared fixtures for backtest tests.

The big one is `synthetic_features` — a 200-bar pandas DataFrame matching
exactly what data_ingest writes to Parquet, so the engine and data_loader
can run against in-memory or on-disk versions identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_ticker(ticker: str, n: int, seed: int) -> pd.DataFrame:
    """Geometric-Brownian-motion price series + the standard features."""
    rng = np.random.default_rng(seed=seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    log_returns = rng.normal(0.0, 0.005, n)
    closes = 100.0 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * (1 + np.abs(rng.normal(0, 0.001, n))),
            "Low": closes * (1 - np.abs(rng.normal(0, 0.001, n))),
            "Close": closes,
            "Volume": rng.integers(1e5, 1e6, n).astype(float),
            "ticker": ticker,
            "return_1": pd.Series(log_returns).fillna(0).values,
            "return_24": pd.Series(log_returns).rolling(24, min_periods=1).sum().fillna(0).values,
            "vol_24": pd.Series(log_returns).rolling(24, min_periods=2).std().fillna(0.005).values,
            "rsi_14": rng.uniform(20, 80, n),
            "sentiment_24h": rng.uniform(0, 5, n),
        },
        index=idx,
    )
    return df


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """200 hourly bars of AAPL — enough for encoder=96 + horizon=4 + buffer."""
    return _make_synthetic_ticker("AAPL", n=200, seed=42)


@pytest.fixture
def parquet_dataset(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Write synthetic AAPL features to a Hive-partitioned tree.

    Returns (base_dir, dataframe). Used by data_loader tests.
    """
    df = _make_synthetic_ticker("AAPL", n=200, seed=42)
    part = tmp_path / "dt=2026-01-01" / "ticker=AAPL"
    part.mkdir(parents=True)
    df.to_parquet(part / "00-00-00.parquet")
    return tmp_path, df
