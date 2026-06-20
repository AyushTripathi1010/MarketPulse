"""Tests for the historical features data loader."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from backtest.data_loader import (
    FeatureFrameError,
    load_historical_features,
)


def test_load_finds_matching_partition(parquet_dataset: tuple[Path, pd.DataFrame]) -> None:
    base_dir, df = parquet_dataset
    out = load_historical_features(
        base_dir, ticker="AAPL",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 31, tzinfo=UTC),
    )
    assert len(out) == len(df)
    assert (out["ticker"] == "AAPL").all()


def test_load_ticker_filter_excludes_others(parquet_dataset: tuple[Path, pd.DataFrame]) -> None:
    base_dir, _df = parquet_dataset
    out = load_historical_features(
        base_dir, ticker="MSFT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 31, tzinfo=UTC),
    )
    # No MSFT partitions exist in this fixture → empty frame, not error.
    assert out.empty


def test_load_date_filter_prunes_partitions(parquet_dataset: tuple[Path, pd.DataFrame]) -> None:
    base_dir, _df = parquet_dataset
    # Date range outside the partition's dt=2026-01-01 → empty result.
    out = load_historical_features(
        base_dir, ticker="AAPL",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )
    assert out.empty


def test_load_missing_base_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_historical_features(
            tmp_path / "missing", ticker="AAPL",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 31, tzinfo=UTC),
        )


def test_load_missing_columns_raises(tmp_path: Path, synthetic_features: pd.DataFrame) -> None:
    """A Parquet file missing required columns → FeatureFrameError."""
    bad = synthetic_features.drop(columns=["rsi_14"])
    part = tmp_path / "dt=2026-01-01" / "ticker=AAPL"
    part.mkdir(parents=True)
    bad.to_parquet(part / "00-00-00.parquet")

    with pytest.raises(FeatureFrameError):
        load_historical_features(
            tmp_path, ticker="AAPL",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 31, tzinfo=UTC),
        )


def test_load_returns_sorted_by_timestamp(parquet_dataset: tuple[Path, pd.DataFrame]) -> None:
    base_dir, _df = parquet_dataset
    out = load_historical_features(
        base_dir, ticker="AAPL",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 31, tzinfo=UTC),
    )
    assert out.index.is_monotonic_increasing
