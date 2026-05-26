"""Tests for the Parquet data loader + training-frame transform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecast.data.loader import (
    FeatureFrameError,
    load_features,
    to_training_frame,
)
from forecast.data.splits import time_based_split


def test_load_features_finds_all_tickers(parquet_dataset: Path) -> None:
    """Both tickers must come back; the (ticker, timestamp) dedup must not drop one."""
    df = load_features(parquet_dataset)
    assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}
    counts = df["ticker"].value_counts()
    # 500 rows per ticker from the fixture — both should survive dedup.
    assert counts["AAPL"] == 500
    assert counts["MSFT"] == 500


def test_load_features_filter_by_ticker(parquet_dataset: Path) -> None:
    """ticker= filter should narrow the result to one symbol."""
    df = load_features(parquet_dataset, ticker="AAPL")
    assert set(df["ticker"].unique()) == {"AAPL"}


def test_load_features_missing_dir_raises(tmp_path: Path) -> None:
    """Bad path → FileNotFoundError, not a silent empty frame."""
    with pytest.raises(FileNotFoundError):
        load_features(tmp_path / "does-not-exist")


def test_load_features_empty_dir_returns_empty(tmp_path: Path) -> None:
    """Empty (but existing) dir → empty frame with the right columns."""
    df = load_features(tmp_path)
    assert df.empty
    # Caller can still check columns without KeyError.
    assert "ticker" in df.columns


def test_load_features_missing_columns_raises(tmp_path: Path, synthetic_features: pd.DataFrame) -> None:
    """A Parquet missing required columns → FeatureFrameError."""
    bad = synthetic_features.drop(columns=["rsi_14"])
    part = tmp_path / "dt=2026-01-01" / "ticker=AAPL"
    part.mkdir(parents=True)
    bad.to_parquet(part / "00-00-00.parquet")
    with pytest.raises(FeatureFrameError):
        load_features(tmp_path)


def test_to_training_frame_adds_time_idx(synthetic_features: pd.DataFrame) -> None:
    """time_idx must be monotonic and ticker-scoped (each ticker starts at 0)."""
    df = to_training_frame(synthetic_features)
    for tk in df["ticker"].unique():
        sub = df[df["ticker"] == tk]
        assert sub["time_idx"].is_monotonic_increasing
        assert sub["time_idx"].iloc[0] == 0


def test_time_based_split_no_leakage(synthetic_features: pd.DataFrame) -> None:
    """Per-ticker, every val time_idx must come AFTER every train time_idx."""
    df = to_training_frame(synthetic_features)
    train_df, val_df = time_based_split(df, train_frac=0.8)
    for tk in df["ticker"].unique():
        train_max = train_df.loc[train_df["ticker"] == tk, "time_idx"].max()
        val_min = val_df.loc[val_df["ticker"] == tk, "time_idx"].min()
        assert val_min > train_max, f"Leakage detected for {tk}: train_max={train_max}, val_min={val_min}"


def test_time_based_split_rejects_invalid_frac(synthetic_features: pd.DataFrame) -> None:
    """train_frac outside (0, 1) must raise — not silently corrupt the split."""
    df = to_training_frame(synthetic_features)
    with pytest.raises(ValueError):
        time_based_split(df, train_frac=0.0)
    with pytest.raises(ValueError):
        time_based_split(df, train_frac=1.5)
