"""
Tests for the local-FS storage backend.

We don't test s3.py here — that would require live AWS creds or a moto/
LocalStack fixture. We'll cover S3 in Phase 9 (CI/CD) with a moto-mocked
integration test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from data_ingest.storage.local_fs import load_latest_features, save_features
from data_ingest.storage.parquet_io import read_parquet, write_parquet


def test_parquet_roundtrip(synthetic_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    """write_parquet → read_parquet must preserve dtypes and tz-aware index."""
    target = tmp_path / "round.parquet"
    write_parquet(synthetic_ohlcv, target)

    loaded = read_parquet(target)
    pd.testing.assert_frame_equal(loaded, synthetic_ohlcv, check_freq=False)
    assert loaded.index.tz is not None
    assert str(loaded.index.tz) == "UTC"


def test_save_features_creates_partition_path(
    synthetic_ohlcv: pd.DataFrame, tmp_path: Path
) -> None:
    """The Hive-style dt=YYYY-MM-DD/ticker=AAPL/HH-MM-SS.parquet path must appear on disk."""
    when = datetime(2026, 5, 24, 9, 0, 0, tzinfo=UTC)
    uri = save_features(synthetic_ohlcv, "AAPL", base_dir=tmp_path, as_of=when)

    assert uri.startswith("file://")
    # Carve the filesystem path out of the URI and check it exists.
    fs_path = Path(uri.removeprefix("file://"))
    assert fs_path.exists()
    assert "dt=2026-05-24" in str(fs_path)
    assert "ticker=AAPL" in str(fs_path)


def test_load_latest_features_returns_empty_when_nothing_written(tmp_path: Path) -> None:
    """If no file exists for the ticker, return an empty DataFrame — not raise."""
    out = load_latest_features("AAPL", base_dir=tmp_path)
    assert out.empty


def test_load_latest_features_returns_most_recent(
    synthetic_ohlcv: pd.DataFrame, tmp_path: Path
) -> None:
    """When multiple files exist, the lexically-last one (= most recent) wins."""
    early = datetime(2026, 5, 24, 6, 0, 0, tzinfo=UTC)
    late = datetime(2026, 5, 24, 9, 0, 0, tzinfo=UTC)

    # Save two versions; the second one has different Close values.
    df_early = synthetic_ohlcv.assign(Close=synthetic_ohlcv["Close"] * 0.5)
    df_late = synthetic_ohlcv

    save_features(df_early, "AAPL", base_dir=tmp_path, as_of=early)
    save_features(df_late, "AAPL", base_dir=tmp_path, as_of=late)

    loaded = load_latest_features("AAPL", base_dir=tmp_path)
    # The late one should match df_late, not df_early.
    assert loaded["Close"].iloc[-1] == df_late["Close"].iloc[-1]
