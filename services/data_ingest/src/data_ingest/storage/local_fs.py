"""
Local filesystem storage backend.

Used in local development (no AWS credentials needed). Same interface as
the S3 backend so the FastAPI handler doesn't care which one is active —
the only difference is the URI scheme (`file://...` vs `s3://...`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data_ingest.storage.parquet_io import read_parquet, write_parquet


def _build_path(base_dir: Path, ticker: str, dt: datetime) -> Path:
    """Compute the partitioned path for a (ticker, datetime) pair.

    Partitioning by date keeps individual files small and lets pandas /
    Spark prune partitions when querying by date range.
    """
    # Hive-style partitioning: dt=2026-05-24/ticker=AAPL/000.parquet
    return (
        base_dir
        / f"dt={dt:%Y-%m-%d}"
        / f"ticker={ticker}"
        / f"{dt:%H-%M-%S}.parquet"
    )


def save_features(
    df: pd.DataFrame,
    ticker: str,
    *,
    base_dir: str | Path = "./data/features",
    as_of: datetime | None = None,
) -> str:
    """Save a feature DataFrame to local disk, partitioned by date + ticker.

    Returns the path as a `file://` URI so callers can store it in the
    response payload the same way they would an `s3://` URI.
    """
    base = Path(base_dir).resolve()
    when = as_of or datetime.now(tz=df.index.tz if df.index.tz else None) or datetime.now()
    target = _build_path(base, ticker, when)
    write_parquet(df, target)
    return f"file://{target}"


def load_latest_features(ticker: str, *, base_dir: str | Path = "./data/features") -> pd.DataFrame:
    """Read the most recently written feature file for a ticker.

    Scans the partition directories and picks the lexically-last file —
    which works because we name files with sortable HH-MM-SS prefixes.
    Returns an empty DataFrame if nothing exists yet (lets callers
    distinguish "no data" from "I/O error").
    """
    base = Path(base_dir).resolve()
    if not base.exists():
        return pd.DataFrame()

    candidates = sorted(base.rglob(f"ticker={ticker}/*.parquet"))
    if not candidates:
        return pd.DataFrame()

    return read_parquet(candidates[-1])
