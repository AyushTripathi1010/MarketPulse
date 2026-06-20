"""
Load historical features for backtesting.

This module is INTENTIONALLY a near-duplicate of services/forecast/data/loader.py.
A "shared" package would be cleaner, but services are deployed independently
and a circular dep would be worse. The two implementations stay in sync by
convention — see learning/04-the-overall-flow-how-files-talk.md.

Why we re-implement instead of HTTP-fetching from data_ingest:
  Backtests run over months of bars (~5K rows for 6 months hourly). One
  HTTP call per row would be ~5K round-trips. Reading Parquet directly
  is two orders of magnitude faster and the BARS ARE STATIC — no race
  with live ingest.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Columns we expect in the feature frame written by data_ingest.
REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ticker",
    "return_1",
    "return_24",
    "vol_24",
    "rsi_14",
    "sentiment_24h",
)


class FeatureFrameError(ValueError):
    """Raised when a feature Parquet is missing columns the backtest needs."""


def _read_one_parquet(path: Path) -> pd.DataFrame:
    """Read one Parquet file with the same workaround as forecast's loader.

    Uses ParquetFile.read() because Qdrant-style partition inference
    conflicts with our `ticker` data column when using pq.read_table().
    """
    return pq.ParquetFile(str(path)).read().to_pandas()


def load_historical_features(
    base_dir: str | Path,
    *,
    ticker: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load Parquet feature snapshots for one ticker over [start, end].

    Returns a DataFrame sorted by timestamp ascending, with the required
    columns. Empty DataFrame (not an exception) when no data for the window.

    The Hive layout is: dt=YYYY-MM-DD/ticker=AAPL/HH-MM-SS.parquet
    We scan all subdirs matching the date range AND the ticker — Python
    string compare on dt=YYYY-MM-DD does the right thing because the
    format is lexically sortable.

    Naive datetimes are interpreted as UTC; otherwise their existing tz
    is preserved. This matters because data_ingest writes a tz-aware UTC
    index — comparing it against a naive datetime would raise.
    """
    from datetime import UTC

    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    base = Path(base_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Feature base dir does not exist: {base}")

    # Date filter at the directory level avoids opening Parquet files
    # we don't need. With years of history this matters a lot.
    start_date = start.date().isoformat()
    end_date = end.date().isoformat()

    matching_files: list[Path] = []
    for dt_dir in sorted(base.glob("dt=*")):
        # dt=YYYY-MM-DD - extract the date suffix and compare lexically.
        dir_date = dt_dir.name.removeprefix("dt=")
        if dir_date < start_date or dir_date > end_date:
            continue
        for parquet_file in sorted(dt_dir.glob(f"ticker={ticker}/*.parquet")):
            matching_files.append(parquet_file)

    if not matching_files:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    frames = [_read_one_parquet(p) for p in matching_files]
    out = pd.concat(frames, axis=0)

    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise FeatureFrameError(
            f"Loaded frame is missing required columns: {missing}. "
            f"Got: {list(out.columns)}"
        )

    # Dedup on (ticker, timestamp) — same fix as the forecast loader.
    out = out.reset_index()
    out = out.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)]
    out = out.sort_values("timestamp").set_index("timestamp")
    return out
