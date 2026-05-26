"""
Load feature Parquet files into a pandas DataFrame ready for TFT training.

The data_ingest service writes Parquet files in Hive-partitioned layout:
  data/features/dt=YYYY-MM-DD/ticker=AAPL/HH-MM-SS.parquet
We read all of them into one frame here, then hand off to TimeSeriesDataSet.

This module intentionally does NOT import from data_ingest — the services
deploy independently. We re-implement the small Parquet read helper rather
than create a runtime coupling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# Columns we expect in the feature frame coming out of data_ingest.
# If any of these is missing we fail loudly — better than training on a
# silently-truncated dataset.
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
    """Raised when a Parquet file doesn't have the columns the TFT expects."""


def _read_one_parquet(path: Path) -> pd.DataFrame:
    """Read a single Parquet file, restoring the tz-aware index.

    Uses ParquetFile.read() instead of pq.read_table() because the latter
    auto-infers Hive partitions and clashes with our `ticker` data column —
    see services/data_ingest/src/data_ingest/storage/parquet_io.py for the
    same workaround.
    """
    table = pq.ParquetFile(str(path)).read()
    return table.to_pandas()


def load_features(base_dir: str | Path, ticker: str | None = None) -> pd.DataFrame:
    """Load every Parquet under base_dir (optionally filtered by ticker) into one frame.

    Parameters
    ----------
    base_dir : str or Path
        Root of the Hive-partitioned features tree, e.g. "./data/features".
    ticker : str or None
        If provided, only read partitions for this ticker. None reads all.

    Returns
    -------
    pd.DataFrame
        Concatenated frame across all partitions, with index "timestamp"
        (tz-aware UTC), columns matching REQUIRED_COLUMNS, sorted by
        (ticker, timestamp), de-duplicated on (ticker, timestamp).

    Raises
    ------
    FileNotFoundError
        base_dir doesn't exist.
    FeatureFrameError
        A loaded frame is missing one of REQUIRED_COLUMNS.
    """
    base = Path(base_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Feature base dir does not exist: {base}")

    pattern = f"ticker={ticker}/*.parquet" if ticker else "*.parquet"
    files = sorted(base.rglob(pattern))
    if not files:
        # Empty result is valid (nothing ingested yet). Caller decides what to do.
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    frames = [_read_one_parquet(p) for p in files]
    out = pd.concat(frames, axis=0)

    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise FeatureFrameError(
            f"Loaded frame is missing required columns: {missing}. "
            f"Got: {list(out.columns)}"
        )

    # The partition files are time-windowed snapshots — running data_ingest
    # twice for the same window writes overlapping rows. De-dup on the
    # (ticker, timestamp) pair, NOT timestamp alone — different tickers can
    # legitimately share timestamps and we must keep both.
    out = out.reset_index()
    out = out.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
    out = out.sort_values(by=["ticker", "timestamp"]).set_index("timestamp")
    return out


def to_training_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Transform a loaded feature frame into the shape pytorch-forecasting wants.

    TimeSeriesDataSet expects:
      - A flat (non-indexed) DataFrame.
      - An integer `time_idx` column that is monotonic within each group.
      - A `group_ids` column (here: ticker).
      - All feature columns as regular columns.

    We compute time_idx as the rank of the timestamp WITHIN each ticker,
    which gives an integer that goes 0, 1, 2, ... and naturally handles
    any gaps. (TFT can tolerate gaps; the time_idx just has to be ordered.)
    """
    df = features.reset_index().rename(columns={"timestamp": "ts"})
    # Use Series.groupby().cumcount() to get integer position within group.
    df["time_idx"] = df.groupby("ticker").cumcount().astype("int64")
    # pytorch-forecasting wants its known/observed features as float32.
    numeric_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "return_1", "return_24", "vol_24", "rsi_14", "sentiment_24h",
    ]
    df[numeric_cols] = df[numeric_cols].astype("float32")
    # Drop rows that still have NaN in the target (typically the first bar
    # of each ticker where return_1 is undefined).
    df = df.dropna(subset=["return_1"])
    return df.reset_index(drop=True)
