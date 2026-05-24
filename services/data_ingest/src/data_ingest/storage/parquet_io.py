"""
Parquet reader/writer helpers.

Why Parquet?
  - Columnar: when we query "Close + return_1 over 90 days" we read ONLY
    those columns, not the whole file. 10x faster than CSV for analytics.
  - Compressed: snappy compression gives ~5x size reduction over CSV.
  - Schema-preserving: dtypes survive a round-trip. CSV turns everything
    into strings; you fight float-vs-int every time you read it back.
  - Industry-standard: every Spark, Athena, Redshift Spectrum job reads it.

We use pyarrow as the engine; it's the canonical Parquet library and what
pandas defaults to from version 2+.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to a single Parquet file.

    The parent directory is created if missing. Existing file is overwritten.
    Uses snappy compression by default — best speed-vs-size ratio in 2026.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # pyarrow round-trips pandas tz-aware datetime indices correctly only if
    # we go through Table.from_pandas. Direct df.to_parquet works too but the
    # explicit table conversion is friendlier for debugging schema issues.
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path, compression="snappy")


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file back into a DataFrame.

    The original index (including tz-aware timestamps) is restored.

    We use ParquetFile().read() instead of pq.read_table() because the latter
    tries to infer Hive-style partition columns from the directory structure
    (e.g. `ticker=AAPL/...`) and conflicts with our actual `ticker` data column.
    """
    table = pq.ParquetFile(str(path)).read()
    return table.to_pandas()
