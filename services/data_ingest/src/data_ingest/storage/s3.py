"""
S3 storage backend (parallel to local_fs.py).

Activated when `S3_BUCKET` env var is set AND AWS credentials are present.
Falls back to local_fs.py if either is missing (handled in the FastAPI
handler, not here — this module assumes you know you want S3).
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _build_key(ticker: str, dt: datetime) -> str:
    """Same Hive-style partition layout as local_fs._build_path."""
    return (
        f"features/dt={dt:%Y-%m-%d}/"
        f"ticker={ticker}/"
        f"{dt:%H-%M-%S}.parquet"
    )


def save_features_s3(
    df: pd.DataFrame,
    ticker: str,
    *,
    bucket: str,
    as_of: datetime | None = None,
    aws_region: str | None = None,
) -> str:
    """Upload a feature DataFrame to S3 as a single Parquet object.

    Returns the s3:// URI of the uploaded object.
    """
    when = as_of or datetime.now(tz=df.index.tz if df.index.tz else None) or datetime.now()
    key = _build_key(ticker, when)

    # Serialize Parquet to an in-memory buffer; one PUT instead of streaming.
    # For our row counts (a few thousand at most) this is fine.
    buffer = BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, buffer, compression="snappy")
    buffer.seek(0)

    s3 = boto3.client("s3", region_name=aws_region)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

    return f"s3://{bucket}/{key}"
