"""
Clean and join OHLCV with news.

Pure functions — no I/O, no network calls. Take dataframes in, return
dataframes out. This makes them trivial to unit-test (no mocks needed).
"""

from __future__ import annotations

import pandas as pd

from data_ingest.fetchers.alpaca_fetcher import NewsItem


def news_to_dataframe(items: list[NewsItem]) -> pd.DataFrame:
    """Convert a list of NewsItem dataclasses to a tidy DataFrame.

    Returns a DataFrame indexed by `created_at` (tz-aware UTC) with
    columns: headline, summary, symbols (csv string), source, url.

    An empty input returns an empty DataFrame with the right schema —
    callers can always assume the column set is stable.
    """
    if not items:
        # Returning an empty frame with the right columns avoids special-casing
        # downstream. KeyError on a missing column is a 5 AM debugging nightmare.
        return pd.DataFrame(
            columns=["headline", "summary", "symbols", "source", "url"]
        ).rename_axis(index="created_at")

    rows = [
        {
            "created_at": item.created_at,
            "headline": item.headline,
            "summary": item.summary,
            # Store symbols as comma-separated string — Parquet doesn't love
            # list-typed columns in some readers.
            "symbols": ",".join(item.symbols),
            "source": item.source,
            "url": item.url,
        }
        for item in items
    ]

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.set_index("created_at").sort_index()
    return df


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Apply hygiene to a yFinance OHLCV frame.

    Steps:
      1. Drop rows where Close is NaN (yfinance sometimes emits one per bad bar).
      2. Forward-fill the remaining gaps in OHLC (NOT Volume — that should be 0).
      3. Drop duplicate timestamps if any (keep last).
      4. Sort by timestamp ascending.

    Returns a new frame; never mutates the input.
    """
    out = df.copy()

    # 1. Drop bars with missing close — they're unrecoverable.
    out = out.dropna(subset=["Close"])

    # 2. Forward-fill the OHLC quartet only. Volume of 0 != Volume of NaN;
    #    we'd rather see real zeros than fabricate volume data.
    out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]].ffill()
    out["Volume"] = out["Volume"].fillna(0)

    # 3. De-dup. yfinance has been observed to emit two bars for the same
    #    timestamp around DST transitions.
    out = out[~out.index.duplicated(keep="last")]

    # 4. Ensure ascending order. Always good hygiene before time-series joins.
    out = out.sort_index()

    return out
