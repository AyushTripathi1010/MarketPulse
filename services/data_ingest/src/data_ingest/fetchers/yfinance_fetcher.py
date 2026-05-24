"""
yFinance OHLCV fetcher with retry-on-failure.

yFinance is an UNOFFICIAL wrapper around Yahoo Finance's web API. The "official"
Yahoo Finance API shut down in 2017. yFinance scrapes the public web endpoints
that Yahoo uses for its own charts. That means:

  - It's free (no API key, no per-request cost).
  - It WILL fail occasionally — Yahoo changes selectors, rate-limits IPs,
    returns empty dataframes for fresh tickers, etc.
  - You MUST wrap it in retry logic with backoff. We use the `tenacity`
    library — the de-facto standard for retries in Python.

We name this module `yfinance_fetcher.py` (not `yfinance.py`) to avoid
Python's module-shadowing trap: `import yfinance` inside a file called
`yfinance.py` would re-import the file itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Tickers we currently support. Centralised in shared/constants for the
# whole project, but we re-import the type here for clarity.
from marketplus_shared.constants import SUPPORTED_TICKERS  # noqa: F401  (used by callers)

# Required OHLCV columns. If yfinance returns something missing one of these,
# we treat the response as malformed and retry.
EXPECTED_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")


class YFinanceError(RuntimeError):
    """Raised when yFinance returns malformed data after all retries.

    We define our own exception so callers don't need to know about
    yfinance internals — they catch `YFinanceError` and decide what to do.
    """


@retry(
    # Try up to 3 times before giving up. yFinance hiccups are usually transient.
    stop=stop_after_attempt(3),
    # Wait 2s, then 4s, then 8s between attempts. Standard exponential backoff.
    wait=wait_exponential(multiplier=2, min=2, max=10),
    # Only retry on our own malformed-data error or generic IO/network errors.
    # We do NOT retry on programmer errors like wrong ticker symbol.
    retry=retry_if_exception_type((YFinanceError, ConnectionError, TimeoutError)),
    reraise=True,
)
def fetch_ohlcv(ticker: str, lookback_days: int = 90, interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV (Open/High/Low/Close/Volume) for one ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL". Crypto uses suffix, e.g. "BTC-USD".
    lookback_days : int
        How many days back to fetch. yFinance's 1h interval is limited to
        about 730 days; for daily you can go years.
    interval : str
        yFinance interval string. "1h" for hourly, "1d" for daily.

    Returns
    -------
    pandas.DataFrame
        Indexed by timestamp (tz-aware, UTC). Columns: Open, High, Low,
        Close, Volume. Adds a `ticker` column for downstream joins.

    Raises
    ------
    YFinanceError
        If yFinance returns an empty dataframe or one missing columns
        after the retry budget is exhausted.
    """
    # yfinance accepts "period" as a string like "90d", "1y", "max".
    period = f"{lookback_days}d"

    # Use Ticker.history rather than download() — gives cleaner errors and
    # doesn't print to stdout.
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(period=period, interval=interval, auto_adjust=False)

    # Defensive checks: yfinance has been known to return an empty frame
    # when Yahoo rate-limits us OR when the ticker doesn't exist OR when
    # the interval is too granular for the lookback.
    if df.empty:
        raise YFinanceError(
            f"yfinance returned empty dataframe for ticker={ticker!r} "
            f"period={period!r} interval={interval!r}. "
            "Likely causes: rate limit, invalid ticker, or interval/period mismatch."
        )

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise YFinanceError(
            f"yfinance response for {ticker!r} is missing columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    # Normalize the index name and ensure UTC timezone — downstream code
    # joins on this and assumes tz-aware UTC.
    df.index = df.index.rename("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Add ticker column for joining with news later.
    df["ticker"] = ticker

    # Drop columns yfinance occasionally adds that we don't need (e.g. Dividends, Stock Splits).
    df = df[[*EXPECTED_COLUMNS, "ticker"]]

    return df


def fetch_last_close(ticker: str) -> tuple[datetime, float]:
    """Convenience helper: just the most recent timestamp + close price.

    Useful for the backtest service and for sanity-checking the forecast.
    Uses the daily interval so it's robust against the hourly endpoint's
    intraday gaps.
    """
    df = fetch_ohlcv(ticker, lookback_days=5, interval="1d")
    last_row = df.iloc[-1]
    # last_row.name is the index value (timestamp).
    timestamp: datetime = last_row.name  # type: ignore[assignment]
    return timestamp, float(last_row["Close"])
