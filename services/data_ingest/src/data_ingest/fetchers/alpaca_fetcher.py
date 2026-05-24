"""
Alpaca News API fetcher with graceful no-key fallback.

Alpaca's News API gives us financial-grade news headlines with optional
sentiment data, free of charge, with both REST and WebSocket interfaces.
Free tier is generous (no published RPS cap for news as of 2026).

If no Alpaca key is configured we DO NOT crash — we return an empty list
so local development works without secrets. That's a deliberate choice:
demos should never require live credentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One news headline with optional sentiment.

    Why a dataclass and not a Pydantic model?
      - This is INTERNAL to the fetcher, never crosses an HTTP boundary.
      - Dataclass is faster (no validation overhead) and slots=True saves
        memory when we hold lists of hundreds of items.
      - When we ship results out via /fetch we convert to the Pydantic
        models in shared/marketplus_shared/models.py.
    """

    headline: str
    summary: str
    created_at: datetime
    symbols: tuple[str, ...]
    source: str
    url: str


class AlpacaNewsError(RuntimeError):
    """Raised when Alpaca returns malformed data after all retries."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((AlpacaNewsError, ConnectionError, TimeoutError)),
    reraise=True,
)
def fetch_news(
    ticker: str,
    *,
    api_key: str | None,
    api_secret: str | None,
    hours: int = 24,
    limit: int = 50,
) -> list[NewsItem]:
    """Fetch the last `hours` of news headlines for one ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL". For crypto use the bare symbol "BTC"
        (Alpaca strips the "-USD" suffix internally).
    api_key, api_secret : str or None
        Alpaca paper-trading credentials. If EITHER is empty/None we
        return [] and log a warning — never crash.
    hours : int
        Look-back window in hours.
    limit : int
        Max news items to return (server-side cap).

    Returns
    -------
    list[NewsItem]
        Empty list if no credentials OR if no news in the window. Never None.
    """
    # Graceful degradation — local dev runs without keys.
    if not api_key or not api_secret:
        logger.warning(
            "No Alpaca credentials configured; returning empty news list. "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET in .env to enable."
        )
        return []

    # Import inside the function so the module loads even when alpaca-py
    # isn't installed (e.g. during static analysis of the package metadata).
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(api_key=api_key, secret_key=api_secret)

    # Normalize crypto-style tickers — Alpaca expects "BTC" not "BTC-USD".
    alpaca_symbol = ticker.split("-", 1)[0]

    request = NewsRequest(
        symbols=alpaca_symbol,
        start=datetime.now(UTC) - timedelta(hours=hours),
        end=datetime.now(UTC),
        limit=limit,
        include_content=False,  # summary only, saves bandwidth
    )

    try:
        response = client.get_news(request)
    except Exception as e:
        # Wrap any alpaca exception in our own type so retry logic can catch it.
        raise AlpacaNewsError(f"Alpaca news API failed for {ticker!r}: {e}") from e

    # response is an alpaca.data.models.NewsSet. Iterate its .news attribute.
    items: list[NewsItem] = []
    for n in response.news:
        items.append(
            NewsItem(
                headline=n.headline or "",
                summary=n.summary or "",
                created_at=n.created_at,
                symbols=tuple(n.symbols or ()),
                source=n.source or "",
                url=n.url or "",
            )
        )

    return items
