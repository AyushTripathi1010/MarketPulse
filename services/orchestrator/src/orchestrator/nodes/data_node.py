"""
data_node — first node in the graph.

Calls the data_ingest service's /fetch endpoint to materialize the latest
OHLCV + news for the ticker. Returns the storage URI so subsequent nodes
know where to find the data.

Every node has the same signature: `(state: MarketState) -> MarketState`.
LangGraph merges the returned dict into the running state.
"""

from __future__ import annotations

import logging

import httpx

from orchestrator.config import settings
from orchestrator.state import MarketState

logger = logging.getLogger(__name__)


def run(state: MarketState) -> MarketState:
    """Fetch fresh OHLCV + news for the ticker."""
    ticker = state["ticker"]
    # Lookback large enough to feed the TFT encoder (96 hours) plus headroom.
    response = httpx.post(
        f"{settings.data_ingest_url}/fetch",
        json={"ticker": ticker, "lookback_days": 90, "interval": "1h"},
        timeout=60.0,
    )
    response.raise_for_status()
    body = response.json()

    logger.info(
        "data_node: ticker=%s rows=%d news=%d uri=%s",
        ticker,
        body["rows_written"],
        body["news_items"],
        body["storage_uri"],
    )

    # data_ingest's response includes news_items count but not the headlines
    # themselves (Phase 1 design decision — payloads stay small). For the
    # Critic we want headlines; Phase 4/5 will add an /news endpoint on
    # data_ingest. Until then, the critic operates on an empty news list
    # and the orchestrator passes a small synthesized note.
    return {
        "ohlcv_uri": body["storage_uri"],
        "recent_news": [],  # Phase 4 adds news passthrough
    }
