"""
MarketState — the typed dict that LangGraph passes between nodes.

This is the "ArtifactStore" from the master plan. Each node receives the
current state, does its work, and returns a partial update that LangGraph
merges in. By the end of the graph the state has accumulated everything
from raw OHLCV through final report URL.

Why TypedDict and not a Pydantic BaseModel?
  - LangGraph's State type expects a TypedDict.
  - The state is INTERNAL to the graph — no HTTP/JSON serialization layer
    needs runtime validation on it. Pydantic's checks would be wasted work.
  - Each node returns a *partial* dict; LangGraph merges. TypedDict
    `total=False` semantics give us the right shape implicitly.
"""

from __future__ import annotations

from typing import TypedDict

from marketplus_shared.models import Critique, Forecast


class MarketState(TypedDict, total=False):
    """Accumulating state across the orchestrator graph.

    Fields are added by the corresponding nodes:
      data_node     → ohlcv_uri, recent_news
      forecast_node → forecast
      critic_node   → critique
      report_node   → report_markdown
      drift_node    → drift_detected, retrain_triggered
    """

    # Input fields — set when the user calls /run.
    ticker: str
    horizon_hours: int

    # Filled by data_node.
    ohlcv_uri: str
    recent_news: list[str]

    # Filled by forecast_node.
    forecast: Forecast

    # Filled by critic_node.
    critique: Critique

    # Filled by report_node.
    report_markdown: str

    # Filled by drift_node (the conditional retrigger step).
    drift_detected: bool
    retrain_triggered: bool

    # Cross-cutting: trace id for debugging this entire run end-to-end.
    trace_id: str
