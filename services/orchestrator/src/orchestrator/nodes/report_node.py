"""
report_node — fourth node.

Composes the markdown brief by calling the report service. Like the critic,
if the report service is down we don't fail the pipeline — we synthesize a
minimal markdown stub so the user at least sees the forecast.
"""

from __future__ import annotations

import logging

import httpx

from orchestrator.config import settings
from orchestrator.state import MarketState

logger = logging.getLogger(__name__)


def _fallback_markdown(ticker: str, predicted_price: float) -> str:
    """Bare-bones markdown if the report service is unreachable."""
    return (
        f"# {ticker} forecast (degraded mode)\n\n"
        f"Predicted price: ${predicted_price:.2f}\n\n"
        f"_Report service unavailable — full brief not generated._\n"
    )


def run(state: MarketState) -> MarketState:
    """Generate the intelligence brief."""
    forecast = state["forecast"]
    critique = state["critique"]
    try:
        response = httpx.post(
            f"{settings.report_url}/generate",
            json={
                "forecast": forecast.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
            },
            timeout=60.0,
        )
        response.raise_for_status()
        body = response.json()
        logger.info(
            "report_node: ticker=%s len=%d polished=%s",
            body["ticker"],
            len(body["markdown"]),
            body["polished"],
        )
        return {"report_markdown": body["markdown"]}
    except httpx.HTTPError as e:
        logger.warning("report_node: degraded due to %s", e)
        return {
            "report_markdown": _fallback_markdown(
                forecast.ticker, forecast.predicted_price
            )
        }
