"""
critic_node — third node.

Sends the forecast to the critic service. If the critic is unreachable
or 502s (Groq down), we DEGRADE GRACEFULLY: synthesize an UNKNOWN
critique so the pipeline can still produce a report. This is the
"never serve fresh-looking content without a confidence tag" principle.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from marketplus_shared.models import Critique, Regime, RegimeLabel

from orchestrator.config import settings
from orchestrator.state import MarketState

logger = logging.getLogger(__name__)


def _degraded_critique(reason: str) -> Critique:
    """Build a Critique that flags the run as degraded."""
    return Critique(
        regime=Regime(
            label=RegimeLabel.SIDEWAYS,
            confidence=0.5,
            classified_at=datetime.now(UTC),
        ),
        analogues=[],
        confidence="UNKNOWN",
        reasoning=f"Critic agent unavailable; pipeline ran in degraded mode. Reason: {reason}",
    )


def run(state: MarketState) -> MarketState:
    """Grade the forecast's confidence, degrading gracefully on failure."""
    forecast = state["forecast"]
    try:
        response = httpx.post(
            f"{settings.critic_url}/critique",
            json={
                "forecast": forecast.model_dump(mode="json"),
                "recent_news": state.get("recent_news", []),
            },
            timeout=30.0,
        )
        response.raise_for_status()
        critique = Critique.model_validate(response.json())
        logger.info(
            "critic_node: regime=%s confidence=%s",
            critique.regime.label.value,
            critique.confidence,
        )
        return {"critique": critique}
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("critic_node: degraded due to %s", e)
        return {"critique": _degraded_critique(str(e))}
