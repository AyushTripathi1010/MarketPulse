"""
drift_node — fifth node, conditional terminus of the graph.

After producing a report we check whether the model's recent error has
crept above the drift threshold (defined in marketplus_shared.constants).
If yes, we'd kick off a retraining job (Phase 4+ behavior). For Phase 3
we just SET a flag in state — the actual retraining trigger is wired
in Phase 4 when fine-tuning lands.

The drift signal itself is approximate in Phase 3:
  We compare the latest forecast's confidence-band width to a recent
  baseline. A widening band is a cheap proxy for "model is becoming
  less certain about this ticker." Phase 4 will replace this with a
  proper rolling-MAE-from-MLflow check.
"""

from __future__ import annotations

import logging

from marketplus_shared.constants import DRIFT_MAE_THRESHOLD

from orchestrator.state import MarketState

logger = logging.getLogger(__name__)


def run(state: MarketState) -> MarketState:
    """Compute a drift signal and mark whether to retrigger fine-tune."""
    forecast = state["forecast"]
    # Phase 3 proxy: relative band width. Phase 4 will replace this with
    # a proper 7-day rolling MAE pulled from MLflow.
    band_width = forecast.confidence_high - forecast.confidence_low
    relative_width = band_width / forecast.predicted_price if forecast.predicted_price else 0.0

    drift_detected = relative_width > DRIFT_MAE_THRESHOLD
    logger.info(
        "drift_node: relative_band_width=%.4f threshold=%.4f drift=%s",
        relative_width,
        DRIFT_MAE_THRESHOLD,
        drift_detected,
    )

    return {
        "drift_detected": drift_detected,
        # Phase 4 will set this True after enqueueing a real retrain job.
        "retrain_triggered": False,
    }


def should_retrigger(state: MarketState) -> bool:
    """Conditional edge: True → run the retrigger node, False → __end__."""
    return bool(state.get("drift_detected"))
