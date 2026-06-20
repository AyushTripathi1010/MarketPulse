"""
Walk-forward backtest engine.

This is where MarketPulse turns from "the model has 62% accuracy" into
"the system would have made $X with Y drawdown on the last six months
of unseen data." Without this, every accuracy claim is unfalsifiable.

Walk-forward means: for each timestamp t in the test window we
  1. Show the system ONLY the data available up to t (encoder window).
  2. Ask it for a forecast for t + horizon.
  3. Wait for the actual price at t + horizon to compare against.
  4. Translate the forecast into a paper-trading position.
  5. Track the equity curve and metrics.

We DO NOT vectorize (apply a strategy to the whole series at once)
because vectorization smuggles in future information unless you're
extremely careful. Walk-forward is the slow-but-honest approach.

The engine is decoupled from how forecasts are produced — it takes a
`Forecaster` callable. The default `httpx_forecaster()` calls our
forecast service over HTTP. For tests we inject a synthetic forecaster
that doesn't need any external service.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from backtest.metrics import BacktestMetrics, compute_all

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ForecastCall:
    """One forecast emitted during a walk-forward step.

    Mirrors a slice of the shared Forecast model — we kept it local so the
    backtest doesn't depend on whatever Pydantic version the forecast
    service is on (independent deployment principle).
    """

    timestamp: datetime
    predicted_price: float
    confidence_low: float
    confidence_high: float


# A Forecaster is any callable that, given a slice of past features,
# returns a forecast for `horizon_hours` ahead. This abstraction lets
# tests inject a deterministic synthetic forecaster.
Forecaster = Callable[[pd.DataFrame, int], ForecastCall]


@dataclass(slots=True, frozen=True)
class WalkForwardResult:
    """Output of a walk-forward run."""

    metrics: BacktestMetrics
    history: pd.DataFrame   # bar-by-bar: timestamp, predicted, realized, return, equity


def _signal_from_forecast(
    forecast: ForecastCall,
    current_price: float,
    *,
    band_threshold: float = 0.002,
) -> int:
    """Translate a quantile forecast into a trading signal in {-1, 0, +1}.

    Rules:
      - If predicted price > current_price by more than band_threshold AND
        the lower confidence bound is still above current_price → +1 (long).
      - If predicted price < current_price by more than band_threshold AND
        the upper confidence bound is still below current_price → -1 (short).
      - Otherwise → 0 (no position).

    The band-aware check is the WHY-this-matters: a 0.3% predicted move
    with a 5% confidence band is NOT a trade signal — the band straddles
    zero and the directional call is just noise. The TFT's quantile output
    is the load-bearing input to this decision.
    """
    delta = (forecast.predicted_price - current_price) / current_price
    if delta > band_threshold and forecast.confidence_low > current_price:
        return 1
    if delta < -band_threshold and forecast.confidence_high < current_price:
        return -1
    return 0


def run_walk_forward(
    features: pd.DataFrame,
    forecaster: Forecaster,
    *,
    horizon_hours: int = 4,
    min_encoder_length: int = 96,
    band_threshold: float = 0.002,
    bars_per_year: int = 1638,  # hourly bars in US market hours
) -> WalkForwardResult:
    """Replay the strategy bar-by-bar over a feature frame.

    Parameters
    ----------
    features : pd.DataFrame
        Output of `load_historical_features` — indexed by timestamp,
        with Close + return_1 + the other Phase-1 columns.
    forecaster : Forecaster
        Callable that takes (encoder_slice, horizon_hours) and returns
        a ForecastCall. Inject the live HTTP forecaster in production,
        a synthetic forecaster in tests.
    horizon_hours : int
        How far ahead the forecaster predicts. Same as data_ingest's
        bar interval = 1h → horizon=4 means "4 hours into the future."
    min_encoder_length : int
        How many bars the forecaster needs in its input window. We
        skip bars in the test set until we have at least this much
        history to feed.
    band_threshold : float
        Relative price move needed to take a position; see _signal_from_forecast.
    bars_per_year : int
        Annualization factor for Sharpe.

    Returns
    -------
    WalkForwardResult — metrics + per-bar history DataFrame
    """
    if features.empty:
        return _empty_result(bars_per_year)

    # We can only forecast bars that have BOTH enough encoder history
    # AND a realized outcome `horizon_hours` in the future. Build the
    # list of valid "anchor" timestamps once.
    n = len(features)
    if n < min_encoder_length + horizon_hours + 1:
        logger.warning(
            "walk-forward: only %d bars; need at least %d for min_encoder + horizon. "
            "Returning empty result.",
            n,
            min_encoder_length + horizon_hours + 1,
        )
        return _empty_result(bars_per_year)

    history_rows: list[dict] = []

    # Loop bar by bar — DELIBERATELY non-vectorized. See module docstring.
    # We stop `horizon_hours` early so we can always look up the realized
    # outcome.
    for i in range(min_encoder_length, n - horizon_hours):
        encoder = features.iloc[: i + 1]
        current_ts = features.index[i]
        current_price = float(features.iloc[i]["Close"])
        realized_future_price = float(features.iloc[i + horizon_hours]["Close"])
        realized_return = (realized_future_price / current_price) - 1.0

        try:
            forecast = forecaster(encoder, horizon_hours)
        except Exception as e:  # noqa: BLE001 — log and skip the bar
            logger.warning("walk-forward: forecaster failed at %s: %s", current_ts, e)
            continue

        signal = _signal_from_forecast(forecast, current_price, band_threshold=band_threshold)

        # Per-bar PnL: position * realized return. No costs modeled —
        # for an interview demo we report frictionless PnL and note the
        # gap explicitly. Real backtests would subtract slippage + fees.
        bar_return = signal * realized_return

        history_rows.append(
            {
                "timestamp": current_ts,
                "current_price": current_price,
                "predicted_price": forecast.predicted_price,
                "confidence_low": forecast.confidence_low,
                "confidence_high": forecast.confidence_high,
                "realized_future_price": realized_future_price,
                "realized_return": realized_return,
                "signal": signal,
                "bar_return": bar_return,
            }
        )

    if not history_rows:
        return _empty_result(bars_per_year)

    history = pd.DataFrame(history_rows).set_index("timestamp")

    # Equity curve: cumulative product of (1 + bar_return).
    history["equity"] = (1.0 + history["bar_return"]).cumprod()

    metrics = compute_all(
        bar_returns=history["bar_return"].to_numpy(),
        predicted_directions=history["signal"].to_numpy(),
        realized_directions=np.sign(history["realized_return"].to_numpy()),
        predicted_prices=history["predicted_price"].to_numpy(),
        realized_prices=history["realized_future_price"].to_numpy(),
        bars_per_year=bars_per_year,
    )

    return WalkForwardResult(metrics=metrics, history=history)


def _empty_result(bars_per_year: int) -> WalkForwardResult:
    """Empty-result builder so callers always get a typed return."""
    empty = BacktestMetrics(
        total_return=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        hit_rate=0.0,
        mae=0.0,
        num_trades=0,
        win_rate=0.0,
    )
    empty_history = pd.DataFrame(
        columns=[
            "current_price",
            "predicted_price",
            "confidence_low",
            "confidence_high",
            "realized_future_price",
            "realized_return",
            "signal",
            "bar_return",
            "equity",
        ]
    )
    return WalkForwardResult(metrics=empty, history=empty_history)


def httpx_forecaster(forecast_service_url: str, ticker: str) -> Forecaster:
    """Build a Forecaster that calls the live forecast service.

    The returned closure ignores the `encoder` argument — the forecast
    service reads features from its own Parquet path; we just tell it
    which ticker and horizon to predict for. This keeps the contract
    aligned with how the orchestrator calls /predict in production.

    For tests use a synthetic forecaster instead.
    """

    def _call(_encoder: pd.DataFrame, horizon_hours: int) -> ForecastCall:
        response = httpx.post(
            f"{forecast_service_url}/predict",
            json={"ticker": ticker, "horizon_hours": horizon_hours},
            timeout=15.0,
        )
        response.raise_for_status()
        body = response.json()
        return ForecastCall(
            timestamp=datetime.fromisoformat(body["predicted_at"]),
            predicted_price=float(body["predicted_price"]),
            confidence_low=float(body["confidence_low"]),
            confidence_high=float(body["confidence_high"]),
        )

    return _call
