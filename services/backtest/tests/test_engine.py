"""
Tests for the walk-forward engine.

We inject a deterministic synthetic forecaster — no httpx, no forecast
service required. The same engine code runs against the live forecaster
in `httpx_forecaster()`; we just don't test that path here because
it would need a running forecast service.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    ForecastCall,
    _signal_from_forecast,
    run_walk_forward,
)


# ---------------------------------------------------------------------------
# _signal_from_forecast — pure function tests
# ---------------------------------------------------------------------------
def test_signal_long_when_band_above_current() -> None:
    """Predicted above current AND confidence_low also above → long (+1)."""
    f = ForecastCall(
        timestamp=datetime.now(UTC),
        predicted_price=101.0,
        confidence_low=100.5,
        confidence_high=101.5,
    )
    assert _signal_from_forecast(f, current_price=100.0) == 1


def test_signal_short_when_band_below_current() -> None:
    """Predicted below current AND confidence_high also below → short (-1)."""
    f = ForecastCall(
        timestamp=datetime.now(UTC),
        predicted_price=99.0,
        confidence_low=98.5,
        confidence_high=99.5,
    )
    assert _signal_from_forecast(f, current_price=100.0) == -1


def test_signal_no_trade_when_band_straddles_current() -> None:
    """If the confidence band crosses the current price, no trade (0)."""
    # Predicted +1% but the band reaches DOWN to current-2% → ambiguous.
    f = ForecastCall(
        timestamp=datetime.now(UTC),
        predicted_price=101.0,
        confidence_low=98.0,
        confidence_high=104.0,
    )
    assert _signal_from_forecast(f, current_price=100.0) == 0


def test_signal_respects_band_threshold() -> None:
    """A predicted move smaller than `band_threshold` is not a signal."""
    f = ForecastCall(
        timestamp=datetime.now(UTC),
        predicted_price=100.05,  # +0.05% — too small to clear default 0.2%
        confidence_low=100.04,
        confidence_high=100.06,
    )
    assert _signal_from_forecast(f, current_price=100.0) == 0


# ---------------------------------------------------------------------------
# run_walk_forward — full engine
# ---------------------------------------------------------------------------
def test_walk_forward_returns_empty_on_short_window(
    synthetic_features: pd.DataFrame,
) -> None:
    """If features < encoder + horizon + 1, the engine returns an empty result."""

    def _always_bullish(_e: pd.DataFrame, _h: int) -> ForecastCall:
        return ForecastCall(
            timestamp=datetime.now(UTC),
            predicted_price=200.0, confidence_low=199.0, confidence_high=201.0,
        )

    short = synthetic_features.head(10)
    result = run_walk_forward(short, _always_bullish, horizon_hours=4, min_encoder_length=96)
    assert result.history.empty
    assert result.metrics.num_trades == 0
    assert result.metrics.total_return == 0.0


def test_walk_forward_bullish_forecaster_takes_long_positions(
    synthetic_features: pd.DataFrame,
) -> None:
    """A forecaster that always predicts up should open longs every bar."""

    def _always_bullish(encoder: pd.DataFrame, _h: int) -> ForecastCall:
        last = float(encoder["Close"].iloc[-1])
        return ForecastCall(
            timestamp=datetime.now(UTC),
            predicted_price=last * 1.01,
            confidence_low=last * 1.005,
            confidence_high=last * 1.015,
        )

    result = run_walk_forward(synthetic_features, _always_bullish, horizon_hours=4, min_encoder_length=96)
    assert len(result.history) > 0
    # Every bar should have taken a long.
    assert (result.history["signal"] == 1).all()
    assert result.metrics.num_trades == len(result.history)


def test_walk_forward_no_signal_forecaster_zero_trades(
    synthetic_features: pd.DataFrame,
) -> None:
    """A forecaster with a wide band straddling current → no trades."""

    def _flat(encoder: pd.DataFrame, _h: int) -> ForecastCall:
        last = float(encoder["Close"].iloc[-1])
        return ForecastCall(
            timestamp=datetime.now(UTC),
            predicted_price=last,
            confidence_low=last * 0.95,
            confidence_high=last * 1.05,
        )

    result = run_walk_forward(synthetic_features, _flat, horizon_hours=4, min_encoder_length=96)
    assert result.metrics.num_trades == 0
    assert result.metrics.total_return == 0.0


def test_walk_forward_perfect_forecaster_makes_money(
    synthetic_features: pd.DataFrame,
) -> None:
    """A 'cheating' forecaster that knows the future should have positive total_return."""
    closes = synthetic_features["Close"].to_numpy()

    def _cheating(encoder: pd.DataFrame, horizon: int) -> ForecastCall:
        # `len(encoder)` is 1-indexed (includes current bar); peek `horizon` ahead.
        i = len(encoder) - 1
        true_future = float(closes[min(i + horizon, len(closes) - 1)])
        return ForecastCall(
            timestamp=datetime.now(UTC),
            predicted_price=true_future,
            # Tight band so it always clears the threshold.
            confidence_low=true_future * 0.999,
            confidence_high=true_future * 1.001,
        )

    result = run_walk_forward(synthetic_features, _cheating, horizon_hours=4, min_encoder_length=96)
    # An oracle MUST beat 0 — if it doesn't, the signal logic has a bug.
    assert result.metrics.total_return > 0
    # Hit rate should be high but not necessarily 100%: bars where the true
    # future move is smaller than band_threshold get filtered to signal=0,
    # which counts as a miss in our hit_rate metric (a 0 signal is never a
    # hit regardless of outcome). On a random-walk price series with std~0.5%
    # per bar, ~85% of bars should clear the 0.2% threshold over 4 hours.
    assert result.metrics.hit_rate > 0.7


def test_walk_forward_handles_forecaster_exception(
    synthetic_features: pd.DataFrame,
) -> None:
    """A forecaster that throws → that bar is skipped, others still run."""
    call_count = {"n": 0}

    def _flaky(encoder: pd.DataFrame, _h: int) -> ForecastCall:
        call_count["n"] += 1
        if call_count["n"] % 3 == 0:
            raise RuntimeError("simulated forecast service outage")
        last = float(encoder["Close"].iloc[-1])
        return ForecastCall(
            timestamp=datetime.now(UTC),
            predicted_price=last * 1.01, confidence_low=last * 1.005, confidence_high=last * 1.015,
        )

    result = run_walk_forward(synthetic_features, _flaky, horizon_hours=4, min_encoder_length=96)
    # Some bars succeeded, some were skipped. Engine should NOT have raised.
    assert 0 < len(result.history) < (200 - 96 - 4)
