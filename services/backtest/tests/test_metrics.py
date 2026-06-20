"""
Pure-function tests for the metrics module.

These are the most fundamental tests in Phase 6: if the math is wrong,
every backtest result is meaningless. We pin exact expected values where
possible and use generous tolerance only where rounding/float drift is
genuinely unavoidable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtest.metrics import (
    BARS_PER_YEAR_DAILY,
    BARS_PER_YEAR_HOURLY,
    compute_all,
    equity_curve,
    hit_rate,
    mae,
    max_drawdown,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------
def test_sharpe_zero_returns() -> None:
    """All-zero returns → Sharpe 0 (no edge, no volatility)."""
    assert sharpe_ratio(np.zeros(100)) == 0.0


def test_sharpe_constant_positive_returns() -> None:
    """Constant positive returns → +inf Sharpe in theory; we return 0.0 to keep ranking sortable."""
    # All returns identical → zero std → division by zero in raw formula.
    assert sharpe_ratio(np.full(100, 0.001)) == 0.0


def test_sharpe_too_few_samples() -> None:
    """A single sample isn't statistically meaningful → 0.0, not NaN."""
    assert sharpe_ratio(np.array([0.01])) == 0.0
    assert sharpe_ratio(np.array([])) == 0.0


def test_sharpe_annualizes_with_sqrt_n() -> None:
    """Annualization scales the per-bar Sharpe by sqrt(bars_per_year)."""
    returns = np.array([0.01, -0.005, 0.002, -0.001, 0.003])
    s_daily = sharpe_ratio(returns, bars_per_year=BARS_PER_YEAR_DAILY)
    s_hourly = sharpe_ratio(returns, bars_per_year=BARS_PER_YEAR_HOURLY)
    # hourly bars_per_year > daily → higher Sharpe for same returns
    assert s_hourly > s_daily > 0


def test_sharpe_negative_when_losing() -> None:
    """A clearly-losing strategy gets negative Sharpe."""
    rng = np.random.default_rng(seed=0)
    losing = rng.normal(loc=-0.001, scale=0.005, size=1000)
    assert sharpe_ratio(losing) < 0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------
def test_max_drawdown_monotonic_curve_is_zero() -> None:
    """A monotonically increasing curve never goes below its running max → 0."""
    curve = np.array([1.0, 1.05, 1.1, 1.15, 1.2])
    assert max_drawdown(curve) == 0.0


def test_max_drawdown_exact_value() -> None:
    """100 → 200 → 50 → 100  =  peak 200 → trough 50  =  -75%."""
    curve = np.array([100.0, 200.0, 50.0, 100.0])
    assert math.isclose(max_drawdown(curve), -0.75, abs_tol=1e-9)


def test_max_drawdown_handles_single_point() -> None:
    """A 1-point curve has no drawdown."""
    assert max_drawdown(np.array([1.0])) == 0.0
    assert max_drawdown(np.array([])) == 0.0


def test_max_drawdown_always_non_positive() -> None:
    """Drawdown values must always be ≤ 0 regardless of the curve."""
    rng = np.random.default_rng(seed=1)
    curve = np.cumprod(1.0 + rng.normal(0, 0.01, 200))
    assert max_drawdown(curve) <= 0.0


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------
def test_hit_rate_all_correct() -> None:
    pred = np.array([1, 1, -1, -1, 1])
    real = np.array([1, 1, -1, -1, 1])
    assert hit_rate(pred, real) == 1.0


def test_hit_rate_no_position_counts_as_miss() -> None:
    """A 0 prediction never counts as a hit even if the real direction was also 0."""
    pred = np.array([0, 0, 0, 0])
    real = np.array([1, -1, 1, -1])
    assert hit_rate(pred, real) == 0.0


def test_hit_rate_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        hit_rate(np.array([1, 1]), np.array([1, 1, 1]))


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------
def test_mae_exact_match_is_zero() -> None:
    arr = np.array([100.0, 101.0, 99.5])
    assert mae(arr, arr) == 0.0


def test_mae_simple() -> None:
    pred = np.array([100.0, 101.0, 99.0])
    real = np.array([100.5, 100.0, 99.5])
    # diffs: 0.5, 1.0, 0.5  →  mean 0.6666...
    assert math.isclose(mae(pred, real), 2.0 / 3.0, abs_tol=1e-9)


def test_mae_empty_inputs_zero() -> None:
    assert mae([], []) == 0.0


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------
def test_equity_curve_compounding() -> None:
    """+10%, +10%, -10% → 1.1, 1.21, 1.089."""
    curve = equity_curve(np.array([0.1, 0.1, -0.1]))
    assert math.isclose(curve[0], 1.1, abs_tol=1e-9)
    assert math.isclose(curve[1], 1.21, abs_tol=1e-9)
    assert math.isclose(curve[2], 1.089, abs_tol=1e-9)


def test_equity_curve_starting_capital() -> None:
    curve = equity_curve(np.array([0.1, 0.1]), starting_capital=1000.0)
    assert math.isclose(curve[-1], 1210.0, abs_tol=1e-6)


def test_equity_curve_empty() -> None:
    """Empty returns → [starting_capital]."""
    curve = equity_curve(np.array([]))
    assert curve.shape == (1,)
    assert curve[0] == 1.0


# ---------------------------------------------------------------------------
# compute_all integration
# ---------------------------------------------------------------------------
def test_compute_all_returns_dataclass() -> None:
    bar_returns = np.array([0.01, -0.005, 0.002, -0.001, 0.003])
    pred = np.array([1, -1, 1, -1, 1])
    real = np.array([1, -1, 1, -1, 1])
    pred_prices = np.array([100.0, 101.0, 100.5, 100.6, 100.8])
    real_prices = np.array([101.0, 100.5, 100.7, 100.5, 100.9])

    m = compute_all(bar_returns, pred, real, pred_prices, real_prices)
    assert m.num_trades == 5
    assert m.hit_rate == 1.0
    assert m.sharpe > 0
    assert m.max_drawdown <= 0.0
    assert m.win_rate > 0.0
