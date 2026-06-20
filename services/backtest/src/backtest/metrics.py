"""
Pure-function metric computations for a backtest run.

These functions are deliberately decoupled from the engine: numpy arrays
in, scalars (or arrays) out. No I/O, no global state, no surprises. This
makes them trivially unit-testable — you build a synthetic returns series
by hand, you assert the expected value, you're done.

The four metrics we compute:

1. **Sharpe ratio (annualized)** — risk-adjusted return.
2. **Max drawdown** — worst peak-to-trough loss the strategy ever felt.
3. **Hit rate** — fraction of predicted-direction calls that were right.
4. **MAE** — mean absolute error of predicted vs realized prices.

Sharpe + drawdown answer "is this strategy any good?"
Hit rate + MAE answer "is the underlying forecast model any good?"
Together they let us bisect blame: bad strategy with good forecast vs
good strategy with bad forecast.

See learning/17-sharpe-drawdown-and-finance-metrics.md for the deeper
discussion of what these mean and when each one lies to you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The risk-free rate convention. 0% is the right default in 2026 — short-end
# T-bill rates fluctuate, and using a non-zero rate just adds a constant
# offset to Sharpe without changing the relative comparison of strategies.
# If the user cares about an absolute Sharpe vs a real treasury benchmark,
# they pass a non-zero rate explicitly.
DEFAULT_RISK_FREE_RATE: float = 0.0

# How many bars are in a "year" — used to annualize Sharpe.
# 252 trading days for daily bars; for hourly bars during US market hours
# (6.5h/day × 252d) ≈ 1638 hours/year.
BARS_PER_YEAR_DAILY: int = 252
BARS_PER_YEAR_HOURLY: int = 1638


@dataclass(slots=True, frozen=True)
class BacktestMetrics:
    """All headline numbers from one backtest run.

    Frozen + slotted because these objects are RESULTS — they should never
    be mutated after creation, and we hold thousands of them in memory
    during parameter sweeps.
    """

    total_return: float            # cumulative return over the test window
    sharpe: float                  # annualized Sharpe (0 risk-free rate by default)
    max_drawdown: float            # max peak-to-trough drawdown, in [-1, 0]
    hit_rate: float                # fraction of correct directional predictions, [0, 1]
    mae: float                     # mean absolute price-prediction error, in price units
    num_trades: int                # how many positions the strategy opened
    win_rate: float                # fraction of trades with positive PnL, [0, 1]


def sharpe_ratio(
    returns: np.ndarray | pd.Series,
    *,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    bars_per_year: int = BARS_PER_YEAR_DAILY,
) -> float:
    """Annualized Sharpe ratio.

    Formula:  (mean(returns) - risk_free_per_bar) / std(returns) * sqrt(bars_per_year)

    Notes
    -----
    - We use SAMPLE std (ddof=1) to match the financial-industry convention.
      Numpy defaults to population std (ddof=0); we override.
    - Returns of length < 2 are not statistically meaningful — we return 0.0
      rather than NaN so downstream sorting/comparison doesn't break.
    - A zero-variance return series returns 0.0 instead of dividing by zero.

    Why we annualize:
      Sharpe is reported per-year industry-wide so analysts can compare
      strategies trading on different bar frequencies. Without annualization
      an hourly strategy would show ~10× the Sharpe of a daily one trading
      the same edge, which is just a unit choice, not real outperformance.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0

    rf_per_bar = risk_free_rate / bars_per_year
    excess = arr - rf_per_bar
    sigma = float(np.std(excess, ddof=1))
    # Epsilon-tolerance zero check: np.std of [0.001]*100 returns ~1e-19 due
    # to float roundoff, not exactly 0.0. Without this guard we'd compute
    # Sharpe = mu / 1e-19 ≈ 1e+16 — which is "infinity" dressed up as a
    # finite number and ruins downstream comparisons. Anything smaller than
    # 1e-12 is noise, not signal.
    if sigma < 1e-12 or not np.isfinite(sigma):
        return 0.0
    mu = float(np.mean(excess))
    return mu / sigma * np.sqrt(bars_per_year)


def max_drawdown(equity_curve: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough drawdown of an equity curve.

    Returns a NON-positive number, e.g. -0.085 = 8.5% peak-to-trough loss.
    A monotonically increasing curve returns 0.0.

    Algorithm:
      For each point, the drawdown = (curr_value / running_max) - 1.
      The min of that series is the worst drawdown.

    Why it matters more than total return:
      A strategy that goes 100 → 200 → 50 → 100 has total return 0 but
      max drawdown -75%. If you held it through the dip you went through
      a 75% loss — most real-world investors would have liquidated. Total
      return alone hides this.
    """
    arr = np.asarray(equity_curve, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    # Cumulative running max. cummax in numpy is np.maximum.accumulate.
    running_max = np.maximum.accumulate(arr)
    # Guard against zero or negative equity (rare but possible if strategy
    # blows up). Drawdown for non-positive equity is undefined; clip to 0.
    safe = np.where(running_max > 0, running_max, 1.0)
    drawdowns = arr / safe - 1.0
    return float(np.min(drawdowns))


def hit_rate(
    predicted_direction: np.ndarray | pd.Series,
    realized_direction: np.ndarray | pd.Series,
) -> float:
    """Fraction of times the predicted direction matched the realized direction.

    Both inputs are arrays of {-1, 0, +1} (or boolean-equivalent floats).
    A prediction of 0 (no view) is counted as a MISS regardless of outcome —
    a model that hedges by predicting 0 should not earn credit.

    For binary classification this collapses to accuracy. Reporting as
    "hit rate" matches industry convention.
    """
    pred = np.sign(np.asarray(predicted_direction, dtype=np.float64).ravel())
    real = np.sign(np.asarray(realized_direction, dtype=np.float64).ravel())
    if pred.size == 0:
        return 0.0
    if pred.size != real.size:
        raise ValueError(
            f"hit_rate: predicted ({pred.size}) and realized ({real.size}) must be the same length."
        )
    correct = (pred == real) & (pred != 0)
    return float(np.sum(correct) / pred.size)


def mae(predicted: np.ndarray | pd.Series, realized: np.ndarray | pd.Series) -> float:
    """Mean absolute error between predicted and realized values.

    Same length required. Returns 0.0 for empty inputs (rather than NaN)
    so downstream summary tables stay clean.

    Why MAE and not MSE for a backtest:
      MSE punishes outliers quadratically, which inflates the number for a
      single bad prediction and obscures average performance. MAE is in the
      original units (price dollars) and is robust to one-off shocks.
    """
    p = np.asarray(predicted, dtype=np.float64).ravel()
    r = np.asarray(realized, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    if p.size != r.size:
        raise ValueError(
            f"mae: predicted ({p.size}) and realized ({r.size}) must be the same length."
        )
    return float(np.mean(np.abs(p - r)))


def equity_curve(returns: np.ndarray | pd.Series, starting_capital: float = 1.0) -> np.ndarray:
    """Translate a returns series into a running equity curve.

    starting_capital=1.0 by default → curve in "multiples of initial capital".
    Use 10_000.0 if you want a dollar-shaped curve for plots.

    We use ARITHMETIC compounding here (not log-additive) because our
    `returns` array is bar-level percentage returns and arithmetic
    compounding is what an actual brokerage account experiences.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.asarray([starting_capital])
    growth = 1.0 + arr
    return starting_capital * np.cumprod(growth)


def compute_all(
    bar_returns: np.ndarray | pd.Series,
    predicted_directions: np.ndarray | pd.Series,
    realized_directions: np.ndarray | pd.Series,
    predicted_prices: np.ndarray | pd.Series,
    realized_prices: np.ndarray | pd.Series,
    *,
    bars_per_year: int = BARS_PER_YEAR_HOURLY,
) -> BacktestMetrics:
    """Compute every headline metric in one call.

    Convenience wrapper used by the engine. The individual functions stay
    callable so unit tests can target each metric in isolation.
    """
    curve = equity_curve(bar_returns)
    return BacktestMetrics(
        total_return=float(curve[-1] - 1.0) if curve.size else 0.0,
        sharpe=sharpe_ratio(bar_returns, bars_per_year=bars_per_year),
        max_drawdown=max_drawdown(curve),
        hit_rate=hit_rate(predicted_directions, realized_directions),
        mae=mae(predicted_prices, realized_prices),
        num_trades=int(np.sum(np.asarray(predicted_directions) != 0)),
        win_rate=_win_rate(bar_returns),
    )


def _win_rate(bar_returns: np.ndarray | pd.Series) -> float:
    """Fraction of bars with strictly positive return.

    Distinct from hit_rate: a strategy can predict direction correctly but
    still LOSE money on a bar (e.g., directional call right, but the move
    was smaller than transaction costs). Tracking both surfaces that gap.
    """
    arr = np.asarray(bar_returns, dtype=np.float64).ravel()
    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        return 0.0
    return float(np.sum(nonzero > 0) / nonzero.size)
