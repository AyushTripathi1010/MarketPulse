"""
Tests for the report module.

We exercise the Markdown writer, JSON writer, and the PNG renderer.
matplotlib uses the Agg backend (headless) so tests don't need a display.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import WalkForwardResult
from backtest.metrics import BacktestMetrics
from backtest.report import render_equity_curve, write_report, _interpret


def _make_result(num_bars: int = 50) -> WalkForwardResult:
    """A non-trivial result with a non-flat equity curve."""
    rng = np.random.default_rng(seed=7)
    bar_returns = rng.normal(0.001, 0.005, num_bars)
    history = pd.DataFrame(
        {
            "current_price": np.full(num_bars, 100.0),
            "predicted_price": 100.0 + rng.normal(0, 0.5, num_bars),
            "confidence_low": 99.0,
            "confidence_high": 101.0,
            "realized_future_price": 100.0 + rng.normal(0, 0.6, num_bars),
            "realized_return": rng.normal(0.001, 0.005, num_bars),
            "signal": rng.choice([-1, 1], num_bars),
            "bar_return": bar_returns,
        },
        index=pd.date_range("2026-01-01", periods=num_bars, freq="1h", tz="UTC"),
    )
    history["equity"] = (1.0 + history["bar_return"]).cumprod()

    metrics = BacktestMetrics(
        total_return=float(history["equity"].iloc[-1] - 1.0),
        sharpe=1.2,
        max_drawdown=-0.06,
        hit_rate=0.55,
        mae=0.4,
        num_trades=num_bars,
        win_rate=0.52,
    )
    return WalkForwardResult(metrics=metrics, history=history)


# ---------------------------------------------------------------------------
# write_report — all three artifacts
# ---------------------------------------------------------------------------
def test_write_report_creates_json_md_png(tmp_path: Path) -> None:
    result = _make_result()
    artifacts = write_report(
        result,
        ticker="AAPL",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 3, tzinfo=UTC),
        output_dir=tmp_path,
    )
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()
    assert artifacts["png"] != ""
    assert Path(artifacts["png"]).exists()


def test_write_report_json_is_valid_and_includes_ticker(tmp_path: Path) -> None:
    result = _make_result()
    artifacts = write_report(
        result,
        ticker="MSFT",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 3, tzinfo=UTC),
        output_dir=tmp_path,
    )
    payload = json.loads(Path(artifacts["json"]).read_text())
    assert payload["ticker"] == "MSFT"
    assert "sharpe" in payload
    assert "max_drawdown" in payload


def test_write_report_markdown_renders_metrics(tmp_path: Path) -> None:
    """The markdown should contain the ticker + Sharpe value + drawdown value."""
    result = _make_result()
    artifacts = write_report(
        result, ticker="NVDA",
        start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 3, tzinfo=UTC),
        output_dir=tmp_path,
    )
    text = Path(artifacts["markdown"]).read_text()
    assert "NVDA" in text
    assert "1.200" in text  # sharpe
    assert "-6.00%" in text  # drawdown


def test_write_report_handles_empty_result(tmp_path: Path) -> None:
    """An empty result still produces all three artifacts (PNG may be empty path)."""
    empty = WalkForwardResult(
        metrics=BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0),
        history=pd.DataFrame(),
    )
    artifacts = write_report(
        empty, ticker="AAPL",
        start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 2, tzinfo=UTC),
        output_dir=tmp_path,
    )
    # JSON + MD always written.
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()
    # PNG returns empty path when history is empty (graceful skip).
    assert artifacts["png"] == ""


# ---------------------------------------------------------------------------
# render_equity_curve — direct call
# ---------------------------------------------------------------------------
def test_render_equity_curve_creates_png(tmp_path: Path) -> None:
    result = _make_result(num_bars=30)
    out = tmp_path / "test_equity.png"
    path = render_equity_curve(result.history, out, ticker="AAPL")
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 1000  # PNG is at least a few KB


def test_render_equity_curve_empty_history_returns_none(tmp_path: Path) -> None:
    """Empty history → render returns None (no crash)."""
    path = render_equity_curve(pd.DataFrame(), tmp_path / "x.png", ticker="AAPL")
    assert path is None


# ---------------------------------------------------------------------------
# _interpret — deterministic prose
# ---------------------------------------------------------------------------
def test_interpret_strong_sharpe() -> None:
    m = BacktestMetrics(0.1, 2.5, -0.05, 0.6, 0.3, 100, 0.55)
    text = _interpret(m)
    assert "strong" in text.lower() or "edge" in text.lower()


def test_interpret_losing_strategy() -> None:
    m = BacktestMetrics(-0.1, -0.5, -0.15, 0.4, 0.3, 100, 0.4)
    text = _interpret(m)
    assert "lost money" in text.lower() or "≤ 0" in text


def test_interpret_zero_trades_flagged() -> None:
    m = BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    text = _interpret(m)
    assert "zero trades" in text.lower()
