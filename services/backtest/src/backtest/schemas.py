"""Pydantic request/response schemas for the backtest service."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """Run a walk-forward backtest for a ticker over a date range."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    start_date: date
    end_date: date
    horizon_hours: int = Field(default=4, ge=1, le=24)
    # Optional overrides — defaults come from config.
    min_encoder_length: int | None = Field(default=None, ge=24, le=512)
    band_threshold: float | None = Field(default=None, ge=0.0, le=0.5)


class BacktestMetricsSchema(BaseModel):
    """Aggregate performance metrics computed at the end of a backtest run.

    Mirrors backtest.metrics.BacktestMetrics — Pydantic for HTTP boundary,
    dataclass for in-process. We keep them separate so the engine doesn't
    pay Pydantic validation overhead during a 10K-bar walk-forward.
    """

    total_return: float
    sharpe: float
    max_drawdown: float = Field(..., le=0.0)  # always non-positive
    hit_rate: float = Field(..., ge=0.0, le=1.0)
    win_rate: float = Field(..., ge=0.0, le=1.0)
    mae: float = Field(..., ge=0.0)
    num_trades: int = Field(..., ge=0)


class BacktestResponse(BaseModel):
    """Result of one backtest run."""

    ticker: str
    metrics: BacktestMetricsSchema
    num_bars: int
    # Paths to the rendered artifacts (file:// in local dev, s3:// in prod).
    report_uri: str
    json_uri: str
    equity_curve_uri: str  # empty string if the plot rendering failed


class ResultsRequest(BaseModel):
    """Fetch a previously-run backtest by ticker + date range."""

    ticker: str = Field(..., min_length=1, max_length=10)
    start_date: date
    end_date: date
