"""Pydantic request/response schemas for the backtest service."""

from datetime import date

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """Run a walk-forward backtest for a ticker over a date range."""

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    start_date: date
    end_date: date
    horizon_hours: int = Field(default=4, ge=1, le=24)


class BacktestMetrics(BaseModel):
    """Aggregate performance metrics computed at the end of a backtest run."""

    total_predictions: int
    hit_rate: float = Field(..., ge=0.0, le=1.0)  # fraction predicted-direction-correct
    mae: float  # mean absolute error in price units
    sharpe_ratio: float
    max_drawdown: float  # negative number (e.g. -0.085 = 8.5% drawdown)


class BacktestResponse(BaseModel):
    """Result of one backtest run."""

    ticker: str
    metrics: BacktestMetrics
    equity_curve_s3_path: str  # link to plot stored in S3
    report_s3_path: str  # link to markdown summary stored in S3
