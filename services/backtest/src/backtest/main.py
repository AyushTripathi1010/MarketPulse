"""
backtest — FastAPI entry point.

Phase 6: real /run endpoint that loads historical features, replays them
through the forecast service via httpx, computes Sharpe/drawdown/hit-rate,
writes JSON + Markdown + PNG to disk (or S3 in production), and returns
the URIs.

The backtest service is the ONLY one that calls the forecast service
during a backtest run. It does NOT go through the orchestrator — we
specifically want to swap forecasters (live model vs. synthetic baseline)
to compute ablation numbers, and routing through LangGraph would force
every backtest to also pay for critic + report calls.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, time

from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse

from backtest.config import settings
from backtest.data_loader import load_historical_features
from backtest.engine import httpx_forecaster, run_walk_forward
from backtest.report import write_report
from backtest.schemas import (
    BacktestMetricsSchema,
    BacktestRequest,
    BacktestResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Just log config — no heavy startup work."""
    logger.info(
        "backtest starting. data_dir=%s output_dir=%s forecast_url=%s",
        settings.local_data_dir,
        settings.local_output_dir,
        settings.forecast_url,
    )
    yield


app = FastAPI(
    title="MarketPulse — Backtest",
    version="0.1.0",
    description="Walk-forward backtester. Replays the forecast pipeline over historical data.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service="backtest",
        extras={
            "environment": settings.environment,
            "forecast_url": settings.forecast_url,
        },
    )


@app.post("/run", response_model=BacktestResponse)
def run(req: BacktestRequest) -> BacktestResponse:
    """Execute a walk-forward backtest.

    Failure modes:
      - No feature data in the requested window → 404 (caller should
        run data_ingest /fetch first).
      - Forecast service unreachable → 502.
      - Internal failure (e.g. matplotlib backend) → 500 with details.
    """
    start_dt = datetime.combine(req.start_date, time.min)
    end_dt = datetime.combine(req.end_date, time.max)

    try:
        features = load_historical_features(
            settings.local_data_dir,
            ticker=req.ticker,
            start=start_dt,
            end=end_dt,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if features.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No features in window for ticker={req.ticker!r}. "
                "Run data_ingest /fetch over this range first."
            ),
        )

    forecaster = httpx_forecaster(settings.forecast_url, req.ticker)

    try:
        result = run_walk_forward(
            features,
            forecaster,
            horizon_hours=req.horizon_hours,
            min_encoder_length=req.min_encoder_length or settings.default_min_encoder_length,
            band_threshold=req.band_threshold or settings.default_band_threshold,
            bars_per_year=settings.bars_per_year,
        )
    except Exception as e:  # noqa: BLE001 — the engine should not raise; if it does, surface it
        logger.exception("walk-forward failed")
        raise HTTPException(status_code=500, detail=f"walk-forward failed: {e}") from e

    artifacts = write_report(
        result,
        ticker=req.ticker,
        start=start_dt,
        end=end_dt,
        output_dir=settings.local_output_dir,
    )

    return BacktestResponse(
        ticker=req.ticker,
        metrics=BacktestMetricsSchema(
            total_return=result.metrics.total_return,
            sharpe=result.metrics.sharpe,
            max_drawdown=result.metrics.max_drawdown,
            hit_rate=result.metrics.hit_rate,
            win_rate=result.metrics.win_rate,
            mae=result.metrics.mae,
            num_trades=result.metrics.num_trades,
        ),
        num_bars=len(result.history),
        report_uri=f"file://{artifacts['markdown']}",
        json_uri=f"file://{artifacts['json']}",
        equity_curve_uri=f"file://{artifacts['png']}" if artifacts["png"] else "",
    )
