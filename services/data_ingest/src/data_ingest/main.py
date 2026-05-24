"""
data_ingest — FastAPI entry point.

Phase 1: real /fetch endpoint that pulls OHLCV + news, joins them, computes
features, and writes Parquet to local FS (dev) or S3 (prod).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from marketplus_shared.models import HealthResponse

from data_ingest.config import settings
from data_ingest.fetchers.alpaca_fetcher import (
    AlpacaNewsError,
    fetch_news,
)
from data_ingest.fetchers.yfinance_fetcher import (
    YFinanceError,
    fetch_ohlcv,
)
from data_ingest.schemas import (
    FetchRequest,
    FetchResponse,
    LatestRequest,
    LatestResponse,
)
from data_ingest.storage.local_fs import (
    load_latest_features,
    save_features,
)
from data_ingest.storage.s3 import save_features_s3
from data_ingest.transforms.clean import clean_ohlcv, news_to_dataframe
from data_ingest.transforms.features import build_feature_frame

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks.

    Phase 2 will use this to warm up the boto3 client and validate credentials.
    """
    logger.info(
        "data_ingest starting. environment=%s use_s3=%s",
        settings.environment,
        settings.use_s3,
    )
    yield


app = FastAPI(
    title="MarketPulse — Data Ingest",
    version="0.1.0",
    description="Pulls OHLCV (yFinance) + news (Alpaca), lands Parquet on S3 or local FS.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe used by docker-compose, Lambda, and Make."""
    return HealthResponse(
        service="data_ingest",
        extras={
            "environment": settings.environment,
            "use_s3": settings.use_s3,
        },
    )


@app.post("/fetch", response_model=FetchResponse)
def fetch(req: FetchRequest) -> FetchResponse:
    """End-to-end ingest: fetch OHLCV + news, build features, persist.

    Returns the storage URI of the written Parquet file. Other services
    read it directly from there — we never ship the whole frame over HTTP.
    """
    # 1. OHLCV. Errors propagate as 502 — yFinance is the bad citizen, not us.
    try:
        ohlcv = fetch_ohlcv(
            req.ticker,
            lookback_days=req.lookback_days,
            interval=req.interval,
        )
    except YFinanceError as e:
        raise HTTPException(status_code=502, detail=f"yFinance failure: {e}") from e

    # 2. News. Failures here are non-fatal — we still want the OHLCV pipeline
    #    to land features, even if news is unavailable.
    try:
        news_items = fetch_news(
            req.ticker,
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            hours=24,
        )
    except AlpacaNewsError as e:
        logger.warning("Alpaca news failed (continuing without): %s", e)
        news_items = []

    # 3. Clean + feature-engineer. Pure functions, no I/O.
    cleaned = clean_ohlcv(ohlcv)
    news_df = news_to_dataframe(news_items)
    features = build_feature_frame(cleaned, news_df)

    # 4. Persist. Local FS if S3 isn't configured, S3 otherwise.
    now = datetime.now(UTC)
    if settings.use_s3:
        uri = save_features_s3(
            features,
            req.ticker,
            bucket=settings.s3_bucket,
            as_of=now,
            aws_region=settings.aws_region,
        )
    else:
        uri = save_features(
            features,
            req.ticker,
            base_dir=settings.local_data_dir,
            as_of=now,
        )

    return FetchResponse(
        ticker=req.ticker,
        rows_written=len(features),
        news_items=len(news_items),
        storage_uri=uri,
    )


@app.post("/latest", response_model=LatestResponse)
def latest(req: LatestRequest) -> LatestResponse:
    """Report the most recently persisted feature frame for a ticker.

    Used by other services to find out whether ingest has run yet and
    where to read the data from. Returns 404 if nothing has been written.
    """
    # We only support local-fs lookup here in Phase 1 — S3 listing is
    # cheap but we want to keep this endpoint dependency-light. The
    # orchestrator can call /fetch when it needs fresh data anyway.
    df = load_latest_features(req.ticker, base_dir=settings.local_data_dir)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No features stored yet for ticker={req.ticker!r}. Call POST /fetch first.",
        )

    return LatestResponse(
        ticker=req.ticker,
        rows=len(df),
        last_close=float(df["Close"].iloc[-1]),
        storage_uri=f"file://{settings.local_data_dir}",  # base; partitioned underneath
    )
