"""Configuration for the backtest service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backtest service settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    # Where data_ingest landed the historical Parquet features. Same
    # default as data_ingest.config so a single docker-compose run shares.
    local_data_dir: str = Field(default="./data/features")

    # Where the backtest service writes its JSON/MD/PNG artifacts.
    local_output_dir: str = Field(default="./data/backtests")

    # The forecast service URL. Default matches docker-compose service name.
    forecast_url: str = Field(default="http://forecast:8000", alias="FORECAST_URL")

    # Hourly bars per US market year (6.5h × 252d) — used to annualize Sharpe.
    bars_per_year: int = Field(default=1638)

    # Backtest engine knobs. Sensible defaults; the API lets callers override.
    default_horizon_hours: int = 4
    default_min_encoder_length: int = 96
    default_band_threshold: float = 0.002

    # S3 destination for production (Phase 7). Empty disables uploads.
    s3_bucket: str = Field(default="", alias="S3_BUCKET")


settings = Settings()
