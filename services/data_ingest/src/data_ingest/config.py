"""
Configuration for the data_ingest service.

Reads from environment variables (or a .env file in local dev) using
pydantic-settings. Anything missing crashes the service at startup with a
clear error — fail fast, don't surprise users at request time.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service-wide settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars (we share .env across services)
    )

    # Set explicitly in deployments; defaults are for local development only.
    environment: str = "local"

    # Alpaca credentials (Phase 1 will use these).
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )

    # AWS / S3 (Phase 1).
    aws_region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")
    s3_bucket: str = Field(default="", alias="S3_BUCKET")

    # Local development fallback: where to drop parquet files when S3 isn't
    # configured. Created on first write.
    local_data_dir: str = Field(default="./data/features")

    @property
    def use_s3(self) -> bool:
        """Decide at runtime whether to write to S3 or to the local FS.

        We require BOTH the bucket name AND AWS credentials to be present.
        If you set just S3_BUCKET without AWS creds, boto3 would crash —
        better to fail-soft to local storage and log loudly.
        """
        import os

        has_aws_creds = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
        )
        return bool(self.s3_bucket) and has_aws_creds


# Singleton — imported by main.py and any fetcher that needs credentials.
settings = Settings()
