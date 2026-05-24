"""
Configuration for the data_ingest service.

Reads from environment variables (or a .env file in local dev) using
pydantic-settings. Anything missing crashes the service at startup with a
clear error — fail fast, don't surprise users at request time.
"""

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
    s3_bucket: str = Field(default="marketplus-local", alias="S3_BUCKET")


# Singleton — imported by main.py and any fetcher that needs credentials.
settings = Settings()
