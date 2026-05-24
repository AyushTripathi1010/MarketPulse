"""Configuration for the backtest service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backtest service settings — S3 bucket for results, orchestrator URL."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"
    s3_bucket: str = Field(default="marketplus-local", alias="S3_BUCKET")
    # Backtest replays through the orchestrator to use the SAME logic as live.
    orchestrator_url: str = Field(default="http://orchestrator:8000")


settings = Settings()
