"""Configuration for the api_gateway service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API Gateway settings — URLs of internal services we proxy to."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    orchestrator_url: str = Field(default="http://orchestrator:8000")
    backtest_url: str = Field(default="http://backtest:8000")

    # CORS — list of origins allowed to hit this API. Set narrowly in prod.
    cors_allow_origins: list[str] = Field(
        default=["http://localhost:3000"]  # Next.js dev server
    )


settings = Settings()
