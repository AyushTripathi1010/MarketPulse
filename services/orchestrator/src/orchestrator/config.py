"""Configuration for the orchestrator service.

The orchestrator is the only service that talks to all the others, so it
holds the URLs of every internal service. In local dev (docker-compose)
these resolve via the compose network. In Lambda, they're API Gateway URLs.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Orchestrator service settings — URLs for every downstream service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    # Downstream service URLs.
    data_ingest_url: str = Field(default="http://data_ingest:8000", alias="DATA_INGEST_URL")
    forecast_url: str = Field(default="http://forecast:8000", alias="FORECAST_URL")
    critic_url: str = Field(default="http://critic:8000", alias="CRITIC_URL")
    report_url: str = Field(default="http://report:8000", alias="REPORT_URL")

    # Langfuse — observability for every LLM call in the graph (Phase 3).
    langfuse_host: str = Field(default="http://langfuse:3000", alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")


settings = Settings()
