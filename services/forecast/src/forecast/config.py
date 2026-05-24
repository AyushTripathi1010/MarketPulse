"""Configuration for the forecast service. See data_ingest/config.py for the pattern."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Forecast service settings — model location and MLflow tracking."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    # S3 path to the latest TFT checkpoint (Phase 2 will populate this).
    s3_bucket: str = Field(default="marketplus-local", alias="S3_BUCKET")
    tft_checkpoint_key: str = Field(default="models/tft/latest.ckpt")

    # MLflow tracking — DagsHub-hosted (free 10GB).
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")


settings = Settings()
