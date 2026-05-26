"""Configuration for the forecast service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Forecast service settings — model location and MLflow tracking."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    # Where to read input features written by data_ingest. Same default
    # as data_ingest.config so a single docker-compose run shares the dir.
    local_data_dir: str = Field(default="./data/features")

    # Where the latest trained checkpoint lives on disk.
    # Production loads from S3; dev loads from this local path.
    local_checkpoint_path: str = Field(default="./data/models/tft-best.ckpt")
    local_dataset_path: str = Field(default="./data/models/train_ds.pkl")

    # S3 (Phase 7 will populate these for prod inference).
    s3_bucket: str = Field(default="", alias="S3_BUCKET")
    tft_checkpoint_key: str = Field(default="models/tft/latest.ckpt")
    aws_region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")

    # MLflow tracking — DagsHub-hosted in prod, file:./mlruns in dev/tests.
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field(default="marketplus-tft")


settings = Settings()
