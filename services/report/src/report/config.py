"""Configuration for the report service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Report service settings — Groq for optional LLM polish."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    s3_bucket: str = Field(default="", alias="S3_BUCKET")


settings = Settings()
