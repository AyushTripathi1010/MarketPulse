"""Configuration for the report service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Report service settings — Groq for LLM-driven report drafting."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    s3_bucket: str = Field(default="marketplus-local", alias="S3_BUCKET")


settings = Settings()
