"""Configuration for the critic service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Critic service settings — LLM keys, vector DB, classifier model."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    # LLM for the judgment step.
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # Hugging Face for the Phi-3 regime classifier.
    hf_token: str = Field(default="", alias="HF_TOKEN")
    phi3_model_id: str = Field(default="marketplus/phi3-regime-classifier")

    # Qdrant for semantic retrieval of historical analogues.
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")


settings = Settings()
