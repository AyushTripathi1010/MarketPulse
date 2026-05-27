"""Configuration for the critic service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Critic service settings — LLM keys, vector DB, classifier model."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    # LLM for the judgment step.
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # Hugging Face — Phi-3 regime classifier served via HF Inference API.
    # The model is fine-tuned by notebooks/03_phi3_qlora_finetune.ipynb on
    # Colab T4 and pushed to the repo named in `phi3_model_id`. Empty token
    # disables the classifier; critic falls back to stub_regime().
    hf_token: str = Field(default="", alias="HF_TOKEN")
    phi3_model_id: str = Field(
        default="marketplus/phi3-regime-classifier",
        alias="HF_REGIME_MODEL_ID",
    )

    # Qdrant for semantic retrieval of historical analogues (Phase 5).
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")


settings = Settings()
