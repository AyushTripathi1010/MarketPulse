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

    # Phase 5 — Hybrid RAG. Models used at retrieve-time via HF Inference API.
    # Both default to the bge family (state-of-the-art on free tier).
    rag_embed_model_id: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="HF_EMBED_MODEL_ID",
    )
    rag_reranker_model_id: str = Field(
        default="BAAI/bge-reranker-base",
        alias="HF_RERANKER_MODEL_ID",
    )

    # Master switch for the rerank pass. Disable (env: USE_RERANKER=false)
    # to skip the second HF call when the reranker endpoint is flaky and
    # you'd rather take the RRF order than wait through retries.
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")

    # Qdrant. Empty / ":memory:" uses in-process mode; perfect for tests
    # and a single-user demo. Set to http://qdrant:6333 in docker-compose
    # and to https://<your-cluster>.qdrant.cloud:6333 in production.
    qdrant_url: str = Field(default=":memory:", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="marketplus_analogues")

    # How many analogues we return to the orchestrator. 3 is the literature
    # sweet spot — enough context for the judge LLM, few enough to fit in
    # the prompt without diluting attention.
    rag_top_k_final: int = Field(default=3, ge=1, le=10)
    rag_top_k_each: int = Field(default=30, ge=5, le=100)

    # Whether to seed the index from the built-in synthetic corpus on startup.
    # In production this is False; the indexer is run separately against
    # real S3 features. In dev/demo we leave it True so a fresh checkout
    # produces a working /critique without manual setup.
    seed_index_on_startup: bool = Field(default=True, alias="SEED_INDEX_ON_STARTUP")


settings = Settings()
