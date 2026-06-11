"""
Text embedding via HuggingFace Inference API.

We use BAAI/bge-small-en-v1.5 (default) — a 384-dim sentence embedding model
that's the 2024-2026 sweet spot for English retrieval. Tiny (133MB on disk),
fast on HF Inference API, well-calibrated cosine distance.

Why not sentence-transformers locally?
  Same reason we don't load Phi-3 locally: torch + the model would push the
  critic container past Lambda's 10GB cap. HF Inference API gives us
  embeddings via one HTTP call, no GPU needed, free tier sufficient for
  our hourly cron.

Returned vectors are always L2-normalized (unit length). We do this client-
side defensively even though bge-small returns normalized vectors by default
— different model versions occasionally don't, and normalization is cheap
insurance against cosine-vs-dot-product confusion downstream.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# bge-small-en-v1.5 outputs 384-dimensional embeddings. Hard-coded here so
# downstream code (Qdrant collection creation) doesn't have to introspect.
EMBED_DIM = 384

# Default model. Override via critic.config.bge_model_id if you want to
# swap to bge-base (768d, slightly better quality) or bge-large (1024d).
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"


class EmbeddingError(RuntimeError):
    """Raised when the HF Inference API call fails after retries."""


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Guards against zero vectors (returns as-is)."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((EmbeddingError, ConnectionError, TimeoutError)),
    reraise=True,
)
def embed_text(
    text: str,
    *,
    hf_api_key: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[float]:
    """Embed a single text. Returns a list of `EMBED_DIM` floats, unit-normalized.

    Why list[float] instead of np.ndarray?
      Qdrant's client accepts both, but lists serialize cleanly through JSON
      for caching, logging, and integration tests. Numpy arrays have to be
      .tolist()'d at every boundary; lists round-trip naturally.
    """
    if not text or not text.strip():
        # Embedding an empty string gives a zero vector; we'd rather fail
        # loud than silently corrupt the index.
        raise EmbeddingError("Cannot embed empty/whitespace-only text.")

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_api_key)
    try:
        # feature_extraction returns the raw embedding from the underlying
        # model. For bge-* this is the [CLS]-pooled vector.
        result = client.feature_extraction(text, model=model_id)
    except Exception as e:
        raise EmbeddingError(
            f"HF embedding failed for model={model_id!r}: {e}"
        ) from e

    # HF returns ndarray-shaped output. Coerce to a flat 1D vector — some
    # versions return [[...]] (batched-of-one), some return [...].
    vec = np.asarray(result, dtype=np.float32).reshape(-1)
    if vec.shape[0] != EMBED_DIM:
        raise EmbeddingError(
            f"Expected {EMBED_DIM}-dim embedding from {model_id!r}, "
            f"got shape {vec.shape}. Did you change models?"
        )

    return _normalize(vec).tolist()


def embed_batch(
    texts: Sequence[str],
    *,
    hf_api_key: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> list[list[float]]:
    """Embed a batch of texts sequentially.

    The HF feature_extraction endpoint accepts a single text per call.
    For batches we loop client-side. For our indexing-time use case
    (a few hundred docs, one-shot) this is fine; if we ever needed to
    embed millions we'd swap to a batched local model on a GPU.
    """
    return [embed_text(t, hf_api_key=hf_api_key, model_id=model_id) for t in texts]
