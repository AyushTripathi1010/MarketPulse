"""
Optional cross-encoder reranker.

Why a reranker at all? RRF is great at MERGING two rankers, but neither
bi-encoder semantic search nor BM25 actually *reads* both the query and
the candidate document together. A cross-encoder model does — it takes
(query, candidate) as a PAIR through transformer cross-attention and
scores their relevance. Slow (one model pass per candidate) but much
more accurate at picking the top-3 from a top-30 candidate set.

We use BAAI/bge-reranker-base by default. Free on HF Inference API.

Why is this *optional*?
  HF Inference API's cross-encoder endpoint is occasionally unavailable
  or rate-limited. Our hybrid retriever degrades gracefully: with rerank
  enabled and reachable, it reranks; otherwise it returns the RRF top-K
  unchanged. Quality is "good" without reranker, "great" with it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class RerankerError(RuntimeError):
    """Raised when reranking fails after retries."""


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type((RerankerError, ConnectionError, TimeoutError)),
    reraise=True,
)
def rerank(
    query: str,
    candidates: Sequence[tuple[str, str]],
    *,
    hf_api_key: str,
    model_id: str = DEFAULT_RERANKER_MODEL,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Score every (query, candidate_text) pair via a cross-encoder, return top-K.

    Parameters
    ----------
    query : str
        The user/forecast question we're trying to match analogues to.
    candidates : Sequence[tuple[str, str]]
        List of (doc_id, text) pairs to rerank.
    hf_api_key : str
        HF token (read scope).
    model_id : str
        Cross-encoder model on HF Hub.
    top_k : int
        Number of top results to return.

    Returns
    -------
    list[(doc_id, score)] sorted descending by score.

    Notes
    -----
    HF Inference API surfaces cross-encoders via the `sentence_similarity`
    or `text_classification` endpoints depending on model card config.
    We use `sentence_similarity` which works for bge-reranker family.
    """
    if not candidates:
        return []

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_api_key)

    candidate_texts = [text for _, text in candidates]

    try:
        # sentence_similarity returns one similarity per candidate text.
        scores = client.sentence_similarity(
            sentence=query,
            other_sentences=list(candidate_texts),
            model=model_id,
        )
    except Exception as e:
        raise RerankerError(
            f"HF rerank failed for model={model_id!r}: {e}"
        ) from e

    if len(scores) != len(candidates):
        raise RerankerError(
            f"Rerank returned {len(scores)} scores for {len(candidates)} candidates."
        )

    paired = [
        (doc_id, float(score))
        for (doc_id, _), score in zip(candidates, scores, strict=True)
    ]
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired[:top_k]
