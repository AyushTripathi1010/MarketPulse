"""
Hybrid retriever — the public face of the RAG module.

Pipeline:
    query text
        │
        ├─► embedder ──► Qdrant.semantic_search ──► top_k_semantic doc_ids
        │
        ├─► BM25.search ──► top_k_lexical doc_ids
        │
        ▼
    Reciprocal Rank Fusion ──► top_n_fused doc_ids  (n typically 30)
        │
        ▼
    (optional) cross-encoder rerank ──► top_k_final doc_ids  (k typically 3)
        │
        ▼
    Materialize payloads from Qdrant ──► HistoricalAnalogue list

Every external call (embed, qdrant, rerank) is degradable: if any one
fails, the retriever returns whatever it has, NEVER raising upward.
A failed rerank just falls back to RRF order. A failed embedding falls
back to BM25-only. The Critic always gets *some* list back, even if
empty — and the orchestrator can still produce a Critique.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from marketplus_shared.models import (
    HistoricalAnalogue,
    RegimeLabel,
)

from critic.rag.bm25_index import BM25Index
from critic.rag.embedder import (
    DEFAULT_MODEL_ID as DEFAULT_EMBED_MODEL,
    EmbeddingError,
    embed_text,
)
from critic.rag.fusion import reciprocal_rank_fusion
from critic.rag.qdrant_store import QdrantStore
from critic.rag.reranker import (
    DEFAULT_RERANKER_MODEL,
    RerankerError,
    rerank as cross_encoder_rerank,
)

logger = logging.getLogger(__name__)


def _payload_to_analogue(payload: dict[str, Any], similarity_score: float) -> HistoricalAnalogue:
    """Convert a Qdrant payload dict back into the shared Pydantic model."""
    return HistoricalAnalogue(
        ticker=str(payload.get("ticker", "")),
        occurred_at=datetime.fromisoformat(str(payload.get("occurred_at", ""))),
        regime_then=RegimeLabel(str(payload.get("regime_then", "sideways"))),
        outcome_24h_return=float(payload.get("outcome_24h_return", 0.0)),
        news_summary=str(payload.get("news_summary", "")),
        similarity_score=similarity_score,
    )


class HybridRetriever:
    """Orchestrates embed → Qdrant + BM25 → RRF → optional rerank → Analogues.

    Stateful only in the sense that it holds references to the underlying
    Qdrant store and BM25 index. Both must be populated before retrieve()
    is called (see indexer.populate_index).
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        bm25: BM25Index,
        *,
        hf_api_key: str,
        embed_model_id: str = DEFAULT_EMBED_MODEL,
        reranker_model_id: str = DEFAULT_RERANKER_MODEL,
        use_reranker: bool = True,
    ) -> None:
        self._qdrant = qdrant
        self._bm25 = bm25
        self._hf_api_key = hf_api_key
        self._embed_model_id = embed_model_id
        self._reranker_model_id = reranker_model_id
        self._use_reranker = use_reranker

    def retrieve(
        self,
        query: str,
        *,
        top_k_each: int = 30,
        top_k_final: int = 3,
        ticker_filter: str | None = None,
    ) -> list[HistoricalAnalogue]:
        """Return the top-`top_k_final` historical analogues for `query`.

        Parameters
        ----------
        query : str
            Natural-language description of the current setup — e.g.
            'AAPL +2% over 5d, low vol, mild positive sentiment'.
        top_k_each : int
            How many candidates to pull from each ranker BEFORE fusion.
            Larger = better quality, slower. 30 is the sweet spot in the
            literature.
        top_k_final : int
            How many analogues to return after rerank.
        ticker_filter : str or None
            If set, semantic search will only consider docs for this ticker.
            BM25 doesn't natively filter, but the post-fusion intersection
            naturally narrows to ticker matches in practice.

        Returns
        -------
        list[HistoricalAnalogue]
            Length ≤ top_k_final. Empty if both rankers fail.
        """
        # Step 1: semantic search via Qdrant. Degrades to empty list on failure.
        # We skip the embed call entirely when no HF key is configured —
        # that's an INTENTIONAL state (local dev without secrets) and
        # logging a warning for it would be noise.
        semantic_hits: list = []
        if self._hf_api_key:
            try:
                query_vec = embed_text(
                    query,
                    hf_api_key=self._hf_api_key,
                    model_id=self._embed_model_id,
                )
                semantic_hits = self._qdrant.semantic_search(
                    query_vec,
                    top_k=top_k_each,
                    ticker_filter=ticker_filter,
                )
            except EmbeddingError as e:
                # KEY is set but the call FAILED — that's a real warning.
                logger.warning(
                    "retriever: embedding failed; semantic side disabled (%s)", e
                )
        else:
            logger.debug("retriever: no HF key; running BM25-only.")

        # Step 2: lexical search via BM25 — pure in-process, no external deps.
        lexical_hits = self._bm25.search(query, top_k=top_k_each)

        # Step 3: Reciprocal Rank Fusion.
        fused = reciprocal_rank_fusion(
            semantic_ids=[h.doc_id for h in semantic_hits],
            lexical_ids=[h.doc_id for h in lexical_hits],
            top_k=top_k_each,
        )

        if not fused:
            logger.info("retriever: no candidates from either ranker; returning empty.")
            return []

        # Step 4: materialize payloads. The BM25 index holds a payload cache
        # built at index time — primary source. Qdrant payloads back it up
        # for semantic-only hits (rare; same payload either way for our seed
        # corpus, but in production these can diverge if the BM25 cache is
        # stale relative to a fresh Qdrant write).
        payload_by_id: dict[str, dict] = {}
        for fhit in fused:
            cached = self._bm25.get_payload(fhit.doc_id)
            if cached is not None:
                payload_by_id[fhit.doc_id] = cached
        # Fill in any still-missing from semantic hits (e.g. BM25 cache
        # is empty in a Qdrant-only configuration).
        for h in semantic_hits:
            payload_by_id.setdefault(h.doc_id, h.payload)

        # Build the candidate list with the embedding text for reranking.
        candidates: list[tuple[str, str]] = []
        for fhit in fused:
            payload = payload_by_id.get(fhit.doc_id)
            if not payload:
                continue
            text = str(payload.get("embedding_text") or payload.get("news_summary") or "")
            candidates.append((fhit.doc_id, text))

        # Step 5: optional cross-encoder rerank.
        if self._use_reranker and self._hf_api_key:
            try:
                reranked = cross_encoder_rerank(
                    query,
                    candidates,
                    hf_api_key=self._hf_api_key,
                    model_id=self._reranker_model_id,
                    top_k=top_k_final,
                )
                top_ids_scored = reranked
            except RerankerError as e:
                logger.warning("retriever: rerank failed, falling back to RRF order (%s)", e)
                top_ids_scored = [
                    (fhit.doc_id, fhit.rrf_score)
                    for fhit in fused[:top_k_final]
                ]
        else:
            # No reranker — use RRF score directly.
            top_ids_scored = [
                (fhit.doc_id, fhit.rrf_score)
                for fhit in fused[:top_k_final]
            ]

        # Step 6: convert to HistoricalAnalogue. Skip any doc we couldn't
        # resolve a payload for (shouldn't happen but defensively safe).
        results: list[HistoricalAnalogue] = []
        for doc_id, score in top_ids_scored:
            payload = payload_by_id.get(doc_id)
            if payload is None:
                continue
            try:
                results.append(_payload_to_analogue(payload, similarity_score=score))
            except (ValueError, KeyError) as e:
                logger.warning("retriever: skipping malformed payload for %s: %s", doc_id, e)

        return results
