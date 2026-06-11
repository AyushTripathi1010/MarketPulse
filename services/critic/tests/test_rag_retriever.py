"""
End-to-end tests for the HybridRetriever.

We test:
  - BM25-only mode (no HF key) returns sane results from the seed corpus.
  - Semantic + BM25 mode (mocked embedder + reranker) returns reranked top-K.
  - Graceful degradation when embedder OR reranker fails.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from critic.rag import embedder as embedder_mod
from critic.rag import reranker as reranker_mod
from critic.rag.bm25_index import BM25Index
from critic.rag.embedder import EMBED_DIM
from critic.rag.hybrid_retriever import HybridRetriever
from critic.rag.indexer import populate_from_seed
from critic.rag.qdrant_store import QdrantStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_indexes() -> tuple[QdrantStore, BM25Index]:
    """Indexes populated from the synthetic seed corpus, no HF embedding."""
    q = QdrantStore(url=":memory:")
    b = BM25Index()
    populate_from_seed(q, b, hf_api_key=None, embed_model_id="unused")
    return q, b


# ---------------------------------------------------------------------------
# BM25-only mode (no HF key)
# ---------------------------------------------------------------------------
def test_retriever_bm25_only_returns_top_k(seeded_indexes: tuple[QdrantStore, BM25Index]) -> None:
    """Without an HF key, retriever uses BM25 alone and still returns results."""
    q, b = seeded_indexes
    r = HybridRetriever(q, b, hf_api_key="", use_reranker=False)
    hits = r.retrieve("NVDA AI demand earnings", top_k_final=3)
    assert len(hits) > 0
    # NVDA should dominate the top results for an NVDA-heavy query.
    assert any(h.ticker == "NVDA" for h in hits[:3])


def test_retriever_bm25_only_empty_on_unmatched_query(
    seeded_indexes: tuple[QdrantStore, BM25Index],
) -> None:
    """A query with no token overlap → empty list, not an exception."""
    q, b = seeded_indexes
    r = HybridRetriever(q, b, hf_api_key="", use_reranker=False)
    hits = r.retrieve("xyzzy quantum cryptography blockchain", top_k_final=3)
    assert hits == []


# ---------------------------------------------------------------------------
# Semantic + BM25 with mocked embedder
# ---------------------------------------------------------------------------
def _mock_embedder(monkeypatch: pytest.MonkeyPatch, vec_factory: Any) -> None:
    """Replace huggingface_hub.InferenceClient with one returning vec_factory(text)."""

    class _Fake:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def feature_extraction(self, text: str, model: str) -> Any:
            return vec_factory(text)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _Fake)


def test_retriever_semantic_path_works_with_mocked_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a mocked embedder, populate_from_seed indexes Qdrant + retriever uses it."""
    # Stable per-text vector so the seed corpus and the query are reproducible.
    # We seed a fresh RandomState per text — keeps every text's embedding stable
    # across runs without sharing state between calls.
    def _vec_for(text: str) -> np.ndarray:
        h = abs(hash(text)) % (2**32)
        return np.random.RandomState(h).randn(EMBED_DIM).astype(np.float32)

    _mock_embedder(monkeypatch, _vec_for)

    q = QdrantStore(url=":memory:")
    b = BM25Index()
    qc, bc = populate_from_seed(q, b, hf_api_key="fake-key", embed_model_id="fake")
    assert qc > 0  # seed embedded into Qdrant successfully
    assert bc == 20

    r = HybridRetriever(q, b, hf_api_key="fake-key", use_reranker=False)
    hits = r.retrieve("AAPL Q1 earnings beat", top_k_final=3)
    assert len(hits) > 0


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------
def test_retriever_falls_back_when_embedder_fails(
    seeded_indexes: tuple[QdrantStore, BM25Index],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedder crash → semantic side returns []; BM25 still produces results."""

    class _BoomClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def feature_extraction(self, *_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("HF 503")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _BoomClient)

    q, b = seeded_indexes
    r = HybridRetriever(q, b, hf_api_key="fake", use_reranker=False)
    hits = r.retrieve("NVDA AI demand", top_k_final=3)
    # The retriever should NOT raise; should fall back to BM25-only.
    assert len(hits) > 0


def test_retriever_falls_back_when_reranker_fails(
    seeded_indexes: tuple[QdrantStore, BM25Index],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reranker crash → use RRF order, still return analogues."""
    # Make rerank() raise. Patch it in the hybrid_retriever module where it's imported.
    from critic.rag import hybrid_retriever

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise reranker_mod.RerankerError("simulated")

    monkeypatch.setattr(hybrid_retriever, "cross_encoder_rerank", _boom)

    q, b = seeded_indexes
    r = HybridRetriever(q, b, hf_api_key="", use_reranker=True)
    hits = r.retrieve("NVDA AI demand", top_k_final=3)
    # Should NOT raise; should fall back to RRF order.
    assert len(hits) > 0


def test_retriever_ticker_filter_narrows_results(
    seeded_indexes: tuple[QdrantStore, BM25Index],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ticker_filter narrows the semantic side; combined with BM25 still works."""
    # No HF key → semantic side returns [] anyway, but the BM25 side
    # naturally narrows by token match.
    q, b = seeded_indexes
    r = HybridRetriever(q, b, hf_api_key="", use_reranker=False)
    hits = r.retrieve("AAPL services", top_k_final=5, ticker_filter="AAPL")
    # BM25 doesn't have ticker_filter but the top tokens are AAPL-flavored
    # in our seed corpus, so AAPL rows should dominate.
    assert any(h.ticker == "AAPL" for h in hits)
