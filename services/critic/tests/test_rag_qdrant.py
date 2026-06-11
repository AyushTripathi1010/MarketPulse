"""
Tests for the Qdrant store wrapper.

Uses qdrant-client's in-memory mode (`QdrantClient(":memory:")`) so we
don't need a running Qdrant container. The same code path that talks to
a real server is exercised — qdrant-client uses the same internal
Python client either way, just with a different storage backend.
"""

from __future__ import annotations

import numpy as np

from critic.rag.embedder import EMBED_DIM
from critic.rag.qdrant_store import IndexedDoc, QdrantStore


def _make_doc(ticker: str, occurred_at: str, regime: str = "bull") -> IndexedDoc:
    return IndexedDoc(
        doc_id=f"{ticker}-{occurred_at}",
        ticker=ticker,
        occurred_at=occurred_at,
        regime_then=regime,
        outcome_24h_return=0.01,
        news_summary=f"{ticker} something happened",
        embedding_text=f"{ticker} embedding text",
    )


def _make_vec(seed: int) -> list[float]:
    """Deterministic random unit-vector of EMBED_DIM."""
    rng = np.random.RandomState(seed)
    v = rng.randn(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_qdrant_in_memory_create_and_upsert() -> None:
    store = QdrantStore(url=":memory:")
    store.ensure_collection()
    n = store.upsert(
        [
            (_make_doc("AAPL", "2024-01-01"), _make_vec(0)),
            (_make_doc("MSFT", "2024-01-01"), _make_vec(1)),
        ]
    )
    assert n == 2
    assert store.count() == 2


def test_qdrant_ensure_collection_idempotent() -> None:
    """Calling ensure_collection twice must not crash or duplicate state."""
    store = QdrantStore(url=":memory:")
    store.ensure_collection()
    store.ensure_collection()
    store.upsert([(_make_doc("AAPL", "2024-01-01"), _make_vec(0))])
    assert store.count() == 1


def test_qdrant_upsert_overwrites_on_same_logical_id() -> None:
    """Re-upserting the same (ticker, occurred_at) updates, not duplicates."""
    store = QdrantStore(url=":memory:")
    store.ensure_collection()
    store.upsert([(_make_doc("AAPL", "2024-01-01"), _make_vec(0))])
    store.upsert([(_make_doc("AAPL", "2024-01-01"), _make_vec(99))])
    assert store.count() == 1


def test_qdrant_semantic_search_returns_closest_first() -> None:
    """Querying with vec(seed=0) should rank seed=0 doc above seed=1 doc."""
    store = QdrantStore(url=":memory:")
    store.ensure_collection()
    store.upsert(
        [
            (_make_doc("AAPL", "2024-01-01"), _make_vec(0)),
            (_make_doc("MSFT", "2024-01-01"), _make_vec(1)),
        ]
    )
    hits = store.semantic_search(_make_vec(0), top_k=2)
    assert len(hits) == 2
    # The exact-match doc must rank first.
    assert hits[0].payload["ticker"] == "AAPL"
    assert hits[0].score > hits[1].score


def test_qdrant_ticker_filter_narrows_results() -> None:
    """ticker_filter at the index level prunes the candidate pool."""
    store = QdrantStore(url=":memory:")
    store.ensure_collection()
    store.upsert(
        [
            (_make_doc("AAPL", "2024-01-01"), _make_vec(0)),
            (_make_doc("AAPL", "2024-02-01"), _make_vec(1)),
            (_make_doc("MSFT", "2024-01-01"), _make_vec(2)),
        ]
    )
    hits = store.semantic_search(_make_vec(0), top_k=10, ticker_filter="AAPL")
    assert len(hits) == 2
    assert all(h.payload["ticker"] == "AAPL" for h in hits)


def test_qdrant_empty_collection_returns_empty() -> None:
    """Searching an empty collection → [] (not an error)."""
    store = QdrantStore(url=":memory:")
    store.ensure_collection()
    hits = store.semantic_search(_make_vec(0), top_k=5)
    assert hits == []
