"""
populate_index — build a fresh Qdrant + BM25 index from seed or real data.

Two modes:
  - Seed mode: from `seed_data.SEED_CORPUS` (built-in, no network).
  - Live mode: from feature Parquet on S3 (Phase 7+ wiring).

Either path produces the same data shape on disk: Qdrant points + an
in-process BM25 index. The critic's lifespan handler calls this on
startup.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import hashlib

from critic.rag.bm25_index import BM25Index
from critic.rag.embedder import EmbeddingError, embed_text
from critic.rag.qdrant_store import IndexedDoc, QdrantStore
from critic.rag.seed_data import SEED_CORPUS, SeedRow, build_embedding_text

logger = logging.getLogger(__name__)


def _seed_doc_id(row: SeedRow) -> str:
    """Stable per-row id (same scheme as qdrant_store._stable_doc_id)."""
    raw = f"{row.ticker}|{row.occurred_at}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def seed_to_indexed_docs(rows: Iterable[SeedRow]) -> list[IndexedDoc]:
    """Convert SeedRow objects to IndexedDoc with the embedding-ready text."""
    return [
        IndexedDoc(
            doc_id=_seed_doc_id(r),
            ticker=r.ticker,
            occurred_at=r.occurred_at,
            regime_then=r.regime_then,
            outcome_24h_return=r.outcome_24h_return,
            news_summary=r.news_summary,
            embedding_text=build_embedding_text(r),
        )
        for r in rows
    ]


def populate_index(
    qdrant: QdrantStore,
    bm25: BM25Index,
    docs: list[IndexedDoc],
    *,
    hf_api_key: str | None,
    embed_model_id: str,
) -> tuple[int, int]:
    """Embed every doc, upsert to Qdrant, rebuild the BM25 index.

    Returns (qdrant_count, bm25_count) — useful for /health introspection.

    If `hf_api_key` is empty, Qdrant is skipped (we can't embed without a
    key). BM25 still builds — the retriever degrades to lexical-only.
    """
    qdrant.ensure_collection()

    # BM25 from doc_id + embedding_text + full payload. Always builds.
    # The payload cache lets the retriever materialize analogues even when
    # Qdrant is unavailable.
    bm25.build(
        [
            (
                d.doc_id,
                d.embedding_text,
                {
                    "doc_id": d.doc_id,
                    "ticker": d.ticker,
                    "occurred_at": d.occurred_at,
                    "regime_then": d.regime_then,
                    "outcome_24h_return": d.outcome_24h_return,
                    "news_summary": d.news_summary,
                    "embedding_text": d.embedding_text,
                },
            )
            for d in docs
        ]
    )

    # Qdrant only if we have a key to embed with.
    qdrant_count = 0
    if hf_api_key:
        embedded: list[tuple[IndexedDoc, list[float]]] = []
        for doc in docs:
            try:
                vec = embed_text(
                    doc.embedding_text,
                    hf_api_key=hf_api_key,
                    model_id=embed_model_id,
                )
                embedded.append((doc, vec))
            except EmbeddingError as e:
                # Log and skip the bad doc — we'd rather have a partial
                # index than no index at all.
                logger.warning("indexer: skipping doc %s due to embed failure: %s", doc.doc_id, e)
        if embedded:
            qdrant_count = qdrant.upsert(embedded)
    else:
        logger.info("indexer: no HF token; skipping Qdrant indexing (BM25-only mode).")

    logger.info("indexer: built corpus — qdrant=%d bm25=%d", qdrant_count, len(bm25))
    return qdrant_count, len(bm25)


def populate_from_seed(
    qdrant: QdrantStore,
    bm25: BM25Index,
    *,
    hf_api_key: str | None,
    embed_model_id: str,
) -> tuple[int, int]:
    """Convenience: seed → index. Used on critic startup for demo + tests."""
    docs = seed_to_indexed_docs(SEED_CORPUS)
    return populate_index(
        qdrant,
        bm25,
        docs,
        hf_api_key=hf_api_key,
        embed_model_id=embed_model_id,
    )
