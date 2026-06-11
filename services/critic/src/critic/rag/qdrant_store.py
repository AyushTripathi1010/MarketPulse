"""
Qdrant vector store wrapper.

Why Qdrant over Pinecone / Chroma / Weaviate / pgvector?
  - **Open source + Apache 2.0** (Pinecone is closed; Weaviate is BSL).
  - **Free self-host or 1GB free cloud** — fits our budget.
  - **Rust core** — fastest open-source ANN index in 2026 benchmarks.
  - **Payload filtering at the index level** — we can constrain searches
    by ticker or date range without re-ranking client-side.
  - **In-memory mode** for tests (`QdrantClient(":memory:")`) — no docker
    required to unit-test the retriever.

The wrapper is intentionally thin — Qdrant's API is already clean. We
hide the construction noise (collection creation idempotency, point ID
generation, distance choice) so the rest of the project never touches
qdrant_client.models directly.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from critic.rag.embedder import EMBED_DIM

# Stable namespace UUID for MarketPulse — generated once, fixed forever.
# We feed (ticker|occurred_at) into uuid5 against this namespace so every
# logical doc gets the same UUID across re-indexes. Qdrant requires point
# IDs to be either integers or valid UUIDs (local-mode strictly enforces this).
_NAMESPACE = uuid.UUID("d4f7a0b2-9c3e-4a1f-8b0a-cafe1234abcd")

logger = logging.getLogger(__name__)

# Default collection name. Configurable via critic.config so multiple
# environments (dev / staging / prod) can coexist on one Qdrant instance.
DEFAULT_COLLECTION = "marketplus_analogues"


@dataclass(slots=True, frozen=True)
class IndexedDoc:
    """One row in the Qdrant collection.

    Mirrors HistoricalAnalogue from marketplus_shared, plus the embedding
    text we built from the doc (kept around so we can rerank without
    re-fetching the original payload).
    """

    doc_id: str  # stable hash-based ID (deterministic across re-indexes)
    ticker: str
    occurred_at: str  # ISO datetime
    regime_then: str  # 'bull' | 'bear' | 'sideways'
    outcome_24h_return: float
    news_summary: str
    embedding_text: str  # the text fed to the embedder (= what BM25 also indexes)


@dataclass(slots=True, frozen=True)
class SemanticHit:
    """One result from a semantic search."""

    doc_id: str
    score: float  # cosine similarity in [-1, 1]; higher is better
    payload: dict


def _stable_doc_id(ticker: str, occurred_at: str) -> str:
    """Deterministic UUID5 string — same (ticker, time) always produces the same id.

    Qdrant's local mode requires point IDs to be either int OR valid UUIDs.
    UUID5 is deterministic (vs random UUID4) — feed the same inputs and you
    get the same UUID forever. Means re-running the indexer overwrites in
    place instead of accumulating duplicates.

    NOTE: we use this UUID as the Qdrant point id (line where this is called).
    The corresponding IndexedDoc.doc_id is a separate identifier we keep in
    the payload for our own bookkeeping; it just happens to use the SAME
    deterministic UUID so payload['doc_id'] == point.id.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{ticker}|{occurred_at}"))


class QdrantStore:
    """Thin wrapper over QdrantClient for the analogue collection."""

    def __init__(
        self,
        *,
        url: str | None = None,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """Construct.

        Parameters
        ----------
        url : str | None
            - None or ":memory:" → in-memory mode (tests, demos).
            - http://qdrant:6333 → docker-compose dev.
            - https://xyz.cloud.qdrant.io:6333 → Qdrant Cloud.
        collection : str
            Logical name. Created on first use if missing.
        """
        if not url or url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url)
        self._collection = collection

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent.

        Cosine distance is the standard choice for sentence-embedding
        retrieval and matches what bge-small was trained for.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("qdrant: created collection %s (dim=%d, COSINE)", self._collection, EMBED_DIM)

    def upsert(self, docs: Iterable[tuple[IndexedDoc, list[float]]]) -> int:
        """Insert or update a batch of (doc, embedding) pairs.

        Returns the number of points written. Upsert (not insert) so that
        re-running the indexer doesn't duplicate — `_stable_doc_id` makes
        the same logical row always land on the same point.
        """
        points = [
            PointStruct(
                id=_stable_doc_id(doc.ticker, doc.occurred_at),
                vector=embedding,
                payload={
                    "doc_id": doc.doc_id,
                    "ticker": doc.ticker,
                    "occurred_at": doc.occurred_at,
                    "regime_then": doc.regime_then,
                    "outcome_24h_return": doc.outcome_24h_return,
                    "news_summary": doc.news_summary,
                    "embedding_text": doc.embedding_text,
                },
            )
            for doc, embedding in docs
        ]
        if not points:
            return 0
        self._client.upsert(self._collection, points=points)
        return len(points)

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 30,
        ticker_filter: str | None = None,
    ) -> list[SemanticHit]:
        """Find the top-K most similar docs to a query embedding.

        Optionally filter by ticker (server-side, no client-side post-filter).
        Returns hits with cosine similarity score and full payload.
        """
        qfilter = None
        if ticker_filter:
            qfilter = Filter(
                must=[
                    FieldCondition(key="ticker", match=MatchValue(value=ticker_filter))
                ]
            )

        result = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [
            SemanticHit(
                doc_id=p.payload["doc_id"] if p.payload else "",
                score=float(p.score),
                payload=p.payload or {},
            )
            for p in result.points
        ]

    def count(self) -> int:
        """Total points in the collection. Used by health checks + tests."""
        return self._client.count(self._collection).count

    def drop(self) -> None:
        """Drop the collection (tests only)."""
        try:
            self._client.delete_collection(self._collection)
        except Exception:  # noqa: BLE001  -- best-effort cleanup
            pass
