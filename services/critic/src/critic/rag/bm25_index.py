"""
BM25 lexical index.

Why BM25 even when we have semantic search?
  Pure semantic retrieval **misses exact-match tokens** that matter for
  finance: ticker symbols ('AAPL'), event names ('Q3 earnings beat'),
  numbers ('above $200'). Embedding models compress these into a few
  hundred dimensions and lose lexical specificity.

  Concrete case: query 'AAPL earnings beat'. A semantic-only retriever
  might surface 'MSFT product launch' (similar topic embedding). BM25
  surfaces the actual AAPL earnings rows because it scores on token
  overlap. Hybrid retrieval = both lists, fused (see fusion.py).

We use the pure-Python rank-bm25 library — no torch, no compile step.
Plenty fast for our corpus size (~10K docs at most).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Stop words we strip from queries + docs. Tiny list — bigger lists hurt
# more than they help on a domain corpus where 'of', 'the', 'and' carry
# no semantic load anyway and BM25's IDF naturally downweights them.
_STOP = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "at", "for", "and",
        "or", "but", "is", "are", "was", "were", "be", "been", "being",
        "with", "as", "by",
    }
)

# Token pattern: word characters + dollar-prefixed cashtags ('$AAPL').
# We deliberately keep digits in tokens — 'q3' and '2024' carry signal.
_TOKEN_RE = re.compile(r"\$?[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25.

    Lowercase, strip basic stop words, keep cashtags ('$AAPL') intact.
    We DON'T stem — modern BM25 implementations work fine without it on
    English financial text, and stemming sometimes hurts named-entity match.
    """
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOP
    ]


@dataclass(slots=True, frozen=True)
class LexicalHit:
    """One result from a BM25 search."""

    doc_id: str
    score: float  # BM25 score; bounded but unnormalized (rerank by rank, not absolute)


class BM25Index:
    """In-memory BM25 index + payload cache.

    Why ONE module owns both ranks AND payloads?
      The BM25 index is rebuilt from scratch on startup anyway, and a
      doc_id → payload dict is essentially free memory-wise (we store
      tens of thousands of small dicts at most). Centralizing payloads
      here means the retriever can resolve ANY doc_id without going back
      to Qdrant — important for two failure modes:
        1. BM25-only hits (when Qdrant has the doc but our query embed
           failed): we still need to materialize the doc.
        2. Qdrant entirely unavailable: BM25 + payloads keep working.

      The alternative — payloads only in Qdrant — couples the lexical
      ranker's success to the vector store's availability. Bad design.
    """

    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._payloads: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None

    def build(self, docs: list[tuple[str, str, dict | None]]) -> None:
        """Build the index from (doc_id, text, payload) triples.

        Parameters
        ----------
        docs : list of (doc_id, text, payload)
            payload may be None — in that case lookups return None and
            the retriever will skip that doc when materializing analogues.

        Builds from scratch every time — there's no incremental API in
        rank-bm25. Cheap enough for our scale (<1ms for 1k docs).
        """
        # Reset.
        self._doc_ids = []
        self._tokenized = []
        self._payloads = {}
        self._bm25 = None

        if not docs:
            logger.info("bm25: empty corpus; index disabled")
            return

        # Tokenize; drop rows that tokenize empty (would crash IDF calc).
        kept: list[tuple[str, list[str], dict | None]] = []
        for doc_id, text, payload in docs:
            tokens = tokenize(text)
            if not tokens:
                continue
            kept.append((doc_id, tokens, payload))

        if not kept:
            logger.warning("bm25: all docs tokenized to empty; index disabled")
            return

        self._doc_ids = [d for d, _, _ in kept]
        self._tokenized = [t for _, t, _ in kept]
        self._payloads = {d: p for d, _, p in kept if p is not None}
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info(
            "bm25: built index over %d docs (%d payloads cached)",
            len(self._doc_ids),
            len(self._payloads),
        )

    def get_payload(self, doc_id: str) -> dict | None:
        """Return the payload for a doc_id, or None if missing."""
        return self._payloads.get(doc_id)

    def search(self, query: str, *, top_k: int = 30) -> list[LexicalHit]:
        """Score every doc against the query; return top-K by score."""
        if self._bm25 is None or not self._doc_ids:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        # Pair with doc_ids, sort desc, take top-K.
        paired = sorted(
            zip(self._doc_ids, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        # Filter out non-positive scores — they mean "no token overlap"
        # and the rank carries no information beyond noise.
        return [LexicalHit(doc_id=d, score=float(s)) for d, s in paired if s > 0]

    def __len__(self) -> int:
        return len(self._doc_ids)
