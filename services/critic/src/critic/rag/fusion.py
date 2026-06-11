"""
Reciprocal Rank Fusion (RRF).

The classic 2009 algorithm by Cormack et al. for combining multiple
ranked lists into one. Despite (because of?) being absurdly simple it
beats most learned-to-rank approaches when you have two genuinely
complementary rankers — exactly our case (semantic + BM25).

The formula:
    rrf_score(doc) = sum over rankers r of: 1 / (k + rank_r(doc))

where rank_r(doc) is 1-indexed and k is a small constant (typically 60).
Docs missing from one ranker just contribute 0 from that side.

Why this works:
  - Top-ranked docs from EITHER ranker get boosted.
  - Docs in BOTH lists get an even bigger boost (sum of two reciprocals).
  - Score scales of the two rankers don't matter — only ranks do, which
    sidesteps the "BM25 scores in [0, ∞), cosine in [-1, 1]" comparison
    problem that bites naive normalization.

The constant k=60 is from the paper. Larger k flattens the curve
(less weight to top ranks); smaller k sharpens it. 60 is the sweet
spot the literature has converged on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FusedHit:
    """One result from RRF, with the merged score and per-ranker rank info."""

    doc_id: str
    rrf_score: float
    semantic_rank: int | None  # 1-indexed; None if not in semantic list
    lexical_rank: int | None   # 1-indexed; None if not in lexical list


def reciprocal_rank_fusion(
    semantic_ids: Sequence[str],
    lexical_ids: Sequence[str],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Merge two ranked id-lists into one RRF-scored list.

    Parameters
    ----------
    semantic_ids, lexical_ids : Sequence[str]
        Ranked lists of doc IDs (best first). Length doesn't need to match.
    k : int
        RRF constant; 60 per the original paper.
    top_k : int or None
        Optional cap on the returned list size. None = return everything.

    Returns
    -------
    list[FusedHit]
        Sorted descending by rrf_score. Stable for ties via dict-insertion order.
    """
    # Build {doc_id: (semantic_rank | None, lexical_rank | None)}
    # Using a dict preserves first-seen order for deterministic tie-breaking.
    seen: dict[str, list[int | None]] = {}
    for rank, doc_id in enumerate(semantic_ids, start=1):
        seen.setdefault(doc_id, [None, None])[0] = rank
    for rank, doc_id in enumerate(lexical_ids, start=1):
        seen.setdefault(doc_id, [None, None])[1] = rank

    def _score(semantic_rank: int | None, lexical_rank: int | None) -> float:
        s = 1.0 / (k + semantic_rank) if semantic_rank is not None else 0.0
        l = 1.0 / (k + lexical_rank) if lexical_rank is not None else 0.0
        return s + l

    hits = [
        FusedHit(
            doc_id=doc_id,
            rrf_score=_score(sem_rank, lex_rank),
            semantic_rank=sem_rank,
            lexical_rank=lex_rank,
        )
        for doc_id, (sem_rank, lex_rank) in seen.items()
    ]
    # Sort by score desc; Python's sort is stable so dict-insertion order
    # determines tie-breaking.
    hits.sort(key=lambda h: h.rrf_score, reverse=True)
    if top_k is not None:
        hits = hits[:top_k]
    return hits
