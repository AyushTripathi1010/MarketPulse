"""
Pure-function tests for Reciprocal Rank Fusion.

RRF has known, exact arithmetic behavior — these tests pin the formula
so any future "optimization" that subtly changes scores is caught.
"""

from __future__ import annotations

from critic.rag.fusion import reciprocal_rank_fusion


def test_rrf_basic_merge_top_3() -> None:
    """Two ranked lists with one overlapping doc → fused order respects both ranks."""
    fused = reciprocal_rank_fusion(
        semantic_ids=["a", "b", "c"],
        lexical_ids=["b", "d", "e"],
        k=60,
    )
    assert fused[0].doc_id == "b"  # rank 2 in semantic + rank 1 in lexical → highest sum


def test_rrf_doc_in_one_list_only() -> None:
    """Doc that appears in only ONE list still gets a positive score."""
    fused = reciprocal_rank_fusion(
        semantic_ids=["a"],
        lexical_ids=["b"],
        k=60,
    )
    scores = {f.doc_id: f.rrf_score for f in fused}
    # Both at rank 1 in their list; both get 1/(60+1) = same score.
    assert scores["a"] == scores["b"]
    assert scores["a"] > 0


def test_rrf_exact_formula() -> None:
    """Verify the exact arithmetic: score = sum(1/(k + rank)) over rankers."""
    k = 60
    fused = reciprocal_rank_fusion(
        semantic_ids=["a", "b"],
        lexical_ids=["a"],
        k=k,
    )
    score_a = next(f.rrf_score for f in fused if f.doc_id == "a")
    score_b = next(f.rrf_score for f in fused if f.doc_id == "b")
    # a is rank 1 in both: 1/61 + 1/61 = 2/61
    assert abs(score_a - (2 / 61)) < 1e-12
    # b is rank 2 in semantic only: 1/62
    assert abs(score_b - (1 / 62)) < 1e-12


def test_rrf_empty_lists() -> None:
    """Empty inputs → empty output, no exceptions."""
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_top_k_truncation() -> None:
    """top_k caps the output length."""
    fused = reciprocal_rank_fusion(
        semantic_ids=["a", "b", "c", "d"],
        lexical_ids=["e", "f", "g", "h"],
        top_k=3,
    )
    assert len(fused) == 3


def test_rrf_returns_per_ranker_ranks() -> None:
    """semantic_rank and lexical_rank fields show where the doc came from."""
    fused = reciprocal_rank_fusion(
        semantic_ids=["a", "b"],
        lexical_ids=["b", "c"],
    )
    by_id = {f.doc_id: f for f in fused}
    assert by_id["a"].semantic_rank == 1
    assert by_id["a"].lexical_rank is None
    assert by_id["b"].semantic_rank == 2
    assert by_id["b"].lexical_rank == 1
    assert by_id["c"].semantic_rank is None
    assert by_id["c"].lexical_rank == 2  # c is rank 2 in lexical_ids=["b","c"]


def test_rrf_smaller_k_sharpens_top_rank() -> None:
    """Lower k gives more weight to top ranks (sanity check on the formula)."""
    fused_k_5 = reciprocal_rank_fusion(["a", "b"], [], k=5)
    fused_k_100 = reciprocal_rank_fusion(["a", "b"], [], k=100)
    # With k=5: rank1=1/6, rank2=1/7 — ratio ~1.17
    # With k=100: rank1=1/101, rank2=1/102 — ratio ~1.01
    ratio_5 = fused_k_5[0].rrf_score / fused_k_5[1].rrf_score
    ratio_100 = fused_k_100[0].rrf_score / fused_k_100[1].rrf_score
    assert ratio_5 > ratio_100
