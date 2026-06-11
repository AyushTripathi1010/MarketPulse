"""
Pure-function tests for the BM25 index + tokenizer.

Real BM25 library, no mocks — these are the smallest, most deterministic
tests in the whole project. If anything in the lexical retrieval path
breaks, these fail first.
"""

from __future__ import annotations

from critic.rag.bm25_index import BM25Index, tokenize


# ---------------------------------------------------------------------------
# tokenize — exercises a few financial-domain edge cases
# ---------------------------------------------------------------------------
def test_tokenize_keeps_cashtag() -> None:
    """`$AAPL` must survive as one token (cashtags are signal)."""
    assert "$aapl" in tokenize("$AAPL Q3 beat")


def test_tokenize_keeps_digits() -> None:
    """`Q3` and `2024` carry signal — must not get filtered."""
    tokens = tokenize("Q3 2024 earnings beat")
    assert "q3" in tokens
    assert "2024" in tokens


def test_tokenize_drops_stop_words() -> None:
    """The tiny stop list strips 'of', 'the', 'and' — they carry no signal."""
    tokens = tokenize("growth of the cloud and AI")
    assert "of" not in tokens
    assert "the" not in tokens
    assert "and" not in tokens
    assert "growth" in tokens
    assert "cloud" in tokens
    assert "ai" in tokens


def test_tokenize_lowercases() -> None:
    assert tokenize("AAPL") == ["aapl"]


# ---------------------------------------------------------------------------
# BM25Index — exercise build + search + payload cache + edge cases
# ---------------------------------------------------------------------------
def _sample_docs() -> list[tuple[str, str, dict]]:
    """A small varied corpus exercising different lexical patterns."""
    return [
        ("d1", "AAPL Q3 earnings beat services revenue up", {"id": "d1", "ticker": "AAPL"}),
        ("d2", "MSFT Azure cloud growth accelerates", {"id": "d2", "ticker": "MSFT"}),
        ("d3", "NVDA AI demand surges data center revenue triples", {"id": "d3", "ticker": "NVDA"}),
        ("d4", "AAPL services revenue and iPhone sales", {"id": "d4", "ticker": "AAPL"}),
        ("d5", "GOOGL ad revenue rebounds first dividend announced", {"id": "d5", "ticker": "GOOGL"}),
    ]


def test_bm25_finds_exact_token_match() -> None:
    """Querying for an exact token surfaces docs containing that token."""
    bm = BM25Index()
    bm.build(_sample_docs())
    hits = bm.search("Azure cloud", top_k=3)
    assert len(hits) >= 1
    assert hits[0].doc_id == "d2"


def test_bm25_returns_empty_on_no_match() -> None:
    """Querying for tokens not in any doc returns []."""
    bm = BM25Index()
    bm.build(_sample_docs())
    hits = bm.search("quantum cryptography", top_k=3)
    assert hits == []


def test_bm25_empty_corpus_returns_empty() -> None:
    """Empty corpus → no error, returns []."""
    bm = BM25Index()
    bm.build([])
    hits = bm.search("anything", top_k=3)
    assert hits == []


def test_bm25_handles_empty_query() -> None:
    """Empty query → no error, returns []."""
    bm = BM25Index()
    bm.build(_sample_docs())
    assert bm.search("", top_k=3) == []
    assert bm.search("   ", top_k=3) == []


def test_bm25_payload_cache_returns_dict() -> None:
    """get_payload returns the dict we put in during build()."""
    bm = BM25Index()
    bm.build(_sample_docs())
    p = bm.get_payload("d3")
    assert p is not None
    assert p["ticker"] == "NVDA"


def test_bm25_payload_cache_misses_return_none() -> None:
    """Unknown doc_id → None, not KeyError."""
    bm = BM25Index()
    bm.build(_sample_docs())
    assert bm.get_payload("missing") is None


def test_bm25_drops_all_stop_tokens_doc() -> None:
    """A doc that tokenizes to nothing must not crash the index build."""
    bm = BM25Index()
    bm.build(
        [
            ("d1", "the of and a", {"id": "d1"}),
            ("d2", "AAPL Q3 beat", {"id": "d2"}),
        ]
    )
    # d1's tokens are all stop words → it gets dropped during build.
    # Index should still work with the remaining doc.
    hits = bm.search("AAPL", top_k=2)
    # On a 1-doc corpus IDF can degenerate; we just assert no crash.
    assert isinstance(hits, list)


def test_bm25_len_reflects_kept_docs() -> None:
    """__len__ returns the number of docs that made it past tokenization."""
    bm = BM25Index()
    assert len(bm) == 0
    bm.build(_sample_docs())
    assert len(bm) == 5
