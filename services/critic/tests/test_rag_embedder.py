"""
Tests for the embedder. The HF Inference call is monkey-patched so we
never hit the API in CI. The L2-normalization + dimension checks run
on real numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from critic.rag import embedder
from critic.rag.embedder import EMBED_DIM, EmbeddingError, embed_text


class _FakeInferenceClient:
    """Minimal stub returning a configurable embedding."""

    def __init__(self, response: Any, *, token: str = "") -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def feature_extraction(self, text: str, model: str) -> Any:
        self.calls.append((text, model))
        return self._response


def _patch_client(monkeypatch: pytest.MonkeyPatch, response: Any) -> _FakeInferenceClient:
    """Install a fake InferenceClient on huggingface_hub."""
    fake = _FakeInferenceClient(response)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "InferenceClient", lambda token=None: fake)
    return fake


def test_embed_text_returns_normalized_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero embedding must come back as a unit-length list of floats."""
    fake_embedding = np.random.RandomState(0).randn(EMBED_DIM).astype(np.float32)
    _patch_client(monkeypatch, fake_embedding)

    vec = embed_text("hello world", hf_api_key="fake")
    arr = np.asarray(vec, dtype=np.float32)
    assert len(vec) == EMBED_DIM
    assert abs(np.linalg.norm(arr) - 1.0) < 1e-5  # unit length


def test_embed_text_handles_batched_of_one_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some HF versions return [[...]] instead of [...]; reshape must handle both."""
    fake = np.random.RandomState(1).randn(1, EMBED_DIM).astype(np.float32)
    _patch_client(monkeypatch, fake)
    vec = embed_text("hello", hf_api_key="fake")
    assert len(vec) == EMBED_DIM


def test_embed_text_rejects_empty_input() -> None:
    """Empty/whitespace text → EmbeddingError before any network call."""
    with pytest.raises(EmbeddingError):
        embed_text("", hf_api_key="fake")
    with pytest.raises(EmbeddingError):
        embed_text("   ", hf_api_key="fake")


def test_embed_text_rejects_wrong_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the HF model returns the wrong-dim vector, fail loud — not silently."""
    _patch_client(monkeypatch, np.zeros(100, dtype=np.float32))
    with pytest.raises(EmbeddingError):
        embed_text("hello", hf_api_key="fake")


def test_embed_text_zero_vector_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-vector input (rare but real) shouldn't blow up the normalizer."""
    _patch_client(monkeypatch, np.zeros(EMBED_DIM, dtype=np.float32))
    vec = embed_text("hello", hf_api_key="fake")
    # Normalize-of-zero stays zero; we just want no division-by-zero crash.
    assert all(v == 0.0 for v in vec)


def test_embed_text_wraps_hf_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raised exception from HF should become EmbeddingError after retries."""

    class _BoomClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def feature_extraction(self, *_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("simulated HF 503")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _BoomClient)
    with pytest.raises(EmbeddingError) as exc:
        embed_text("hello", hf_api_key="fake")
    assert "HF embedding failed" in str(exc.value)
