"""
Tests for the Phi-3 regime classifier.

All HF Inference API calls are mocked. The classifier is a thin wrapper
around `huggingface_hub.InferenceClient.text_generation`, so we just need
to verify (a) the prompt is built correctly, (b) outputs are parsed
defensively, (c) failures degrade gracefully.
"""

from __future__ import annotations

from typing import Any

import pytest

from critic.classifier import phi3_regime
from critic.classifier.phi3_regime import (
    MarketSnapshot,
    RegimeClassifierError,
    _parse_label,
    classify,
    stub_regime,
)
from marketplus_shared.models import RegimeLabel


# ---------------------------------------------------------------------------
# _parse_label — pure function, no mocks needed
# ---------------------------------------------------------------------------
def test_parse_label_clean_bull() -> None:
    assert _parse_label("bull") == RegimeLabel.BULL


def test_parse_label_extracts_from_wordy_response() -> None:
    """Phi-3 sometimes returns reasoning prose around the label."""
    raw = "Based on the indicators above, I would classify this as bull market behavior."
    assert _parse_label(raw) == RegimeLabel.BULL


def test_parse_label_case_insensitive() -> None:
    assert _parse_label("BEAR.") == RegimeLabel.BEAR
    assert _parse_label("Sideways") == RegimeLabel.SIDEWAYS


def test_parse_label_unknown_defaults_to_sideways() -> None:
    """Output without any of bull/bear/sideways → SIDEWAYS (safe default)."""
    assert _parse_label("I have no opinion at all here") == RegimeLabel.SIDEWAYS
    assert _parse_label("") == RegimeLabel.SIDEWAYS


# ---------------------------------------------------------------------------
# stub_regime — fallback when no key
# ---------------------------------------------------------------------------
def test_stub_regime_returns_sideways_confidence_half() -> None:
    r = stub_regime("AAPL")
    assert r.label == RegimeLabel.SIDEWAYS
    assert r.confidence == 0.5


# ---------------------------------------------------------------------------
# classify() with mocked HF InferenceClient
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        ticker="AAPL",
        return_5d=0.025,
        return_20d=0.08,
        vol_20d=0.018,
        rsi_14=62.0,
        sentiment_24h=2.3,
    )


class _FakeInferenceClient:
    """Minimal stub of huggingface_hub.InferenceClient."""

    def __init__(self, response: str, *, token: str = "") -> None:
        self._response = response
        self.token = token
        self.last_prompt: str | None = None
        self.last_model: str | None = None

    def text_generation(self, prompt: str, model: str, **_kwargs: Any) -> str:
        self.last_prompt = prompt
        self.last_model = model
        return self._response


def _patch_inference_client(
    monkeypatch: pytest.MonkeyPatch, response: str
) -> _FakeInferenceClient:
    """Install a fake InferenceClient on the phi3_regime module."""
    fake = _FakeInferenceClient(response)

    # The classifier imports InferenceClient INSIDE the function, so patch
    # the module reference that the import line resolves to.
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "InferenceClient",
        lambda token=None: fake,
    )
    return fake


def test_classify_happy_path_high_confidence(
    sample_snapshot: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean 'bull' response → label=BULL, confidence=0.85 (HIGH)."""
    _patch_inference_client(monkeypatch, "bull")
    regime = classify(sample_snapshot, hf_api_key="fake", model_id="fake/phi3")
    assert regime.label == RegimeLabel.BULL
    assert regime.confidence == 0.85


def test_classify_messy_output_low_confidence(
    sample_snapshot: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output without bull/bear/sideways → defensive fallback at 0.5 confidence."""
    _patch_inference_client(monkeypatch, "I am uncertain about this market.")
    regime = classify(sample_snapshot, hf_api_key="fake", model_id="fake/phi3")
    assert regime.label == RegimeLabel.SIDEWAYS
    assert regime.confidence == 0.5


def test_classify_prompt_includes_all_snapshot_fields(
    sample_snapshot: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fine-tuned model needs all five inputs — none silently dropped."""
    fake = _patch_inference_client(monkeypatch, "sideways")
    classify(sample_snapshot, hf_api_key="fake", model_id="fake/phi3")
    assert fake.last_prompt is not None
    assert "AAPL" in fake.last_prompt
    assert "+2.50%" in fake.last_prompt  # return_5d
    assert "+8.00%" in fake.last_prompt  # return_20d
    assert "0.0180" in fake.last_prompt  # vol_20d
    assert "62.0" in fake.last_prompt   # rsi_14
    assert "2.30" in fake.last_prompt   # sentiment_24h


def test_classify_passes_through_model_id(
    sample_snapshot: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_id must be threaded to the HF call so each user can point at their own repo."""
    fake = _patch_inference_client(monkeypatch, "bear")
    classify(sample_snapshot, hf_api_key="fake", model_id="ayush/phi3-regime")
    assert fake.last_model == "ayush/phi3-regime"


def test_classify_wraps_hf_exceptions(
    sample_snapshot: MarketSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raised exception from HF should become RegimeClassifierError after retries."""

    class _BoomClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def text_generation(self, *_a: Any, **_kw: Any) -> str:
            raise RuntimeError("simulated 503 from HF")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "InferenceClient", _BoomClient)

    with pytest.raises(RegimeClassifierError) as exc:
        classify(sample_snapshot, hf_api_key="fake", model_id="fake/phi3")
    assert "HF Inference API failed" in str(exc.value)
