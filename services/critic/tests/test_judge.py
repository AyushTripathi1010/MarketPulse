"""
Tests for the Groq-powered critic judge.

The real Groq call is mocked — we don't hit the API in CI. The stub
fallback IS tested with real code (no Groq involved).
"""

from __future__ import annotations

from typing import Any

import pytest
from marketplus_shared.models import Forecast

from critic.judge import groq_critic
from critic.judge.groq_critic import (
    CriticLLMError,
    judge,
    stub_critique,
    _build_news_block,
    _parse_response,
)


# ---------------------------------------------------------------------------
# stub fallback
# ---------------------------------------------------------------------------
def test_stub_critique_returns_unknown_confidence(sample_forecast: Forecast) -> None:
    """Stub mode must mark confidence as UNKNOWN so downstream sees degraded state."""
    c = stub_critique(sample_forecast)
    assert c.confidence == "UNKNOWN"
    assert c.analogues == []
    assert "GROQ_API_KEY" in c.reasoning  # actionable error message


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------
def test_parse_response_handles_plain_json() -> None:
    raw = '{"regime": "bull", "confidence": "HIGH", "reasoning": "x"}'
    parsed = _parse_response(raw)
    assert parsed["regime"] == "bull"


def test_parse_response_strips_markdown_fences() -> None:
    """Groq sometimes wraps JSON in ```json ... ``` despite instructions."""
    raw = '```json\n{"regime": "bull", "confidence": "HIGH", "reasoning": "x"}\n```'
    parsed = _parse_response(raw)
    assert parsed["confidence"] == "HIGH"


def test_parse_response_raises_on_garbage() -> None:
    with pytest.raises(CriticLLMError):
        _parse_response("not json at all")


def test_news_block_empty_is_explicit() -> None:
    """No news → explicit 'No recent news.' string, not blank."""
    assert _build_news_block([]) == "No recent news."


def test_news_block_caps_at_10() -> None:
    """We never send more than 10 headlines (token cost discipline)."""
    headlines = [f"headline {i}" for i in range(20)]
    out = _build_news_block(headlines)
    assert out.count("\n") == 9  # 10 lines → 9 separators


# ---------------------------------------------------------------------------
# happy-path judge() with mocked Groq client
# ---------------------------------------------------------------------------
class _FakeGroqResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeGroqClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_kwargs: Any) -> _FakeGroqResponse:
        return _FakeGroqResponse(self._content)


def test_judge_happy_path(
    sample_forecast: Forecast, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean Groq response produces a valid Critique."""
    fake_content = '{"regime": "bull", "confidence": "HIGH", "reasoning": "Tight band, positive sentiment."}'
    monkeypatch.setattr(
        groq_critic, "Groq", lambda **_kw: _FakeGroqClient(fake_content)
    )

    critique = judge(sample_forecast, ["AAPL earnings beat"], api_key="fake")
    assert critique.regime.label.value == "bull"
    assert critique.confidence == "HIGH"
    assert critique.reasoning.startswith("Tight band")


def test_judge_defaults_unknown_regime_to_sideways(
    sample_forecast: Forecast, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown regime label → SIDEWAYS (safe default), not a crash."""
    fake_content = '{"regime": "moonwalk", "confidence": "MEDIUM", "reasoning": "x"}'
    monkeypatch.setattr(
        groq_critic, "Groq", lambda **_kw: _FakeGroqClient(fake_content)
    )
    critique = judge(sample_forecast, [], api_key="fake")
    assert critique.regime.label.value == "sideways"


def test_judge_defaults_unknown_confidence_to_medium(
    sample_forecast: Forecast, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confidence outside HIGH/MEDIUM/LOW falls back to MEDIUM."""
    fake_content = '{"regime": "bear", "confidence": "VERY_HIGH", "reasoning": "x"}'
    monkeypatch.setattr(
        groq_critic, "Groq", lambda **_kw: _FakeGroqClient(fake_content)
    )
    critique = judge(sample_forecast, [], api_key="fake")
    assert critique.confidence == "MEDIUM"
