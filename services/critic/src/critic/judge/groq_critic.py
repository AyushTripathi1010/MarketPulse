"""
Groq-powered critic: grades a forecast's confidence in plain English.

Phase 3 ships a SIMPLE prompt — forecast + recent news in, regime/confidence
out. Phase 4 will swap the regime piece for a fine-tuned Phi-3 classifier.
Phase 5 will add hybrid-RAG retrieval of historical analogues to the prompt.

Why Groq instead of OpenAI/Anthropic?
  - Free tier: 30 RPM / 14.4K req/day / Llama-3.3-70B. Plenty for our
    hourly cron firing 7 tickers/hr.
  - Sub-second latency thanks to their custom LPU silicon. Important
    because the Critic is in the critical path of every prediction cycle.
  - Drop-in OpenAI-compatible client API.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from groq import Groq
from marketplus_shared.models import (
    Critique,
    Forecast,
    HistoricalAnalogue,
    Regime,
    RegimeLabel,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are MarketPulse's Critic Agent. Your job is to grade the confidence
of an ML-generated price forecast for a single ticker. You are NOT a trading
advisor; you are calibrating how much downstream consumers should trust this
specific forecast.

Forecast:
- Ticker: {ticker}
- Horizon: {horizon_hours} hours
- Predicted price: {predicted_price:.2f}
- 80% confidence band: [{confidence_low:.2f}, {confidence_high:.2f}]
{regime_block}
{analogues_block}
Recent news headlines (last 24h):
{news_block}

Respond ONLY with valid JSON in EXACTLY this format (no markdown, no commentary):
{{
  "regime": "bull" | "bear" | "sideways",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "<two to four sentences explaining the grade>"
}}
"""


class CriticLLMError(RuntimeError):
    """Raised when the LLM call fails or returns un-parseable JSON."""


def _build_news_block(news: list[str]) -> str:
    """Format news as a bulleted list, or 'No recent news.' if empty."""
    if not news:
        return "No recent news."
    # Cap at 10 headlines to keep token cost predictable.
    capped = news[:10]
    return "\n".join(f"- {h}" for h in capped)


def _parse_response(raw: str) -> dict[str, Any]:
    """Tolerant JSON parser — strips markdown fences if Groq added any."""
    cleaned = raw.strip()
    # LLMs sometimes wrap JSON in ```json ... ``` despite being told not to.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise CriticLLMError(f"Critic LLM returned non-JSON: {raw[:200]}") from e


def _build_regime_block(regime_hint: Regime | None) -> str:
    """Format the optional Phi-3 regime hint as a prompt block.

    Phase 4 adds this — the classifier's label and confidence get fed into
    the judge's context. Phase 3 callers pass None and we emit an empty
    block, preserving the original prompt shape.
    """
    if regime_hint is None:
        return ""
    return (
        f"\nRegime classifier (fine-tuned Phi-3-mini):\n"
        f"- Label: {regime_hint.label.value}\n"
        f"- Classifier confidence: {regime_hint.confidence:.0%}\n"
    )


def _build_analogues_block(analogues: list[HistoricalAnalogue]) -> str:
    """Format the optional retrieved analogues as a prompt block.

    Phase 5 adds this — hybrid RAG returns the 3 most similar historical
    setups and we feed them in so the judge can reason about what tended
    to happen next in comparable conditions.

    Format: bulleted list, one analogue per line, with the outcome and
    a short news summary. We DON'T include the embedding text — that
    would just bloat the prompt with text the model already saw via
    the news_summary field.
    """
    if not analogues:
        return ""

    lines = ["\nHistorical analogues (retrieved by hybrid RAG):"]
    for a in analogues:
        lines.append(
            f"- {a.ticker} on {a.occurred_at.date()} (regime: {a.regime_then.value}) → "
            f"24h outcome: {a.outcome_24h_return:+.2%}. "
            f"News: {a.news_summary}"
        )
    return "\n".join(lines) + "\n"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((CriticLLMError, ConnectionError, TimeoutError)),
    reraise=True,
)
def judge(
    forecast: Forecast,
    recent_news: list[str],
    *,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    regime_hint: Regime | None = None,
    analogues: list[HistoricalAnalogue] | None = None,
) -> Critique:
    """Call Groq to grade the forecast. Returns the shared Critique model.

    Phase 4 passes the Phi-3 classifier's regime as `regime_hint`.
    Phase 5 passes retrieved historical analogues. Both are OPTIONAL — when
    None/empty, the corresponding prompt block is empty and the model
    judges on what it has. This is what makes the critic robust against
    partial outages: the classifier or the retriever or both can be down
    and the judgment still completes (with appropriately tagged confidence).
    """
    prompt = PROMPT_TEMPLATE.format(
        ticker=forecast.ticker,
        horizon_hours=forecast.horizon_hours,
        predicted_price=forecast.predicted_price,
        confidence_low=forecast.confidence_low,
        confidence_high=forecast.confidence_high,
        regime_block=_build_regime_block(regime_hint),
        analogues_block=_build_analogues_block(analogues or []),
        news_block=_build_news_block(recent_news),
    )

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # Low temperature — we want consistent grading, not creativity.
            temperature=0.2,
            # JSON-only response. Groq supports the OpenAI response_format
            # contract; if their model rejects it we fall back to parsing.
            response_format={"type": "json_object"},
            max_tokens=400,
        )
    except Exception as e:
        # Wrap arbitrary Groq SDK errors so tenacity sees a known type.
        raise CriticLLMError(f"Groq API call failed: {e}") from e

    raw = response.choices[0].message.content or ""
    parsed = _parse_response(raw)

    # Map LLM strings → enum. Defensive about case + unknown values.
    label_str = str(parsed.get("regime", "sideways")).lower().strip()
    label = RegimeLabel(label_str) if label_str in {e.value for e in RegimeLabel} else RegimeLabel.SIDEWAYS

    confidence = str(parsed.get("confidence", "MEDIUM")).upper().strip()
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "MEDIUM"

    reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided."

    return Critique(
        regime=Regime(
            label=label,
            # Phase 3: we don't get a soft-max from Groq, just a categorical
            # answer. We assign a heuristic confidence based on the LLM's
            # own HIGH/MEDIUM/LOW grade so this field stays informative.
            confidence={"HIGH": 0.85, "MEDIUM": 0.6, "LOW": 0.35}[confidence],
            classified_at=datetime.now(UTC),
        ),
        analogues=[],  # populated by Phase 5 hybrid RAG
        confidence=confidence,  # type: ignore[arg-type]  # narrowed above
        reasoning=reasoning,
    )


def stub_critique(forecast: Forecast) -> Critique:
    """Deterministic fallback when no GROQ_API_KEY is configured.

    Returns a sensible-shaped Critique with `confidence=UNKNOWN` so downstream
    consumers (and the eventual frontend) can render a "degraded mode" badge.
    """
    return Critique(
        regime=Regime(
            label=RegimeLabel.SIDEWAYS,
            confidence=0.5,
            classified_at=datetime.now(UTC),
        ),
        analogues=[],
        confidence="UNKNOWN",
        reasoning=(
            "Critic LLM not configured (no GROQ_API_KEY). "
            "Forecast served without grading. "
            "Set GROQ_API_KEY in .env to enable LLM critique."
        ),
    )
