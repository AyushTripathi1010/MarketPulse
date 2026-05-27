"""
Phi-3 regime classifier — calls the fine-tuned model on HuggingFace Hub.

The actual fine-tuning runs on a Colab T4 in notebooks/03_phi3_qlora_finetune.ipynb
(QLoRA, 4-bit, ~30-60 min). The trained adapter gets pushed to a Hub repo
(default: `marketplus/phi3-regime-classifier`). At runtime, we call the
HF Inference API — no torch, no transformers, no 7GB model in our
critic container. Just an HTTP call.

If HF_API_KEY isn't configured, this returns a deterministic SIDEWAYS
classification with `confidence=0.5`. Same graceful-degradation pattern
the rest of the project uses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from marketplus_shared.models import Regime, RegimeLabel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Tiny dataclass describing the market state we ask the classifier about.

    These five numbers correspond to what we trained Phi-3 on (see the
    notebook). The classifier's prompt template stringifies them.
    """

    ticker: str
    return_5d: float  # 5-day cumulative return
    return_20d: float  # 20-day cumulative return
    vol_20d: float  # 20-day realized volatility
    rsi_14: float
    sentiment_24h: float


# Instruction-format prompt — matches the format the notebook uses for
# training. Keep them in sync or fine-tuned model will misbehave.
PROMPT_TEMPLATE = (
    "Classify the market regime for the following snapshot as one of: "
    "bull, bear, sideways.\n\n"
    "Ticker: {ticker}\n"
    "5-day return: {return_5d:+.2%}\n"
    "20-day return: {return_20d:+.2%}\n"
    "20-day volatility: {vol_20d:.4f}\n"
    "RSI(14): {rsi_14:.1f}\n"
    "Sentiment(24h): {sentiment_24h:.2f}\n\n"
    "Answer with one word: bull, bear, or sideways.\n"
    "Answer: "
)


class RegimeClassifierError(RuntimeError):
    """Raised when the HF Inference API call fails after retries."""


def _parse_label(raw: str) -> RegimeLabel:
    """Extract the regime label from the model's first generated word.

    Phi-3 sometimes emits extra reasoning even when prompted for one word.
    We look for the first occurrence of bull/bear/sideways (case-insensitive).
    Falls back to SIDEWAYS if the model says something unexpected.
    """
    text = raw.lower()
    match = re.search(r"\b(bull|bear|sideways)\b", text)
    if not match:
        logger.warning("phi3_regime: unparseable output %r — defaulting to sideways", raw[:120])
        return RegimeLabel.SIDEWAYS
    return RegimeLabel(match.group(1))


def stub_regime(ticker: str) -> Regime:
    """Deterministic fallback when no HF_API_KEY is configured."""
    return Regime(
        label=RegimeLabel.SIDEWAYS,
        confidence=0.5,
        classified_at=datetime.now(UTC),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((RegimeClassifierError, ConnectionError, TimeoutError)),
    reraise=True,
)
def classify(
    snapshot: MarketSnapshot,
    *,
    hf_api_key: str,
    model_id: str,
    max_new_tokens: int = 10,
) -> Regime:
    """Ask the fine-tuned Phi-3 model to classify the regime.

    Parameters
    ----------
    snapshot : MarketSnapshot
        Tiny numeric summary; same fields the Colab notebook trains on.
    hf_api_key : str
        HuggingFace Hub access token with `read` scope.
    model_id : str
        Hub repo of the fine-tuned model (e.g. "your-user/phi3-regime").
    max_new_tokens : int
        Tight cap — we only want the one-word label.

    Returns
    -------
    Regime
        With `confidence=0.85` (HIGH) when classifier returns a clean label,
        `confidence=0.5` when output had to be re-parsed defensively.
    """
    # Import inside the function so the module loads even if the HF SDK
    # version drifts under us — keeps the rest of the critic resilient.
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_api_key)
    prompt = PROMPT_TEMPLATE.format(
        ticker=snapshot.ticker,
        return_5d=snapshot.return_5d,
        return_20d=snapshot.return_20d,
        vol_20d=snapshot.vol_20d,
        rsi_14=snapshot.rsi_14,
        sentiment_24h=snapshot.sentiment_24h,
    )

    try:
        # text_generation is the simplest HF endpoint that supports custom
        # fine-tuned causal LMs. We could also use chat_completion if our
        # model has a chat template; text_generation works for both.
        response = client.text_generation(
            prompt=prompt,
            model=model_id,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # low — we want deterministic classification
            do_sample=False,  # greedy decoding for stability
        )
    except Exception as e:
        raise RegimeClassifierError(
            f"HF Inference API failed for model={model_id!r}: {e}"
        ) from e

    label = _parse_label(response or "")
    # If we matched cleanly, the model was confident enough to use a
    # known label. If we fell back to SIDEWAYS via the regex default,
    # mark confidence lower so downstream can see the degradation.
    matched = bool(re.search(r"\b(bull|bear|sideways)\b", (response or "").lower()))
    confidence = 0.85 if matched else 0.5

    return Regime(
        label=label,
        confidence=confidence,
        classified_at=datetime.now(UTC),
    )
