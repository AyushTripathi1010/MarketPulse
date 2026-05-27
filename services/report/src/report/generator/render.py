"""
Render the Jinja markdown template, then optionally polish with Groq.

The TEMPLATE produces a deterministic, structured brief — same inputs always
produce the same output. The OPTIONAL LLM polish pass rewrites prose for
flow while preserving every number. We do the deterministic render first
because:
  1. Easier to test (no LLM in unit tests).
  2. If Groq is down, we still produce a brief.
  3. Numbers/units never get re-interpreted by an LLM.

Phase 7 adds an S3-write step. For now we return the markdown string and
the FastAPI handler decides where to land it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from groq import Groq
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from marketplus_shared.models import Critique, Forecast
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Template directory lives next to this file.
_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    # StrictUndefined: rendering fails LOUDLY if the template references a
    # variable we didn't pass. Silent missing variables are a debugging hell.
    undefined=StrictUndefined,
    # Auto-trim whitespace around block tags so the output isn't full of
    # blank lines from Jinja control flow.
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(forecast: Forecast, critique: Critique) -> str:
    """Render the deterministic markdown brief. No LLM calls.

    This is the fallback when Groq is down. Always produces SOMETHING usable.
    """
    template = _JINJA_ENV.get_template("brief.md.j2")
    return template.render(
        forecast=forecast,
        critique=critique,
        generated_at=datetime.now(UTC),
    )


POLISH_PROMPT = """You are an analyst editing a market intelligence brief.
Rewrite the prose sections for clarity and a professional tone, BUT:

1. Do not change any numbers, prices, percentages, tickers, or dates.
2. Do not add new claims that aren't already in the brief.
3. Preserve all section headings exactly as written.
4. Keep the markdown structure (headings, lists, blockquotes).
5. The closing disclaimer paragraph must remain verbatim.

Return only the rewritten markdown — no commentary.

--- BRIEF ---
{brief}
"""


class ReportPolishError(RuntimeError):
    """Raised when the LLM polish call fails after retries."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((ReportPolishError, ConnectionError, TimeoutError)),
    reraise=True,
)
def polish_with_groq(brief_markdown: str, *, api_key: str, model: str) -> str:
    """Send the rendered brief to Groq for prose polishing.

    Returns the polished markdown. Falls back to the original if Groq returns
    an obviously broken response (e.g. empty body).
    """
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": POLISH_PROMPT.format(brief=brief_markdown)}],
            temperature=0.4,  # a touch of creativity for prose
            max_tokens=1200,
        )
    except Exception as e:
        raise ReportPolishError(f"Groq polish call failed: {e}") from e

    polished = (response.choices[0].message.content or "").strip()

    # Guard against catastrophic failure modes: a too-short or empty response
    # means we should fall back to the deterministic render.
    if len(polished) < len(brief_markdown) * 0.6:
        logger.warning(
            "Polished brief is suspiciously short (%d chars vs %d). "
            "Falling back to deterministic render.",
            len(polished),
            len(brief_markdown),
        )
        return brief_markdown

    return polished


def generate_brief(
    forecast: Forecast,
    critique: Critique,
    *,
    groq_api_key: str | None = None,
    groq_model: str = "llama-3.3-70b-versatile",
) -> str:
    """Full pipeline: render → optionally polish.

    If `groq_api_key` is empty/None, returns the deterministic render
    unchanged (graceful degradation, same pattern as the critic).
    """
    brief = render_template(forecast, critique)

    if not groq_api_key:
        logger.info("No GROQ_API_KEY — skipping LLM polish.")
        return brief

    try:
        return polish_with_groq(brief, api_key=groq_api_key, model=groq_model)
    except ReportPolishError as e:
        # Polishing is OPTIONAL — fall back to deterministic, log loudly.
        logger.warning("Falling back to deterministic brief due to: %s", e)
        return brief
