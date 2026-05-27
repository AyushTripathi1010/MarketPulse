"""Tests for the deterministic Jinja render."""

from __future__ import annotations

from marketplus_shared.models import Critique, Forecast

from report.generator.render import generate_brief, render_template


def test_render_template_contains_ticker_and_price(
    sample_forecast: Forecast, sample_critique: Critique
) -> None:
    """Numbers and ticker must appear verbatim in the rendered markdown."""
    md = render_template(sample_forecast, sample_critique)
    assert sample_forecast.ticker in md
    assert "$187.32" in md
    assert "$185.10" in md
    assert "$189.50" in md


def test_render_template_marks_regime_uppercase(
    sample_forecast: Forecast, sample_critique: Critique
) -> None:
    """The regime label is shown UPPERCASE for scannability."""
    md = render_template(sample_forecast, sample_critique)
    assert "**BULL**" in md


def test_render_template_handles_zero_analogues(
    sample_forecast: Forecast, sample_critique: Critique
) -> None:
    """No analogues yet → 'None retrieved' message, not a Jinja crash."""
    md = render_template(sample_forecast, sample_critique)
    assert "None retrieved" in md


def test_generate_brief_no_key_skips_polish(
    sample_forecast: Forecast, sample_critique: Critique
) -> None:
    """Without GROQ_API_KEY, generate_brief returns the deterministic render."""
    md = generate_brief(sample_forecast, sample_critique, groq_api_key=None)
    assert "$187.32" in md
    assert "Intelligence Brief" in md
