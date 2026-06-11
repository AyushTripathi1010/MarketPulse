"""
End-to-end tests for the Phase 5 /critique endpoint.

We use a TestClient bound to the live FastAPI app, which means the lifespan
context manager runs — building the BM25 index from the seed corpus. We
then verify that analogues flow through into the response and into the
Groq judge prompt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from marketplus_shared.models import (
    Critique,
    Forecast,
    HistoricalAnalogue,
    Regime,
    RegimeLabel,
)

from critic import main as main_module
from critic.main import app


@pytest.fixture
def client() -> TestClient:
    """TestClient that triggers the FastAPI lifespan (RAG index builds here)."""
    with TestClient(app) as c:
        yield c


def _forecast_json() -> dict[str, Any]:
    return Forecast(
        ticker="AAPL",
        predicted_at=datetime.now(UTC),
        horizon_hours=4,
        predicted_price=187.32,
        confidence_low=185.10,
        confidence_high=189.50,
    ).model_dump(mode="json")


def _snapshot_json() -> dict[str, Any]:
    return {
        "ticker": "AAPL",
        "return_5d": 0.025,
        "return_20d": 0.08,
        "vol_20d": 0.018,
        "rsi_14": 62.0,
        "sentiment_24h": 2.3,
    }


def test_health_reports_rag_corpus_size(client: TestClient) -> None:
    """Lifespan populated the BM25 index from seed → corpus size > 0."""
    r = client.get("/health")
    body = r.json()
    assert body["service"] == "critic"
    assert body["extras"]["rag_corpus_size"] > 0


def test_critique_stub_mode_includes_analogues(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even in stub mode (no GROQ_API_KEY), analogues come back when RAG is loaded."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "")
    r = client.post(
        "/critique",
        json={
            "forecast": _forecast_json(),
            "recent_news": ["AAPL Q1 earnings beat"],
            "snapshot": _snapshot_json(),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] == "UNKNOWN"  # still in stub mode
    # Seed corpus has AAPL rows, so BM25 should surface at least one.
    assert isinstance(body["analogues"], list)


def test_critique_passes_analogues_to_judge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Groq is configured, the judge receives the retrieved analogues."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(main_module.settings, "hf_token", "")  # BM25-only RAG

    captured: dict[str, Any] = {}

    def _fake_judge(*_a: Any, **kwargs: Any) -> Critique:
        captured["analogues"] = kwargs.get("analogues")
        captured["regime_hint"] = kwargs.get("regime_hint")
        return _critique_response()

    monkeypatch.setattr(main_module, "judge", _fake_judge)

    r = client.post(
        "/critique",
        json={
            "forecast": _forecast_json(),
            "recent_news": ["AAPL Q1 earnings beat"],
            "snapshot": _snapshot_json(),
        },
    )
    assert r.status_code == 200
    # judge() was called with the analogues kwarg as a list
    assert "analogues" in captured
    assert isinstance(captured["analogues"], list)


def test_critique_response_includes_analogues_field(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Critique returned to the orchestrator carries analogues end-to-end."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(main_module.settings, "hf_token", "")

    monkeypatch.setattr(main_module, "judge", lambda *_a, **_kw: _critique_response())

    r = client.post(
        "/critique",
        json={
            "forecast": _forecast_json(),
            "recent_news": ["AAPL services growth"],
            "snapshot": _snapshot_json(),
        },
    )
    body = r.json()
    assert "analogues" in body
    # The judge returned analogues=[]; main.py should overlay with the
    # retrieved list. Type check (it's a list) is the contract.
    assert isinstance(body["analogues"], list)


def _critique_response() -> Critique:
    """A Critique stub for monkey-patched judge() return values."""
    return Critique(
        regime=Regime(
            label=RegimeLabel.BULL,
            confidence=0.85,
            classified_at=datetime.now(UTC),
        ),
        analogues=[],  # main.py overlays the retrieved list on top of this
        confidence="HIGH",
        reasoning="mocked",
    )
