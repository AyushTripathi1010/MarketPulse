"""HTTP-level tests for the report service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from marketplus_shared.models import Critique, Forecast

from report import main as main_module
from report.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_llm_unconfigured(client: TestClient) -> None:
    r = client.get("/health")
    body = r.json()
    assert body["service"] == "report"
    assert body["extras"]["llm_configured"] is False


def test_generate_returns_markdown(
    client: TestClient,
    sample_forecast: Forecast,
    sample_critique: Critique,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Posting forecast+critique returns a non-empty markdown brief."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "")
    r = client.post(
        "/generate",
        json={
            "forecast": sample_forecast.model_dump(mode="json"),
            "critique": sample_critique.model_dump(mode="json"),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert "Intelligence Brief" in body["markdown"]
    assert body["polished"] is False
