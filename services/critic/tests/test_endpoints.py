"""HTTP-level tests for the critic service."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from marketplus_shared.models import Forecast

from critic import main as main_module
from critic.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _forecast_json() -> dict:
    return Forecast(
        ticker="AAPL",
        predicted_at=datetime.now(UTC),
        horizon_hours=4,
        predicted_price=187.32,
        confidence_low=185.10,
        confidence_high=189.50,
    ).model_dump(mode="json")


def test_health_reports_llm_unconfigured(client: TestClient) -> None:
    """Default (no GROQ_API_KEY) → llm_configured=False."""
    r = client.get("/health")
    body = r.json()
    assert body["service"] == "critic"
    assert body["extras"]["llm_configured"] is False


def test_critique_stub_mode_no_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No GROQ_API_KEY → stub Critique with confidence=UNKNOWN."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "")
    r = client.post(
        "/critique", json={"forecast": _forecast_json(), "recent_news": []}
    )
    assert r.status_code == 200
    assert r.json()["confidence"] == "UNKNOWN"


def test_critique_calls_judge_when_key_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a key, the endpoint should call judge() (mocked) and return its result."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "fake")

    from datetime import UTC, datetime as _dt

    from marketplus_shared.models import Critique, Regime, RegimeLabel

    def _fake_judge(*_a: object, **_kw: object) -> Critique:
        return Critique(
            regime=Regime(
                label=RegimeLabel.BEAR,
                confidence=0.7,
                classified_at=_dt.now(UTC),
            ),
            analogues=[],
            confidence="LOW",
            reasoning="mocked",
        )

    monkeypatch.setattr(main_module, "judge", _fake_judge)
    r = client.post(
        "/critique", json={"forecast": _forecast_json(), "recent_news": []}
    )
    assert r.status_code == 200
    assert r.json()["confidence"] == "LOW"
    assert r.json()["regime"]["label"] == "bear"
