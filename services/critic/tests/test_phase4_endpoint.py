"""
Endpoint tests for the Phase 4 regime-classifier integration.

Asserts that when both GROQ_API_KEY and HF_TOKEN are set, the /critique
endpoint runs the classifier first and the classifier's label appears in
the Groq prompt as a hint.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from marketplus_shared.models import Forecast, Regime, RegimeLabel

from critic import main as main_module
from critic.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _forecast_json() -> dict[str, Any]:
    return Forecast(
        ticker="AAPL",
        predicted_at=datetime.now(UTC),
        horizon_hours=4,
        predicted_price=187.32,
        confidence_low=185.10,
        confidence_high=189.50,
    ).model_dump(mode="json")


def _snapshot_json() -> dict[str, float]:
    return {
        "ticker": "AAPL",
        "return_5d": 0.025,
        "return_20d": 0.08,
        "vol_20d": 0.018,
        "rsi_14": 62.0,
        "sentiment_24h": 2.3,
    }


def test_critique_skips_classifier_when_no_hf_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No HF_TOKEN → classifier disabled, judge runs without regime hint."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(main_module.settings, "hf_token", "")

    captured: dict[str, Any] = {}

    def _fake_judge(*_a: object, **kwargs: object) -> Any:
        captured["regime_hint"] = kwargs.get("regime_hint")
        return _critique_response()

    monkeypatch.setattr(main_module, "judge", _fake_judge)

    r = client.post(
        "/critique",
        json={
            "forecast": _forecast_json(),
            "recent_news": [],
            "snapshot": _snapshot_json(),
        },
    )
    assert r.status_code == 200
    assert captured["regime_hint"] is None


def test_critique_runs_classifier_when_both_creds_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both creds + snapshot → classifier runs, judge receives regime hint."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(main_module.settings, "hf_token", "fake-hf")

    def _fake_classify(*_a: object, **_kw: object) -> Regime:
        return Regime(
            label=RegimeLabel.BULL,
            confidence=0.85,
            classified_at=datetime.now(UTC),
        )

    monkeypatch.setattr(main_module, "classify_regime", _fake_classify)

    captured_hint: list[Regime | None] = []

    def _fake_judge(*_a: object, **kwargs: object) -> Any:
        captured_hint.append(kwargs.get("regime_hint"))  # type: ignore[arg-type]
        return _critique_response()

    monkeypatch.setattr(main_module, "judge", _fake_judge)

    r = client.post(
        "/critique",
        json={
            "forecast": _forecast_json(),
            "recent_news": [],
            "snapshot": _snapshot_json(),
        },
    )
    assert r.status_code == 200
    assert len(captured_hint) == 1
    hint = captured_hint[0]
    assert hint is not None
    assert hint.label == RegimeLabel.BULL
    assert hint.confidence == 0.85


def test_critique_degrades_when_classifier_throws(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Phi-3 fails, the critic still produces a Critique via Groq alone."""
    monkeypatch.setattr(main_module.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(main_module.settings, "hf_token", "fake-hf")

    from critic.classifier.phi3_regime import RegimeClassifierError

    def _boom(*_a: object, **_kw: object) -> Regime:
        raise RegimeClassifierError("simulated HF API outage")

    monkeypatch.setattr(main_module, "classify_regime", _boom)

    captured_hint: list[Regime | None] = []

    def _fake_judge(*_a: object, **kwargs: object) -> Any:
        captured_hint.append(kwargs.get("regime_hint"))  # type: ignore[arg-type]
        return _critique_response()

    monkeypatch.setattr(main_module, "judge", _fake_judge)

    r = client.post(
        "/critique",
        json={
            "forecast": _forecast_json(),
            "recent_news": [],
            "snapshot": _snapshot_json(),
        },
    )
    assert r.status_code == 200
    # Classifier failed → hint passed to judge is None
    assert captured_hint == [None]


def _critique_response() -> Any:
    """Build a valid Critique to return from a mocked judge()."""
    from marketplus_shared.models import Critique

    return Critique(
        regime=Regime(
            label=RegimeLabel.BULL,
            confidence=0.85,
            classified_at=datetime.now(UTC),
        ),
        analogues=[],
        confidence="HIGH",
        reasoning="mocked",
    )
