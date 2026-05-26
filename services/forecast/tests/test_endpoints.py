"""
HTTP-level tests for the forecast FastAPI app.

Mocks the model so we don't pay training time per test. The training-smoke
test in test_training_smoke.py covers the real training+predict path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from forecast import main as main_module
from forecast.main import app
from marketplus_shared.models import Forecast


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient with the model mocked out."""
    return TestClient(app)


def test_health_reports_model_loaded_false_by_default(client: TestClient) -> None:
    """No checkpoint on disk → /health reports model_loaded=False."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "forecast"
    # In test environment there's no checkpoint, so lifespan won't load one.
    assert body["extras"]["model_loaded"] is False


def test_predict_returns_503_without_model(client: TestClient) -> None:
    """/predict before training → 503, not a 500 crash."""
    r = client.post("/predict", json={"ticker": "AAPL", "horizon_hours": 4})
    assert r.status_code == 503
    assert "not loaded" in r.json()["detail"]


def test_predict_with_mocked_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a fake model populated, /predict returns a valid Forecast JSON."""
    # Patch the module-level handles directly.
    monkeypatch.setattr(main_module, "_model", object(), raising=False)
    monkeypatch.setattr(main_module, "_train_ds", object(), raising=False)

    # Fake features loader returns one row (we only need .empty to be False).
    import pandas as pd

    monkeypatch.setattr(
        main_module,
        "load_features",
        lambda *_a, **_kw: pd.DataFrame({"ticker": ["AAPL"], "Close": [100.0]}),
    )

    # Fake predict_one returns a Forecast directly.
    def _fake_predict(*_a: object, **_kw: object) -> Forecast:
        return Forecast(
            ticker="AAPL",
            predicted_at=datetime.now(UTC),
            horizon_hours=4,
            predicted_price=101.0,
            confidence_low=99.5,
            confidence_high=102.5,
            attention_weights={},
        )

    monkeypatch.setattr(main_module, "predict_one", _fake_predict)

    r = client.post("/predict", json={"ticker": "AAPL", "horizon_hours": 4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["predicted_price"] == 101.0
    assert body["confidence_low"] < body["predicted_price"] < body["confidence_high"]
