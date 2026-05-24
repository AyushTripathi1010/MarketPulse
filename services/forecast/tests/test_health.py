"""Skeleton test for forecast service health endpoint."""

from fastapi.testclient import TestClient

from forecast.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='forecast'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "forecast"
    assert payload["status"] == "ok"
    # Phase 0: model_loaded is always False (no model yet).
    assert payload["extras"]["model_loaded"] is False
