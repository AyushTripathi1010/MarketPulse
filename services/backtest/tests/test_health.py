"""Skeleton test for backtest service health endpoint."""

from fastapi.testclient import TestClient

from backtest.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='backtest'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "backtest"
    assert payload["status"] == "ok"
