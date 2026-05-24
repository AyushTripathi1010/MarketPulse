"""Skeleton test for critic service health endpoint."""

from fastapi.testclient import TestClient

from critic.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='critic'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "critic"
    assert payload["status"] == "ok"
