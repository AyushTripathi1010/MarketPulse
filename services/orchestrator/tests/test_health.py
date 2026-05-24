"""Skeleton test for orchestrator service health endpoint."""

from fastapi.testclient import TestClient

from orchestrator.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='orchestrator'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "orchestrator"
    assert payload["status"] == "ok"
