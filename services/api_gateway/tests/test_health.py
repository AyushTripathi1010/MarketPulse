"""Skeleton test for api_gateway service health endpoint."""

from fastapi.testclient import TestClient

from api_gateway.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='api_gateway'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "api_gateway"
    assert payload["status"] == "ok"
