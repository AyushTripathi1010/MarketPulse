"""Skeleton test for report service health endpoint."""

from fastapi.testclient import TestClient

from report.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='report'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "report"
    assert payload["status"] == "ok"
