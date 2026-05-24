"""Skeleton test — exercise /health to prove the app boots and routes work."""

from fastapi.testclient import TestClient

from data_ingest.main import app


def test_health_returns_ok() -> None:
    """GET /health should return 200 with service='data_ingest', status='ok'."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "data_ingest"
    assert payload["status"] == "ok"
