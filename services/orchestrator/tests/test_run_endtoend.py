"""
End-to-end orchestrator test: drive the LangGraph through every node with
mocked httpx, assert we get markdown back. This is the closest thing to a
production-run assertion we can do without standing up the full docker-compose.
"""

from __future__ import annotations

from typing import Any

import pytest


def test_full_graph_runs_with_mocked_services(
    fake_httpx_post: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock httpx in EVERY node module, then invoke the compiled graph."""
    # Each node imports httpx into its own module namespace; we patch each
    # module's httpx.post separately.
    from orchestrator.nodes import (
        critic_node,
        data_node,
        forecast_node,
        report_node,
    )

    monkeypatch.setattr(data_node.httpx, "post", fake_httpx_post)
    monkeypatch.setattr(forecast_node.httpx, "post", fake_httpx_post)
    monkeypatch.setattr(critic_node.httpx, "post", fake_httpx_post)
    monkeypatch.setattr(report_node.httpx, "post", fake_httpx_post)

    from orchestrator.graph import build_graph

    g = build_graph()
    final_state = g.invoke({"ticker": "AAPL", "horizon_hours": 4, "trace_id": "test"})

    # Every key the graph promised to fill in should be present.
    assert "forecast" in final_state
    assert "critique" in final_state
    assert "report_markdown" in final_state
    assert "drift_detected" in final_state
    # The end-to-end markdown should at least contain the ticker.
    assert "AAPL" in final_state["report_markdown"]
