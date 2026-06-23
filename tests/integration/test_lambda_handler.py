"""
Tests for infra/lambda/handler.py.

We can't test against real Lambda without AWS credentials, but we CAN:
  - Call cron_handler directly with a fake event + mocked graph + mocked S3.
  - Verify api_handler is a Mangum instance that accepts ASGI scopes.

These tests run as part of the normal pytest suite. They live under
tests/integration/ because the handler crosses service boundaries
(orchestrator + S3 client) — not a unit test of any single module.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

# Add infra/lambda to sys.path so we can import handler.py as a module.
INFRA_LAMBDA = Path(__file__).resolve().parents[2] / "infra" / "lambda"
if str(INFRA_LAMBDA) not in sys.path:
    sys.path.insert(0, str(INFRA_LAMBDA))


@pytest.fixture
def fresh_handler(monkeypatch: pytest.MonkeyPatch):
    """Re-import handler with a fresh module state per test.

    The handler module builds its Mangum instance at import time, so
    if a previous test patched env vars, we need a clean import for the
    next test. importlib.reload() does the job.
    """
    monkeypatch.setenv("SUPPORTED_TICKERS", "AAPL,MSFT")
    monkeypatch.setenv("S3_BUCKET", "")  # no-op S3 by default
    if "handler" in sys.modules:
        del sys.modules["handler"]
    return importlib.import_module("handler")


# ---------------------------------------------------------------------------
# Symbol existence
# ---------------------------------------------------------------------------
def test_handler_exports_api_and_cron(fresh_handler: Any) -> None:
    """Both expected callables must be importable — Lambda invokes by string."""
    assert hasattr(fresh_handler, "api_handler")
    assert hasattr(fresh_handler, "cron_handler")
    assert callable(fresh_handler.cron_handler)


def test_api_handler_is_mangum_instance(fresh_handler: Any) -> None:
    """api_handler must be a Mangum object — Lambda calls it as `handler(event, ctx)`."""
    from mangum import Mangum
    assert isinstance(fresh_handler.api_handler, Mangum)


# ---------------------------------------------------------------------------
# _supported_tickers reads env var
# ---------------------------------------------------------------------------
def test_supported_tickers_reads_env(fresh_handler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORTED_TICKERS", "FOO,BAR,BAZ")
    assert fresh_handler._supported_tickers() == ["FOO", "BAR", "BAZ"]


def test_supported_tickers_falls_back_to_defaults(
    fresh_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPPORTED_TICKERS", "")
    from marketplus_shared.constants import SUPPORTED_TICKERS as DEFAULTS
    assert fresh_handler._supported_tickers() == list(DEFAULTS)


def test_supported_tickers_strips_whitespace(
    fresh_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPPORTED_TICKERS", " AAPL ,  MSFT,, NVDA ")
    assert fresh_handler._supported_tickers() == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# cron_handler with mocked graph (no httpx, no S3)
# ---------------------------------------------------------------------------
def test_cron_handler_happy_path_no_s3(
    fresh_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful run produces a summary dict with one entry per ticker."""

    class _FakeGraph:
        def invoke(self, state: dict) -> dict:
            return {
                **state,
                "report_markdown": f"# {state['ticker']} brief",
                "drift_detected": False,
            }

    monkeypatch.setattr(fresh_handler, "build_graph", lambda: _FakeGraph(), raising=False)
    # Patch the module's imports inside cron_handler. Because they're
    # imported INSIDE the function, we patch the module-level reference
    # they'd resolve to. The simplest approach: pre-import + monkeypatch.
    import orchestrator.graph
    monkeypatch.setattr(orchestrator.graph, "build_graph", lambda: _FakeGraph())

    out = fresh_handler.cron_handler({}, None)

    assert out["tickers_processed"] == 2  # SUPPORTED_TICKERS = "AAPL,MSFT"
    assert out["tickers_failed"] == 0
    assert len(out["results"]) == 2
    # Without S3_BUCKET set, report_uri is empty string (not raised).
    assert all(r["report_uri"] == "" for r in out["results"])


def test_cron_handler_continues_on_per_ticker_failure(
    fresh_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If one ticker fails, the cron still processes the others."""
    call_count = {"n": 0}

    class _FlakyGraph:
        def invoke(self, state: dict) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated downstream service outage")
            return {
                **state,
                "report_markdown": f"# {state['ticker']} brief",
                "drift_detected": False,
            }

    import orchestrator.graph
    monkeypatch.setattr(orchestrator.graph, "build_graph", lambda: _FlakyGraph())

    out = fresh_handler.cron_handler({}, None)

    assert out["tickers_processed"] == 1
    assert out["tickers_failed"] == 1
    assert "simulated downstream service outage" in out["failures"][0]["error"]


def test_cron_handler_writes_to_s3_when_bucket_set(
    fresh_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With S3_BUCKET set, _write_report_to_s3 is called per ticker."""
    monkeypatch.setenv("S3_BUCKET", "marketplus-test-bucket")

    class _FakeGraph:
        def invoke(self, state: dict) -> dict:
            return {
                **state,
                "report_markdown": f"# {state['ticker']} brief",
                "drift_detected": False,
            }

    import orchestrator.graph
    monkeypatch.setattr(orchestrator.graph, "build_graph", lambda: _FakeGraph())

    s3_writes: list[tuple[str, str, str]] = []

    def _fake_s3_write(ticker, trace_id, markdown, *, bucket, now):
        s3_writes.append((ticker, bucket, markdown))
        return f"s3://{bucket}/reports/test/{ticker}/{trace_id}.md"

    monkeypatch.setattr(fresh_handler, "_write_report_to_s3", _fake_s3_write)

    out = fresh_handler.cron_handler({}, None)

    assert len(s3_writes) == 2
    assert all(b == "marketplus-test-bucket" for _, b, _ in s3_writes)
    assert all(r["report_uri"].startswith("s3://") for r in out["results"])


def test_cron_handler_s3_failure_does_not_fail_run(
    fresh_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If S3 write fails, the run still completes — just with empty report_uri."""
    monkeypatch.setenv("S3_BUCKET", "marketplus-test-bucket")

    class _FakeGraph:
        def invoke(self, state: dict) -> dict:
            return {**state, "report_markdown": "x", "drift_detected": False}

    import orchestrator.graph
    monkeypatch.setattr(orchestrator.graph, "build_graph", lambda: _FakeGraph())

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated S3 outage")

    monkeypatch.setattr(fresh_handler, "_write_report_to_s3", _boom)

    out = fresh_handler.cron_handler({}, None)
    # Pipeline completed; S3 write was best-effort and failed silently.
    assert out["tickers_processed"] == 2
    assert all(r["report_uri"] == "" for r in out["results"])
