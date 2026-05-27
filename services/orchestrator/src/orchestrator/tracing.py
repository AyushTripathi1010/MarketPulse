"""
Langfuse tracing — opt-in observability over the agentic pipeline.

Same opt-in pattern as MLflow in services/forecast: if LANGFUSE_PUBLIC_KEY
isn't set, this module no-ops everything. That way tests, local dev without
Langfuse, and CI all work without a backend.

Langfuse traces are useful BECAUSE multi-agent pipelines are otherwise
impossible to debug. With 5 nodes calling 4 internal services calling
2 LLMs, a single "the forecast looked weird" report has dozens of
possible failure points. Langfuse shows the timeline + inputs + outputs
of every step in one screen.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Module-level Langfuse client. Lazily initialized on first use; None when
# disabled. We keep it module-level (not a class) so every node sees the
# same instance without dependency injection plumbing.
_client: Any | None = None
_enabled: bool = False


def init_langfuse(public_key: str, secret_key: str, host: str) -> None:
    """Configure the process-global Langfuse client.

    Called from main.py's lifespan if all three creds are present. After
    this, `with span(name)` / `trace(...)` calls become real network traces.
    """
    global _client, _enabled
    if not (public_key and secret_key and host):
        logger.info("Langfuse disabled (missing public_key / secret_key / host).")
        _enabled = False
        return

    # Import inside the function so the orchestrator module loads even when
    # the SDK isn't installed (e.g. during package metadata scanning).
    from langfuse import Langfuse

    _client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    _enabled = True
    logger.info("Langfuse tracing enabled. host=%s", host)


def is_enabled() -> bool:
    return _enabled and _client is not None


def new_trace_id() -> str:
    """Generate a trace id. Always a valid UUID, even when Langfuse is off,
    so downstream consumers (frontend, logs) have something to correlate on.
    """
    return str(uuid.uuid4())


@contextmanager
def trace_run(trace_id: str, ticker: str, horizon_hours: int) -> Any:
    """Wrap a single orchestrator /run invocation in a Langfuse trace.

    When Langfuse is disabled this is a no-op context manager.
    """
    if not is_enabled():
        yield None
        return

    trace = _client.trace(  # type: ignore[union-attr]
        id=trace_id,
        name="orchestrator.run",
        input={"ticker": ticker, "horizon_hours": horizon_hours},
        metadata={"service": "orchestrator", "phase": "3"},
    )
    try:
        yield trace
    except Exception as e:  # noqa: BLE001 — we want to log ANY failure
        trace.update(level="ERROR", status_message=str(e))
        raise
    finally:
        # Flush so the trace appears in the UI promptly (matters for short
        # Lambda invocations where the process exits within ~20s).
        _client.flush()  # type: ignore[union-attr]


@contextmanager
def span(name: str, trace_id: str | None = None, input_data: dict | None = None) -> Any:
    """Wrap a single node in a Langfuse span.

    No-op when disabled. Use:

        with span("forecast_node", trace_id=tid, input_data=state) as s:
            result = httpx.post(...)
            if s: s.update(output=result)
    """
    if not is_enabled() or trace_id is None:
        yield None
        return

    sp = _client.span(  # type: ignore[union-attr]
        trace_id=trace_id,
        name=name,
        input=input_data or {},
    )
    try:
        yield sp
    except Exception as e:  # noqa: BLE001
        sp.update(level="ERROR", status_message=str(e))
        raise
    finally:
        sp.end()
