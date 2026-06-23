"""
AWS Lambda entry points for the MarketPulse orchestrator.

Two distinct handlers because Lambda is invoked by two different event
sources with completely different payload shapes:

1. `api_handler` — wraps the orchestrator FastAPI app via Mangum. Used
   when the Lambda is fronted by API Gateway and a user hits a URL.

2. `cron_handler` — bare-Python handler for EventBridge Scheduler events.
   The cron fires hourly, the handler loops every supported ticker
   through the LangGraph pipeline, and writes the resulting markdown
   report to S3. NO API Gateway in the picture.

Why two handlers and not one detector function?
  We could sniff the event shape and branch (`event.get("source") ==
  "aws.events"`?). But: two SAM functions sharing one Docker image
  with different CMD overrides is the AWS-idiomatic pattern. It gives
  each handler its own IAM policy (the cron needs S3 write; the API
  handler doesn't) and its own concurrency budget. Two small functions
  beat one branchy one.

This module lives under infra/lambda/ (NOT inside services/orchestrator/)
because it's part of the DEPLOYMENT artifact, not the application code.
The Dockerfile copies BOTH services/orchestrator/ AND this file into
the image; at runtime AWS calls `handler.cron_handler` or `handler.api_handler`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

# Lambda's default log handler is configured by AWS. We just need to
# raise the root logger so our INFO messages reach CloudWatch.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# api_handler — Mangum-wrapped FastAPI for API Gateway
# ---------------------------------------------------------------------------
#
# Mangum translates between Lambda's (event, context) calling convention
# and ASGI's (scope, receive, send) — letting our existing FastAPI app run
# unchanged inside Lambda. lifespan="on" means Mangum runs the FastAPI
# startup hooks once per cold start, so the LangGraph compiles once and
# is reused across warm invocations.
def _build_api_handler():
    # Import inside the factory so the module loads even on the cron Lambda
    # (which doesn't need the orchestrator FastAPI app to be importable).
    from mangum import Mangum
    from orchestrator.main import app

    return Mangum(app, lifespan="on")


# Module-level instance — created on first import. On warm invocations
# Lambda reuses this; on cold start it pays the import + ASGI startup cost.
api_handler = _build_api_handler()


# ---------------------------------------------------------------------------
# cron_handler — EventBridge → loop tickers → write S3
# ---------------------------------------------------------------------------
# EventBridge cron events look like:
#   {
#     "version": "0",
#     "source": "aws.events",
#     "detail-type": "Scheduled Event",
#     "time": "2026-05-24T09:00:00Z",
#     ...
#   }
# We don't actually care about the fields — the EVENT itself is the signal.


def _supported_tickers() -> list[str]:
    """Read from env var SUPPORTED_TICKERS (comma-separated) or fall back to defaults.

    Why a runtime env var rather than the shared constant?
      In production we may want to scale ticker coverage up/down without
      redeploying. Lambda env vars are mutable via SAM template update —
      faster than a full container image rebuild.
    """
    raw = os.environ.get("SUPPORTED_TICKERS", "")
    if raw.strip():
        return [t.strip() for t in raw.split(",") if t.strip()]
    # Fall back to the shared list so a fresh deploy with no env override
    # produces useful work.
    from marketplus_shared.constants import SUPPORTED_TICKERS as DEFAULTS

    return list(DEFAULTS)


def _write_report_to_s3(
    ticker: str,
    trace_id: str,
    report_markdown: str,
    *,
    bucket: str,
    now: datetime,
) -> str:
    """Upload one markdown report to S3 and return the s3:// URI.

    We compose the key as `reports/dt=YYYY-MM-DD/ticker=AAPL/<trace_id>.md`
    so the layout mirrors data_ingest's Hive-partitioned Parquet. That
    means the frontend can list "all reports for AAPL today" with a
    single S3 prefix scan.
    """
    import boto3  # imported inside fn so the test path can skip it

    key = (
        f"reports/dt={now:%Y-%m-%d}/ticker={ticker}/"
        f"{trace_id}.md"
    )
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=report_markdown.encode("utf-8"),
        ContentType="text/markdown; charset=utf-8",
    )
    return f"s3://{bucket}/{key}"


def cron_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """EventBridge hourly cron entry point.

    Walks every supported ticker through the orchestrator's LangGraph
    state machine in process (no httpx, no API Gateway hop). For each:
      - Build initial MarketState (ticker + horizon + trace_id)
      - Invoke compiled graph (runs every node in sequence)
      - Persist the resulting report_markdown to S3

    Returns a summary dict — useful in CloudWatch and downstream metrics.

    Why call the graph in-process instead of via HTTP to the orchestrator
    service?
      In Lambda the orchestrator IS the container. Calling its own
      FastAPI app via httpx (localhost) would be a needless serialization
      round-trip. We directly invoke the compiled Runnable.
    """
    # Import inside the function so the cold-start cost is paid only when
    # cron_handler is actually called (api_handler still works without
    # boto3 at import time).
    from orchestrator.graph import build_graph
    from orchestrator.tracing import init_langfuse, new_trace_id, trace_run

    # Set up Langfuse tracing if creds are present (opt-in, same pattern
    # as everywhere else in the project).
    init_langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", ""),
    )

    s3_bucket = os.environ.get("S3_BUCKET", "")
    horizon_hours = int(os.environ.get("FORECAST_HORIZON_HOURS", "4"))

    graph = build_graph()
    tickers = _supported_tickers()
    now = datetime.now(UTC)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for ticker in tickers:
        trace_id = new_trace_id()
        initial = {
            "ticker": ticker,
            "horizon_hours": horizon_hours,
            "trace_id": trace_id,
        }
        try:
            with trace_run(trace_id, ticker, horizon_hours):
                final_state = graph.invoke(initial)
            report_md = final_state.get("report_markdown", "")
            drift = bool(final_state.get("drift_detected", False))

            s3_uri = ""
            if s3_bucket and report_md:
                try:
                    s3_uri = _write_report_to_s3(
                        ticker, trace_id, report_md, bucket=s3_bucket, now=now
                    )
                except Exception as e:  # noqa: BLE001 — log + continue, don't fail the cron
                    logger.warning("cron: S3 write failed for %s: %s", ticker, e)

            results.append(
                {
                    "ticker": ticker,
                    "trace_id": trace_id,
                    "drift_detected": drift,
                    "report_uri": s3_uri,
                }
            )
            logger.info(
                "cron: ticker=%s trace=%s drift=%s uri=%s",
                ticker, trace_id, drift, s3_uri or "(none)",
            )
        except Exception as e:  # noqa: BLE001 — log + continue
            logger.exception("cron: ticker=%s failed", ticker)
            failures.append({"ticker": ticker, "trace_id": trace_id, "error": str(e)})

    summary = {
        "fired_at": now.isoformat(),
        "tickers_processed": len(results),
        "tickers_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    logger.info("cron summary: %s", json.dumps({k: v for k, v in summary.items() if k != "results"}))
    return summary
