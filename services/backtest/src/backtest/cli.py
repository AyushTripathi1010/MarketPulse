"""
CLI entry point for the backtest service.

Lets you run a backtest from the terminal without standing up the FastAPI
server. Useful for one-off experiments, parameter sweeps, and the kind of
exploratory work that doesn't fit cleanly into an HTTP request.

Invoke as:
    uv run backtest run --ticker AAPL --from 2025-11-01 --to 2026-05-01

The CLI re-uses the SAME engine + report code paths as the HTTP endpoint,
so a parameter sweep run via shell loops and a single HTTP call produce
identical artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, time

from backtest.config import settings
from backtest.data_loader import load_historical_features
from backtest.engine import httpx_forecaster, run_walk_forward
from backtest.report import write_report

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _parse_date(s: str) -> datetime:
    """Accept either YYYY-MM-DD or full ISO datetime."""
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise SystemExit(f"Invalid date {s!r}: {e}") from e


def cmd_run(args: argparse.Namespace) -> int:
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    # If user passed bare dates, expand to full day boundaries so the
    # filter captures every bar.
    if start.time() == time.min and not args.start_inclusive_only:
        start = start.replace(hour=0, minute=0, second=0)
    if end.time() == time.min:
        end = end.replace(hour=23, minute=59, second=59)

    logger.info(
        "Loading features  ticker=%s window=%s..%s data_dir=%s",
        args.ticker,
        start.isoformat(),
        end.isoformat(),
        args.data_dir,
    )
    features = load_historical_features(
        args.data_dir, ticker=args.ticker, start=start, end=end
    )
    if features.empty:
        logger.error("No features in window — did you run data_ingest /fetch?")
        return 2

    logger.info(
        "Running walk-forward  bars=%d horizon=%dh",
        len(features),
        args.horizon,
    )
    forecaster = httpx_forecaster(args.forecast_url, args.ticker)
    result = run_walk_forward(
        features,
        forecaster,
        horizon_hours=args.horizon,
        min_encoder_length=args.min_encoder,
        band_threshold=args.band_threshold,
        bars_per_year=settings.bars_per_year,
    )

    artifacts = write_report(
        result,
        ticker=args.ticker,
        start=start,
        end=end,
        output_dir=args.output_dir,
    )

    logger.info("Backtest complete.")
    # Print to stdout for easy piping into jq / other tools.
    print(json.dumps({"metrics": asdict(result.metrics), "artifacts": artifacts}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="MarketPulse walk-forward backtester.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a walk-forward backtest.")
    run_p.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    run_p.add_argument(
        "--from", dest="start", required=True,
        help="Start date (YYYY-MM-DD or ISO datetime).",
    )
    run_p.add_argument(
        "--to", dest="end", required=True,
        help="End date (YYYY-MM-DD or ISO datetime).",
    )
    run_p.add_argument("--horizon", type=int, default=4, help="Forecast horizon in hours.")
    run_p.add_argument(
        "--min-encoder", type=int, default=96,
        help="Min encoder window length before backtest starts taking trades.",
    )
    run_p.add_argument(
        "--band-threshold", type=float, default=0.002,
        help="Min predicted relative move to trigger a position.",
    )
    run_p.add_argument(
        "--data-dir", default=settings.local_data_dir,
        help="Path to feature Parquet files.",
    )
    run_p.add_argument(
        "--output-dir", default=settings.local_output_dir,
        help="Where to write the JSON+MD+PNG artifacts.",
    )
    run_p.add_argument(
        "--forecast-url", default=settings.forecast_url,
        help="URL of the forecast service to call.",
    )
    run_p.add_argument(
        "--start-inclusive-only", action="store_true",
        help="Use EXACTLY the start datetime instead of expanding to 00:00:00.",
    )
    run_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logs.")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
