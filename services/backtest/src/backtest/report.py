"""
Backtest report generator.

Three outputs from one engine result:
  1. JSON metrics — for the frontend / API consumers / downstream pipelines.
  2. Markdown summary — for humans to read.
  3. Equity-curve PNG — for the dashboard.

All three land in a configurable output directory (local FS by default;
S3 in production). The functions are I/O-bound and exception-tolerant —
a failed plot doesn't kill the report; a failed S3 upload doesn't lose
the local file.

Why matplotlib over Plotly/Bokeh for the chart:
  - matplotlib renders to PNG without a browser. Plotly needs a kernel or
    Kaleido (extra binary). The dashboard is Next.js (no Python at render
    time), so we ship a static PNG it can <img src=...> against.
  - The chart is one-shot — we never need interactive zoom on a deployed
    backtest. Static is fine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

# `Agg` is the headless backend — no Tcl/Tk dependency, no GUI, no DISPLAY
# env var required. Critical for Lambda + Docker environments. Must be
# set BEFORE pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- intentional after backend select
import pandas as pd  # noqa: E402

from backtest.engine import WalkForwardResult  # noqa: E402

logger = logging.getLogger(__name__)


MARKDOWN_TEMPLATE = """# Backtest Report — {ticker}

**Window:** {start_iso} → {end_iso}
**Bars evaluated:** {num_bars}
**Trades opened:** {num_trades}

## Headline metrics

| Metric | Value |
|---|---|
| Total return | {total_return:+.2%} |
| Annualized Sharpe | {sharpe:.3f} |
| Max drawdown | {max_drawdown:.2%} |
| Hit rate (directional) | {hit_rate:.1%} |
| Win rate (positive bars) | {win_rate:.1%} |
| Mean absolute price error | ${mae:.4f} |

## Interpretation

{interpretation}

## Equity curve

![Equity curve]({equity_curve_filename})

---

_Frictionless backtest — no transaction costs or slippage modeled. The Sharpe
number is an UPPER bound on what a real account would have earned. See
`learning/17-sharpe-drawdown-and-finance-metrics.md` for the trade-offs and
how to interpret these numbers honestly._
"""


def _interpret(metrics) -> str:
    """Compose a short prose interpretation of the metrics.

    Deterministic — not an LLM call. The frontend can ALWAYS render the
    report even when Groq is down. (We could optionally LLM-polish the
    prose later, but the deterministic baseline always exists.)
    """
    parts: list[str] = []

    # Sharpe interpretation
    if metrics.sharpe > 2:
        parts.append("Sharpe > 2 suggests a strong risk-adjusted edge — but verify with longer windows; thin-window Sharpe is unreliable.")
    elif metrics.sharpe > 1:
        parts.append("Sharpe in the 1–2 band is a plausible real edge, well above a buy-and-hold baseline for individual stocks.")
    elif metrics.sharpe > 0:
        parts.append("Sharpe between 0 and 1 — the strategy beats cash but doesn't compensate for the volatility taken on.")
    else:
        parts.append("Sharpe ≤ 0 — the strategy lost money on a risk-adjusted basis. Hit rate above 50% (if any) was eaten by adverse moves.")

    # Drawdown interpretation
    if metrics.max_drawdown < -0.20:
        parts.append(f"A {metrics.max_drawdown:.0%} drawdown is severe — most real-world investors would have liquidated mid-trough.")
    elif metrics.max_drawdown < -0.10:
        parts.append(f"A {metrics.max_drawdown:.0%} drawdown is meaningful but recoverable; volatile-strategy normal.")
    else:
        parts.append(f"Drawdown of {metrics.max_drawdown:.1%} is mild — strategy was relatively calm through the window.")

    # Hit rate vs PnL gap
    if metrics.hit_rate > 0.55 and metrics.total_return < 0:
        parts.append(
            "Note the gap: directional hit rate > 55% but total return is negative. "
            "The strategy was RIGHT more often than wrong, but the wrong calls were larger losers than the right calls were winners — "
            "a classic 'asymmetric tails' problem the band-aware signal logic doesn't fully resolve."
        )

    if metrics.num_trades == 0:
        parts.append("Zero trades were taken — the band-threshold filter rejected every forecast. Either the threshold is too strict or the forecaster's bands are wider than the predicted moves.")

    return " ".join(parts)


def render_equity_curve(history: pd.DataFrame, out_path: Path, *, ticker: str) -> Path | None:
    """Render the equity curve PNG. Returns the path on success, None on failure.

    Exception-tolerant by design — a failed plot should not kill the report.
    The markdown will reference the file regardless; the frontend handles
    the broken-img case with a placeholder.
    """
    if history.empty or "equity" not in history.columns:
        logger.warning("render_equity_curve: empty history, skipping plot.")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        history["equity"].plot(ax=ax, color="#1f77b4", linewidth=1.5)
        ax.set_title(f"{ticker} — backtest equity curve")
        ax.set_ylabel("Equity (multiples of initial capital)")
        ax.set_xlabel("")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="png")
        plt.close(fig)
        return out_path
    except Exception as e:  # noqa: BLE001  -- best-effort plot
        logger.warning("render_equity_curve: failed to render plot: %s", e)
        return None


def write_report(
    result: WalkForwardResult,
    *,
    ticker: str,
    start: datetime,
    end: datetime,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON + Markdown + PNG to output_dir.

    Returns
    -------
    dict mapping artifact name → file path. Keys: 'json', 'markdown', 'png'.
    PNG key may map to '' if rendering failed (callers should test, not assert).
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    base_name = f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}"
    json_path = out / f"{base_name}.json"
    md_path = out / f"{base_name}.md"
    png_path = out / f"{base_name}_equity.png"

    # 1. JSON — always writes. Even on empty results we want a stable
    #    artifact so downstream batch processes don't crash.
    metrics_dict = asdict(result.metrics)
    metrics_dict.update(
        {
            "ticker": ticker,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "num_bars": len(result.history),
        }
    )
    json_path.write_text(json.dumps(metrics_dict, indent=2))

    # 2. PNG — best effort.
    png_result = render_equity_curve(result.history, png_path, ticker=ticker)

    # 3. Markdown — references the PNG file regardless.
    md_text = MARKDOWN_TEMPLATE.format(
        ticker=ticker,
        start_iso=start.isoformat(),
        end_iso=end.isoformat(),
        num_bars=len(result.history),
        num_trades=result.metrics.num_trades,
        total_return=result.metrics.total_return,
        sharpe=result.metrics.sharpe,
        max_drawdown=result.metrics.max_drawdown,
        hit_rate=result.metrics.hit_rate,
        win_rate=result.metrics.win_rate,
        mae=result.metrics.mae,
        interpretation=_interpret(result.metrics),
        equity_curve_filename=png_path.name,
    )
    md_path.write_text(md_text)

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "png": str(png_result) if png_result else "",
    }
