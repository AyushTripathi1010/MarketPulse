"""
Constants used across services.

Anything that's NOT a secret (those live in .env) but IS used in more than
one service belongs here — feature names, default thresholds, enum-like
strings, etc.
"""

from typing import Final

# Default lookback window for OHLCV data (90 days is enough for most TFT
# context windows while staying small enough to fit in Lambda memory).
DEFAULT_LOOKBACK_DAYS: Final[int] = 90

# Default forecast horizon: 4 hours ahead. The TFT model is trained to
# predict on this horizon — change requires retraining.
DEFAULT_HORIZON_HOURS: Final[int] = 4

# Error threshold above which we trigger a fine-tune. Computed as 7-day
# rolling MAE on closing-price predictions. Picked empirically — see
# learning/15-difficulties-and-interview-qa.md for the calibration story.
DRIFT_MAE_THRESHOLD: Final[float] = 0.025  # 2.5% MAE

# Tickers we currently support. The TFT was trained on these. Adding
# a new ticker requires either fine-tuning or a zero-shot foundation model.
SUPPORTED_TICKERS: Final[tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "TSLA",
    "BTC-USD",
    "ETH-USD",
)
