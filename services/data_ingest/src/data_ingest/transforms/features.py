"""
Feature engineering for the TFT.

Why feature-engineer here instead of inside the model?
  - Reproducibility. Features computed here get stored in Parquet alongside
    the raw OHLCV. The TFT sees the SAME features at training and inference.
  - Speed. Doing it once at ingest is cheaper than re-computing per request.
  - Debuggability. You can eyeball a Parquet file in pandas and see the
    features. You can't easily inspect tensors inside a forward pass.

Every function here is pure: pandas in, pandas out. Easy to unit-test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Add `return_1` (one-bar log return) and `return_24` (24-bar log return).

    Why log returns instead of percent returns?
      - Log returns are additive over time (sum of log returns = log of total
        return) — convenient for backtests.
      - They're symmetric: a +10% log return cancels a -10% log return.
        Percent returns don't have this property (+10% then -10% ≠ flat).
      - The TFT's loss function (typically quantile/MAE) behaves better on
        log-transformed targets when prices span many orders of magnitude
        (e.g. AAPL at $187 vs BTC at $60,000).
    """
    out = df.copy()
    out["return_1"] = np.log(out[close_col] / out[close_col].shift(1))
    out["return_24"] = np.log(out[close_col] / out[close_col].shift(24))
    return out


def add_volatility(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling std of 1-bar returns over `window` bars.

    Volatility is one of the most predictive features for short-horizon
    movement — markets that just moved a lot tend to keep moving.
    """
    if "return_1" not in df.columns:
        df = add_returns(df)
    out = df.copy()
    out[f"vol_{window}"] = out["return_1"].rolling(window=window, min_periods=2).std()
    return out


def add_rsi(df: pd.DataFrame, period: int = 14, close_col: str = "Close") -> pd.DataFrame:
    """Relative Strength Index (RSI) — classic momentum oscillator.

    RSI is the ratio of average up-moves to average down-moves over `period`
    bars, scaled to 0-100. Above 70 = "overbought", below 30 = "oversold".
    Even if you don't believe in TA, RSI is a useful feature because the
    market behaves AS IF other people believe in it (self-fulfilling).
    """
    out = df.copy()
    delta = out[close_col].diff()
    # Separate gains and losses; treat NaN-on-first-row as 0.
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Wilder's smoothing (Exponential moving average with alpha = 1/period).
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Avoid division by zero — when there are no losses, RS is huge → RSI=100.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    out[f"rsi_{period}"] = out[f"rsi_{period}"].fillna(50.0)  # neutral default
    return out


def add_sentiment_decay(
    ohlcv: pd.DataFrame,
    news: pd.DataFrame,
    half_life_hours: float = 6.0,
    sentiment_col: str | None = None,
) -> pd.DataFrame:
    """Attach a decaying sentiment score to each OHLCV bar.

    News doesn't have a clean numeric sentiment from Alpaca by default
    (Phase 5 may add one via FinBERT). For now: count news items in the
    last 24h, weighted by exponential decay so a headline 6h ago has half
    the weight of one just published.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Cleaned OHLCV indexed by tz-aware timestamp.
    news : pd.DataFrame
        Output of `news_to_dataframe`, indexed by `created_at`.
    half_life_hours : float
        How fast a news item's influence decays. 6h means a 6-hour-old
        headline contributes half what a fresh one does.
    sentiment_col : str or None
        Name of a sentiment column in `news` (when Phase 5 lands FinBERT
        scores). If None, treat every news item as sentiment 1.0
        (i.e. pure news-volume).

    Returns
    -------
    pd.DataFrame
        Copy of ohlcv with a `sentiment_24h` column attached.
    """
    out = ohlcv.copy()

    if news.empty:
        out["sentiment_24h"] = 0.0
        return out

    # Pre-decay each news item to its base weight.
    # exp(-ln(2) * age / half_life) = 0.5 ^ (age / half_life)
    decay_factor = np.log(2) / half_life_hours

    sentiment_values: list[float] = []
    for bar_ts in out.index:
        # Hours from each news item to the current bar. Future news = ignored.
        ages_hours = (bar_ts - news.index).total_seconds() / 3600.0
        # Only count items in the past 24h.
        mask = (ages_hours >= 0) & (ages_hours <= 24)
        if not mask.any():
            sentiment_values.append(0.0)
            continue

        relevant_ages = ages_hours[mask]
        weights = np.exp(-decay_factor * relevant_ages)

        if sentiment_col and sentiment_col in news.columns:
            scores = news.loc[mask, sentiment_col].to_numpy()
            sentiment_values.append(float(np.sum(weights * scores)))
        else:
            # Pure news-volume signal.
            sentiment_values.append(float(np.sum(weights)))

    out["sentiment_24h"] = sentiment_values
    return out


def build_feature_frame(ohlcv: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature pipeline: clean OHLCV + features + decayed news sentiment.

    This is the single entry point the FastAPI handler calls. Returning one
    frame keeps the contract simple: caller doesn't need to know about
    intermediate steps.
    """
    df = add_returns(ohlcv)
    df = add_volatility(df, window=24)
    df = add_rsi(df, period=14)
    df = add_sentiment_decay(df, news, half_life_hours=6.0)
    return df
