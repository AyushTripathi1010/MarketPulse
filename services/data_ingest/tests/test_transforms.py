"""
Tests for the pure-function transforms.

These are the easiest tests in the project — no mocks, no I/O, just
dataframes in and dataframes out. If these don't pass, nothing else will.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_ingest.transforms.clean import clean_ohlcv, news_to_dataframe
from data_ingest.transforms.features import (
    add_returns,
    add_rsi,
    add_sentiment_decay,
    add_volatility,
    build_feature_frame,
)


# ---------------------------------------------------------------------------
# clean_ohlcv
# ---------------------------------------------------------------------------
def test_clean_ohlcv_drops_nan_close(synthetic_ohlcv: pd.DataFrame) -> None:
    """Rows with NaN Close should be dropped."""
    df = synthetic_ohlcv.copy()
    df.iloc[5, df.columns.get_loc("Close")] = np.nan
    cleaned = clean_ohlcv(df)
    assert len(cleaned) == len(synthetic_ohlcv) - 1
    assert cleaned["Close"].isna().sum() == 0


def test_clean_ohlcv_dedupes_index(synthetic_ohlcv: pd.DataFrame) -> None:
    """Duplicate timestamps should collapse to one (keep last)."""
    df = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[[10]]])
    cleaned = clean_ohlcv(df)
    assert cleaned.index.is_unique


def test_clean_ohlcv_is_sorted(synthetic_ohlcv: pd.DataFrame) -> None:
    """Result must be sorted ascending by timestamp regardless of input order."""
    shuffled = synthetic_ohlcv.sample(frac=1.0, random_state=0)
    cleaned = clean_ohlcv(shuffled)
    assert cleaned.index.is_monotonic_increasing


def test_clean_ohlcv_does_not_mutate_input(synthetic_ohlcv: pd.DataFrame) -> None:
    """The cleaner returns a NEW frame; original must be untouched."""
    before = synthetic_ohlcv.copy()
    _ = clean_ohlcv(synthetic_ohlcv)
    pd.testing.assert_frame_equal(synthetic_ohlcv, before)


# ---------------------------------------------------------------------------
# news_to_dataframe
# ---------------------------------------------------------------------------
def test_news_to_dataframe_empty_returns_empty_with_schema() -> None:
    """Empty input must still have the right columns (so downstream code never KeyErrors)."""
    out = news_to_dataframe([])
    assert list(out.columns) == ["headline", "summary", "symbols", "source", "url"]
    assert len(out) == 0


def test_news_to_dataframe_sorted_by_time(synthetic_news: list) -> None:
    """News should come out time-ordered, oldest first."""
    out = news_to_dataframe(synthetic_news)
    assert out.index.is_monotonic_increasing
    assert len(out) == 3


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def test_add_returns_first_row_is_nan(synthetic_ohlcv: pd.DataFrame) -> None:
    """return_1 of the first row is undefined (no previous price)."""
    out = add_returns(synthetic_ohlcv)
    assert np.isnan(out["return_1"].iloc[0])
    assert not np.isnan(out["return_1"].iloc[1])


def test_add_returns_log_property(synthetic_ohlcv: pd.DataFrame) -> None:
    """Log returns should additively reconstruct the total log change."""
    out = add_returns(synthetic_ohlcv)
    total_log_return = out["return_1"].iloc[1:].sum()
    expected = np.log(out["Close"].iloc[-1] / out["Close"].iloc[0])
    assert np.isclose(total_log_return, expected, atol=1e-10)


def test_add_volatility_is_nonnegative(synthetic_ohlcv: pd.DataFrame) -> None:
    """Volatility is a std — must be >= 0 wherever defined."""
    out = add_volatility(synthetic_ohlcv)
    vol = out["vol_24"].dropna()
    assert (vol >= 0).all()


def test_add_rsi_bounds(synthetic_ohlcv: pd.DataFrame) -> None:
    """RSI must be between 0 and 100 inclusive."""
    out = add_rsi(synthetic_ohlcv)
    rsi = out["rsi_14"].dropna()
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0


def test_add_sentiment_decay_zero_when_no_news(synthetic_ohlcv: pd.DataFrame) -> None:
    """No news → sentiment is 0 everywhere."""
    empty_news = news_to_dataframe([])
    out = add_sentiment_decay(synthetic_ohlcv, empty_news)
    assert (out["sentiment_24h"] == 0.0).all()


def test_add_sentiment_decay_recent_is_higher(synthetic_news: list) -> None:
    """A bar AFTER all news items should see more decayed signal than one BEFORE."""
    news_df = news_to_dataframe(synthetic_news)
    # Build a tiny OHLCV frame: one bar before all news, one after all news.
    early = news_df.index.min() - pd.Timedelta(hours=1)
    late = news_df.index.max() + pd.Timedelta(hours=1)
    ohlcv = pd.DataFrame(
        {"Close": [100.0, 100.0], "ticker": ["AAPL", "AAPL"]},
        index=pd.DatetimeIndex([early, late], tz="UTC", name="timestamp"),
    )
    out = add_sentiment_decay(ohlcv, news_df)
    # The "late" bar has news in its past 24h; the "early" bar has only future news.
    assert out["sentiment_24h"].iloc[0] == 0.0
    assert out["sentiment_24h"].iloc[1] > 0.0


def test_build_feature_frame_has_expected_columns(
    synthetic_ohlcv: pd.DataFrame, synthetic_news: list
) -> None:
    """Smoke test the whole pipeline: every expected feature column appears."""
    news_df = news_to_dataframe(synthetic_news)
    out = build_feature_frame(synthetic_ohlcv, news_df)
    for col in ("return_1", "return_24", "vol_24", "rsi_14", "sentiment_24h"):
        assert col in out.columns, f"missing feature column: {col}"
