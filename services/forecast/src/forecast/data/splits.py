"""
Time-aware train/val split.

Random splits are the cardinal sin of time-series ML — you'd train on rows
that come after your "test" rows, leaking the future into the past. Every
metric you'd report would be wildly optimistic.

We split by time: the first `train_frac` of each ticker's history goes
to training; the rest is validation. Same cutoff index across all tickers
keeps the splits aligned in wallclock time.
"""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a per-ticker time-indexed frame into train and validation.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `to_training_frame` — must have a `time_idx` column.
    train_frac : float
        Fraction of each ticker's history in the training split.

    Returns
    -------
    (train_df, val_df) : tuple of pd.DataFrame
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    # Per-ticker max time_idx so the cutoff is consistent within each group.
    per_ticker_cutoff = df.groupby("ticker")["time_idx"].transform(
        lambda s: int(s.max() * train_frac)
    )

    train_df = df[df["time_idx"] <= per_ticker_cutoff].copy()
    val_df = df[df["time_idx"] > per_ticker_cutoff].copy()
    return train_df, val_df
