"""
Build pytorch-forecasting TimeSeriesDataSet objects from a training frame.

Why this is its own module:
  - The TimeSeriesDataSet config has *many* knobs (encoder length, target,
    known/observed/static features, normalizers). Getting them right is the
    difference between a TFT that learns and one that overfits to nothing.
  - We want training and inference to build the dataset IDENTICALLY so that
    column normalizations match. Extracting this into one function ensures
    that.
"""

from __future__ import annotations

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# The target the TFT is trained to predict. We use log returns rather than
# raw price because:
#   1. Log returns are stationary (mean ~0, std ~constant) — friendlier loss
#      surface than non-stationary prices that range from $50 to $300+.
#   2. They're additive over horizons, simplifying multi-step backtests.
TARGET_COLUMN = "return_1"

# How many past time steps the model SEES before predicting. 96 hours = 4 days.
# Big enough to capture daily patterns; small enough to fit in memory.
MAX_ENCODER_LENGTH = 96

# How many future time steps the model PREDICTS. We default to 4 hours so the
# TFT learns to do multi-step forecasting; at inference we usually just look
# at the first or last step of the prediction window.
MAX_PREDICTION_LENGTH = 4

# Columns that are KNOWN at prediction time (calendar features).
# Future hour-of-day is known; future RSI is not.
TIME_VARYING_KNOWN_REALS = ("time_idx",)

# Columns that are OBSERVED (revealed only as time progresses).
# These are most of our features.
TIME_VARYING_UNKNOWN_REALS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "return_1",
    "return_24",
    "vol_24",
    "rsi_14",
    "sentiment_24h",
)


def build_training_dataset(
    train_df: pd.DataFrame,
    *,
    max_encoder_length: int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
) -> TimeSeriesDataSet:
    """Build the training-side TimeSeriesDataSet.

    The training dataset includes the GroupNormalizer fit; validation/inference
    datasets are built from THIS one (via `.from_dataset(...)`) so the same
    normalization is reused.
    """
    return TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["ticker"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        # Per-ticker target normalization — different tickers have wildly
        # different return scales (crypto vs blue-chip stock).
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        time_varying_known_reals=list(TIME_VARYING_KNOWN_REALS),
        time_varying_unknown_reals=list(TIME_VARYING_UNKNOWN_REALS),
        # Categorical that identifies each series — TFT learns per-ticker
        # embeddings without us hand-engineering them.
        static_categoricals=["ticker"],
        # add_relative_time_idx + add_target_scales are common TFT recipe
        # additions that give the model richer position information.
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


def build_validation_dataset(
    train_ds: TimeSeriesDataSet,
    val_df: pd.DataFrame,
) -> TimeSeriesDataSet:
    """Build the validation dataset from the training dataset's config.

    Reusing the train normalizer prevents target leakage: validation rows
    get rescaled using statistics computed only on the training split.
    """
    return TimeSeriesDataSet.from_dataset(
        train_ds,
        val_df,
        # predict=False is correct for the val set — we want the model to
        # see val time steps as ordinary training-style sequences during
        # eval (not as one-shot predictions).
        predict=False,
        stop_randomization=True,
    )
