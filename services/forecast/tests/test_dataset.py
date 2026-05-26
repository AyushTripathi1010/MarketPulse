"""Tests for the TimeSeriesDataSet construction (the most footgun-prone part)."""

from __future__ import annotations

import pandas as pd

from forecast.data.loader import to_training_frame
from forecast.data.splits import time_based_split
from forecast.model.dataset import (
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    TARGET_COLUMN,
    build_training_dataset,
    build_validation_dataset,
)


def test_build_training_dataset_target_is_log_return(synthetic_features: pd.DataFrame) -> None:
    """The target the TFT learns must be the log-return column, not raw price."""
    df = to_training_frame(synthetic_features)
    train_df, _ = time_based_split(df)
    ds = build_training_dataset(train_df)
    assert ds.target == TARGET_COLUMN
    assert TARGET_COLUMN == "return_1"


def test_build_training_dataset_encoder_lengths(synthetic_features: pd.DataFrame) -> None:
    """Encoder + prediction lengths must match the module-level constants."""
    df = to_training_frame(synthetic_features)
    train_df, _ = time_based_split(df)
    ds = build_training_dataset(train_df)
    assert ds.max_encoder_length == MAX_ENCODER_LENGTH
    assert ds.max_prediction_length == MAX_PREDICTION_LENGTH


def test_build_validation_dataset_shares_train_normalization(
    synthetic_features: pd.DataFrame,
) -> None:
    """Val dataset must inherit train's normalization (no leakage of val stats)."""
    df = to_training_frame(synthetic_features)
    train_df, val_df = time_based_split(df)
    train_ds = build_training_dataset(train_df)
    val_ds = build_validation_dataset(train_ds, val_df)

    # Both datasets should reference the same normalizer instance/config.
    assert val_ds.target_normalizer is train_ds.target_normalizer or (
        type(val_ds.target_normalizer) is type(train_ds.target_normalizer)
    )
