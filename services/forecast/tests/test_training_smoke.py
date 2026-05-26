"""
Smoke test for the full training loop.

This trains a TINY TFT for 1 epoch on synthetic data — the goal isn't a
useful model, it's to prove every wire holds: data loader → dataset →
TFT → Lightning trainer → checkpoint → reload → predict.

If this passes, we know the whole Phase 2 plumbing works. If something
breaks in real Colab training later, it's a hyperparameter issue, not a
pipeline issue.

Marked @pytest.mark.slow because even a 1-epoch run takes ~20s.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from forecast.data.loader import to_training_frame
from forecast.model.predict import load_artifacts, predict_one
from forecast.model.train import train


@pytest.mark.slow
def test_train_then_predict_endtoend(
    synthetic_features: pd.DataFrame, tmp_path: Path
) -> None:
    """Train 1 epoch on synthetic data, reload checkpoint, run a prediction."""
    out = tmp_path / "models"

    # CPU-only + 1 epoch keeps wall time under ~30s on a laptop.
    result = train(
        synthetic_features,
        output_dir=out,
        max_epochs=1,
        batch_size=32,
        train_frac=0.8,
        early_stopping_patience=10,  # disable for the smoke test
        accelerator="cpu",
        mlflow_tracking_uri=None,  # no MLflow in the smoke test
    )

    # Checkpoint produced and val loss is finite (not NaN — would indicate
    # gradient explosion or a bug in normalization).
    assert result.best_checkpoint.exists()
    assert torch.isfinite(torch.tensor(result.best_val_loss))
    assert result.n_epochs_actually_trained >= 1

    # Reload and predict — proves the inference path works against the
    # checkpoint we just wrote.
    model, train_ds = load_artifacts(
        result.best_checkpoint, out / "train_ds.pkl"
    )
    forecast = predict_one(
        model,
        train_ds,
        synthetic_features,
        ticker="AAPL",
        horizon_hours=1,
    )

    assert forecast.ticker == "AAPL"
    assert forecast.predicted_price > 0
    # The quantile order must hold: low <= predicted <= high. If this fails
    # the model is producing inverted quantiles, which means we're reading
    # the wrong tensor dimension somewhere in predict.py.
    assert forecast.confidence_low <= forecast.predicted_price <= forecast.confidence_high
