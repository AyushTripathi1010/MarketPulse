"""
Inference: load a trained TFT and produce a Forecast for one ticker.

This is what the FastAPI /predict endpoint calls. It is *intentionally*
synchronous and CPU-friendly — Lambda inference must work without a GPU.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from forecast.data.loader import to_training_frame
from marketplus_shared.models import Forecast


def load_artifacts(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    """Load model + training-dataset config from disk.

    The dataset is needed to rebuild the matching normalization at inference;
    without it the model would see un-normalized features and produce nonsense.

    Returns (model, training_dataset).
    """
    train_ds = torch.load(dataset_path, weights_only=False)
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return model, train_ds


def predict_one(
    model: TemporalFusionTransformer,
    train_ds: TimeSeriesDataSet,
    features: pd.DataFrame,
    *,
    ticker: str,
    horizon_hours: int,
) -> Forecast:
    """Run inference for a single ticker and pack the result into a Forecast.

    The TFT outputs quantile predictions for the entire prediction horizon
    (max_prediction_length steps). We pick the step closest to the user's
    requested horizon — clamped to what the model was trained on.

    Returns
    -------
    Forecast (Pydantic) — the cross-service contract from marketplus_shared.
    """
    # Slice to just this ticker's history; rebuild the training frame shape.
    ticker_features = features[features["ticker"] == ticker]
    if ticker_features.empty:
        raise ValueError(f"No features in DataFrame for ticker={ticker!r}.")

    training_df = to_training_frame(ticker_features)

    # Build an inference dataset using the train normalization. predict=True
    # tells pytorch-forecasting to return one prediction window per group.
    inference_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        training_df,
        predict=True,
        stop_randomization=True,
    )
    inference_loader = inference_ds.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    # raw_predictions returns the full distributional output (quantile heads).
    raw = model.predict(inference_loader, mode="raw", return_x=True)

    # raw.output.prediction has shape (batch, prediction_length, n_quantiles).
    # Quantiles are in the order we passed to QuantileLoss: (0.1, 0.5, 0.9).
    quantile_tensor = raw.output.prediction  # type: ignore[attr-defined]
    # We take the LAST step within the prediction window (= furthest forecast)
    # if the user asked for a horizon at/past the training horizon, else step
    # `horizon_hours - 1`.
    pred_len = quantile_tensor.shape[1]
    step_index = max(0, min(horizon_hours - 1, pred_len - 1))

    # quantile_tensor[batch=0, step, quantile]
    q_low = float(quantile_tensor[0, step_index, 0])
    q_med = float(quantile_tensor[0, step_index, 1])
    q_high = float(quantile_tensor[0, step_index, 2])

    # The model predicts log returns. Convert back to a price using the
    # most recent observed close as the base.
    last_close = float(ticker_features["Close"].iloc[-1])
    predicted_price = last_close * float(torch.exp(torch.tensor(q_med)))
    confidence_low = last_close * float(torch.exp(torch.tensor(q_low)))
    confidence_high = last_close * float(torch.exp(torch.tensor(q_high)))

    return Forecast(
        ticker=ticker,
        predicted_at=datetime.now(UTC),
        horizon_hours=horizon_hours,
        predicted_price=predicted_price,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        # attention_weights are a richer story we'll expose in Phase 3.
        attention_weights={},
    )
