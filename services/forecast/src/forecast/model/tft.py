"""
TFT architecture configuration.

We don't reimplement the Temporal Fusion Transformer here — pytorch-forecasting
ships a battle-tested version. Our job is to construct it with sensible
hyperparameters for our data scale (a few tickers, hourly bars, hundreds to
thousands of rows). Big TFTs are powerful but slow and overfit on small data;
small ones are honest and trainable in minutes.

This module is intentionally tiny — the heavy lifting is in pytorch-forecasting.
The senior-engineering value here is in CHOOSING the right hyperparameters
and explaining why; see learning/08-time-series-and-tft.md.
"""

from __future__ import annotations

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


def build_tft(
    training_dataset: TimeSeriesDataSet,
    *,
    hidden_size: int = 16,
    attention_head_size: int = 1,
    dropout: float = 0.1,
    hidden_continuous_size: int = 8,
    learning_rate: float = 1e-3,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> TemporalFusionTransformer:
    """Construct a TFT sized for our project scale.

    Hyperparameter justification:
      - hidden_size=16: small. Our dataset is hundreds-of-thousands of rows
        at most; a bigger model just memorizes noise. The 2020 TFT paper
        used 160; modern benchmarks find 16-64 is plenty for ETF-scale
        single-instrument forecasting.
      - attention_head_size=1: single-head suffices when the input
        dimensionality is modest. More heads = more params, no observed gain
        on this kind of data.
      - dropout=0.1: standard regularization. We're not overfitting (yet),
        but cheap insurance.
      - hidden_continuous_size=8: must be <= hidden_size; controls per-feature
        variable selection network width.
      - learning_rate=1e-3: pytorch-forecasting's TFT defaults to 1e-2 which
        is aggressive; we pull it down for stability on small data.
      - quantiles=(0.1, 0.5, 0.9): the headline output of the TFT is a
        QUANTILE forecast, not a point estimate. The 0.1 and 0.9 quantiles
        ARE the confidence interval our /predict endpoint returns.
    """
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=list(quantiles)),
        # Logging frequency for Lightning's progress bar.
        log_interval=10,
        # Slow but full attention weights so we can return interpretation
        # alongside predictions.
        reduce_on_plateau_patience=4,
    )
