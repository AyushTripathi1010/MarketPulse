"""
Training loop.

Combines pytorch-forecasting's TFT with Lightning's Trainer. We deliberately
keep this minimal — Lightning handles 95% of the bookkeeping (checkpointing,
gradient clipping, mixed precision, multi-device); we just wire MLflow
logging on top.

For real training (multi-hour) the Phase 2 plan is to run this from a
notebook on Colab T4. The same `train()` function is callable both ways.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from forecast.data.loader import to_training_frame
from forecast.data.splits import time_based_split
from forecast.model.dataset import (
    build_training_dataset,
    build_validation_dataset,
)
from forecast.model.tft import build_tft

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainResult:
    """Return value from a training run."""

    best_checkpoint: Path
    best_val_loss: float
    n_epochs_actually_trained: int
    mlflow_run_id: str | None


def train(
    features: pd.DataFrame,
    *,
    output_dir: str | Path,
    max_epochs: int = 20,
    batch_size: int = 64,
    train_frac: float = 0.8,
    early_stopping_patience: int = 5,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "marketplus-tft",
    accelerator: str = "auto",
) -> TrainResult:
    """Run a full TFT training cycle.

    Parameters
    ----------
    features : pd.DataFrame
        Output of `load_features(...)` — already validated and sorted.
    output_dir : str or Path
        Where to write checkpoints. Best-by-val-loss is kept.
    max_epochs : int
        Cap on training epochs. Early stopping usually fires sooner.
    batch_size : int
    train_frac : float
        Time-based split fraction.
    early_stopping_patience : int
        Stop if val loss doesn't improve for this many epochs.
    mlflow_tracking_uri : str or None
        File URI (file:./mlruns) or remote (https://dagshub.com/...). None
        disables MLflow.
    mlflow_experiment : str
        Experiment name within MLflow.
    accelerator : str
        "auto" picks MPS/CUDA/CPU. "cpu" forces CPU.

    Returns
    -------
    TrainResult
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_df = to_training_frame(features)
    train_df, val_df = time_based_split(training_df, train_frac=train_frac)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Empty train ({len(train_df)}) or val ({len(val_df)}) split. "
            "Need more data."
        )

    train_ds = build_training_dataset(train_df)
    val_ds = build_validation_dataset(train_ds, val_df)

    # num_workers=0 keeps the dataloader single-process — avoids macOS
    # multiprocessing footguns and makes tests deterministic.
    train_loader = train_ds.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    model = build_tft(train_ds)

    callbacks: list[L.pytorch.callbacks.Callback] = [
        # Save the best-by-val-loss checkpoint so /predict can load it later.
        ModelCheckpoint(
            dirpath=output_dir,
            filename="tft-best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            min_delta=1e-4,
        ),
    ]

    pytorch_lightning_logger: MLFlowLogger | bool = False
    mlflow_run_id: str | None = None
    if mlflow_tracking_uri:
        pytorch_lightning_logger = MLFlowLogger(
            experiment_name=mlflow_experiment,
            tracking_uri=mlflow_tracking_uri,
        )
        mlflow_run_id = pytorch_lightning_logger.run_id
        logger.info(
            "MLflow run started: uri=%s experiment=%s run_id=%s",
            mlflow_tracking_uri,
            mlflow_experiment,
            mlflow_run_id,
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        # Disable Lightning's noisy progress bar in test mode; production
        # callers can override via env var.
        enable_progress_bar=True,
        # Lightning sets logging defaults appropriate to local vs. cloud.
        # MLFlowLogger is plugged in only if a tracking URI is provided.
        logger=pytorch_lightning_logger,
        gradient_clip_val=0.1,
        # Deterministic mode is slower but reproducible — important for tests.
        deterministic=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore[union-attr]
    best_val = float(trainer.checkpoint_callback.best_model_score)  # type: ignore[union-attr]

    # Save the dataset object alongside the checkpoint so /predict can rebuild
    # validation/inference TimeSeriesDataSets with matching normalization.
    dataset_pickle = output_dir / "train_ds.pkl"
    torch.save(train_ds, dataset_pickle)

    logger.info(
        "Training complete. best_ckpt=%s best_val_loss=%.6f",
        best_ckpt,
        best_val,
    )

    return TrainResult(
        best_checkpoint=best_ckpt,
        best_val_loss=best_val,
        n_epochs_actually_trained=trainer.current_epoch,
        mlflow_run_id=mlflow_run_id,
    )
