"""
Thin wrapper around MLflow's tracking client.

MLflow is the de-facto standard for experiment tracking in 2026: it records
every run's parameters, metrics, artifacts, and (optionally) the trained
model itself, all keyed by an experiment + run_id. You can compare runs
side-by-side in a web UI and recover any past model by its run_id.

We use DagsHub's hosted MLflow tracking server (free 10GB) in production
and an on-disk file:./mlruns backend for local dev + tests. The same
mlflow.<X> API works against both.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def configure_mlflow(
    tracking_uri: str | None,
    experiment_name: str = "marketplus-tft",
) -> bool:
    """Configure MLflow's process-wide tracking URI + experiment.

    Returns True if MLflow is active, False if disabled (no URI given).

    Why this is a global mutation rather than a Tracker object:
      MLflow's API is designed around a process-global client. We bow to
      that — fighting it just creates leaky abstractions.
    """
    if not tracking_uri:
        logger.info("MLflow disabled (no tracking_uri configured).")
        return False

    # File-backed URIs (file:./mlruns) work out of the box; remote (DagsHub
    # https://dagshub.com/...) requires MLFLOW_TRACKING_USERNAME and
    # MLFLOW_TRACKING_PASSWORD env vars (set in .env). MLflow picks them up
    # automatically.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow active: tracking_uri=%s experiment=%s",
        tracking_uri,
        experiment_name,
    )
    return True


def log_checkpoint_as_artifact(checkpoint_path: str | Path, run_id: str) -> None:
    """Attach a trained checkpoint to a specific MLflow run.

    Lightning's MLFlowLogger logs metrics automatically; checkpoints are NOT
    auto-logged. We do it here explicitly so the run-id can be used by the
    /predict endpoint to fetch the right checkpoint from S3 (Phase 7).
    """
    # mlflow.start_run(run_id=...) with nested=False is required to attach
    # artifacts to an EXISTING run (the one Lightning's logger created).
    with mlflow.start_run(run_id=run_id, nested=False):
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
