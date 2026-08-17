"""Log one train/validation model run to the local MLflow backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator

from src.config import AppConfig, get_config
from src.evaluation.latency import measure_inference_latency
from src.mlops.fingerprints import dataframe_fingerprint, split_fingerprint
from src.mlops.serialization import LoggedModel, log_sklearn_pipeline, verify_logged_pipeline
from src.mlops.tracking import current_git_commit


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested hyperparameters to MLflow param strings."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=name))
        else:
            flat[name] = str(value)
    return flat


def log_candidate_run(
    *,
    run_name: str,
    model_family: str,
    pipeline: BaseEstimator,
    hyperparameters: dict[str, Any],
    val_metrics: dict[str, float],
    x_val: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    split_manifest_path: str | Path | None,
    threshold: float,
    threshold_policy: str,
    cv_mean: float | None,
    cv_std: float | None,
    class_weight: str,
    calibration_status: str,
    preprocess_config: dict[str, Any],
    extra_tags: dict[str, str] | None = None,
    extra_metrics: dict[str, float] | None = None,
    artifact_files: list[Path] | None = None,
    latency_repeats: int = 50,
    fixture: pd.DataFrame | None = None,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Create an MLflow run with lineage, validation metrics and the fitted pipeline."""
    cfg = config or get_config()
    if any(str(key).startswith("test_") for key in val_metrics):
        raise ValueError("Test metrics must not be logged into experiment runs used for selection")

    tags = {
        "model_family": model_family,
        "calibration_status": calibration_status,
        "git_commit": current_git_commit(),
        "split": "validation",
        "experiment": str(
            cfg.mlops.get("experiment_name") or cfg.training.get("mlflow_experiment") or ""
        ),
        **(extra_tags or {}),
    }
    params = {
        "model_family": model_family,
        "random_seed": str(cfg.random_seed),
        "class_weight": class_weight,
        "calibration_status": calibration_status,
        "threshold": str(threshold),
        "threshold_policy": threshold_policy,
        "n_train": str(len(train_df)),
        "n_val": str(len(val_df)),
        **flatten_params(hyperparameters, prefix="hp"),
        **flatten_params(preprocess_config, prefix="preprocess"),
    }
    metrics = {
        "val_pr_auc": float(val_metrics.get("pr_auc") or val_metrics.get("ranking_pr_auc") or 0.0),
        "val_roc_auc": float(val_metrics.get("roc_auc", 0.0)),
        "val_precision": float(
            val_metrics.get("precision", val_metrics.get("precision_positive", 0.0))
        ),
        "val_recall": float(val_metrics.get("recall", val_metrics.get("recall_positive", 0.0))),
        "val_f1": float(val_metrics.get("f1", val_metrics.get("f1_positive", 0.0))),
        "val_brier": float(val_metrics.get("brier_score", 0.0)),
        "val_ece": float(val_metrics.get("ece", 0.0)),
    }
    if cv_mean is not None:
        metrics["cv_pr_auc_mean"] = float(cv_mean)
    if cv_std is not None:
        metrics["cv_pr_auc_std"] = float(cv_std)
    if extra_metrics:
        if any(str(key).startswith("test_") for key in extra_metrics):
            raise ValueError(
                "Test metrics must not be logged into experiment runs used for selection"
            )
        metrics.update(extra_metrics)

    latency = measure_inference_latency(
        pipeline, x_val, repeats=latency_repeats, seed=cfg.random_seed
    )
    metrics.update(
        {
            "latency_p50_ms": latency["latency_p50_ms"],
            "latency_p95_ms": latency["latency_p95_ms"],
            "latency_p99_ms": latency["latency_p99_ms"],
        }
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "train_fingerprint": dataframe_fingerprint(train_df),
                "val_fingerprint": dataframe_fingerprint(val_df),
                "split": split_fingerprint(split_manifest_path),
            },
            "dataset_fingerprint.json",
        )
        for path in artifact_files or []:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="evaluation")

        logged: LoggedModel = log_sklearn_pipeline(pipeline)
        fixture_frame = fixture if fixture is not None else x_val.head(min(16, len(x_val)))
        if logged.method.startswith("mlflow.sklearn"):
            logged = verify_logged_pipeline(pipeline, logged, fixture_frame)
            mlflow.log_metrics({"roundtrip_ok": 1.0 if logged.roundtrip_ok else 0.0})
        else:
            mlflow.log_metrics({"roundtrip_ok": 0.0})

        payload = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model_family": model_family,
            "hyperparameters": hyperparameters,
            "validation_metrics": {
                **metrics,
                "threshold": float(threshold),
            },
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "threshold": float(threshold),
            "threshold_policy": threshold_policy,
            "calibration_status": calibration_status,
            "class_weight": class_weight,
            "artifact_uri": logged.artifact_uri,
            "serialization": logged.method,
            "roundtrip_ok": logged.roundtrip_ok,
            "fallback_reason": logged.fallback_reason,
            "dataset_fingerprint": dataframe_fingerprint(train_df),
            "val_fingerprint": dataframe_fingerprint(val_df),
            "git_commit": tags["git_commit"],
            "latency": latency,
        }
        mlflow.log_dict(payload, "run_summary.json")
        return payload
