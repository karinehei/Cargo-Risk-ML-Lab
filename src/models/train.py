"""Model training, persistence and inference utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import AppConfig, get_config, resolve_path, set_seed, setup_logging
from src.features import get_feature_names, prepare_xy
from src.models.estimators import build_estimator as build_named_estimator
from src.models.estimators import build_model_pipeline as build_named_pipeline

logger = setup_logging(name="src.models")


@dataclass
class TrainedModelBundle:
    """Container for a fitted pipeline, metadata and feature names."""

    pipeline: Pipeline
    feature_names: list[str]
    metadata: dict[str, Any]


def build_estimator(config: AppConfig | None = None) -> Any:
    """Construct the default XGBoost classifier (single-model helper)."""
    y_placeholder = np.array([0, 1], dtype=int)
    return build_named_estimator("xgboost", y_placeholder, config)


def build_model_pipeline(config: AppConfig | None = None) -> Pipeline:
    """Create an unfitted preprocessing + estimator pipeline."""
    y_placeholder = np.array([0, 1], dtype=int)
    return build_named_pipeline("xgboost", y_placeholder, config)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    config: AppConfig | None = None,
    *,
    estimator_name: str | None = None,
) -> TrainedModelBundle:
    """Fit one model pipeline on training data (no test set involved)."""
    cfg = config or get_config()
    seed = set_seed(cfg.random_seed)
    x_train, y_train = prepare_xy(train_df, cfg, fit_derived_reference=train_df)
    name = str(estimator_name or cfg.model.get("algorithm", "xgboost"))
    pipeline = build_named_pipeline(name, y_train, cfg)

    logger.info("Fitting %s pipeline on %s rows", name, len(x_train))
    pipeline.fit(x_train, y_train)

    feature_names = get_feature_names(pipeline.named_steps["preprocess"])
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(),
        "random_seed": seed,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)) if val_df is not None else 0,
        "positive_rate_train": float(np.mean(y_train)),
        "algorithm": name,
        "threshold": float(cfg.model.get("threshold", 0.5)),
        "feature_names": feature_names,
        "target_column": str(cfg.data.get("target_column", "requires_review")),
        "derived_feature_stats": {},
        "disclaimer": cfg.disclaimer,
    }
    return TrainedModelBundle(pipeline=pipeline, feature_names=feature_names, metadata=metadata)


def predict_proba(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a feature frame."""
    proba = pipeline.predict_proba(features)
    return np.asarray(proba[:, 1], dtype=float)


def predict_labels(
    pipeline: Pipeline,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return binary predictions using a probability threshold."""
    scores = predict_proba(pipeline, features)
    return (scores >= threshold).astype(int)


def save_model_bundle(bundle: TrainedModelBundle, artifact_dir: str | Path | None = None) -> Path:
    """Persist the complete preprocessing-and-model pipeline."""
    cfg = get_config()
    out_dir = resolve_path(artifact_dir or str(cfg.training.get("artifact_dir", "artifacts")))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    pipeline_path = out_dir / "preprocess_pipeline.joblib"
    metadata_path = out_dir / "model_metadata.json"

    steps = bundle.pipeline.named_steps
    if "preprocess" not in steps or "model" not in steps:
        raise ValueError("Saved pipeline must contain both preprocess and model steps")
    joblib.dump(bundle.pipeline, model_path)
    joblib.dump(bundle.pipeline.named_steps["preprocess"], pipeline_path)
    serialisable = dict(bundle.metadata)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2, default=str)

    logger.info("Saved model artifacts to %s", out_dir)
    return out_dir


def load_model_bundle(artifact_dir: str | Path | None = None) -> TrainedModelBundle:
    """Load a previously saved model bundle from disk."""
    cfg = get_config()
    out_dir = resolve_path(artifact_dir or str(cfg.training.get("artifact_dir", "artifacts")))
    model_path = out_dir / "model.joblib"
    metadata_path = out_dir / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline: Pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)

    feature_names = list(
        metadata.get("feature_names") or get_feature_names(pipeline.named_steps["preprocess"])
    )
    return TrainedModelBundle(pipeline=pipeline, feature_names=feature_names, metadata=metadata)
