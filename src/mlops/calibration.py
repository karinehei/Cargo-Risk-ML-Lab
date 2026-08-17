"""Train-only probability calibration. Validation is used only to compare fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import AppConfig, get_config
from src.evaluation.metrics import compute_classification_metrics, expected_calibration_error
from src.evaluation.threshold import select_threshold
from src.features import build_preprocess_pipeline
from src.models.estimators import build_model_pipeline
from src.models.train import predict_proba

ISOTONIC_MIN_TRAIN_ROWS = 1000
ISOTONIC_MIN_POSITIVES = 50
FROZEN_OPERATING_THRESHOLD = 0.525


@dataclass
class CalibrationCandidate:
    """One calibration strategy fitted on training rows only."""

    name: str
    pipeline: Pipeline | CalibratedClassifierCV
    calibration_status: str
    class_weight: str
    val_metrics_at_frozen_threshold: dict[str, float]
    val_metrics_at_selected_threshold: dict[str, float]
    selected_threshold: float
    threshold_policy: str
    hyperparameters: dict[str, Any]


def _unweighted_logreg_pipeline(config: AppConfig, y_train: np.ndarray) -> Pipeline:
    seed = int(config.random_seed)
    _ = y_train
    return Pipeline(
        steps=[
            ("preprocess", build_preprocess_pipeline(config, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    class_weight=None,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def _score(
    estimator: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_prob = predict_proba(estimator, x_val)
    metrics = compute_classification_metrics(y_val, y_prob, threshold=threshold)
    metrics["ece"] = expected_calibration_error(y_val, y_prob)
    metrics["ranking_pr_auc"] = metrics["pr_auc"]
    return metrics


def _logreg_c(estimator: Any) -> float:
    if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
        return float(getattr(estimator.named_steps["model"], "C", 1.0))
    return 1.0


def fit_calibration_candidates(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    config: AppConfig | None = None,
    *,
    frozen_threshold: float = FROZEN_OPERATING_THRESHOLD,
    base_logreg: Pipeline | None = None,
) -> list[CalibrationCandidate]:
    """Fit uncalibrated and calibrated logistic models on **training** data only.

    ``CalibratedClassifierCV`` uses stratified CV inside the training fold, so
    validation labels never enter calibration. Validation is scored afterwards.
    When ``base_logreg`` is provided (the comparison-grid winner), that fitted
    pipeline is the uncalibrated reference and the clone source for calibration.
    """
    cfg = config or get_config()
    seed = int(cfg.random_seed)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    candidates: list[CalibrationCandidate] = []

    if base_logreg is not None:
        uncalibrated = base_logreg
        calibrator_template: Pipeline = clone(base_logreg)
    else:
        uncalibrated = build_model_pipeline("logreg", y_train, cfg)
        uncalibrated.fit(x_train, y_train)
        calibrator_template = clone(uncalibrated)

    candidates.append(
        _package(
            "logreg_uncalibrated_weighted",
            uncalibrated,
            "none",
            "balanced",
            x_val,
            y_val,
            frozen_threshold,
            cfg,
            {"C": _logreg_c(uncalibrated), "class_weight": "balanced"},
        )
    )

    sigmoid = CalibratedClassifierCV(
        estimator=clone(calibrator_template), method="sigmoid", cv=inner_cv
    )
    sigmoid.fit(x_train, y_train)
    candidates.append(
        _package(
            "logreg_sigmoid",
            sigmoid,
            "sigmoid",
            "balanced",
            x_val,
            y_val,
            frozen_threshold,
            cfg,
            {
                "method": "sigmoid",
                "cv_folds": 3,
                "base": "logreg_weighted",
                "C": _logreg_c(uncalibrated),
            },
        )
    )

    n_pos = int(np.sum(y_train == 1))
    if len(y_train) >= ISOTONIC_MIN_TRAIN_ROWS and n_pos >= ISOTONIC_MIN_POSITIVES:
        isotonic = CalibratedClassifierCV(
            estimator=clone(calibrator_template), method="isotonic", cv=inner_cv
        )
        isotonic.fit(x_train, y_train)
        candidates.append(
            _package(
                "logreg_isotonic",
                isotonic,
                "isotonic",
                "balanced",
                x_val,
                y_val,
                frozen_threshold,
                cfg,
                {
                    "method": "isotonic",
                    "cv_folds": 3,
                    "base": "logreg_weighted",
                    "C": _logreg_c(uncalibrated),
                },
            )
        )

    unweighted = _unweighted_logreg_pipeline(cfg, y_train)
    unweighted.fit(x_train, y_train)
    candidates.append(
        _package(
            "logreg_unweighted",
            unweighted,
            "none",
            "none",
            x_val,
            y_val,
            frozen_threshold,
            cfg,
            {"C": float(unweighted.named_steps["model"].C), "class_weight": "none"},
        )
    )
    return candidates


def _package(
    name: str,
    estimator: Any,
    calibration_status: str,
    class_weight: str,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    frozen_threshold: float,
    cfg: AppConfig,
    hyperparameters: dict[str, Any],
) -> CalibrationCandidate:
    frozen_metrics = _score(estimator, x_val, y_val, frozen_threshold)
    y_prob = predict_proba(estimator, x_val)
    selected = select_threshold(y_val, y_prob, cfg, split_name="validation")
    selected_threshold = float(selected["threshold"])
    selected_metrics = _score(estimator, x_val, y_val, selected_threshold)
    policy = f"validation_fbeta_beta{selected['beta']}_min_precision_{selected['min_precision']}"
    return CalibrationCandidate(
        name=name,
        pipeline=estimator,
        calibration_status=calibration_status,
        class_weight=class_weight,
        val_metrics_at_frozen_threshold=frozen_metrics,
        val_metrics_at_selected_threshold=selected_metrics,
        selected_threshold=selected_threshold,
        threshold_policy=policy,
        hyperparameters=hyperparameters,
    )
