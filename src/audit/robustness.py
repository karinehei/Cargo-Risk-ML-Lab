"""Modest Random Forest / XGBoost robustness search on training data only.

Validation is used solely to score fitted candidates. Early stopping uses an
inner split of the training fold, never validation or test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.audit.diagnostics import expected_calibration_error
from src.config import AppConfig, get_config, set_seed, setup_logging
from src.evaluation.metrics import compute_classification_metrics
from src.features import build_preprocess_pipeline
from src.models.estimators import _positive_scale_weight, build_model_pipeline
from src.models.train import predict_proba

logger = setup_logging(name="src.audit.robustness")

ORIGINAL_BEST_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 4.0},
    "random_forest": {"max_depth": 6, "n_estimators": 80},
    "xgboost": {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 80},
}

RF_ROBUSTNESS_GRID: tuple[dict[str, Any], ...] = (
    {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "class_weight": "balanced",
    },
    {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_leaf": 8,
        "max_features": "log2",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    },
)

XGB_ROBUSTNESS_GRID: tuple[dict[str, Any], ...] = (
    {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.10,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 7,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 250,
        "learning_rate": 0.08,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
)


def _score_fitted(
    name: str,
    pipeline: Pipeline,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    y_prob = predict_proba(pipeline, x_val)
    metrics = compute_classification_metrics(y_val, y_prob, threshold=0.5)
    return {
        "name": name,
        "val_ranking_pr_auc": float(average_precision_score(y_val, y_prob)),
        "val_pr_auc": float(metrics["pr_auc"]),
        "val_roc_auc": float(metrics["roc_auc"]),
        "val_brier": float(brier_score_loss(y_val, y_prob)),
        "val_ece": expected_calibration_error(y_val, y_prob, n_bins=8),
        "val_f1_at_0_5": float(metrics["f1_positive"]),
        "params": {str(key): _json_safe(value) for key, value in params.items()},
        "notes": notes,
        "pipeline": pipeline,
        "val_probability": y_prob,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def _fit_named(
    estimator_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    config: AppConfig,
) -> Pipeline:
    pipeline = build_model_pipeline(estimator_name, y_train, config)
    pipeline.set_params(**{f"model__{key}": value for key, value in params.items()})
    pipeline.fit(x_train, y_train)
    return pipeline


def fit_xgboost_early_stopping(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    config: AppConfig,
    *,
    inner_frac: float = 0.15,
    n_estimators: int = 400,
    early_stopping_rounds: int = 30,
) -> tuple[Pipeline, dict[str, Any]]:
    """Early-stop on an inner split of **training** rows only."""
    seed = int(config.random_seed)
    x_fit, x_stop, y_fit, y_stop = train_test_split(
        x_train,
        y_train,
        test_size=inner_frac,
        stratify=y_train,
        random_state=seed,
    )
    preprocess = build_preprocess_pipeline(config, scale_numeric=False)
    x_fit_t = preprocess.fit_transform(x_fit)
    x_stop_t = preprocess.transform(x_stop)
    scale_pos_weight = _positive_scale_weight(np.asarray(y_fit))
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0,
    )
    model.fit(x_fit_t, y_fit, eval_set=[(x_stop_t, y_stop)])
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    best_iteration = getattr(model, "best_iteration", None)
    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "inner_train_frac": 1.0 - inner_frac,
        "scale_pos_weight_from": "inner_train_fit_labels",
    }
    return pipeline, params


def run_robustness_experiment(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Fit original best configs plus a modest expanded RF/XGB grid. No test data."""
    cfg = config or get_config()
    set_seed(cfg.random_seed)
    results: list[dict[str, Any]] = []

    for name, params in ORIGINAL_BEST_PARAMS.items():
        logger.info("Refitting original %s %s on training data", name, params)
        pipeline = _fit_named(name, params, x_train, y_train, cfg)
        results.append(
            _score_fitted(
                f"{name}_original_best",
                pipeline,
                x_val,
                y_val,
                params,
                "Refit of frozen-search best params on full training fold.",
            )
        )

    for index, params in enumerate(RF_ROBUSTNESS_GRID, start=1):
        logger.info("Robustness RF config %s %s", index, params)
        pipeline = _fit_named("random_forest", params, x_train, y_train, cfg)
        results.append(
            _score_fitted(
                f"random_forest_robust_{index}",
                pipeline,
                x_val,
                y_val,
                params,
                "Expanded RF grid; fitted on training rows only.",
            )
        )

    for index, params in enumerate(XGB_ROBUSTNESS_GRID, start=1):
        logger.info("Robustness XGB config %s %s", index, params)
        pipeline = _fit_named("xgboost", params, x_train, y_train, cfg)
        results.append(
            _score_fitted(
                f"xgboost_robust_{index}",
                pipeline,
                x_val,
                y_val,
                params,
                "Expanded XGBoost grid; fitted on training rows only.",
            )
        )

    logger.info("Fitting XGBoost with early stopping on an inner training split")
    es_pipeline, es_params = fit_xgboost_early_stopping(x_train, y_train, cfg)
    results.append(
        _score_fitted(
            "xgboost_early_stopping_inner_train",
            es_pipeline,
            x_val,
            y_val,
            es_params,
            "Early stopping monitor is an inner split of train, not validation or test.",
        )
    )
    return results
