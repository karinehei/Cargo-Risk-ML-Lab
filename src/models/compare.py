"""Train/validation model comparison. The test set is not an input."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import AppConfig, get_config, set_seed, setup_logging
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.threshold import select_threshold
from src.features import get_feature_names, prepare_xy
from src.models.estimators import build_model_pipeline
from src.models.train import TrainedModelBundle, predict_proba

logger = setup_logging(name="src.models.compare")


@dataclass
class ModelCandidateResult:
    """Validation-set outcome for one estimator family."""

    name: str
    pipeline: Pipeline
    cv_mean: float
    cv_std: float
    best_params: dict[str, Any]
    val_metrics: dict[str, float]
    val_probability: np.ndarray


@dataclass
class ComparisonResult:
    """Selected model plus all candidates. Contains no test-set metrics."""

    selected_name: str
    bundle: TrainedModelBundle
    threshold_info: dict[str, Any]
    candidates: list[ModelCandidateResult] = field(default_factory=list)
    comparison_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _param_grid(config: AppConfig, name: str) -> dict[str, list[Any]]:
    grids = dict(config.training.get("search_grids", {}))
    raw = dict(grids.get(name, {}) or {})
    return {str(key): list(value) for key, value in raw.items() if value}


def _fit_candidate(
    name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    config: AppConfig,
) -> ModelCandidateResult:
    seed = int(config.random_seed)
    cv_folds = int(config.training.get("cv_folds", 3))
    scoring = str(config.training.get("cv_scoring", "average_precision"))
    search_enabled = bool(config.training.get("search_enabled", True))
    n_jobs = int(config.training.get("n_jobs", 1))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pipeline = build_model_pipeline(name, y_train, config)
    grid = _param_grid(config, name)

    best_params: dict[str, Any] = {}
    if search_enabled and grid:
        logger.info("Grid search for %s with %s", name, grid)
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted = search.best_estimator_
        cv_mean = float(search.best_score_)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        best_params = {str(k): _json_safe(v) for k, v in search.best_params_.items()}
    else:
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        pipeline.fit(x_train, y_train)
        fitted = pipeline
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))

    val_prob = predict_proba(fitted, x_val)
    ranking_pr_auc = float(average_precision_score(y_val, val_prob))
    # Threshold 0.5 is only for display of precision/recall/F1. PR-AUC ignores it.
    val_metrics = compute_classification_metrics(y_val, val_prob, threshold=0.5)
    if abs(ranking_pr_auc - float(val_metrics["pr_auc"])) > 1e-12:
        raise RuntimeError("Validation PR-AUC must be independent of the display threshold")
    val_metrics["ranking_pr_auc"] = ranking_pr_auc
    val_metrics["cv_mean"] = cv_mean
    val_metrics["cv_std"] = cv_std
    return ModelCandidateResult(
        name=name,
        pipeline=fitted,
        cv_mean=cv_mean,
        cv_std=cv_std,
        best_params=best_params,
        val_metrics=val_metrics,
        val_probability=val_prob,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppConfig | None = None,
) -> ComparisonResult:
    """Train candidates on train, select on validation. Test data is not accepted.

    Model ranking uses validation PR-AUC (threshold-free). The decision threshold
    is then chosen on the same validation predictions.
    """
    cfg = config or get_config()
    seed = set_seed(cfg.random_seed)
    x_train, y_train = prepare_xy(train_df, cfg, fit_derived_reference=train_df)
    x_val, y_val = prepare_xy(val_df, cfg, fit_derived_reference=train_df)

    names = list(cfg.training.get("candidates", ["dummy", "logreg", "random_forest", "xgboost"]))
    candidates = [_fit_candidate(name, x_train, y_train, x_val, y_val, cfg) for name in names]
    # Always rank by threshold-free validation PR-AUC (probabilities), never by F1 at 0.5.
    selection_metric = "ranking_pr_auc"
    winner = max(candidates, key=lambda item: float(item.val_metrics["ranking_pr_auc"]))
    logger.info(
        "Selected %s by validation %s=%.4f",
        winner.name,
        selection_metric,
        winner.val_metrics.get(selection_metric, float("nan")),
    )

    threshold_info = select_threshold(
        y_val,
        winner.val_probability,
        cfg,
        split_name="validation",
    )
    feature_names = get_feature_names(winner.pipeline.named_steps["preprocess"])
    table = pd.DataFrame(
        [
            {
                "model": item.name,
                "val_pr_auc": item.val_metrics.get("ranking_pr_auc"),
                "val_roc_auc": item.val_metrics.get("roc_auc"),
                "val_f1_positive": item.val_metrics.get("f1_positive"),
                "val_precision_positive": item.val_metrics.get("precision_positive"),
                "val_recall_positive": item.val_metrics.get("recall_positive"),
                "cv_mean": item.cv_mean,
                "cv_std": item.cv_std,
                "selected": item.name == winner.name,
                "best_params": item.best_params,
            }
            for item in candidates
        ]
    )
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(),
        "trained_at_seed": seed,
        "selected_model": winner.name,
        "selection_metric": selection_metric,
        "selection_split": "validation",
        "threshold": threshold_info["threshold"],
        "threshold_info": {k: v for k, v in threshold_info.items() if k != "curve"},
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_val": float(np.mean(y_val)),
        "cv_mean": winner.cv_mean,
        "cv_std": winner.cv_std,
        "best_params": winner.best_params,
        "feature_names": feature_names,
        "target_column": str(cfg.data.get("target_column", "requires_review")),
        "algorithm": winner.name,
        "disclaimer": cfg.disclaimer,
        "notes": (
            "Test set was not used for hyperparameter search, model selection, or threshold choice."
        ),
    }
    bundle = TrainedModelBundle(
        pipeline=winner.pipeline,
        feature_names=feature_names,
        metadata=metadata,
    )
    return ComparisonResult(
        selected_name=winner.name,
        bundle=bundle,
        threshold_info=threshold_info,
        candidates=candidates,
        comparison_table=table,
    )
