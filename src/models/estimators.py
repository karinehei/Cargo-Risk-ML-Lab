"""Leakage-safe estimators used in the model comparison.

Class imbalance is handled with ``class_weight`` / ``scale_pos_weight``
computed from the **training** labels only. No resampling is applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import AppConfig, get_config
from src.features import build_preprocess_pipeline

NEEDS_SCALING: dict[str, bool] = {
    "dummy": False,
    "logreg": True,
    "random_forest": False,
    "xgboost": False,
}


def _positive_scale_weight(y_train: np.ndarray) -> float:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    if positives <= 0:
        return 1.0
    return negatives / positives


class XGBTrainWeightedClassifier(XGBClassifier):
    """XGBoost classifier that sets ``scale_pos_weight`` from the ``y`` passed to ``fit``.

    GridSearchCV clones this estimator per training fold, so the class ratio is
    computed from that fold's labels rather than from validation or test.
    """

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> XGBTrainWeightedClassifier:  # noqa: N803
        self.set_params(scale_pos_weight=_positive_scale_weight(np.asarray(y)))
        super().fit(X, y, **kwargs)
        return self


def build_estimator(
    name: str,
    y_train: np.ndarray,
    config: AppConfig | None = None,
) -> Any:
    """Construct an unfitted classifier with imbalance handling from train labels."""
    cfg = config or get_config()
    seed = int(cfg.random_seed)
    key = name.lower()
    if key == "dummy":
        return DummyClassifier(strategy="prior", random_state=seed)
    if key == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=seed,
        )
    if key == "xgboost":
        return XGBTrainWeightedClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
            scale_pos_weight=_positive_scale_weight(y_train),
        )
    raise ValueError(f"Unknown estimator: {name}")


def build_model_pipeline(
    name: str,
    y_train: np.ndarray,
    config: AppConfig | None = None,
) -> Pipeline:
    """Preprocessing + estimator pipeline. Imputation lives inside the pipeline."""
    scale = NEEDS_SCALING.get(name.lower(), True)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocess_pipeline(config, scale_numeric=scale)),
            ("model", build_estimator(name, y_train, config)),
        ]
    )
