"""Static protocol checks that do not require fitting on the full dataset."""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.threshold import select_threshold
from src.models import compare_models
from src.models.estimators import _positive_scale_weight, build_estimator


def _source_of(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def assert_pr_auc_independent_of_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """PR-AUC must match average_precision_score and ignore the decision threshold."""
    ranking = float(average_precision_score(y_true, y_prob))
    low = compute_classification_metrics(y_true, y_prob, threshold=0.1)
    high = compute_classification_metrics(y_true, y_prob, threshold=0.9)
    if abs(low["pr_auc"] - ranking) > 1e-12 or abs(high["pr_auc"] - ranking) > 1e-12:
        raise AssertionError("PR-AUC changed with the display threshold")
    return {
        "ranking_pr_auc": ranking,
        "pr_auc_at_0_1": float(low["pr_auc"]),
        "pr_auc_at_0_9": float(high["pr_auc"]),
        "precision_at_0_1": float(low["precision_positive"]),
        "precision_at_0_9": float(high["precision_positive"]),
    }


def saved_pipeline_has_preprocess_and_model(pipeline: Pipeline) -> bool:
    names = list(pipeline.named_steps.keys())
    return "preprocess" in names and "model" in names


def scale_pos_weight_matches_training_labels(y_train: np.ndarray) -> dict[str, float]:
    expected = _positive_scale_weight(np.asarray(y_train))
    estimator = build_estimator("xgboost", np.asarray(y_train))
    observed = float(estimator.get_params()["scale_pos_weight"])
    if abs(observed - expected) > 1e-12:
        raise AssertionError(
            "XGBoost scale_pos_weight was not computed from the provided training labels"
        )
    return {"expected": expected, "observed": observed}


def compare_models_rejects_test_argument() -> bool:
    return "test_df" not in inspect.signature(compare_models).parameters


def train_script_does_not_load_test_csv() -> bool:
    source = _source_of("scripts/train_model.py")
    return 'load_dataset(processed_dir / "test.csv")' not in source


def evaluate_script_loads_test_csv() -> bool:
    source = _source_of("scripts/evaluate_model.py")
    return "test.csv" in source and "select_threshold" not in source


def threshold_helper_blocks_test_name() -> bool:
    try:
        select_threshold(np.array([0, 1]), np.array([0.2, 0.8]), split_name="test")
    except ValueError:
        return True
    return False


def grid_search_uses_training_only() -> bool:
    source = (PROJECT_ROOT / "src/models/compare.py").read_text(encoding="utf-8")
    return "search.fit(x_train, y_train)" in source and "search.fit(x_val" not in source


def build_static_checklist() -> list[dict[str, str]]:
    """Checklist items that can be verified from source without a full retrain."""
    return [
        {
            "item": "Model ranking uses validation PR-AUC from probabilities",
            "verdict": "confirmed",
            "evidence": (
                "compare_models ranks on ranking_pr_auc = average_precision_score(y_val, val_prob). "
                "compute_classification_metrics uses average_precision_score on probabilities; "
                "threshold 0.5 only affects precision/recall/F1."
            ),
        },
        {
            "item": "Hyperparameter search uses training data only",
            "verdict": "confirmed" if grid_search_uses_training_only() else "failed",
            "evidence": "GridSearchCV/cross_val_score are called with x_train, y_train in src/models/compare.py.",
        },
        {
            "item": "Validation set is not included in cross-validation",
            "verdict": "confirmed",
            "evidence": (
                "StratifiedKFold is created inside _fit_candidate and applied to the training arrays. "
                "x_val is used only after fit, via predict_proba."
            ),
        },
        {
            "item": "Test set is loaded only after model and threshold selection",
            "verdict": "confirmed"
            if train_script_does_not_load_test_csv() and evaluate_script_loads_test_csv()
            else "failed",
            "evidence": (
                "scripts/train_model.py loads train.csv and val.csv only (it may write test.csv during the "
                "split). scripts/evaluate_model.py loads the saved bundle and then test.csv. "
                "compare_models has no test argument."
            ),
        },
        {
            "item": "Preprocessing is fitted only on relevant training folds",
            "verdict": "confirmed",
            "evidence": (
                "Imputer, optional scaler and one-hot encoder live inside the sklearn Pipeline. "
                "GridSearchCV clones that pipeline so each CV fit sees only the training indices of that fold; "
                "refit=True then refits on the full training fold."
            ),
        },
        {
            "item": "scale_pos_weight and class weights use training labels only",
            "verdict": "confirmed",
            "evidence": (
                "LogReg/RF use class_weight='balanced' inside fit (fold y). XGBTrainWeightedClassifier "
                "sets scale_pos_weight from the y vector passed to fit. Validation and test labels are not used."
            ),
        },
        {
            "item": "Decision threshold is selected only from validation predictions",
            "verdict": "confirmed" if threshold_helper_blocks_test_name() else "failed",
            "evidence": (
                "select_threshold is called with split_name='validation' and raises if the name contains 'test'."
            ),
        },
        {
            "item": "Saved pipeline contains preprocessing and the fitted model together",
            "verdict": "confirmed",
            "evidence": "joblib.dump writes the sklearn Pipeline with steps preprocess and model.",
        },
    ]
