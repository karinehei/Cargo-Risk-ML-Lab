"""Tests for the methodological audit helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.audit.diagnostics import toy_score_term_contributions
from src.audit.leakage import feature_leakage_inventory, id_split_audit
from src.audit.markdown import markdown_table
from src.audit.protocol import assert_pr_auc_independent_of_threshold
from src.data import generate_synthetic_shipments, split_dataset
from src.data.generate import _assert_disjoint_ids, _raw_review_scores
from src.data.validate import DatasetValidationError
from src.evaluation import (
    bootstrap_metric_intervals,
    compute_classification_metrics,
    operational_rates_per_1000,
)
from src.models.estimators import _positive_scale_weight, build_estimator


def test_pr_auc_independent_of_threshold() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.05, 0.2, 0.4, 0.35, 0.8, 0.7, 0.9, 0.1, 0.65, 0.3])
    low = compute_classification_metrics(y_true, y_prob, threshold=0.1)
    high = compute_classification_metrics(y_true, y_prob, threshold=0.9)
    assert low["pr_auc"] == high["pr_auc"]
    assert low["precision_positive"] != high["precision_positive"]
    checked = assert_pr_auc_independent_of_threshold(y_true, y_prob)
    assert checked["pr_auc_at_0_1"] == checked["pr_auc_at_0_9"]


def test_markdown_table_requires_a_header_per_column() -> None:
    table = markdown_table(
        ["Model", "Validation PR-AUC", "Selected"],
        [["logreg", "0.227", "yes"], ["xgboost", "0.213", "no"]],
    )
    lines = table.splitlines()
    assert lines[0] == "| Model | Validation PR-AUC | Selected |"
    assert lines[1] == "| --- | --- | --- |"
    with pytest.raises(ValueError, match="non-empty header"):
        markdown_table(["Model", ""], [["logreg", "0.2"]])
    with pytest.raises(ValueError, match="column headers"):
        markdown_table(["Model", "PR-AUC"], [["logreg"]])


def test_bootstrap_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0] * 20)
    y_prob = np.clip(y_true * 0.4 + rng.random(len(y_true)) * 0.5, 0, 1)
    intervals = bootstrap_metric_intervals(y_true, y_prob, threshold=0.5, n_bootstrap=200, seed=42)
    for metric_name in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        stats = intervals[metric_name]
        assert stats["low"] <= stats["point"] <= stats["high"]
        assert stats["n_bootstrap"] == 200


def test_operational_rates_scale_to_one_thousand() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.4, 0.3, 0.05])
    rates = operational_rates_per_1000(y_true, y_prob, threshold=0.5)
    assert rates["reviews_per_1000"] == pytest.approx(375.0)
    assert rates["true_positives_per_1000"] == pytest.approx(250.0)
    assert rates["false_positives_per_1000"] == pytest.approx(125.0)
    assert rates["missed_positives_per_1000"] == pytest.approx(125.0)


def test_scale_pos_weight_from_training_labels_only() -> None:
    y_train = np.array([0, 0, 0, 1])
    estimator = build_estimator("xgboost", y_train)
    assert estimator.get_params()["scale_pos_weight"] == pytest.approx(3.0)
    estimator.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
    assert estimator.get_params()["scale_pos_weight"] == pytest.approx(
        _positive_scale_weight(np.array([0, 0, 1, 1]))
    )


def test_duplicate_ids_within_fold_are_rejected() -> None:
    df = generate_synthetic_shipments(n_samples=80, seed=1, validate=False)
    bundle = split_dataset(df, seed=1, strategy="stratified")
    train = bundle.train.copy()
    train.loc[1, "shipment_id"] = train.loc[0, "shipment_id"]
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        _assert_disjoint_ids(train, bundle.val, bundle.test, "shipment_id")


def test_ids_disjoint_and_not_model_features() -> None:
    df = generate_synthetic_shipments(n_samples=400, seed=42, validate=False)
    bundle = split_dataset(df, seed=42, strategy="stratified")
    audit = id_split_audit(bundle.train, bundle.val, bundle.test)
    assert audit["unique_within_folds"]
    assert audit["disjoint_across_folds"]
    assert not audit["shipment_id_is_model_feature"]
    assert not audit["sender_or_entity_id_present"]
    inventory = feature_leakage_inventory(bundle.train)
    used = [row for row in inventory if row["used_in_model"]]
    assert used
    assert all(not row["direct_leakage"] and not row["indirect_leakage"] for row in used)
    excluded = {row["feature"] for row in inventory if not row["used_in_model"]}
    assert {"shipment_id", "event_date", "generation_period", "requires_review"} <= excluded


def test_toy_score_components_match_generator() -> None:
    df = generate_synthetic_shipments(n_samples=300, seed=3, validate=False)
    filled = df.copy()
    numeric = filled.select_dtypes(include=["number"])
    filled[numeric.columns] = numeric.fillna(numeric.median())
    terms = toy_score_term_contributions(filled)
    reconstructed = np.sum(list(terms.values()), axis=0)
    original = _raw_review_scores(filled)
    assert np.allclose(reconstructed, original)


def test_grid_search_fits_preprocess_on_training_folds_only() -> None:
    recorded: list[int] = []

    class RecordFitSize(BaseEstimator, TransformerMixin):
        def __init__(self, tag: str = "rec") -> None:
            self.tag = tag

        def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RecordFitSize:
            recorded.append(int(np.asarray(X).shape[0]))
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(X, dtype=float)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 3))
    y = np.array([0] * 78 + [1] * 12)
    pipe = Pipeline([("rec", RecordFitSize()), ("clf", LogisticRegression(max_iter=200))])
    search = GridSearchCV(
        pipe,
        param_grid={"clf__C": [1.0]},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        scoring="average_precision",
        refit=True,
    )
    search.fit(x, y)
    assert recorded[-1] == 90
    assert all(size < 90 for size in recorded[:-1])
