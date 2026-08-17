"""Pipeline integrity, reproducibility, metrics and train/test separation."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from src.config import get_config
from src.data import generate_synthetic_shipments, split_dataset
from src.evaluation import compute_classification_metrics, select_threshold
from src.features import build_preprocess_pipeline, prepare_xy
from src.models import compare_models, save_model_bundle, train_model


def test_preprocess_handles_missing_and_unseen_categories() -> None:
    df = generate_synthetic_shipments(n_samples=180, seed=42, validate=False)
    x, _ = prepare_xy(df)
    pipe = build_preprocess_pipeline(scale_numeric=True)
    pipe.fit(x)
    x_dirty = x.copy()
    x_dirty.loc[0, "declaration_completeness_score"] = np.nan
    x_dirty.loc[1, "transport_mode"] = "teleport"
    transformed = np.asarray(pipe.transform(x_dirty), dtype=float)
    assert transformed.shape[0] == len(x_dirty)
    assert np.isfinite(transformed).all()


def test_stratified_splits_are_disjoint_and_cover_all_ids() -> None:
    df = generate_synthetic_shipments(n_samples=400, seed=42, validate=False)
    bundle = split_dataset(df, seed=42, strategy="stratified")
    train_ids = set(bundle.train["shipment_id"])
    val_ids = set(bundle.val["shipment_id"])
    test_ids = set(bundle.test["shipment_id"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == set(df["shipment_id"])
    assert abs(bundle.test["requires_review"].mean() - df["requires_review"].mean()) < 0.08


def test_metric_calculation_known_example() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["precision_positive"] == 1.0
    assert metrics["recall_positive"] == 1.0
    assert metrics["f1_positive"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics


def test_threshold_selection_rejects_test_split_name() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.05, 0.9])
    selected = select_threshold(y_true, y_prob, split_name="validation")
    assert 0.05 <= selected["threshold"] <= 0.95
    with pytest.raises(ValueError, match="test"):
        select_threshold(y_true, y_prob, split_name="test")


def test_compare_models_signature_excludes_test_set() -> None:
    params = inspect.signature(compare_models).parameters
    assert "test_df" not in params
    assert "y_test" not in params
    assert "train_df" in params
    assert "val_df" in params


def test_compare_models_ranks_on_ranking_pr_auc() -> None:
    df = generate_synthetic_shipments(n_samples=280, seed=7, validate=False)
    bundle = split_dataset(df, seed=7, strategy="stratified")
    cfg = get_config()
    training = dict(cfg.training)
    training["search_enabled"] = False
    training["candidates"] = ["logreg"]
    training["cv_folds"] = 2
    cfg = replace(cfg, training=training)
    first = compare_models(bundle.train, bundle.val, config=cfg)
    second = compare_models(bundle.train, bundle.val, config=cfg)
    metrics = first.candidates[0].val_metrics
    assert first.selected_name == second.selected_name == "logreg"
    assert metrics["ranking_pr_auc"] == metrics["pr_auc"]
    assert first.candidates[0].val_metrics["pr_auc"] == second.candidates[0].val_metrics["pr_auc"]
    assert first.bundle.metadata["selection_metric"] == "ranking_pr_auc"


def test_saved_pipeline_contains_preprocess_and_model(tmp_path: Path) -> None:
    df = generate_synthetic_shipments(n_samples=220, seed=3, validate=False)
    bundle = split_dataset(df, seed=3, strategy="stratified")
    trained = train_model(bundle.train, val_df=bundle.val, estimator_name="logreg")
    assert trained.metadata["algorithm"] == "logreg"
    assert "test" not in trained.metadata.get("selection_split", "")
    assert list(trained.pipeline.named_steps.keys()) == ["preprocess", "model"]
    save_model_bundle(trained, artifact_dir=tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "preprocess_pipeline.joblib").exists()
