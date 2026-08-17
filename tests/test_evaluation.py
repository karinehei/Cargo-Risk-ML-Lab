"""Tests for evaluation metrics helpers."""

from __future__ import annotations

import numpy as np
from src.evaluation import compute_classification_metrics


def test_compute_classification_metrics_range() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_samples"] == 8
    assert metrics["f1_macro"] >= 0.0
    assert metrics["false_negatives"] + metrics["true_positives"] == 4
