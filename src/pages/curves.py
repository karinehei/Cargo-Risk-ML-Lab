"""Validation ranking and calibration summaries for the public bundle.

Curve coordinates are aggregate functions of the validation scores. Individual
scores are not returned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import expected_calibration_error


def equal_width_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[list[dict[str, Any]], float]:
    """Bin the review score on [0, 1] and return bin aggregates plus ECE."""
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    total = max(len(labels), 1)
    ece = 0.0
    for index in range(n_bins):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_index": index,
                    "score_left": left,
                    "score_right": right,
                    "count": 0,
                    "mean_review_score": None,
                    "fraction_positive": None,
                }
            )
            continue
        mean_score = float(scores[mask].mean())
        fraction = float(labels[mask].mean())
        ece += (count / total) * abs(fraction - mean_score)
        rows.append(
            {
                "bin_index": index,
                "score_left": left,
                "score_right": right,
                "count": count,
                "mean_review_score": mean_score,
                "fraction_positive": fraction,
            }
        )
    reference = expected_calibration_error(labels, scores, n_bins=n_bins)
    if abs(ece - reference) > 1e-12:
        raise ValueError("Equal-width calibration bins do not match the project ECE")
    return rows, float(ece)


def validation_curve_payload(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Build PR, ROC and calibration payloads from validation labels and scores."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    precision, recall, _thresholds = precision_recall_curve(labels, scores)
    fpr, tpr, _roc_thresholds = roc_curve(labels, scores)
    bins, ece = equal_width_calibration(labels, scores, n_bins=10)
    positive_rate = float(np.mean(labels))
    return {
        "split": "validation",
        "n_rows": int(len(labels)),
        "positive_labels": int(np.sum(labels)),
        "positive_rate": positive_rate,
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": ece,
        "ece_bins": 10,
        "ece_binning": "equal_width_on_review_score",
        "precision_recall": {
            "recall": [float(value) for value in recall],
            "precision": [float(value) for value in precision],
        },
        "roc": {
            "false_positive_rate": [float(value) for value in fpr],
            "true_positive_rate": [float(value) for value in tpr],
        },
        "calibration_bins": bins,
        "score_axis": "review_score",
        "score_is_calibrated": False,
    }
