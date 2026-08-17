"""Operational workload summaries at a chosen probability threshold.

Rates are scaled per 1,000 shipments so a fictional review queue can be
discussed without implying a real customs policy. Threshold choice still
depends on the (unknown, policy-specific) cost of missed cases versus
unnecessary reviews.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


def operational_rates_per_1000(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    per: int = 1000,
) -> dict[str, float]:
    """Reviews, true positives, false positives and missed positives per 1,000."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= float(threshold)).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel())
    n_samples = max(int(len(y_true_arr)), 1)
    scale = float(per) / float(n_samples)
    reviews = tp + fp
    return {
        "threshold": float(threshold),
        "n_samples": float(n_samples),
        "per": float(per),
        "reviews_per_1000": float(reviews) * scale,
        "true_positives_per_1000": float(tp) * scale,
        "false_positives_per_1000": float(fp) * scale,
        "missed_positives_per_1000": float(fn) * scale,
        "true_negatives_per_1000": float(tn) * scale,
        "precision_positive": float(tp / reviews) if reviews else 0.0,
        "recall_positive": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "positive_rate": float(np.mean(y_true_arr)),
    }


def compare_operating_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
    *,
    selected_threshold: float,
    split_name: str,
) -> list[dict[str, Any]]:
    """Describe several thresholds on the same predictions (not for reselection)."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row: dict[str, Any] = operational_rates_per_1000(y_true, y_prob, threshold=threshold)
        row["split"] = split_name
        row["is_selected_threshold"] = abs(float(threshold) - float(selected_threshold)) < 1e-12
        rows.append(row)
    return rows
