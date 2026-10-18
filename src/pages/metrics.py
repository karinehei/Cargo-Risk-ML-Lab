"""Aggregate confusion counts and rates for the public threshold grid.

Precision is null when no row is flagged. Recall is null when the labelled
split has no positives. A zero from ``zero_division=0`` is not used for those
undefined cases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

DEFAULT_BETA = 2.0


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def confusion_counts(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, int]:
    """Count TP, FP, TN and FN at one threshold. Labels must be 0 or 1."""
    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have the same length")
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = 1 if float(score) >= float(threshold) else 0
        actual = int(label)
        if actual not in (0, 1):
            raise ValueError("Labels must be 0 or 1")
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        else:
            false_negatives += 1
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }


def metrics_from_counts(
    *,
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
    beta: float = DEFAULT_BETA,
) -> dict[str, Any]:
    """Derive rates from counts. Undefined rates are None, not zero."""
    flagged = true_positives + false_positives
    positives = true_positives + false_negatives
    negatives = true_negatives + false_positives
    total = flagged + true_negatives + false_negatives
    precision = (true_positives / flagged) if flagged else None
    recall = (true_positives / positives) if positives else None
    precision_status = "defined" if precision is not None else "undefined_no_flagged_rows"
    recall_status = "defined" if recall is not None else "undefined_no_positive_labels"
    if precision is None or recall is None or (precision + recall) == 0.0:
        fbeta = None
        fbeta_status = "undefined"
    else:
        beta_sq = float(beta) * float(beta)
        fbeta = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        fbeta_status = "defined"
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "flagged_for_review": flagged,
        "positive_labels": positives,
        "negative_labels": negatives,
        "n_rows": total,
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "precision_status": precision_status,
        "recall_status": recall_status,
        "fbeta_status": fbeta_status,
        "beta": float(beta),
    }


def counts_close(stored: dict[str, Any], recomputed: dict[str, Any], *, tol: float = 1e-9) -> bool:
    """Return whether stored aggregate rates match ``metrics_from_counts``."""
    for key in (
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "flagged_for_review",
    ):
        if int(stored[key]) != int(recomputed[key]):
            return False
    for key in ("precision", "recall", "fbeta"):
        left = stored[key]
        right = recomputed[key]
        if left is None or right is None:
            if left is not None or right is not None:
                return False
            continue
        if abs(float(left) - float(right)) > tol:
            return False
    return True
