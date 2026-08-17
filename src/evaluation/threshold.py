"""Validation-only decision threshold selection.

The operational objective is fictional: false negatives (missed reviews) are
treated as more costly than false positives (extra human reviews). Thresholds
are chosen by maximising F-beta with beta=2 on the **validation** set, subject
to an optional minimum precision. The test set must never be passed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

from src.config import AppConfig, get_config


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: AppConfig | None = None,
    *,
    split_name: str = "validation",
) -> dict[str, Any]:
    """Sweep thresholds on a labelled split and return the selected operating point.

    Raises:
        ValueError: If ``split_name`` looks like a test fold.
    """
    if "test" in split_name.lower():
        raise ValueError("Threshold selection must not use the test set")

    cfg = config or get_config()
    settings = dict(cfg.training.get("threshold", {}))
    beta = float(settings.get("beta", 2.0))
    min_precision = float(settings.get("min_precision", 0.0))
    grid_size = int(settings.get("grid_size", 37))
    thresholds = np.linspace(0.05, 0.95, grid_size)

    records: list[dict[str, float]] = []
    feasible: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        fbeta = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        row = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }
        records.append(row)
        if precision + 1e-12 >= min_precision:
            feasible.append(row)

    pool = feasible or records
    best = max(pool, key=lambda item: (item["fbeta"], item["recall"], -item["threshold"]))
    return {
        "threshold": best["threshold"],
        "beta": beta,
        "min_precision": min_precision,
        "objective": str(settings.get("objective", "fbeta")),
        "split": split_name,
        "used_precision_constraint": bool(feasible),
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_fbeta": best["fbeta"],
        "curve": records,
        "disclaimer": (
            "Threshold chosen on validation only. Fictional review-queue objective; "
            "not an operational customs policy."
        ),
    }
