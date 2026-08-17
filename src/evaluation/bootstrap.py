"""Bootstrap confidence intervals for held-out classification metrics.

Intervals are descriptive. Overlapping intervals do not support strong claims
that one model or threshold is better than another.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import compute_classification_metrics

BOOTSTRAP_METRICS: tuple[str, ...] = ("pr_auc", "roc_auc", "precision", "recall", "f1")


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int = 2000,
    seed: int = 42,
    metric_names: tuple[str, ...] = BOOTSTRAP_METRICS,
) -> dict[str, dict[str, float]]:
    """Percentile 95% CIs for test metrics using a deterministic seed.

    ROC-AUC and PR-AUC are computed from probabilities. Precision, recall and
    F1 use ``threshold``. Degenerate resamples with a single class are skipped
    and redrawn so ranking metrics stay defined.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n_samples = int(len(y_true_arr))
    if n_samples == 0:
        raise ValueError("Cannot bootstrap an empty prediction vector")
    if n_samples != len(y_prob_arr):
        raise ValueError("y_true and y_prob must have the same length")

    point = compute_classification_metrics(y_true_arr, y_prob_arr, threshold=threshold)
    rng = np.random.default_rng(int(seed))
    collected: dict[str, list[float]] = {name: [] for name in metric_names}

    attempts = 0
    max_attempts = n_bootstrap * 20
    while min(len(values) for values in collected.values()) < n_bootstrap:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Too many degenerate bootstrap draws (single-class resamples)")
        indices = rng.integers(0, n_samples, size=n_samples)
        y_resample = y_true_arr[indices]
        if y_resample.min() == y_resample.max():
            continue
        metrics = compute_classification_metrics(
            y_resample,
            y_prob_arr[indices],
            threshold=threshold,
        )
        for name in metric_names:
            collected[name].append(float(metrics[name]))

    intervals: dict[str, dict[str, float]] = {}
    for name in metric_names:
        samples = np.asarray(collected[name], dtype=float)
        intervals[name] = {
            "point": float(point[name]),
            "low": float(np.percentile(samples, 2.5)),
            "high": float(np.percentile(samples, 97.5)),
            "n_bootstrap": float(n_bootstrap),
            "seed": float(seed),
        }
    return intervals


def intervals_to_records(intervals: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Flatten bootstrap intervals for CSV / Markdown tables."""
    records: list[dict[str, Any]] = []
    for metric_name, stats in intervals.items():
        records.append(
            {
                "metric": metric_name,
                "point": stats["point"],
                "ci_low": stats["low"],
                "ci_high": stats["high"],
                "n_bootstrap": int(stats["n_bootstrap"]),
                "seed": int(stats["seed"]),
            }
        )
    return records
