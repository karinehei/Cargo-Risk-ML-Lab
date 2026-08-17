"""Validation-only subgroup performance. Not a fairness proof and not used for tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_classification_metrics
from src.models.train import predict_proba

DEFAULT_GROUP_COLUMNS = [
    "transport_mode",
    "commodity_category",
    "origin_region",
    "destination_region",
    "expedited_shipment",
    "generation_period",
]


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def _rate_with_ci(successes: float, n: float) -> dict[str, float]:
    low, high = wilson_interval(successes, n)
    return {
        "value": float(successes / n) if n else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "n": float(n),
    }


def subgroup_rows(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    group_columns: list[str] | None = None,
    min_n: int = 50,
) -> pd.DataFrame:
    """Compute per-level metrics. Groups with n < min_n are flagged, not used for tuning."""
    columns = group_columns or DEFAULT_GROUP_COLUMNS
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for level, index in frame.groupby(column, dropna=False).groups.items():
            idx = np.asarray(list(index))
            n = int(len(idx))
            y = y_true[idx]
            scores = y_score[idx]
            pred = (scores >= threshold).astype(int)
            tp = float(np.sum((pred == 1) & (y == 1)))
            fp = float(np.sum((pred == 1) & (y == 0)))
            fn = float(np.sum((pred == 0) & (y == 1)))
            tn = float(np.sum((pred == 0) & (y == 0)))
            if n >= 2 and y.min() != y.max() and np.unique(pred).size >= 1:
                metrics = compute_classification_metrics(y, scores, threshold=threshold)
            else:
                metrics = {
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "false_positive_rate": float("nan"),
                    "false_negative_rate": float("nan"),
                }
            precision_ci = _rate_with_ci(tp, tp + fp)
            recall_ci = _rate_with_ci(tp, tp + fn)
            small = n < min_n
            rows.append(
                {
                    "group_column": column,
                    "group_value": _level_label(level),
                    "n": n,
                    "target_prevalence": float(np.mean(y)) if n else float("nan"),
                    "predicted_review_rate": float(np.mean(pred)) if n else float("nan"),
                    "precision": metrics["precision"],
                    "precision_ci_low": precision_ci["ci_low"],
                    "precision_ci_high": precision_ci["ci_high"],
                    "recall": metrics["recall"],
                    "recall_ci_low": recall_ci["ci_low"],
                    "recall_ci_high": recall_ci["ci_high"],
                    "f1": metrics["f1"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "false_negative_rate": metrics["false_negative_rate"],
                    "mean_review_score": float(np.mean(scores)) if n else float("nan"),
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn,
                    "small_sample": small,
                    "min_n": min_n,
                    "warning": (
                        f"n={n} is below min_n={min_n}; interval is wide and this cell is not evidence"
                        if small
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def subgroup_payload(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    min_n: int = 50,
) -> dict[str, Any]:
    table = subgroup_rows(frame, y_true, y_score, threshold=threshold, min_n=min_n)
    return {
        "split": "validation",
        "decision_threshold": float(threshold),
        "min_n": min_n,
        "n_validation": int(len(frame)),
        "rows": table.to_dict(orient="records"),
        "limitations": [
            "Observed differences are descriptive of this synthetic validation fold only.",
            "They are not evidence of unfairness in a real operational system.",
            "Passing these checks does not prove fairness.",
            "Geographic region and commodity are fictional trade-corridor labels, not protected personal characteristics.",
            "Possible proxies: origin/destination region, commodity category, route rarity and sender history can stand in for unmodelled attributes in a real dataset. They are toy generator features here.",
            "Subgroup metrics were not used to tune the champion, threshold or calibration.",
        ],
        "protected_characteristics_present": False,
        "used_for_tuning": False,
    }


def score_validation(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    return predict_proba(pipeline, features)


def _level_label(level: Any) -> str:
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return "missing"
    return str(level)
