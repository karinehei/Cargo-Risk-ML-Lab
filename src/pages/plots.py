"""Scientific figures for the static site.

Axis titles call the model output a review score. They do not call it a
probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure

from src.explainability.plots import COLOR_DECREASE, COLOR_INCREASE, COLOR_NEUTRAL


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_validation_pr(payload: dict[str, Any], path: Path) -> Path:
    """Precision–recall curve for the champion on the validation split."""
    curve = payload["precision_recall"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        curve["recall"],
        curve["precision"],
        color=COLOR_INCREASE,
        linewidth=2.0,
        label=f"Champion PR-AUC = {payload['pr_auc']:.3f}",
    )
    ax.axhline(
        payload["positive_rate"],
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=1.2,
        label=f"Validation positive rate = {payload['positive_rate']:.3f}",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation precision–recall of the review score")
    ax.legend(loc="upper right", frameon=False)
    return _save(fig, path)


def plot_validation_roc(payload: dict[str, Any], path: Path) -> Path:
    """ROC curve for the champion on the validation split."""
    curve = payload["roc"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        curve["false_positive_rate"],
        curve["true_positive_rate"],
        color=COLOR_INCREASE,
        linewidth=2.0,
        label=f"Champion ROC-AUC = {payload['roc_auc']:.3f}",
    )
    ax.plot(
        [0.0, 1.0], [0.0, 1.0], color=COLOR_NEUTRAL, linestyle="--", linewidth=1.2, label="Chance"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC of the review score")
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, path)


def plot_validation_calibration(payload: dict[str, Any], path: Path) -> Path:
    """Equal-width calibration of the uncalibrated review score."""
    bins = [
        row
        for row in payload["calibration_bins"]
        if row["count"] and row["mean_review_score"] is not None
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=COLOR_NEUTRAL,
        linestyle="--",
        linewidth=1.2,
        label="Perfect calibration reference",
    )
    ax.plot(
        [row["mean_review_score"] for row in bins],
        [row["fraction_positive"] for row in bins],
        color=COLOR_DECREASE,
        marker="o",
        linewidth=2.0,
        label=f"Equal-width bins (ECE = {payload['ece']:.3f})",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean review score in bin")
    ax.set_ylabel("Fraction of validation rows labelled for review")
    ax.set_title("Validation calibration — review score, not a probability")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, path)


def plot_model_comparison(models: list[dict[str, Any]], path: Path) -> Path:
    """Grouped bars of training-fold CV PR-AUC and validation PR-AUC."""
    names = [str(model["display_name"]) for model in models]
    validation = [float(model["validation_pr_auc"]) for model in models]
    cross_val = [float(model["training_fold_cv_pr_auc_mean"]) for model in models]
    positions = list(range(len(models)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(
        [pos - width / 2 for pos in positions],
        cross_val,
        width=width,
        color=COLOR_NEUTRAL,
        hatch="//",
        label="Training-fold CV PR-AUC",
    )
    ax.bar(
        [pos + width / 2 for pos in positions],
        validation,
        width=width,
        color=COLOR_INCREASE,
        label="Validation PR-AUC (selection metric)",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Average precision (PR-AUC)")
    ax.set_ylim(0.0, max(max(validation), max(cross_val)) * 1.25)
    ax.set_title("Same experiment: cross-validation and validation PR-AUC")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_threshold_tradeoff(rows: list[dict[str, Any]], selected: float, path: Path) -> Path:
    """Precision, recall and flagged fraction across the validation threshold grid."""
    thresholds = [float(row["threshold"]) for row in rows]
    precision = [row["precision"] if row["precision"] is not None else float("nan") for row in rows]
    recall = [row["recall"] if row["recall"] is not None else float("nan") for row in rows]
    flagged = [float(row["flagged_for_review"]) / float(row["n_rows"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(
        thresholds, precision, color=COLOR_INCREASE, marker="o", markersize=3.5, label="Precision"
    )
    ax.plot(
        thresholds,
        recall,
        color=COLOR_DECREASE,
        marker="s",
        markersize=3.5,
        linestyle="--",
        label="Recall",
    )
    ax.plot(
        thresholds,
        flagged,
        color=COLOR_NEUTRAL,
        marker="^",
        markersize=3.5,
        linestyle=":",
        label="Fraction flagged for review",
    )
    ax.axvline(selected, color="#1f4b6e", linewidth=1.2, label=f"Selected threshold {selected:.3f}")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Review-score threshold (validation grid)")
    ax.set_ylabel("Rate")
    ax.set_title("Validation threshold trade-off")
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, path)
