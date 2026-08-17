"""Model evaluation metrics, plots and error analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import AppConfig, get_config, resolve_path, setup_logging
from src.evaluation.errors import build_error_analysis_report

logger = setup_logging(name="src.evaluation")


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Weighted absolute gap between mean predicted probability and empirical rate."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = max(len(y_true_arr), 1)
    for index in range(n_bins):
        left = bins[index]
        right = bins[index + 1]
        if index == n_bins - 1:
            mask = (y_prob_arr >= left) & (y_prob_arr <= right)
        else:
            mask = (y_prob_arr >= left) & (y_prob_arr < right)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n_samples) * abs(
            float(y_true_arr[mask].mean()) - float(y_prob_arr[mask].mean())
        )
    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary metrics, emphasising PR-AUC and positive-class trade-offs."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    metrics: dict[str, float] = {
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "threshold": float(threshold),
        "n_samples": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    # Backward-compatible aliases used by older scripts/tests.
    metrics["precision"] = metrics["precision_positive"]
    metrics["recall"] = metrics["recall_positive"]
    metrics["f1"] = metrics["f1_positive"]
    metrics["average_precision"] = metrics["pr_auc"]
    return metrics


def confusion_and_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return confusion matrix and sklearn classification report as dicts."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"confusion_matrix": cm, "classification_report": report}


def analyse_errors(
    features: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build an error analysis table for false positives and false negatives."""
    report = build_error_analysis_report(features, y_true, y_prob, threshold)
    labelled: pd.DataFrame = report["labelled"]
    errors = labelled[labelled["error_type"] != "correct"].copy()
    return errors.sort_values("confidence", ascending=False).head(top_n)


def save_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, Path]:
    """Generate and save ROC, PR, calibration and confusion-matrix plots."""
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    stem = f"{prefix}_" if prefix else ""

    sns.set_theme(style="whitegrid")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (synthetic evaluation)")
    ax.legend(loc="lower right")
    roc_path = out / f"{stem}roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    paths["roc_curve"] = roc_path

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve (synthetic evaluation)")
    ax.legend(loc="lower left")
    pr_path = out / f"{stem}pr_curve.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    paths["pr_curve"] = pr_path

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (synthetic evaluation)")
    ax.legend()
    cal_path = out / f"{stem}calibration_curve.png"
    fig.tight_layout()
    fig.savefig(cal_path, dpi=120)
    plt.close(fig)
    paths["calibration_curve"] = cal_path

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (synthetic evaluation)")
    cm_path = out / f"{stem}confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    paths["confusion_matrix"] = cm_path

    logger.info("Saved evaluation plots to %s", out)
    return paths


def save_comparison_plot(table: pd.DataFrame, output_dir: str | Path) -> Path:
    """Bar chart of validation PR-AUC for each candidate model."""
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["model"], table["val_pr_auc"], color="#2c3e50")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Model comparison (validation only, synthetic data)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = out / "model_comparison_pr_auc.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_threshold_plot(
    curve: list[dict[str, float]], selected: float, output_dir: str | Path
) -> Path:
    """Plot validation precision/recall/F-beta against threshold."""
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["threshold"], frame["precision"], label="precision")
    ax.plot(frame["threshold"], frame["recall"], label="recall")
    ax.plot(frame["threshold"], frame["fbeta"], label="fbeta")
    ax.axvline(selected, color="black", linestyle="--", label="selected")
    ax.set_xlabel("Threshold (chosen on validation)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep (validation only)")
    ax.legend()
    fig.tight_layout()
    path = out / "threshold_sweep_validation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    features: pd.DataFrame | None = None,
    config: AppConfig | None = None,
    split_name: str = "test",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run full evaluation and persist metrics/plots/error analysis."""
    cfg = config or get_config()
    resolved_threshold = (
        float(threshold) if threshold is not None else float(cfg.model.get("threshold", 0.5))
    )
    metrics = compute_classification_metrics(y_true, y_prob, threshold=resolved_threshold)
    detail = confusion_and_report(y_true, y_prob, threshold=resolved_threshold)

    artifact_dir = resolve_path(str(cfg.training.get("artifact_dir", "artifacts")))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = resolve_path(str(cfg.evaluation.get("plots_dir", artifact_dir / "plots")))
    plot_paths = save_evaluation_plots(
        y_true,
        y_prob,
        plots_dir,
        threshold=resolved_threshold,
        prefix=split_name,
    )

    result: dict[str, Any] = {
        "split": split_name,
        "metrics": metrics,
        "detail": detail,
        "plots": {key: str(path) for key, path in plot_paths.items()},
        "disclaimer": cfg.disclaimer,
    }

    if features is not None:
        error_report = build_error_analysis_report(features, y_true, y_prob, resolved_threshold)
        grouped: pd.DataFrame = error_report["grouped_table"]
        labelled: pd.DataFrame = error_report["labelled"]
        error_csv = artifact_dir / f"error_analysis_{split_name}.csv"
        group_csv = artifact_dir / f"error_analysis_groups_{split_name}.csv"
        error_json = artifact_dir / f"error_analysis_{split_name}.json"
        labelled[labelled["error_type"] != "correct"].to_csv(error_csv, index=False)
        grouped.to_csv(group_csv, index=False)
        error_json.write_text(
            json.dumps(
                {"summary": error_report["summary"], "by_group": error_report["by_group"]},
                indent=2,
            ),
            encoding="utf-8",
        )
        result["error_analysis_path"] = str(error_csv)
        result["error_analysis_groups_path"] = str(group_csv)
        result["error_analysis_json"] = str(error_json)
        result["error_summary"] = error_report["summary"]

    metrics_path = artifact_dir / f"metrics_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)
    result["metrics_path"] = str(metrics_path)
    return result
