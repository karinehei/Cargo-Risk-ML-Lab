"""Data and prediction drift monitoring (legacy evaluate_model integration).

Uses the project's PSI/KS metrics as the source of truth. Evidently is optional
and never replaces the lightweight implementation.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config, resolve_path, setup_logging
from src.monitoring.metrics import (
    categorical_feature_metrics,
    default_thresholds,
    numeric_feature_metrics,
    score_distribution_metrics,
)

logger = setup_logging(name="src.monitoring.drift")


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Backward-compatible PSI-like statistic for categorical variables."""
    thresholds = default_thresholds()
    row = categorical_feature_metrics(reference, current, thresholds=thresholds)
    return float(row.get("jensen_shannon_divergence", 0.0))


def prediction_drift_score(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, float]:
    """Compare prediction score distributions."""
    payload = score_distribution_metrics(reference_scores, current_scores, threshold=0.5)
    return {
        "ks_statistic": float(payload["ks_statistic"]),
        "ks_pvalue": float(payload["ks_pvalue"]),
        "mean_shift": float(payload["mean_shift"]),
        "psi": float(payload["psi"]),
    }


def _lightweight_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_bins: int,
    threshold: float,
) -> dict[str, Any]:
    thresholds = default_thresholds({"psi_warning": threshold, "psi_critical": threshold})
    feature_reports: list[dict[str, Any]] = []
    for col in numeric_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        row = numeric_feature_metrics(
            reference[col], current[col], thresholds=thresholds, n_bins=n_bins
        )
        feature_reports.append(
            {
                "feature": row["feature"],
                "type": "numeric",
                "psi": row["psi"],
                "drift_detected": row["severity"] != "none",
                "severity": row["severity"],
            }
        )
    for col in categorical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        row = categorical_feature_metrics(reference[col], current[col], thresholds=thresholds)
        feature_reports.append(
            {
                "feature": row["feature"],
                "type": "categorical",
                "psi": row.get("jensen_shannon_divergence"),
                "drift_detected": row["severity"] != "none",
                "severity": row["severity"],
            }
        )
    drifted = [item for item in feature_reports if item["drift_detected"]]
    return {
        "method": "lightweight_psi",
        "n_features_checked": len(feature_reports),
        "n_drifted_features": len(drifted),
        "features": feature_reports,
        "threshold": threshold,
    }


def _evidently_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any] | None:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evidently import failed; using lightweight drift. (%s)", type(exc).__name__)
        return None
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        return {"method": "evidently_data_drift_preset", "report": report.as_dict()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evidently report failed; using lightweight drift. (%s)", type(exc).__name__)
        return None


def run_drift_checks(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Run data (and optional prediction) drift checks and persist a JSON report."""
    cfg = config or get_config()
    threshold = float(cfg.monitoring.get("drift_threshold", 0.15))
    n_bins = int(cfg.monitoring.get("psi_bins", 10))
    numeric_cols = list(cfg.features.get("numeric", []))
    categorical_cols = list(cfg.features.get("categorical", []))

    evidently_payload = _evidently_data_drift(
        reference[numeric_cols + categorical_cols],
        current[numeric_cols + categorical_cols],
    )
    lightweight = _lightweight_data_drift(
        reference,
        current,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_bins=n_bins,
        threshold=threshold,
    )

    result: dict[str, Any] = {
        "data_drift": evidently_payload or lightweight,
        "lightweight_summary": lightweight,
        "disclaimer": cfg.disclaimer,
    }

    if reference_scores is not None and current_scores is not None:
        champion_threshold = float(cfg.monitoring.get("legacy_prediction_threshold", 0.525))
        result["prediction_drift"] = score_distribution_metrics(
            reference_scores,
            current_scores,
            threshold=champion_threshold,
            n_bins=n_bins,
        )

    out_dir = resolve_path(str(cfg.monitoring.get("report_dir", "artifacts/monitoring")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drift_report.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    logger.info("Wrote drift report")
    result["report_path"] = str(out_path)
    return result
