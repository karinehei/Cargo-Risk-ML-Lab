"""Monitoring-policy aggregation and alert-reason extraction.

Policy 1.0.0 raised overall severity to the maximum of any single feature or
score metric (union / max rule). Policy 1.1.0 keeps the same effect-size
thresholds but aggregates with:

* isolated weak warnings that do not become an overall warning;
* a minimum number of warning features, or a score-level warning;
* persistence across consecutive windows;
* immediate-critical exceptions for schema / unseen-category / extreme missingness
  and large review-rate shifts.

P-values never raise severity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from src.monitoring.metrics import default_thresholds

PolicyStatus = Literal[
    "insufficient_data",
    "no_material_drift",
    "warning",
    "critical",
    "monitoring_error",
]
Severity = Literal["none", "warning", "critical"]

MONITORING_POLICY_VERSION = "1.1.0"
MONITORING_POLICY_V1 = "1.0.0"

NUMERIC_METRIC_MAP: tuple[tuple[str, str, bool], ...] = (
    ("psi", "psi", False),
    ("ks_statistic", "ks", False),
    ("standardized_mean_difference", "smd", False),
    ("missing_rate_change", "missing_rate_delta", True),
)
CATEGORICAL_METRIC_MAP: tuple[tuple[str, str, bool], ...] = (
    ("jensen_shannon_divergence", "js_divergence", False),
    ("total_variation_distance", "tv_distance", False),
    ("unseen_category_rate", "unseen_category_rate", False),
    ("missing_rate_change", "missing_rate_delta", True),
)
SCORE_METRIC_MAP: tuple[tuple[str, str, bool], ...] = (
    ("psi", "score_psi", False),
    ("ks_statistic", "ks", False),
    ("predicted_review_rate_change", "predicted_review_rate_delta", True),
)

INTERPRETATIONS: dict[str, str] = {
    "psi": "Population Stability Index exceeds the configured effect-size threshold.",
    "ks_statistic": "Kolmogorov–Smirnov statistic exceeds the effect-size threshold (p-value is not used to raise severity).",
    "standardized_mean_difference": "Standardized mean difference exceeds the effect-size threshold.",
    "missing_rate_change": "Absolute missing-value-rate change exceeds the effect-size threshold.",
    "jensen_shannon_divergence": "Jensen–Shannon divergence exceeds the effect-size threshold.",
    "total_variation_distance": "Total-variation distance exceeds the effect-size threshold.",
    "unseen_category_rate": "Share of rows with categories absent from the reference exceeds the threshold.",
    "predicted_review_rate_change": "Absolute change in predicted review rate at the fixed threshold exceeds the effect-size threshold.",
}


def expected_union_alert_rate(per_comparison_rates: Sequence[float]) -> float:
    """Independence approximation of P(at least one comparison alerts)."""
    survival = 1.0
    for rate in per_comparison_rates:
        clipped = min(max(float(rate), 0.0), 1.0)
        survival *= 1.0 - clipped
    return 1.0 - survival


def _as_severity(value: str) -> Severity:
    if value == "warning":
        return "warning"
    if value == "critical":
        return "critical"
    return "none"


def _metric_severity(observed: float, warning: float, critical: float) -> Severity:
    if observed >= critical:
        return "critical"
    if observed >= warning:
        return "warning"
    return "none"


def extract_alert_reasons(
    feature_rows: list[dict[str, Any]],
    score_row: dict[str, Any] | None,
    thresholds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Return every metric that meets warning or critical effect-size thresholds."""
    reasons: list[dict[str, Any]] = []
    for row in feature_rows:
        mapping = NUMERIC_METRIC_MAP if row.get("type") == "numeric" else CATEGORICAL_METRIC_MAP
        name = str(row.get("feature") or "")
        for field, threshold_key, use_abs in mapping:
            raw = row.get(field)
            if raw is None:
                continue
            observed = abs(float(raw)) if use_abs else float(raw)
            bounds = thresholds[threshold_key]
            severity = _metric_severity(observed, bounds["warning"], bounds["critical"])
            if severity == "none":
                continue
            reasons.append(
                {
                    "name": name,
                    "metric": field,
                    "observed_value": float(raw),
                    "warning_threshold": bounds["warning"],
                    "critical_threshold": bounds["critical"],
                    "severity": severity,
                    "interpretation": INTERPRETATIONS.get(field, "Effect-size threshold exceeded."),
                    "immediate_critical": False,
                }
            )
    if score_row is not None:
        for field, threshold_key, use_abs in SCORE_METRIC_MAP:
            raw = score_row.get(field)
            if raw is None:
                continue
            observed = abs(float(raw)) if use_abs else float(raw)
            bounds = thresholds[threshold_key]
            severity = _metric_severity(observed, bounds["warning"], bounds["critical"])
            if severity == "none":
                continue
            reasons.append(
                {
                    "name": "review_score",
                    "metric": field,
                    "observed_value": float(raw),
                    "warning_threshold": bounds["warning"],
                    "critical_threshold": bounds["critical"],
                    "severity": severity,
                    "interpretation": INTERPRETATIONS.get(field, "Effect-size threshold exceeded."),
                    "immediate_critical": False,
                }
            )
    return reasons


def _mark_immediate_critical(reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for reason in reasons:
        immediate = False
        if reason["metric"] == "unseen_category_rate" and reason["severity"] == "critical":
            immediate = True
            reason = {
                **reason,
                "interpretation": (
                    "High unseen-category rate is treated as a schema violation. "
                    "Investigate before relying on scores."
                ),
            }
        if reason["metric"] == "missing_rate_change" and reason["severity"] == "critical":
            immediate = True
            reason = {
                **reason,
                "interpretation": (
                    "Extreme missingness change is an immediate-critical exception."
                ),
            }
        if (
            reason["name"] == "review_score"
            and reason["metric"] == "predicted_review_rate_change"
            and reason["severity"] == "critical"
        ):
            immediate = True
            reason = {
                **reason,
                "interpretation": (
                    "Major change in predicted review rate at the fixed threshold. "
                    "Investigate operational mix; do not auto-retrain."
                ),
            }
        out.append({**reason, "immediate_critical": immediate})
    return out


def _feature_severity_counts(feature_rows: list[dict[str, Any]]) -> tuple[int, int]:
    n_warning = sum(1 for row in feature_rows if row.get("severity") == "warning")
    n_critical = sum(1 for row in feature_rows if row.get("severity") == "critical")
    return n_warning, n_critical


def aggregate_status_v1_0_0(
    feature_rows: list[dict[str, Any]],
    score_row: dict[str, Any] | None,
) -> PolicyStatus:
    """Legacy max-of-all-metrics rule used for policy 1.0.0 evidence."""
    from src.monitoring.metrics import overall_severity

    severity = overall_severity(feature_rows, score_row)
    if severity == "critical":
        return "critical"
    if severity == "warning":
        return "warning"
    return "no_material_drift"


def aggregate_status_v1_1_0(
    feature_rows: list[dict[str, Any]],
    score_row: dict[str, Any] | None,
    reasons: list[dict[str, Any]],
    *,
    min_warning_features: int = 2,
    previous_status: str | None = None,
    previous_warning_names: list[str] | None = None,
) -> PolicyStatus:
    """Aggregate per-metric findings into an operational status."""
    if any(bool(item.get("immediate_critical")) for item in reasons):
        return "critical"
    n_warning, n_critical = _feature_severity_counts(feature_rows)
    score_severity = _as_severity(str((score_row or {}).get("severity") or "none"))
    if n_critical >= 1 or score_severity == "critical":
        return "critical"

    schema_warning = any(
        item["metric"] == "unseen_category_rate" and item["severity"] != "none" for item in reasons
    )
    warning_names = sorted(
        {
            str(row.get("feature"))
            for row in feature_rows
            if row.get("severity") in {"warning", "critical"}
        }
    )
    persistent = False
    if previous_warning_names:
        overlap = set(warning_names) & set(previous_warning_names)
        persistent = bool(overlap) and previous_status in {
            "warning",
            "no_material_drift",
            "critical",
        }

    coordinated = n_warning >= int(min_warning_features)
    score_warn = score_severity == "warning"
    if schema_warning or coordinated or score_warn or persistent:
        return "warning"
    return "no_material_drift"


def recommended_action_for_status(status: PolicyStatus) -> str:
    if status == "critical":
        return (
            "Investigate data pipeline and operational context before relying on outputs. "
            "Do not automatically retrain or change the threshold."
        )
    if status == "warning":
        return "Investigate data pipeline and operational context. Continue routine monitoring."
    if status == "insufficient_data":
        return "Collect a larger current batch before interpreting drift."
    if status == "monitoring_error":
        return "Monitoring did not complete. Investigate the monitoring job; do not treat this as healthy."
    return "Continue routine monitoring. Isolated weak findings, if listed, do not constitute material drift."


def severity_from_status(status: PolicyStatus) -> Severity:
    if status == "critical":
        return "critical"
    if status == "warning":
        return "warning"
    return "none"


def apply_policy(
    feature_rows: list[dict[str, Any]],
    score_row: dict[str, Any] | None,
    *,
    thresholds: dict[str, dict[str, float]] | None = None,
    min_warning_features: int = 2,
    previous_status: str | None = None,
    previous_warning_names: list[str] | None = None,
    policy_version: str = MONITORING_POLICY_VERSION,
) -> dict[str, Any]:
    """Return status, alert reasons and counts for a completed comparison."""
    resolved = thresholds or default_thresholds()
    reasons = _mark_immediate_critical(extract_alert_reasons(feature_rows, score_row, resolved))
    n_warning, n_critical = _feature_severity_counts(feature_rows)
    score_alerts = [item for item in reasons if item["name"] == "review_score"]
    if score_row is not None and score_row.get("severity") == "warning":
        n_warning_score = 1
    elif score_row is not None and score_row.get("severity") == "critical":
        n_warning_score = 0
        n_critical += 1
    else:
        n_warning_score = 0
    if policy_version == MONITORING_POLICY_V1:
        status = aggregate_status_v1_0_0(feature_rows, score_row)
    else:
        status = aggregate_status_v1_1_0(
            feature_rows,
            score_row,
            reasons,
            min_warning_features=min_warning_features,
            previous_status=previous_status,
            previous_warning_names=previous_warning_names,
        )
    warning_names = sorted(
        {
            str(row.get("feature"))
            for row in feature_rows
            if row.get("severity") in {"warning", "critical"}
        }
    )
    isolated = (
        status == "no_material_drift" and n_warning == 1 and n_critical == 0 and not score_alerts
    )
    for reason in reasons:
        if isolated and reason["name"] in warning_names:
            reason["role"] = "isolated_weak_warning"
        elif reason.get("immediate_critical"):
            reason["role"] = "immediate_critical"
        else:
            reason["role"] = "contributing"
    return {
        "status": status,
        "overall_severity": severity_from_status(status),
        "alert_reasons": reasons,
        "n_warning_features": n_warning,
        "n_critical_features": n_critical,
        "n_score_warnings": n_warning_score,
        "n_alert_reasons": len(reasons),
        "warning_feature_names": warning_names,
        "isolated_weak_warning": isolated,
        "recommended_action": recommended_action_for_status(status),
        "policy_version": policy_version,
    }
