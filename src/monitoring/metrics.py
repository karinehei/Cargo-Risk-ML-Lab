"""Drift metrics with configurable severity thresholds.

Effect-size thresholds drive ``none`` / ``warning`` / ``critical`` severities.
P-values are reported for context but never used alone to raise severity.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

Severity = Literal["none", "warning", "critical"]

EPS = 1e-12


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _severity(
    value: float, warning: float, critical: float, *, higher_is_worse: bool = True
) -> Severity:
    if not math.isfinite(value):
        return "none"
    if higher_is_worse:
        if value >= critical:
            return "critical"
        if value >= warning:
            return "warning"
        return "none"
    if value <= critical:
        return "critical"
    if value <= warning:
        return "warning"
    return "none"


def _max_severity(*severities: Severity) -> Severity:
    order = {"none": 0, "warning": 1, "critical": 2}
    return max(severities, key=lambda item: order[item])


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index for a numeric feature."""
    ref = _finite(reference)
    cur = _finite(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    if np.allclose(ref, ref[0]) and np.allclose(cur, cur[0]):
        return 0.0
    quantiles = np.linspace(0, 1, n_bins + 1)
    breakpoints = np.unique(np.quantile(ref, quantiles))
    if len(breakpoints) < 2:
        return 0.0
    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)
    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()
    if ref_total <= 0 or cur_total <= 0:
        return float("nan")
    ref_perc = np.clip(ref_counts / ref_total, EPS, None)
    cur_perc = np.clip(cur_counts / cur_total, EPS, None)
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def kolmogorov_smirnov(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """Two-sample KS statistic and p-value."""
    ref = _finite(reference)
    cur = _finite(current)
    if len(ref) == 0 or len(cur) == 0:
        return {"ks_statistic": float("nan"), "ks_pvalue": float("nan")}
    if np.allclose(ref, ref[0]) and np.allclose(cur, cur[0]):
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0}
    statistic, pvalue = stats.ks_2samp(ref, cur)
    return {"ks_statistic": float(statistic), "ks_pvalue": float(pvalue)}


def standardized_mean_difference(reference: np.ndarray, current: np.ndarray) -> float:
    """Standardized mean difference (Cohen-style using pooled std)."""
    ref = _finite(reference)
    cur = _finite(current)
    if len(ref) < 2 or len(cur) < 2:
        return float("nan")
    ref_std = float(np.std(ref, ddof=1))
    cur_std = float(np.std(cur, ddof=1))
    pooled = math.sqrt((ref_std**2 + cur_std**2) / 2.0)
    if pooled <= EPS:
        return 0.0 if math.isclose(float(np.mean(ref)), float(np.mean(cur))) else float("inf")
    return abs(float(np.mean(cur) - np.mean(ref))) / pooled


def missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def jensen_shannon_divergence(reference: pd.Series, current: pd.Series) -> float:
    """Jensen–Shannon divergence between categorical distributions."""
    ref_counts = reference.astype(str).replace("nan", "__MISSING__").value_counts(normalize=True)
    cur_counts = current.astype(str).replace("nan", "__MISSING__").value_counts(normalize=True)
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref = np.array([max(float(ref_counts.get(cat, 0.0)), EPS) for cat in categories], dtype=float)
    cur = np.array([max(float(cur_counts.get(cat, 0.0)), EPS) for cat in categories], dtype=float)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    midpoint = 0.5 * (ref + cur)
    kl_ref = np.sum(ref * np.log(ref / midpoint))
    kl_cur = np.sum(cur * np.log(cur / midpoint))
    return float(0.5 * (kl_ref + kl_cur))


def total_variation_distance(reference: pd.Series, current: pd.Series) -> float:
    ref_counts = reference.astype(str).replace("nan", "__MISSING__").value_counts(normalize=True)
    cur_counts = current.astype(str).replace("nan", "__MISSING__").value_counts(normalize=True)
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    distance = 0.0
    for cat in categories:
        distance += abs(float(ref_counts.get(cat, 0.0)) - float(cur_counts.get(cat, 0.0)))
    return float(0.5 * distance)


def unseen_category_rate(reference: pd.Series, current: pd.Series) -> tuple[float, list[str]]:
    ref_values = set(reference.dropna().astype(str).unique())
    cur_values = current.dropna().astype(str)
    if cur_values.empty:
        return 0.0, []
    unseen_mask = ~cur_values.isin(ref_values)
    unseen = sorted(cur_values[unseen_mask].unique().tolist())
    rate = float(unseen_mask.mean())
    return rate, unseen


def score_distribution_metrics(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    threshold: float,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Review-score and decision drift metrics."""
    ref = _finite(reference_scores)
    cur = _finite(current_scores)
    ks = kolmogorov_smirnov(ref, cur)
    ref_rate = float(np.mean(ref >= threshold)) if len(ref) else float("nan")
    cur_rate = float(np.mean(cur >= threshold)) if len(cur) else float("nan")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    ref_q = {f"q{int(q * 100)}": float(np.quantile(ref, q)) for q in quantiles} if len(ref) else {}
    cur_q = {f"q{int(q * 100)}": float(np.quantile(cur, q)) for q in quantiles} if len(cur) else {}
    q_shift = {
        key: float(cur_q.get(key, float("nan")) - ref_q.get(key, float("nan"))) for key in ref_q
    }
    return {
        "psi": population_stability_index(ref, cur, n_bins=n_bins),
        "ks_statistic": ks["ks_statistic"],
        "ks_pvalue": ks["ks_pvalue"],
        "mean_shift": float(np.mean(cur) - np.mean(ref)) if len(ref) and len(cur) else float("nan"),
        "reference_predicted_review_rate": ref_rate,
        "current_predicted_review_rate": cur_rate,
        "predicted_review_rate_change": float(cur_rate - ref_rate)
        if math.isfinite(ref_rate)
        else float("nan"),
        "reference_quantiles": ref_q,
        "current_quantiles": cur_q,
        "quantile_shift": q_shift,
    }


def numeric_feature_metrics(
    reference: pd.Series,
    current: pd.Series,
    *,
    thresholds: dict[str, Any],
    n_bins: int,
) -> dict[str, Any]:
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)
    psi = population_stability_index(ref, cur, n_bins=n_bins)
    ks = kolmogorov_smirnov(ref, cur)
    smd = standardized_mean_difference(ref, cur)
    ref_missing = missing_rate(reference)
    cur_missing = missing_rate(current)
    missing_delta = float(cur_missing - ref_missing)
    psi_sev = _severity(psi, thresholds["psi"]["warning"], thresholds["psi"]["critical"])
    ks_sev = _severity(
        ks["ks_statistic"], thresholds["ks"]["warning"], thresholds["ks"]["critical"]
    )
    smd_sev = _severity(smd, thresholds["smd"]["warning"], thresholds["smd"]["critical"])
    miss_sev = _severity(
        abs(missing_delta),
        thresholds["missing_rate_delta"]["warning"],
        thresholds["missing_rate_delta"]["critical"],
    )
    severity = _max_severity(psi_sev, ks_sev, smd_sev, miss_sev)
    return {
        "feature": str(reference.name),
        "type": "numeric",
        "psi": psi,
        "ks_statistic": ks["ks_statistic"],
        "ks_pvalue": ks["ks_pvalue"],
        "standardized_mean_difference": smd,
        "reference_missing_rate": ref_missing,
        "current_missing_rate": cur_missing,
        "missing_rate_change": missing_delta,
        "severity": severity,
        "p_value_note": "KS p-values can be significant with large samples even for small shifts.",
    }


def categorical_feature_metrics(
    reference: pd.Series,
    current: pd.Series,
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    js = jensen_shannon_divergence(reference, current)
    tv = total_variation_distance(reference, current)
    unseen_rate, unseen_values = unseen_category_rate(reference, current)
    ref_missing = missing_rate(reference)
    cur_missing = missing_rate(current)
    missing_delta = float(cur_missing - ref_missing)
    js_sev = _severity(
        js, thresholds["js_divergence"]["warning"], thresholds["js_divergence"]["critical"]
    )
    tv_sev = _severity(
        tv, thresholds["tv_distance"]["warning"], thresholds["tv_distance"]["critical"]
    )
    unseen_sev = _severity(
        unseen_rate,
        thresholds["unseen_category_rate"]["warning"],
        thresholds["unseen_category_rate"]["critical"],
    )
    miss_sev = _severity(
        abs(missing_delta),
        thresholds["missing_rate_delta"]["warning"],
        thresholds["missing_rate_delta"]["critical"],
    )
    severity = _max_severity(js_sev, tv_sev, unseen_sev, miss_sev)
    return {
        "feature": str(reference.name),
        "type": "categorical",
        "jensen_shannon_divergence": js,
        "total_variation_distance": tv,
        "unseen_category_rate": unseen_rate,
        "unseen_categories": unseen_values[:20],
        "reference_missing_rate": ref_missing,
        "current_missing_rate": cur_missing,
        "missing_rate_change": missing_delta,
        "severity": severity,
    }


def overall_severity(
    feature_rows: list[dict[str, Any]], score_row: dict[str, Any] | None
) -> Severity:
    severities: list[Severity] = []
    for row in feature_rows:
        value = str(row.get("severity", "none"))
        if value in ("none", "warning", "critical"):
            severities.append(value)  # type: ignore[arg-type]
    if score_row is not None:
        value = str(score_row.get("severity", "none"))
        if value in ("none", "warning", "critical"):
            severities.append(value)  # type: ignore[arg-type]
    return _max_severity(*severities) if severities else "none"


def recommended_action(severity: Severity) -> str:
    if severity == "critical":
        return "Investigate data pipeline and operational context before relying on outputs. Do not automatically retrain or change the threshold."
    if severity == "warning":
        return "Investigate data pipeline and operational context. Continue routine monitoring."
    return "Continue routine monitoring."


def default_thresholds(config: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    cfg = dict(config or {})
    return {
        "psi": {
            "warning": float(cfg.get("psi_warning", 0.10)),
            "critical": float(cfg.get("psi_critical", 0.25)),
        },
        "ks": {
            "warning": float(cfg.get("ks_warning", 0.10)),
            "critical": float(cfg.get("ks_critical", 0.20)),
        },
        "smd": {
            "warning": float(cfg.get("smd_warning", 0.20)),
            "critical": float(cfg.get("smd_critical", 0.50)),
        },
        "js_divergence": {
            "warning": float(cfg.get("js_warning", 0.05)),
            "critical": float(cfg.get("js_critical", 0.15)),
        },
        "tv_distance": {
            "warning": float(cfg.get("tv_warning", 0.10)),
            "critical": float(cfg.get("tv_critical", 0.25)),
        },
        "unseen_category_rate": {
            "warning": float(cfg.get("unseen_warning", 0.01)),
            "critical": float(cfg.get("unseen_critical", 0.05)),
        },
        "missing_rate_delta": {
            "warning": float(cfg.get("missing_delta_warning", 0.05)),
            "critical": float(cfg.get("missing_delta_critical", 0.15)),
        },
        "score_psi": {
            "warning": float(cfg.get("score_psi_warning", 0.10)),
            "critical": float(cfg.get("score_psi_critical", 0.25)),
        },
        "predicted_review_rate_delta": {
            "warning": float(cfg.get("review_rate_delta_warning", 0.05)),
            "critical": float(cfg.get("review_rate_delta_critical", 0.08)),
        },
    }
