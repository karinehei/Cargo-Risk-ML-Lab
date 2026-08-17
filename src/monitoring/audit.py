"""Null Monte Carlo and independent detection validation for monitoring policy."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

from src.config import AppConfig, get_config, resolve_path, setup_logging
from src.monitoring.metrics import default_thresholds
from src.monitoring.policy import MONITORING_POLICY_V1, MONITORING_POLICY_VERSION, apply_policy
from src.monitoring.runner import (
    _load_train_reference_frame,
    _score_batch,
    evaluate_comparison,
)
from src.monitoring.scenarios import (
    NULL_SEED_BASE,
    VALIDATION_SEEDS,
    generate_monitoring_batch,
    generate_null_batch,
)

logger = setup_logging(name="src.monitoring.audit")


def _summarise_distribution(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    array = np.asarray(finite, dtype=float)
    return {
        "n": float(len(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def run_null_monte_carlo(
    *,
    n_replications: int = 150,
    seed_base: int = NULL_SEED_BASE,
    config: AppConfig | None = None,
    policy_version: str = MONITORING_POLICY_VERSION,
    persist: bool = True,
) -> dict[str, Any]:
    """Estimate false-alert rates under matched no-intentional-shift batches."""
    cfg = config or get_config()
    if n_replications < 1:
        raise ValueError("n_replications must be >= 1")
    batch_size = int(cfg.monitoring.get("scenario_batch_size", 1200))
    reference = _load_train_reference_frame(cfg)
    ref_scores, champion, pipeline = _score_batch(reference)
    thresholds = default_thresholds(dict(cfg.monitoring.get("severity_thresholds") or {}))
    min_warning = int(cfg.monitoring.get("min_warning_features", 2))

    metric_values: dict[str, list[float]] = defaultdict(list)
    per_feature_alert: dict[str, int] = defaultdict(int)
    per_metric_alert: dict[str, int] = defaultdict(int)
    overall_v11 = {"warning": 0, "critical": 0, "no_material_drift": 0}
    overall_v10 = {"warning": 0, "critical": 0, "no_material_drift": 0}
    union_any_reason = 0
    replications: list[dict[str, Any]] = []

    for index in range(n_replications):
        seed = int(seed_base) + index
        current = generate_null_batch(seed, n_samples=batch_size, config=cfg)
        result = evaluate_comparison(
            current,
            reference_frame=reference,
            reference_scores=ref_scores,
            champion=champion,
            config=cfg,
            scenario="none",
            seed=seed,
            policy_version=policy_version,
            write=False,
            pipeline=pipeline,
        )
        legacy = apply_policy(
            result["feature_metrics"],
            result["score_drift"],
            thresholds=thresholds,
            min_warning_features=min_warning,
            policy_version=MONITORING_POLICY_V1,
        )
        status = str(result["status"])
        overall_v11[status] = overall_v11.get(status, 0) + 1
        overall_v10[str(legacy["status"])] = overall_v10.get(str(legacy["status"]), 0) + 1
        reasons = list(result.get("alert_reasons") or [])
        if reasons:
            union_any_reason += 1
        names_this: set[str] = set()
        metrics_this: set[str] = set()
        for reason in reasons:
            names_this.add(str(reason["name"]))
            metrics_this.add(f"{reason['name']}:{reason['metric']}")
        for name in names_this:
            per_feature_alert[name] += 1
        for key in metrics_this:
            per_metric_alert[key] += 1

        for row in result["feature_metrics"]:
            prefix = str(row["feature"])
            if row.get("type") == "numeric":
                metric_values[f"{prefix}:psi"].append(float(row.get("psi") or 0.0))
                metric_values[f"{prefix}:ks"].append(float(row.get("ks_statistic") or 0.0))
                metric_values[f"{prefix}:smd"].append(
                    float(row.get("standardized_mean_difference") or 0.0)
                )
                metric_values[f"{prefix}:missing_rate_change"].append(
                    abs(float(row.get("missing_rate_change") or 0.0))
                )
            else:
                metric_values[f"{prefix}:js"].append(
                    float(row.get("jensen_shannon_divergence") or 0.0)
                )
                metric_values[f"{prefix}:tv"].append(
                    float(row.get("total_variation_distance") or 0.0)
                )
                metric_values[f"{prefix}:missing_rate_change"].append(
                    abs(float(row.get("missing_rate_change") or 0.0))
                )
        score = result["score_drift"]
        metric_values["review_score:psi"].append(float(score.get("psi") or 0.0))
        metric_values["review_score:ks"].append(float(score.get("ks_statistic") or 0.0))
        metric_values["review_score:review_rate_change"].append(
            abs(float(score.get("predicted_review_rate_change") or 0.0))
        )
        replications.append(
            {
                "seed": seed,
                "status": status,
                "legacy_status": legacy["status"],
                "n_alert_reasons": len(reasons),
                "isolated_weak_warning": result.get("isolated_weak_warning"),
            }
        )

    n = float(n_replications)
    summary = {
        "n_replications": n_replications,
        "seed_base": seed_base,
        "batch_size": batch_size,
        "reference_n_rows": int(len(reference)),
        "champion_version": champion.get("model_version"),
        "decision_threshold": champion.get("decision_threshold"),
        "policy_version": policy_version,
        "false_alert_rate_overall_v1_1_0": {
            "warning": overall_v11.get("warning", 0) / n,
            "critical": overall_v11.get("critical", 0) / n,
            "no_material_drift": overall_v11.get("no_material_drift", 0) / n,
            "any_warning_or_critical": (
                overall_v11.get("warning", 0) + overall_v11.get("critical", 0)
            )
            / n,
        },
        "false_alert_rate_overall_v1_0_0": {
            "warning": overall_v10.get("warning", 0) / n,
            "critical": overall_v10.get("critical", 0) / n,
            "no_material_drift": overall_v10.get("no_material_drift", 0) / n,
            "any_warning_or_critical": (
                overall_v10.get("warning", 0) + overall_v10.get("critical", 0)
            )
            / n,
        },
        "union_any_metric_reason_rate": union_any_reason / n,
        "per_feature_alert_rate": {
            key: value / n for key, value in sorted(per_feature_alert.items())
        },
        "per_metric_alert_rate": {
            key: value / n for key, value in sorted(per_metric_alert.items())
        },
        "distributions": {
            key: _summarise_distribution(vals) for key, vals in sorted(metric_values.items())
        },
        "multiple_comparison_note": (
            "Monitoring many features increases the chance that at least one metric crosses a "
            "threshold by random variation even when no feature is individually unreliable. "
            "Policy 1.1.0 therefore distinguishes isolated weak warnings from overall warning."
        ),
        "replications": replications,
    }
    if persist:
        out_dir = (
            resolve_path(str(cfg.monitoring.get("report_dir", "artifacts/monitoring")))
            / "policy_audit"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "null_monte_carlo.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote null Monte Carlo summary (%s replications)", n_replications)
        summary["artifact"] = path.name
    return summary


def run_independent_validation(
    *,
    config: AppConfig | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run hold-out seeds that were not used to set demonstration scenarios."""
    cfg = config or get_config()
    reference = _load_train_reference_frame(cfg)
    ref_scores, champion, pipeline = _score_batch(reference)
    batch_size = int(cfg.monitoring.get("scenario_batch_size", 1200))
    rows: list[dict[str, Any]] = []
    for scenario, seeds in VALIDATION_SEEDS.items():
        match_training = scenario == "none"
        for seed in seeds:
            current = generate_monitoring_batch(
                scenario,  # type: ignore[arg-type]
                n_samples=batch_size,
                seed=seed,
                config=cfg,
                match_training_generator=match_training,
            )
            result = evaluate_comparison(
                current,
                reference_frame=reference,
                reference_scores=ref_scores,
                champion=champion,
                config=cfg,
                scenario=scenario,  # type: ignore[arg-type]
                seed=seed,
                write=False,
                pipeline=pipeline,
            )
            score = result["score_drift"]
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "status": result["status"],
                    "overall_severity": result["overall_severity"],
                    "n_alert_reasons": result.get("n_alert_reasons"),
                    "alert_reasons": result.get("alert_reasons"),
                    "warning_feature_names": result.get("warning_feature_names"),
                    "isolated_weak_warning": result.get("isolated_weak_warning"),
                    "score_psi": score.get("psi"),
                    "predicted_review_rate_change": score.get("predicted_review_rate_change"),
                    "reference_predicted_review_rate": score.get("reference_predicted_review_rate"),
                    "current_predicted_review_rate": score.get("current_predicted_review_rate"),
                }
            )
    payload = {
        "policy_version": MONITORING_POLICY_VERSION,
        "champion_version": champion.get("model_version"),
        "decision_threshold": champion.get("decision_threshold"),
        "results": rows,
        "note": "Independent seeds. Not used to calibrate thresholds.",
    }
    if persist:
        out_dir = (
            resolve_path(str(cfg.monitoring.get("report_dir", "artifacts/monitoring")))
            / "policy_audit"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "independent_validation.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["artifact"] = path.name
    return payload
