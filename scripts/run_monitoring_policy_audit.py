"""Run monitoring-policy verification: null Monte Carlo, validation, scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import get_config
from src.monitoring.audit import run_independent_validation, run_null_monte_carlo
from src.monitoring.runner import generate_scenario_batch, run_monitoring

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = get_config()
    n_rep = int(cfg.monitoring.get("null_replications", 150))
    seed_base = int(cfg.monitoring.get("null_seed_base", 92001))
    print(f"null_monte_carlo replications={n_rep} seed_base={seed_base}")
    summary = run_null_monte_carlo(n_replications=n_rep, seed_base=seed_base, config=cfg)
    compact = {
        key: summary[key]
        for key in (
            "n_replications",
            "seed_base",
            "batch_size",
            "reference_n_rows",
            "champion_version",
            "decision_threshold",
            "policy_version",
            "false_alert_rate_overall_v1_1_0",
            "false_alert_rate_overall_v1_0_0",
            "union_any_metric_reason_rate",
            "per_feature_alert_rate",
            "per_metric_alert_rate",
            "multiple_comparison_note",
            "artifact",
        )
        if key in summary
    }
    dist_keys = [
        "review_score:psi",
        "review_score:ks",
        "review_score:review_rate_change",
        "declaration_completeness_score:ks",
        "declaration_completeness_score:smd",
        "declaration_completeness_score:psi",
    ]
    compact["selected_distributions"] = {
        key: summary.get("distributions", {}).get(key) for key in dist_keys
    }
    print(json.dumps(compact, indent=2, default=str))

    print("independent_validation")
    validation = run_independent_validation(config=cfg)
    rows = []
    for row in validation.get("results") or []:
        rows.append(
            {
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "status": row.get("status"),
                "overall_severity": row.get("overall_severity"),
                "n_alert_reasons": row.get("n_alert_reasons"),
                "warning_feature_names": row.get("warning_feature_names"),
                "isolated_weak_warning": row.get("isolated_weak_warning"),
                "score_psi": row.get("score_psi"),
                "predicted_review_rate_change": row.get("predicted_review_rate_change"),
                "alert_reasons": [
                    {
                        "name": item.get("name"),
                        "metric": item.get("metric"),
                        "observed_value": item.get("observed_value"),
                        "severity": item.get("severity"),
                        "role": item.get("role"),
                    }
                    for item in (row.get("alert_reasons") or [])
                ],
            }
        )
    print(
        json.dumps({"policy_version": validation.get("policy_version"), "results": rows}, indent=2)
    )

    print("standard_scenarios")
    scenario_out = {}
    for scenario in ("none", "subtle", "moderate", "major", "missingness", "unseen_category"):
        generate_scenario_batch(scenario, config=cfg)
        report = run_monitoring(scenario, mode="unlabelled_monitoring", config=cfg)
        scenario_out[scenario] = {
            "status": report.get("status"),
            "overall_severity": report.get("overall_severity"),
            "policy_version": report.get("policy_version"),
            "n_alert_reasons": report.get("n_alert_reasons"),
            "isolated_weak_warning": report.get("isolated_weak_warning"),
            "warning_feature_names": report.get("warning_feature_names"),
            "score_psi": (report.get("score_drift") or {}).get("psi"),
            "predicted_review_rate_change": (report.get("score_drift") or {}).get(
                "predicted_review_rate_change"
            ),
            "alert_reasons": [
                {
                    "name": item.get("name"),
                    "metric": item.get("metric"),
                    "observed_value": item.get("observed_value"),
                    "warning_threshold": item.get("warning_threshold"),
                    "critical_threshold": item.get("critical_threshold"),
                    "severity": item.get("severity"),
                    "role": item.get("role"),
                }
                for item in (report.get("alert_reasons") or [])
            ],
        }
    out_path = ROOT / "artifacts" / "monitoring" / "policy_audit" / "scenario_results_v1_1_0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scenario_out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(scenario_out, indent=2, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
