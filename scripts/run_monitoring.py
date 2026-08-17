"""CLI for reference profiles, monitoring scenarios and drift reports."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import get_config, setup_logging
from src.monitoring.audit import run_independent_validation, run_null_monte_carlo
from src.monitoring.runner import (
    create_reference_profile,
    generate_scenario_batch,
    run_monitoring,
    show_latest_status,
)
from src.monitoring.scenarios import SCENARIO_SEEDS

SCENARIOS = tuple(SCENARIO_SEEDS.keys())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cargo Risk ML Lab monitoring CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create-reference", help="Build a train-derived reference profile")

    generate = sub.add_parser("generate-scenario", help="Generate a monitoring current batch")
    generate.add_argument(
        "scenario", choices=[name for name in SCENARIOS if name != "labelled_simulation"]
    )
    generate.add_argument(
        "--with-labels", action="store_true", help="Include synthetic labels (simulation only)"
    )

    unlabelled = sub.add_parser(
        "run-unlabelled", help="Run unlabelled input and score drift monitoring"
    )
    unlabelled.add_argument(
        "scenario", choices=[name for name in SCENARIOS if name != "labelled_simulation"]
    )

    labelled = sub.add_parser(
        "run-labelled-simulation", help="Run synthetic labelled performance simulation"
    )
    labelled.add_argument(
        "scenario", choices=[name for name in SCENARIOS if name != "labelled_simulation"]
    )

    sub.add_parser("status", help="Show latest monitoring status")
    sub.add_parser("run-all", help="Create reference and run all standard scenarios")
    null_audit = sub.add_parser(
        "run-null-audit", help="Null Monte Carlo false-alert audit (no test data)"
    )
    null_audit.add_argument("--replications", type=int, default=None)
    null_audit.add_argument("--seed-base", type=int, default=None)
    sub.add_parser(
        "run-validation",
        help="Independent detection validation with hold-out seeds",
    )
    return parser


def main() -> None:
    logger = setup_logging(name="scripts.run_monitoring")
    cfg = get_config()
    parser = _build_parser()
    args = parser.parse_args()
    logger.info("Disclaimer: %s", cfg.disclaimer)

    if args.command == "create-reference":
        profile = create_reference_profile(cfg)
        print(
            json.dumps(
                {"status": "ok", "dataset_fingerprint": profile["dataset_fingerprint"]}, indent=2
            )
        )
        return

    if args.command == "generate-scenario":
        meta = generate_scenario_batch(args.scenario, config=cfg, include_labels=args.with_labels)
        print(json.dumps(meta, indent=2, default=str))
        return

    if args.command == "run-unlabelled":
        report = run_monitoring(args.scenario, mode="unlabelled_monitoring", config=cfg)
        print(
            json.dumps(
                {
                    "monitoring_run_id": report["monitoring_run_id"],
                    "status": report.get("status"),
                    "overall_severity": report.get("overall_severity"),
                    "policy_version": report.get("policy_version"),
                    "n_alert_reasons": report.get("n_alert_reasons"),
                    "alert_reasons": report.get("alert_reasons") or [],
                    "predicted_review_rate_change": (report.get("score_drift") or {}).get(
                        "predicted_review_rate_change"
                    ),
                    "score_psi": (report.get("score_drift") or {}).get("psi"),
                },
                indent=2,
                default=str,
            )
        )
        return

    if args.command == "run-labelled-simulation":
        report = run_monitoring(args.scenario, mode="labelled_simulation", config=cfg)
        print(json.dumps(report.get("simulated_performance") or {}, indent=2, default=str))
        return

    if args.command == "status":
        print(json.dumps(show_latest_status(), indent=2, default=str))
        return

    if args.command == "run-all":
        profile = create_reference_profile(cfg)
        summaries = {"reference_fingerprint": profile["dataset_fingerprint"], "scenarios": {}}
        for scenario in ("none", "subtle", "moderate", "major", "missingness", "unseen_category"):
            generate_scenario_batch(scenario, config=cfg)
            report = run_monitoring(scenario, mode="unlabelled_monitoring", config=cfg)
            summaries["scenarios"][scenario] = {
                "status": report.get("status"),
                "overall_severity": report.get("overall_severity"),
                "policy_version": report.get("policy_version"),
                "n_alert_reasons": report.get("n_alert_reasons"),
                "alert_reasons": [
                    {
                        "name": item.get("name"),
                        "metric": item.get("metric"),
                        "observed_value": item.get("observed_value"),
                        "severity": item.get("severity"),
                        "role": item.get("role"),
                    }
                    for item in (report.get("alert_reasons") or [])
                ],
                "predicted_review_rate_change": (report.get("score_drift") or {}).get(
                    "predicted_review_rate_change"
                ),
                "score_psi": (report.get("score_drift") or {}).get("psi"),
            }
        print(json.dumps(summaries, indent=2, default=str))
        return

    if args.command == "run-null-audit":
        replications = int(
            args.replications
            if args.replications is not None
            else cfg.monitoring.get("null_replications", 150)
        )
        seed_base = int(
            args.seed_base
            if args.seed_base is not None
            else cfg.monitoring.get("null_seed_base", 92001)
        )
        summary = run_null_monte_carlo(n_replications=replications, seed_base=seed_base, config=cfg)
        printable = dict(summary)
        printable.pop("replications", None)
        printable.pop("distributions", None)
        print(json.dumps(printable, indent=2, default=str))
        return

    if args.command == "run-validation":
        payload = run_independent_validation(config=cfg)
        print(json.dumps(payload, indent=2, default=str))
        return

    parser.error(f"Unknown command: {args.command}")
    sys.exit(2)


if __name__ == "__main__":
    main()
