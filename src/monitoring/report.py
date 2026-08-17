"""Human-readable and machine-readable monitoring report writers."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any, cast

from src.config import get_config, resolve_path, setup_logging

logger = setup_logging(name="src.monitoring.report")


def _feature_rows_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {}
            for key, value in row.items():
                if isinstance(value, list):
                    serializable[key] = "|".join(map(str, value))
                else:
                    serializable[key] = value
            writer.writerow(serializable)


def render_markdown_report(report: dict[str, Any]) -> str:
    champion = report.get("champion") or {}
    lines = [
        "# Cargo Risk ML Lab monitoring report",
        "",
        f"- Monitoring run ID: `{report.get('monitoring_run_id')}`",
        f"- Timestamp: {report.get('timestamp')}",
        f"- Mode: **{report.get('mode')}**",
        f"- Scenario: `{report.get('scenario')}`",
        f"- Policy version: `{report.get('policy_version') or report.get('monitoring_policy_version')}`",
        f"- Status: **{report.get('status')}**",
        f"- Overall severity: **{report.get('overall_severity')}**",
        f"- Recommended action: {report.get('recommended_action')}",
        f"- Reference batch size: {report.get('reference_batch_size')}",
        f"- Current batch size: {report.get('batch_size')}",
        f"- Monitored features: {report.get('n_monitored_features')}",
        f"- Warning features: {report.get('n_warning_features')}",
        f"- Critical features: {report.get('n_critical_features')}",
        f"- Ground truth available: {report.get('ground_truth_available')}",
        f"- Report complete: {report.get('report_complete')}",
        "",
        "## Alert reasons",
        "",
    ]
    reasons = report.get("alert_reasons") or []
    if not reasons:
        lines.append("No metric exceeded a warning or critical effect-size threshold.")
    else:
        for reason in reasons:
            lines.append(
                f"- `{reason.get('name')}` / `{reason.get('metric')}`: "
                f"observed={reason.get('observed_value')} "
                f"warning>={reason.get('warning_threshold')} "
                f"critical>={reason.get('critical_threshold')} "
                f"severity={reason.get('severity')} "
                f"role={reason.get('role')}. {reason.get('interpretation')}"
            )
    lines.extend(
        [
            "",
            "## Champion",
            "",
            f"- Model version: `{champion.get('model_version')}`",
            f"- MLflow run ID: `{champion.get('mlflow_run_id')}`",
            f"- Decision threshold: {champion.get('decision_threshold')}",
            "",
            "## Batch identities",
            "",
            f"- Reference fingerprint: `{report.get('reference_fingerprint')}`",
            f"- Current fingerprint: `{report.get('current_fingerprint')}`",
            f"- Current batch size: {report.get('batch_size')}",
            "",
            "## Score drift",
            "",
        ]
    )
    score = report.get("score_drift") or {}
    lines.extend(
        [
            f"- Review-score PSI: {score.get('psi')}",
            f"- Predicted review rate (reference): {score.get('reference_predicted_review_rate')}",
            f"- Predicted review rate (current): {score.get('current_predicted_review_rate')}",
            f"- Predicted review rate change: {score.get('predicted_review_rate_change')}",
            f"- Score drift severity: **{score.get('severity')}**",
            "",
            "## Feature drift summary",
            "",
        ]
    )
    for row in report.get("feature_metrics") or []:
        if row.get("type") == "numeric":
            lines.append(
                f"- `{row['feature']}` severity={row['severity']} psi={row.get('psi')} "
                f"smd={row.get('standardized_mean_difference')} missing_change={row.get('missing_rate_change')}"
            )
        else:
            lines.append(
                f"- `{row['feature']}` severity={row['severity']} js={row.get('jensen_shannon_divergence')} "
                f"tv={row.get('total_variation_distance')} unseen_rate={row.get('unseen_category_rate')}"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *(f"- {item}" for item in report.get("limitations") or []),
            "",
            report.get("disclaimer", ""),
        ]
    )
    if report.get("mode") == "labelled_simulation":
        lines.extend(["", "## Simulated performance (not production evidence)", ""])
        perf = report.get("simulated_performance") or {}
        for key, value in perf.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("These outcomes are synthetic simulation only.")
    else:
        lines.extend(
            [
                "",
                "## Ground truth",
                "",
                "Ground-truth labels are unavailable in unlabelled monitoring. "
                "Performance degradation cannot be measured from this report alone.",
            ]
        )
    return "\n".join(lines)


def write_monitoring_report(
    report: dict[str, Any], *, output_dir: str | None = None
) -> dict[str, str]:
    cfg = get_config()
    directory = resolve_path(
        output_dir or str(cfg.monitoring.get("report_dir", "artifacts/monitoring"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    run_id = str(report.get("monitoring_run_id") or uuid.uuid4())
    mode = str(report.get("mode") or "unlabelled_monitoring")
    scenario = str(report.get("scenario") or "unknown")
    stem = f"{mode}_{scenario}_{run_id[:8]}"
    json_path = directory / f"{stem}.json"
    csv_path = directory / f"{stem}_features.csv"
    md_path = directory / f"{stem}.md"
    latest_path = directory / "latest_report.json"
    status_path = directory / "status.json"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _feature_rows_to_csv(list(report.get("feature_metrics") or []), csv_path)
    md_path.write_text(render_markdown_report(report), encoding="utf-8")

    status = {
        "available": bool(report.get("available", True))
        and bool(report.get("report_complete", True)),
        "monitoring_run_id": run_id,
        "timestamp": report.get("timestamp"),
        "mode": mode,
        "scenario": scenario,
        "overall_severity": report.get("overall_severity"),
        "status": report.get("status") or "insufficient_data",
        "policy_version": report.get("policy_version") or report.get("monitoring_policy_version"),
        "recommended_action": report.get("recommended_action"),
        "alert_reasons": report.get("alert_reasons") or [],
        "warning_feature_names": report.get("warning_feature_names") or [],
        "n_warnings": report.get("n_warnings", report.get("n_warning_features")),
        "n_critical_findings": report.get("n_critical_findings", report.get("n_critical_features")),
        "n_monitored_features": report.get("n_monitored_features"),
        "reference_batch_size": report.get("reference_batch_size"),
        "batch_size": report.get("batch_size"),
        "report_complete": report.get("report_complete", True),
        "ground_truth_available": bool(report.get("ground_truth_available")),
        "reference_fingerprint": report.get("reference_fingerprint"),
        "current_fingerprint": report.get("current_fingerprint"),
        "champion_version": (report.get("champion") or {}).get("model_version"),
        "report_json": json_path.name,
        "report_markdown": md_path.name,
        "feature_csv": csv_path.name,
    }
    latest_payload = {
        **report,
        "artifact_names": {"json": json_path.name, "markdown": md_path.name, "csv": csv_path.name},
    }
    latest_path.write_text(json.dumps(latest_payload, indent=2, default=str), encoding="utf-8")
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote monitoring report %s", json_path.name)
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "latest": str(latest_path),
        "status": str(status_path),
    }


def load_latest_status() -> dict[str, Any]:
    cfg = get_config()
    status_path = (
        resolve_path(str(cfg.monitoring.get("report_dir", "artifacts/monitoring"))) / "status.json"
    )
    if not status_path.exists():
        return {
            "available": False,
            "status": "insufficient_data",
            "overall_severity": None,
            "message": "No monitoring report is available.",
            "report_complete": False,
            "ground_truth_available": False,
            "alert_reasons": [],
            "n_warnings": 0,
            "n_critical_findings": 0,
            "n_monitored_features": None,
            "reference_batch_size": None,
            "batch_size": None,
            "policy_version": None,
        }
    return cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))


def load_latest_report() -> dict[str, Any]:
    cfg = get_config()
    latest_path = (
        resolve_path(str(cfg.monitoring.get("report_dir", "artifacts/monitoring")))
        / "latest_report.json"
    )
    if not latest_path.exists():
        raise FileNotFoundError("Latest monitoring report is unavailable.")
    return cast(dict[str, Any], json.loads(latest_path.read_text(encoding="utf-8")))
