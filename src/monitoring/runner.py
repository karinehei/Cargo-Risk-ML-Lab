"""Monitoring orchestration for unlabelled drift and labelled simulation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config, resolve_path, setup_logging
from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation.metrics import compute_classification_metrics
from src.features import prepare_xy
from src.mlops.fingerprints import dataframe_fingerprint
from src.mlops.serving import load_champion
from src.models import predict_proba
from src.monitoring.metrics import (
    categorical_feature_metrics,
    default_thresholds,
    numeric_feature_metrics,
    score_distribution_metrics,
)
from src.monitoring.policy import (
    MONITORING_POLICY_VERSION,
    PolicyStatus,
    apply_policy,
    recommended_action_for_status,
)
from src.monitoring.reference import (
    build_reference_profile,
    load_reference_profile,
    save_reference_profile,
)
from src.monitoring.report import load_latest_status, write_monitoring_report
from src.monitoring.scenarios import (
    SCENARIO_SEEDS,
    ScenarioName,
    generate_monitoring_batch,
    load_monitoring_batch,
    save_monitoring_batch,
    scenario_metadata,
)

logger = setup_logging(name="src.monitoring.runner")

Mode = Literal["unlabelled_monitoring", "labelled_simulation"]

LIMITATIONS = [
    "Input drift does not automatically imply model failure or performance degradation.",
    "Large batches can make small differences statistically significant; effect-size thresholds are primary.",
    "Ground-truth labels are usually delayed and are unavailable in default unlabelled monitoring.",
    "This educational pipeline must not trigger automatic retraining, threshold changes or shipment blocking.",
]


def _reference_sample_path(config: AppConfig) -> str:
    return str(
        config.monitoring.get("reference_sample_path", "data/monitoring/reference_sample.csv")
    )


def _load_train_reference_frame(
    config: AppConfig,
    *,
    use_saved_sample: bool = True,
) -> pd.DataFrame:
    sample_path = resolve_path(_reference_sample_path(config))
    if use_saved_sample and sample_path.exists():
        frame = pd.read_csv(sample_path)
    else:
        train_path = resolve_path(
            str(config.monitoring.get("reference_dataset_path", "data/processed/train.csv"))
        )
        if not train_path.exists():
            raise FileNotFoundError(
                "Train reference dataset is unavailable. Run `make generate-data` first."
            )
        frame = pd.read_csv(train_path)
        sample_size = int(config.monitoring.get("reference_sample_size", min(5000, len(frame))))
        seed = int(config.monitoring.get("reference_seed", config.random_seed))
        frame = frame.sample(n=min(sample_size, len(frame)), random_state=seed)
    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])
    drop_cols = [
        col for col in ("shipment_id", "event_date", "generation_period") if col in frame.columns
    ]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    feature_cols = [col for col in FEATURE_COLUMNS if col in frame.columns]
    return frame[feature_cols].reset_index(drop=True)


def _persist_reference_sample(frame: pd.DataFrame, config: AppConfig) -> str:
    path = resolve_path(_reference_sample_path(config))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def create_reference_profile(config: AppConfig | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    bundle = load_champion()
    sample = _load_train_reference_frame(cfg, use_saved_sample=False)
    _persist_reference_sample(sample, cfg)
    scoring_frame = sample.copy()
    scoring_frame[TARGET_COLUMN] = 0
    x_ref, _ = prepare_xy(scoring_frame, fit_derived_reference=sample)
    scores = predict_proba(bundle.pipeline, x_ref)
    profile = build_reference_profile(
        sample,
        champion_metadata=bundle.metadata,
        scores=scores,
        threshold=float(bundle.threshold),
        seed=int(cfg.monitoring.get("reference_seed", cfg.random_seed)),
        source="train_reference_sample",
        config=cfg,
    )
    save_reference_profile(
        profile,
        path=str(cfg.monitoring.get("reference_profile_path")),
        config=cfg,
    )
    return profile


def generate_scenario_batch(
    scenario: ScenarioName,
    *,
    config: AppConfig | None = None,
    include_labels: bool = False,
) -> dict[str, Any]:
    cfg = config or get_config()
    seed = SCENARIO_SEEDS[str(scenario)]
    frame = generate_monitoring_batch(scenario, config=cfg, include_labels=include_labels)
    path = save_monitoring_batch(frame, scenario, config=cfg)
    meta = scenario_metadata(scenario, frame, seed)
    meta["path"] = path
    return meta


def _champion_payload(bundle: Any) -> dict[str, Any]:
    return {
        "model_name": str(bundle.metadata.get("model_name") or ""),
        "model_version": str(bundle.metadata.get("model_version") or ""),
        "mlflow_run_id": str(bundle.metadata.get("mlflow_run_id") or ""),
        "decision_threshold": float(bundle.threshold),
    }


def _score_batch(
    frame: pd.DataFrame,
    *,
    pipeline: Any | None = None,
    champion: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    if pipeline is None or champion is None:
        bundle = load_champion()
        pipeline = bundle.pipeline
        champion = _champion_payload(bundle)
    scoring_frame = frame.copy()
    if TARGET_COLUMN not in scoring_frame.columns:
        scoring_frame[TARGET_COLUMN] = 0
    x_batch, _ = prepare_xy(scoring_frame, fit_derived_reference=scoring_frame)
    scores = predict_proba(pipeline, x_batch)
    return scores, champion, pipeline


def _validate_batch(frame: pd.DataFrame, config: AppConfig) -> None:
    minimum = int(config.monitoring.get("min_batch_size", 50))
    if len(frame) < minimum:
        raise ValueError(f"Monitoring batch size {len(frame)} is below minimum {minimum}.")


def _incomplete_report(
    *,
    status: PolicyStatus,
    message: str,
    mode: Mode,
    scenario: ScenarioName,
    config: AppConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "monitoring_run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode,
        "scenario": scenario,
        "status": status,
        "overall_severity": None,
        "available": False,
        "report_complete": False,
        "ground_truth_available": False,
        "policy_version": str(config.monitoring.get("policy_version") or MONITORING_POLICY_VERSION),
        "recommended_action": recommended_action_for_status(status),
        "alert_reasons": [],
        "n_warnings": 0,
        "n_critical_findings": 0,
        "message": message,
        "disclaimer": config.disclaimer,
    }
    if extra:
        report.update(extra)
    return report


def _build_feature_metrics(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    config: AppConfig,
) -> list[dict[str, Any]]:
    threshold_cfg = dict(config.monitoring.get("severity_thresholds") or {})
    thresholds = default_thresholds(threshold_cfg)
    n_bins = int(config.monitoring.get("psi_bins", 10))
    numeric_cols = list(config.features.get("numeric", []))
    categorical_cols = list(config.features.get("categorical", []))
    rows: list[dict[str, Any]] = []
    for col in numeric_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        rows.append(
            numeric_feature_metrics(
                reference[col],
                current[col],
                thresholds=thresholds,
                n_bins=n_bins,
            )
        )
    for col in categorical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        rows.append(
            categorical_feature_metrics(
                reference[col],
                current[col],
                thresholds=thresholds,
            )
        )
    return rows


def _score_drift_row(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    threshold: float,
    config: AppConfig,
) -> dict[str, Any]:
    threshold_cfg = dict(config.monitoring.get("severity_thresholds") or {})
    thresholds = default_thresholds(threshold_cfg)
    n_bins = int(config.monitoring.get("psi_bins", 10))
    metrics = score_distribution_metrics(
        reference_scores,
        current_scores,
        threshold=threshold,
        n_bins=n_bins,
    )
    from src.monitoring.metrics import _max_severity, _severity

    psi_sev = _severity(
        float(metrics["psi"]),
        thresholds["score_psi"]["warning"],
        thresholds["score_psi"]["critical"],
    )
    rate_sev = _severity(
        abs(float(metrics["predicted_review_rate_change"])),
        thresholds["predicted_review_rate_delta"]["warning"],
        thresholds["predicted_review_rate_delta"]["critical"],
    )
    ks_sev = _severity(
        float(metrics["ks_statistic"]),
        thresholds["ks"]["warning"],
        thresholds["ks"]["critical"],
    )
    metrics["severity"] = _max_severity(psi_sev, rate_sev, ks_sev)
    metrics["concept"] = "review_score_and_decision_drift"
    return metrics


def evaluate_comparison(
    current: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame,
    reference_scores: np.ndarray,
    champion: dict[str, Any],
    config: AppConfig,
    scenario: ScenarioName,
    seed: int,
    mode: Mode = "unlabelled_monitoring",
    policy_version: str = MONITORING_POLICY_VERSION,
    previous_status: str | None = None,
    previous_warning_names: list[str] | None = None,
    write: bool = True,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Compare a current batch to the reference without retraining."""
    if mode == "unlabelled_monitoring" and TARGET_COLUMN in current.columns:
        current = current.drop(columns=[TARGET_COLUMN])
    feature_cols = [col for col in FEATURE_COLUMNS if col in current.columns]
    current_features = current[feature_cols]
    cur_scores, _, pipeline = _score_batch(current_features, pipeline=pipeline, champion=champion)
    feature_metrics = _build_feature_metrics(reference_frame, current_features, config=config)
    score_drift = _score_drift_row(
        reference_scores,
        cur_scores,
        threshold=float(champion["decision_threshold"]),
        config=config,
    )
    thresholds = default_thresholds(dict(config.monitoring.get("severity_thresholds") or {}))
    policy = apply_policy(
        feature_metrics,
        score_drift,
        thresholds=thresholds,
        min_warning_features=int(config.monitoring.get("min_warning_features", 2)),
        previous_status=previous_status,
        previous_warning_names=previous_warning_names,
        policy_version=policy_version,
    )
    try:
        reference_profile = load_reference_profile()
        reference_fingerprint = str(reference_profile.get("dataset_fingerprint") or "")
    except FileNotFoundError:
        from src.mlops.fingerprints import dataframe_fingerprint as _fp

        reference_fingerprint = _fp(reference_frame)
        reference_profile = {}
    report: dict[str, Any] = {
        "monitoring_run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode,
        "scenario": scenario,
        "seed": seed,
        "monitoring_policy_version": policy_version,
        "policy_version": policy_version,
        "status": policy["status"],
        "overall_severity": policy["overall_severity"],
        "recommended_action": policy["recommended_action"],
        "alert_reasons": policy["alert_reasons"],
        "isolated_weak_warning": policy["isolated_weak_warning"],
        "warning_feature_names": policy["warning_feature_names"],
        "n_alert_reasons": policy["n_alert_reasons"],
        "n_warning_features": policy["n_warning_features"],
        "n_critical_features": policy["n_critical_features"],
        "reference_fingerprint": reference_fingerprint,
        "current_fingerprint": dataframe_fingerprint(current_features),
        "reference_batch_size": int(len(reference_frame)),
        "batch_size": int(len(current_features)),
        "n_monitored_features": int(len(feature_metrics)),
        "n_warnings": int(policy["n_warning_features"]) + int(policy["n_score_warnings"]),
        "n_critical_findings": int(policy["n_critical_features"]),
        "champion": champion,
        "feature_metrics": feature_metrics,
        "score_drift": score_drift,
        "input_data_drift": {
            "n_features_checked": len(feature_metrics),
            "n_warning_or_critical": sum(1 for row in feature_metrics if row["severity"] != "none"),
        },
        "limitations": LIMITATIONS,
        "disclaimer": config.disclaimer,
        "ground_truth_available": False,
        "report_complete": True,
    }
    if write:
        paths = write_monitoring_report(
            report, output_dir=str(config.monitoring.get("report_dir", "artifacts/monitoring"))
        )
        report["artifact_paths"] = paths
    return report


def _previous_window(config: AppConfig, scenario: str) -> tuple[str | None, list[str] | None]:
    status = load_latest_status()
    if not status.get("available"):
        return None, None
    if str(status.get("scenario") or "") != str(scenario):
        return None, None
    names = status.get("warning_feature_names")
    if not isinstance(names, list):
        names = None
    return (
        str(status.get("status") or "") or None,
        [str(item) for item in names] if names else None,
    )


def run_monitoring(
    scenario: ScenarioName,
    *,
    mode: Mode = "unlabelled_monitoring",
    config: AppConfig | None = None,
    current: pd.DataFrame | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run monitoring for a scenario against the saved reference profile."""
    cfg = config or get_config()
    try:
        try:
            reference_profile = load_reference_profile()
        except FileNotFoundError:
            report = _incomplete_report(
                status="insufficient_data",
                message="Reference profile is unavailable.",
                mode=mode,
                scenario=scenario,
                config=cfg,
            )
            if write:
                paths = write_monitoring_report(
                    report, output_dir=str(cfg.monitoring.get("report_dir", "artifacts/monitoring"))
                )
                report["artifact_paths"] = paths
            return report
        reference_frame = _load_train_reference_frame(cfg)
        if current is None:
            try:
                current = load_monitoring_batch(scenario, config=cfg)
            except FileNotFoundError:
                report = _incomplete_report(
                    status="insufficient_data",
                    message="Current monitoring batch is unavailable.",
                    mode=mode,
                    scenario=scenario,
                    config=cfg,
                )
                if write:
                    paths = write_monitoring_report(
                        report,
                        output_dir=str(cfg.monitoring.get("report_dir", "artifacts/monitoring")),
                    )
                    report["artifact_paths"] = paths
                return report
        try:
            _validate_batch(current, cfg)
        except ValueError as exc:
            report = _incomplete_report(
                status="insufficient_data",
                message=str(exc),
                mode=mode,
                scenario=scenario,
                config=cfg,
                extra={"batch_size": int(len(current))},
            )
            if write:
                paths = write_monitoring_report(
                    report, output_dir=str(cfg.monitoring.get("report_dir", "artifacts/monitoring"))
                )
                report["artifact_paths"] = paths
            return report
        ref_scores, champion, pipeline = _score_batch(reference_frame)
        previous_status, previous_names = _previous_window(cfg, str(scenario))
        report = evaluate_comparison(
            current,
            reference_frame=reference_frame,
            reference_scores=ref_scores,
            champion=champion,
            config=cfg,
            scenario=scenario,
            seed=int(SCENARIO_SEEDS.get(str(scenario), 0)),
            mode=mode,
            policy_version=str(cfg.monitoring.get("policy_version") or MONITORING_POLICY_VERSION),
            previous_status=previous_status,
            previous_warning_names=previous_names,
            write=False,
            pipeline=pipeline,
        )
        report["monitoring_policy_version"] = report["policy_version"]
        _ = reference_profile
        if mode == "labelled_simulation":
            if TARGET_COLUMN not in current.columns:
                labelled = generate_monitoring_batch(scenario, config=cfg, include_labels=True)
            else:
                labelled = current
            y_true = labelled[TARGET_COLUMN].astype(int).to_numpy()
            scoring = labelled.copy()
            x_labelled, _ = prepare_xy(scoring, fit_derived_reference=scoring)
            y_prob = predict_proba(load_champion().pipeline, x_labelled)
            metrics = compute_classification_metrics(
                y_true,
                y_prob,
                threshold=float(champion["decision_threshold"]),
            )
            report["ground_truth_available"] = True
            report["simulated_performance"] = {
                "pr_auc": metrics.get("average_precision"),
                "roc_auc": metrics.get("roc_auc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "brier_score": metrics.get("brier_score"),
                "predicted_review_rate": float(np.mean(y_prob >= champion["decision_threshold"])),
                "positive_rate": float(np.mean(y_true)),
                "review_workload_per_1000": float(
                    np.mean(y_prob >= champion["decision_threshold"]) * 1000.0
                ),
                "note": "Synthetic labelled simulation only. Not production monitoring evidence.",
            }
        if write:
            paths = write_monitoring_report(
                report, output_dir=str(cfg.monitoring.get("report_dir", "artifacts/monitoring"))
            )
            report["artifact_paths"] = paths
        return report
    except Exception as exc:  # noqa: BLE001
        logger.warning("Monitoring computation failed: %s", type(exc).__name__)
        error_report = _incomplete_report(
            status="monitoring_error",
            message="Monitoring computation failed.",
            mode=mode,
            scenario=scenario,
            config=cfg,
        )
        if write:
            paths = write_monitoring_report(
                error_report,
                output_dir=str(cfg.monitoring.get("report_dir", "artifacts/monitoring")),
            )
            error_report["artifact_paths"] = paths
        return error_report


def show_latest_status() -> dict[str, Any]:
    return load_latest_status()
