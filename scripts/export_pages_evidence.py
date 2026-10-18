"""Write the public Pages evidence bundle from local validation artifacts.

The command scores the registered champion on the validation split, writes
aggregate curves and threshold counts, and scores a few fictional inputs.
It does not evaluate the frozen test set or modify ``artifacts/``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
from src.config import PROJECT_ROOT, get_config
from src.explainability.linear import LinearExplanationModel, global_linear_explanation
from src.explainability.plots import plot_coefficient_bars
from src.explainability.semantics import SCORE_WARNING, score_metadata_from_champion
from src.mlops.integrity import config_fingerprint, sha256_file
from src.mlops.serving import load_champion
from src.mlops.tracking import current_git_commit
from src.models.inference import prepare_inference_frame
from src.models.train import predict_proba
from src.pages.curves import validation_curve_payload
from src.pages.metrics import confusion_counts, metrics_from_counts
from src.pages.plots import (
    plot_model_comparison,
    plot_threshold_tradeoff,
    plot_validation_calibration,
    plot_validation_pr,
    plot_validation_roc,
)
from src.pages.scenarios import FICTIONAL_SCENARIOS, REQUIRED_INPUTS, validate_scenario_specs
from src.pages.schema import (
    DATA_FILES,
    DISPLAY_NAMES,
    FIGURE_FILES,
    PRIMARY_FAMILIES,
    SCHEMA_VERSION,
    find_forbidden_substrings,
    find_row_level_keys,
)
from src.pages.selection import gate_reason, select_family
from src.pages.verify import verify_site

FROZEN_TEST_FILES = (
    PROJECT_ROOT / "artifacts" / "frozen_v1" / "metrics_test.json",
    PROJECT_ROOT / "artifacts" / "metrics_test.json",
)
SITE_DIR = PROJECT_ROOT / "site"
DATA_DIR = SITE_DIR / "data"
FIGURE_DIR = SITE_DIR / "figures"
HISTORICAL_NOTE = (
    "This export did not score the frozen test set and did not build test curves "
    "from these summary numbers."
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: None)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    row_keys = find_row_level_keys(payload)
    if row_keys:
        raise RuntimeError(f"Refusing to write row-level keys into {path.name}: {row_keys}")
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    forbidden = find_forbidden_substrings(text)
    if forbidden:
        raise RuntimeError(f"Refusing to write forbidden tokens into {path.name}: {forbidden}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _frozen_hashes() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in FROZEN_TEST_FILES
    }


def _signature(inputs: dict[str, Any]) -> tuple[str, ...]:
    items: list[str] = []
    for key in REQUIRED_INPUTS:
        value = inputs[key]
        if isinstance(value, float):
            items.append(f"{key}={value!r}")
        else:
            items.append(f"{key}={value}")
    return tuple(items)


def _dataset_signatures() -> dict[str, set[tuple[str, ...]]]:
    paths = {
        "train": PROJECT_ROOT / "data" / "processed" / "train.csv",
        "validation": PROJECT_ROOT / "data" / "processed" / "val.csv",
        "test": PROJECT_ROOT / "data" / "processed" / "test.csv",
    }
    found: dict[str, set[tuple[str, ...]]] = {}
    for split, path in paths.items():
        frame = pd.read_csv(path, usecols=list(REQUIRED_INPUTS))
        found[split] = {_signature(record) for record in frame.to_dict(orient="records")}
    return found


def _public_model(
    record: dict[str, Any], policy: dict[str, Any], champion_family: str
) -> dict[str, Any]:
    family = str(record["model_family"])
    role = (
        "primary"
        if family in PRIMARY_FAMILIES and str(record["run_name"]).startswith("compare_")
        else "variant"
    )
    metrics = record["validation_metrics"]
    reason = gate_reason(record, policy)
    cv_mean = record.get("cv_mean")
    cv_std = record.get("cv_std")
    return {
        "role": role,
        "model_family": family,
        "display_name": DISPLAY_NAMES.get(family, family),
        "run_id": str(record["run_id"]),
        "run_name": str(record["run_name"]),
        "selected": role == "primary" and family == champion_family,
        "eligible": reason is None,
        "ineligible_reason": reason,
        "hyperparameters": record.get("hyperparameters") or {},
        "class_weight": record.get("class_weight"),
        "calibration_status": record.get("calibration_status"),
        "decision_threshold": _finite(record.get("threshold")),
        "threshold_split": "validation",
        "training_fold_cv_pr_auc_mean": None if cv_mean is None else float(cv_mean),
        "training_fold_cv_pr_auc_std": None if cv_std is None else float(cv_std),
        "validation_pr_auc": float(metrics["val_pr_auc"]),
        "validation_roc_auc": float(metrics["val_roc_auc"]),
        "validation_precision_at_threshold": float(metrics["val_precision"]),
        "validation_recall_at_threshold": float(metrics["val_recall"]),
        "validation_f1_at_threshold": float(metrics["val_f1"]),
        "validation_brier": float(metrics["val_brier"]),
        "validation_ece": float(metrics["val_ece"]),
        "latency_p99_ms": float(metrics["latency_p99_ms"]),
        "roundtrip_ok": bool(record.get("roundtrip_ok")),
        "dataset_fingerprint": str(record.get("dataset_fingerprint")),
        "validation_metrics": {
            "val_pr_auc": float(metrics["val_pr_auc"]),
            "val_recall": float(metrics["val_recall"]),
            "val_brier": float(metrics["val_brier"]),
            "latency_p99_ms": float(metrics["latency_p99_ms"]),
        },
    }


def _historical_summary() -> dict[str, Any]:
    metrics_path = PROJECT_ROOT / "artifacts" / "frozen_v1" / "metrics_test.json"
    bootstrap_path = PROJECT_ROOT / "artifacts" / "audit" / "bootstrap_test_v1.csv"
    operational_path = PROJECT_ROOT / "artifacts" / "audit" / "operational_rates.csv"
    stored = _load_json(metrics_path)
    metrics = stored["metrics"]
    summary = {
        "pr_auc": float(metrics["pr_auc"]),
        "roc_auc": float(metrics["roc_auc"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "brier_score": float(metrics["brier_score"]),
        "true_positives": int(metrics["true_positives"]),
        "false_positives": int(metrics["false_positives"]),
        "false_negatives": int(metrics["false_negatives"]),
        "true_negatives": int(metrics["true_negatives"]),
        "n_samples": int(metrics["n_samples"]),
        "positive_rate": float(metrics["positive_rate"]),
        "threshold": float(metrics["threshold"]),
    }
    with bootstrap_path.open(encoding="utf-8", newline="") as handle:
        intervals = [
            {
                "metric": row["metric"],
                "point": float(row["point"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n_bootstrap": int(float(row["n_bootstrap"])),
                "seed": int(float(row["seed"])),
            }
            for row in csv.DictReader(handle)
        ]
    per_thousand: dict[str, Any] | None = None
    with operational_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test_characterisation" and row["is_selected_threshold"] == "True":
                per_thousand = {
                    "threshold": float(row["threshold"]),
                    "reviews_per_1000": float(row["reviews_per_1000"]),
                    "true_positives_per_1000": float(row["true_positives_per_1000"]),
                    "false_positives_per_1000": float(row["false_positives_per_1000"]),
                    "missed_positives_per_1000": float(row["missed_positives_per_1000"]),
                    "precision": float(row["precision_positive"]),
                    "recall": float(row["recall_positive"]),
                }
    if per_thousand is None:
        raise RuntimeError("Selected frozen-test operational row is missing")
    pr_point = next(row["point"] for row in intervals if row["metric"] == "pr_auc")
    if abs(pr_point - summary["pr_auc"]) > 1e-12:
        raise RuntimeError("Bootstrap PR-AUC does not match frozen test metrics")
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.historical_test",
        "evidence_class": "documented_historical",
        "reproduced_by_this_export": False,
        "curves_included": False,
        "validation_rows_included": False,
        "split": "test",
        "source_metrics_path": "artifacts/frozen_v1/metrics_test.json",
        "source_metrics_sha256": sha256_file(metrics_path),
        "source_bootstrap_path": "artifacts/audit/bootstrap_test_v1.csv",
        "source_bootstrap_sha256": sha256_file(bootstrap_path),
        "source_operational_path": "artifacts/audit/operational_rates.csv",
        "source_operational_sha256": sha256_file(operational_path),
        "threshold": summary["threshold"],
        "n_samples": summary["n_samples"],
        "metrics": summary,
        "bootstrap_intervals": intervals,
        "selected_threshold_per_1000": per_thousand,
        "documentation": [
            "docs/methodological_audit.md",
            "docs/model_card.md",
        ],
        "note": HISTORICAL_NOTE,
    }


def _coefficient_rows(table: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in table[:limit]:
        rows.append(
            {
                "transformed_feature": item["transformed_feature"],
                "display_name": item["display_name"],
                "source_feature": item["source_feature"],
                "kind": item["kind"],
                "coefficient_transformed": float(item["coefficient_transformed"]),
                "coefficient_original_unit": _finite(item.get("coefficient_original_unit")),
                "direction": item["direction"],
                "interpretation": item["interpretation"],
            }
        )
    return rows


def _trim_contributions(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for item in rows[:limit]:
        trimmed.append(
            {
                "display_name": item["display_name"],
                "source_feature": item["source_feature"],
                "kind": item["kind"],
                "log_odds_contribution": float(item["log_odds_contribution"]),
                "original_value": item["original_value"],
            }
        )
    return trimmed


def _score_examples(
    pipeline: Any, threshold: float, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_scenario_specs()
    signatures = _dataset_signatures()
    explainer = LinearExplanationModel.from_pipeline(pipeline)
    semantics = score_metadata_from_champion(metadata)
    examples: list[dict[str, Any]] = []
    for scenario in FICTIONAL_SCENARIOS:
        inputs = dict(scenario["inputs"])
        signature = _signature(inputs)
        collisions = [split for split, seen in signatures.items() if signature in seen]
        if collisions:
            raise RuntimeError(f"{scenario['example_id']} matches a {collisions[0]} row")
        explained = explainer.explain_row(
            prepare_inference_frame([inputs]),
            threshold=threshold,
            metadata=metadata,
            top_n=6,
        )
        contributions = explained["contributions"]
        total = float(sum(float(item["log_odds_contribution"]) for item in contributions))
        examples.append(
            {
                "example_id": scenario["example_id"],
                "title": scenario["title"],
                "description": scenario["description"],
                "precomputed_offline": True,
                "inputs": inputs,
                "review_score": float(explained["review_score"]),
                "decision_threshold": float(explained["decision_threshold"]),
                "flagged_for_review": bool(explained["requires_review"]),
                "intercept_log_odds": float(explained["intercept_log_odds"]),
                "contributions_log_odds_sum": total,
                "logit": float(explained["logit"]),
                "reconstruction_error": float(explained["reconstruction_error"]),
                "strongest_increasing": _trim_contributions(explained["strongest_increasing"], 6),
                "strongest_decreasing": _trim_contributions(explained["strongest_decreasing"], 6),
                "score_is_calibrated": False,
                "score_semantics": semantics["score_semantics"],
                "score_warning": SCORE_WARNING,
                "causation_disclaimer": explained["causation_disclaimer"],
            }
        )
    return examples


def _threshold_payload(
    labels: list[int],
    scores: list[float],
    *,
    selected_threshold: float,
    stored_curve: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    grid = [float(value) for value in np.linspace(0.05, 0.95, 37)]
    if len(stored_curve) != len(grid):
        raise RuntimeError("Stored validation threshold curve does not have 37 points")
    rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(grid):
        counts = confusion_counts(labels, scores, threshold)
        metrics = metrics_from_counts(**counts, beta=2.0)
        stored = stored_curve[index]
        if abs(float(stored["threshold"]) - threshold) > 1e-8:
            raise RuntimeError("Stored threshold grid does not match linspace(0.05, 0.95, 37)")
        if (
            metrics["precision_status"] == "defined"
            and abs(float(stored["precision"]) - float(metrics["precision"])) > 1e-9
        ):
            raise RuntimeError(
                f"Precision at {threshold} does not match the stored validation sweep"
            )
        if (
            metrics["recall_status"] == "defined"
            and abs(float(stored["recall"]) - float(metrics["recall"])) > 1e-9
        ):
            raise RuntimeError(f"Recall at {threshold} does not match the stored validation sweep")
        rows.append(
            {
                "grid_index": index,
                "threshold": threshold,
                "selected_operating_point": False,
                **metrics,
            }
        )
    operating_index = min(
        range(len(rows)), key=lambda index: abs(rows[index]["threshold"] - selected_threshold)
    )
    if abs(rows[operating_index]["threshold"] - selected_threshold) > 1e-8:
        raise RuntimeError("Selected threshold is not on the validation grid")
    rows[operating_index]["selected_operating_point"] = True
    operating = dict(rows[operating_index])
    return {
        "rows": rows,
        "operating_point": operating,
        "grid": {"minimum": 0.05, "maximum": 0.95, "size": 37, "method": "numpy.linspace"},
    }


def build_bundle() -> dict[str, Any]:
    """Create the in-memory public bundle from the local champion and validation split."""
    before = _frozen_hashes()
    config = get_config()
    champion = _load_json(PROJECT_ROOT / "artifacts" / "mlops" / "champion.json")
    records = _load_json(PROJECT_ROOT / "artifacts" / "mlops" / "experiment_records.json")
    validation_metrics = _load_json(PROJECT_ROOT / "artifacts" / "metrics_validation.json")
    comparison_file = _load_json(PROJECT_ROOT / "artifacts" / "model_comparison.json")
    metadata = _load_json(PROJECT_ROOT / "artifacts" / "model_metadata.json")
    policy = dict(champion["extras"]["policy"])
    bundle = load_champion()
    import mlflow

    champion_run = mlflow.MlflowClient().get_run(str(champion["mlflow_run_id"]))
    if bundle.metadata["mlflow_run_id"] != champion["mlflow_run_id"]:
        raise RuntimeError("Loaded champion run id does not match champion.json")

    validation = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val.csv")
    if len(validation) != int(metadata["n_val"]):
        raise RuntimeError("Validation row count does not match model metadata")
    labels = [int(value) for value in validation["requires_review"].tolist()]
    features = prepare_inference_frame(validation.to_dict(orient="records"))
    scores = [float(value) for value in predict_proba(bundle.pipeline, features)]
    curves = validation_curve_payload(pd.Series(labels).to_numpy(), pd.Series(scores).to_numpy())
    if abs(curves["pr_auc"] - float(validation_metrics["metrics"]["pr_auc"])) > 1e-12:
        raise RuntimeError("Champion validation PR-AUC does not match metrics_validation.json")
    if abs(curves["roc_auc"] - float(validation_metrics["metrics"]["roc_auc"])) > 1e-12:
        raise RuntimeError("Champion validation ROC-AUC does not match metrics_validation.json")
    if abs(curves["ece"] - float(champion["validation_metrics"]["val_ece"])) > 1e-9:
        raise RuntimeError("Champion validation ECE does not match champion.json")

    models = [_public_model(record, policy, str(champion["model_name"])) for record in records]
    families = [row["model_family"] for row in models if row["role"] == "primary"]
    if families != list(PRIMARY_FAMILIES):
        raise RuntimeError(f"Primary models are {families}, expected {PRIMARY_FAMILIES}")
    fingerprints = {row["dataset_fingerprint"] for row in models}
    if fingerprints != {str(champion["dataset_fingerprint"])}:
        raise RuntimeError("Experiment records do not share the champion dataset fingerprint")
    selection = select_family(models, policy)
    if selection.chosen_family != champion["model_name"]:
        raise RuntimeError("Policy replay did not select the registered champion")
    saved_rejections = list(champion["extras"].get("rejections") or [])
    if set(saved_rejections) != set(selection.rejections):
        raise RuntimeError("Replayed rejections do not match champion.json")

    logreg = next(
        row for row in models if row["model_family"] == "logreg" and row["role"] == "primary"
    )
    if abs(logreg["validation_pr_auc"] - curves["pr_auc"]) > 1e-12:
        raise RuntimeError("Comparison logistic regression PR-AUC does not match the champion")
    threshold_payload = _threshold_payload(
        labels,
        scores,
        selected_threshold=float(bundle.threshold),
        stored_curve=list(comparison_file["threshold"]["curve"]),
    )
    operating = threshold_payload["operating_point"]
    expected_counts = validation_metrics["detail"]["confusion_matrix"]
    if [
        operating["true_negatives"],
        operating["false_positives"],
        operating["false_negatives"],
        operating["true_positives"],
    ] != [
        int(expected_counts[0][0]),
        int(expected_counts[0][1]),
        int(expected_counts[1][0]),
        int(expected_counts[1][1]),
    ]:
        raise RuntimeError("Operating-point counts do not match metrics_validation.json")
    if abs(float(operating["precision"]) - logreg["validation_precision_at_threshold"]) > 1e-9:
        raise RuntimeError("Operating-point precision does not match the comparison table")

    run_note = None
    if logreg["run_id"] != champion["mlflow_run_id"]:
        run_note = (
            f"The registered champion MLflow run is {champion['mlflow_run_id']} "
            f"({champion_run.info.run_name}). The logistic regression row in this table is run "
            f"{logreg['run_id']} from artifacts/mlops/experiment_records.json. The two run ids differ. "
            "Their validation PR-AUC values and dataset fingerprint match. Curves, the threshold grid "
            "and the precomputed examples were produced from the registered champion on the validation split."
        )
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.model_comparison",
        "evidence_class": "locally_generated",
        "source_path": "artifacts/mlops/experiment_records.json",
        "source_sha256": sha256_file(
            PROJECT_ROOT / "artifacts" / "mlops" / "experiment_records.json"
        ),
        "experiment_name": str(config.mlops.get("experiment_name")),
        "dataset_fingerprint": str(champion["dataset_fingerprint"]),
        "validation_fingerprint": str(records[0]["val_fingerprint"]),
        "registered_champion_run_id": str(champion["mlflow_run_id"]),
        "registered_champion_run_name": str(champion_run.info.run_name),
        "run_identity_note": run_note,
        "policy": policy,
        "models": models,
        "selection": {
            "chosen_family": selection.chosen_family,
            "leader_family": selection.leader_family,
            "leader_validation_pr_auc": selection.leader_validation_pr_auc,
            "chosen_validation_pr_auc": selection.chosen_validation_pr_auc,
            "absolute_gap": abs(
                selection.leader_validation_pr_auc - selection.chosen_validation_pr_auc
            ),
            "indifference": selection.indifference,
            "simpler_preference_applied": selection.simpler_preference_applied,
            "rejections": list(selection.rejections),
            "champion_reason": str(champion["reason"]),
            "selection_metric": "validation_pr_auc",
        },
    }

    model_identity = {
        "name": str(champion["model_name"]),
        "version": str(champion["model_version"]),
        "mlflow_run_id": str(champion["mlflow_run_id"]),
        "run_name": str(champion_run.info.run_name),
        "threshold": float(bundle.threshold),
        "threshold_selection_method": str(champion["threshold_selection_method"]),
        "calibration_status": str(champion["calibration_status"]),
        "score_is_calibrated": False,
        "score_semantics": score_metadata_from_champion(champion)["score_semantics"],
        "score_warning": SCORE_WARNING,
    }
    curve_document = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.validation_curves",
        "evidence_class": "locally_generated",
        "source_split": "validation",
        "source_metrics_path": "artifacts/metrics_validation.json",
        "source_metrics_sha256": sha256_file(
            PROJECT_ROOT / "artifacts" / "metrics_validation.json"
        ),
        "model": model_identity,
        **curves,
    }
    grid_document = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.threshold_grid",
        "evidence_class": "locally_generated",
        "split": "validation",
        "test_split_used": False,
        "n_rows": int(curves["n_rows"]),
        "positive_labels": int(curves["positive_labels"]),
        "model": model_identity,
        "source_sweep_path": "artifacts/model_comparison.json",
        "source_sweep_sha256": sha256_file(PROJECT_ROOT / "artifacts" / "model_comparison.json"),
        **threshold_payload,
    }

    live_explanation = global_linear_explanation(bundle.pipeline, top_n=15)
    saved_coefficients = _load_json(
        PROJECT_ROOT / "artifacts" / "explanations" / "global_coefficients.json"
    )
    coefficient_match = (
        abs(
            float(saved_coefficients["intercept_log_odds"])
            - float(live_explanation["intercept_log_odds"])
        )
        <= 1e-8
    )
    if coefficient_match:
        saved_top = saved_coefficients["ranked_coefficients"][0]
        live_top = live_explanation["ranked_coefficients"][0]
        coefficient_match = (
            saved_top["transformed_feature"] == live_top["transformed_feature"]
            and abs(
                float(saved_top["coefficient_transformed"])
                - float(live_top["coefficient_transformed"])
            )
            <= 1e-8
        )
    permutation = _load_json(
        PROJECT_ROOT / "artifacts" / "explanations" / "permutation_importance.json"
    )
    explanations = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.explanations",
        "evidence_class": "locally_generated",
        "model": model_identity,
        "score_is_calibrated": False,
        "intercept_log_odds": float(live_explanation["intercept_log_odds"]),
        "coefficient_caveat": live_explanation["coefficient_caveat"],
        "causation_disclaimer": live_explanation["causation_disclaimer"],
        "saved_coefficient_artifact_matches_champion": coefficient_match,
        "saved_coefficient_path": "artifacts/explanations/global_coefficients.json",
        "saved_coefficient_sha256": sha256_file(
            PROJECT_ROOT / "artifacts" / "explanations" / "global_coefficients.json"
        ),
        "top_coefficients": _coefficient_rows(live_explanation["ranked_coefficients"], 15),
        "grouped_original_features": [
            {
                "source_feature": row["source_feature"],
                "l2_transformed_coefficient": float(row["l2_transformed_coefficient"]),
                "n_transformed_features": int(row["n_transformed_features"]),
            }
            for row in live_explanation["grouped_original_features"][:8]
        ],
        "permutation_importance": {
            "evidence_class": "locally_generated_artifact",
            "recomputed_by_this_export": False,
            "source_path": "artifacts/explanations/permutation_importance.json",
            "source_sha256": sha256_file(
                PROJECT_ROOT / "artifacts" / "explanations" / "permutation_importance.json"
            ),
            "source_mlflow_run_id": str(permutation["mlflow_run_id"]),
            "split": str(permutation["split"]),
            "scoring": str(permutation["scoring"]),
            "n_repeats": int(permutation["n_repeats"]),
            "rows": [
                {
                    "feature": row["feature"],
                    "importance_mean": float(row["importance_mean"]),
                    "importance_std": float(row["importance_std"]),
                }
                for row in permutation["rows"][:10]
            ],
            "note": (
                "Permutation importance is the saved validation artifact. "
                "This export did not shuffle features again."
            ),
        },
    }
    examples = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.examples",
        "evidence_class": "precomputed_fictional",
        "scored_live_in_browser": False,
        "split": "not_a_dataset_split",
        "model": model_identity,
        "collision_check": {
            "splits_compared_for_exact_input_match": ["train", "validation", "test"],
            "matches_found": 0,
            "test_split_scored": False,
        },
        "examples": _score_examples(bundle.pipeline, float(bundle.threshold), bundle.metadata),
    }
    historical = _historical_summary()
    if _frozen_hashes() != before:
        raise RuntimeError("A frozen test artifact changed while the bundle was built")
    return {
        "model_comparison.json": comparison,
        "validation_curves.json": curve_document,
        "threshold_grid.json": grid_document,
        "explanations.json": explanations,
        "examples.json": examples,
        "historical_test.json": historical,
        "_model": model_identity,
        "_random_seed": int(config.random_seed),
        "_n_train": int(metadata["n_train"]),
        "_n_val": int(metadata["n_val"]),
        "_training_commit": str(champion.get("git_commit")),
    }


def write_bundle(site_dir: Path = SITE_DIR) -> None:
    """Write ``site/data`` and ``site/figures``, then verify the public site."""
    before = _frozen_hashes()
    built = build_bundle()
    model = built.pop("_model")
    n_train = int(built.pop("_n_train"))
    n_val = int(built.pop("_n_val"))
    training_commit = str(built.pop("_training_commit"))
    random_seed = int(built.pop("_random_seed"))
    data_dir = site_dir / "data"
    figure_dir = site_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in built.items():
        _write_json(data_dir / name, payload)

    curves = built["validation_curves.json"]
    comparison = built["model_comparison.json"]
    grid = built["threshold_grid.json"]
    explanations = built["explanations.json"]
    plot_validation_pr(curves, figure_dir / "validation_pr_curve.png")
    plot_validation_roc(curves, figure_dir / "validation_roc_curve.png")
    plot_validation_calibration(curves, figure_dir / "validation_calibration_curve.png")
    plot_model_comparison(
        [row for row in comparison["models"] if row["role"] == "primary"],
        figure_dir / "model_comparison_pr_auc.png",
    )
    plot_threshold_tradeoff(
        grid["rows"],
        float(grid["operating_point"]["threshold"]),
        figure_dir / "validation_threshold_tradeoff.png",
    )
    plot_coefficient_bars(
        pd.DataFrame(explanations["top_coefficients"]),
        figure_dir / "champion_coefficients.png",
        top_n=12,
    )

    source_artifacts = [
        ("artifacts/mlops/champion.json", "registered champion"),
        ("artifacts/mlops/experiment_records.json", "same-experiment comparison records"),
        ("artifacts/metrics_validation.json", "validation metric summary"),
        ("artifacts/model_comparison.json", "stored validation threshold sweep"),
        ("artifacts/model_metadata.json", "training metadata"),
        ("artifacts/explanations/global_coefficients.json", "saved coefficient table"),
        ("artifacts/explanations/permutation_importance.json", "saved permutation importance"),
        ("artifacts/frozen_v1/metrics_test.json", "historical test summary"),
        ("artifacts/audit/bootstrap_test_v1.csv", "historical bootstrap intervals"),
        ("artifacts/audit/operational_rates.csv", "historical per-1000 test summary"),
        ("configs/default.yaml", "training configuration"),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "pages.manifest",
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts.export_pages_evidence",
        "project_subpath": "Cargo-Risk-ML-Lab",
        "model": model,
        "source_split_for_curves": "validation",
        "historical_test_reproduced": False,
        "configuration": {
            "config_path": "configs/default.yaml",
            "config_sha256": config_fingerprint(),
            "random_seed": random_seed,
            "n_train": n_train,
            "n_validation": n_val,
            "cv_folds": 3,
            "threshold_grid": {
                "minimum": 0.05,
                "maximum": 0.95,
                "size": 37,
                "method": "numpy.linspace",
            },
            "selection_metric": "validation_pr_auc",
            "policy_version": str(comparison["policy"]["version"]),
        },
        "source_artifacts": [
            {"path": relative, "sha256": sha256_file(PROJECT_ROOT / relative), "role": role}
            for relative, role in source_artifacts
        ],
        "files": {
            name: {"sha256": sha256_file(data_dir / name), "schema_name": schema_name}
            for name, schema_name in DATA_FILES.items()
            if name != "manifest.json"
        },
        "figures": {name: {"sha256": sha256_file(figure_dir / name)} for name in FIGURE_FILES},
        "training_git_commit": training_commit,
        "export_git_commit": current_git_commit(),
        "did_not": [
            "evaluate the frozen test set",
            "retune the threshold",
            "overwrite artifacts/frozen_v1",
            "export raw dataset rows",
            "export row-level prediction arrays",
            "export model serialization",
        ],
    }
    _write_json(data_dir / "manifest.json", manifest)
    if _frozen_hashes() != before:
        raise RuntimeError("A frozen test artifact changed while figures were written")
    verify_site(site_dir)


def main() -> None:
    write_bundle()
    print("pages_evidence_ok")


if __name__ == "__main__":
    main()
