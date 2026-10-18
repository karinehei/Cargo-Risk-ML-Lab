"""Integrity checks for the public Pages bundle.

The verifier uses the committed JSON and figures. It does not load MLflow,
score data, or read the frozen test file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from src.pages.metrics import counts_close, metrics_from_counts, sigmoid
from src.pages.schema import (
    DATA_FILES,
    FIGURE_FILES,
    GITHUB_REPOSITORY,
    SCHEMA_VERSION,
    find_forbidden_substrings,
    find_row_level_keys,
    unexpected_site_files,
)
from src.pages.selection import select_family

ASSET_ATTR = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
REMOTE_TOKENS = (
    "localhost",
    "127.0.0.1",
    "cdn.jsdelivr",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "googleapis.com",
    "jsdelivr.net",
)


class VerificationError(ValueError):
    """The public bundle or static site failed a check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> tuple[str, Any]:
    text = path.read_text(encoding="utf-8")
    forbidden = find_forbidden_substrings(text)
    if forbidden:
        raise VerificationError(f"{path.name} contains forbidden public tokens: {forbidden}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{path.name} is not valid JSON") from exc
    row_keys = find_row_level_keys(payload)
    if row_keys:
        raise VerificationError(f"{path.name} contains row-level keys: {row_keys}")
    if _contains_fixture_flag(payload):
        raise VerificationError(f"{path.name} is labelled as a development fixture")
    return text, payload


def _contains_fixture_flag(payload: object) -> bool:
    if isinstance(payload, dict):
        if payload.get("development_fixture") is True:
            return True
        return any(_contains_fixture_flag(item) for item in payload.values())
    if isinstance(payload, list):
        return any(_contains_fixture_flag(item) for item in payload)
    return False


def _require_mapping(payload: object, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VerificationError(f"{name} must be a JSON object")
    return payload


def _close(left: object, right: object, tol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tol  # type: ignore[arg-type]


def verify_bundle(site_dir: Path) -> dict[str, Any]:
    """Check hashes, provenance and numerical consistency of ``site/data``."""
    unexpected = unexpected_site_files(site_dir)
    if unexpected:
        raise VerificationError(f"Unexpected site files: {unexpected}")
    data_dir = site_dir / "data"
    figure_dir = site_dir / "figures"
    if not data_dir.is_dir():
        raise VerificationError("site/data is missing")

    documents: dict[str, dict[str, Any]] = {}
    for name, schema_name in DATA_FILES.items():
        path = data_dir / name
        if not path.is_file():
            raise VerificationError(f"Missing public evidence file data/{name}")
        _text, payload = _load(path)
        document = _require_mapping(payload, name)
        if document.get("schema_version") != SCHEMA_VERSION:
            raise VerificationError(f"{name} schema_version is not {SCHEMA_VERSION}")
        if document.get("schema_name") != schema_name:
            raise VerificationError(f"{name} schema_name is not {schema_name}")
        documents[name] = document

    manifest = documents["manifest.json"]
    _verify_manifest_hashes(site_dir, manifest)
    _verify_model_identity(documents)
    _verify_comparison(documents["model_comparison.json"], manifest)
    _verify_curves(documents["validation_curves.json"], documents["model_comparison.json"])
    _verify_threshold(
        documents["threshold_grid.json"], documents["model_comparison.json"], manifest
    )
    _verify_historical(documents["historical_test.json"], documents["validation_curves.json"])
    _verify_examples(documents["examples.json"], manifest)
    _verify_explanations(documents["explanations.json"], manifest)
    for name in FIGURE_FILES:
        if not (figure_dir / name).is_file():
            raise VerificationError(f"Missing figure figures/{name}")
    return manifest


def _verify_manifest_hashes(site_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    figures = manifest.get("figures")
    if not isinstance(files, dict) or not isinstance(figures, dict):
        raise VerificationError("Manifest is missing file hashes")
    for name in DATA_FILES:
        if name == "manifest.json":
            continue
        recorded = files.get(name, {}).get("sha256") if isinstance(files.get(name), dict) else None
        if not isinstance(recorded, str) or len(recorded) != 64:
            raise VerificationError(f"Manifest hash missing for {name}")
        observed = _sha256(site_dir / "data" / name)
        if observed != recorded:
            raise VerificationError(f"Hash mismatch for data/{name}")
    for name in FIGURE_FILES:
        recorded = (
            figures.get(name, {}).get("sha256") if isinstance(figures.get(name), dict) else None
        )
        if not isinstance(recorded, str) or len(recorded) != 64:
            raise VerificationError(f"Manifest hash missing for {name}")
        observed = _sha256(site_dir / "figures" / name)
        if observed != recorded:
            raise VerificationError(f"Hash mismatch for figures/{name}")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise VerificationError("Manifest model identity is missing")
    for field in ("name", "version", "mlflow_run_id", "threshold", "score_is_calibrated"):
        if field not in model:
            raise VerificationError(f"Manifest model is missing {field}")
    if model.get("score_is_calibrated") is not False:
        raise VerificationError("Published champion must record score_is_calibrated false")
    if manifest.get("historical_test_reproduced") is not False:
        raise VerificationError("Manifest must record that the historical test was not reproduced")
    if manifest.get("source_split_for_curves") != "validation":
        raise VerificationError("Curve split must be validation")
    generated = manifest.get("generated_at")
    if not isinstance(generated, str) or "T" not in generated:
        raise VerificationError("Manifest generated_at is missing")


def _verify_model_identity(documents: dict[str, dict[str, Any]]) -> None:
    run_id = documents["manifest.json"]["model"]["mlflow_run_id"]
    for name in (
        "validation_curves.json",
        "threshold_grid.json",
        "examples.json",
        "explanations.json",
    ):
        model = documents[name].get("model")
        if not isinstance(model, dict) or model.get("mlflow_run_id") != run_id:
            raise VerificationError(f"{name} model run id does not match the manifest")


def _verify_comparison(comparison: dict[str, Any], manifest: dict[str, Any]) -> None:
    if comparison.get("evidence_class") != "locally_generated":
        raise VerificationError("Model comparison evidence class is wrong")
    models = comparison.get("models")
    policy = comparison.get("policy")
    selection = comparison.get("selection")
    if (
        not isinstance(models, list)
        or not isinstance(policy, dict)
        or not isinstance(selection, dict)
    ):
        raise VerificationError("Model comparison is missing models, policy or selection")
    primary = [row for row in models if row.get("role") == "primary"]
    families = [row.get("model_family") for row in primary]
    if families != ["dummy", "logreg", "random_forest", "xgboost"]:
        raise VerificationError("Primary comparison families are missing or reordered")
    fingerprints = {row.get("dataset_fingerprint") for row in models}
    if len(fingerprints) != 1:
        raise VerificationError("Comparison models do not share a dataset fingerprint")
    selected = [row for row in models if row.get("selected") is True]
    if len(selected) != 1:
        raise VerificationError("Exactly one selected model is required")
    result = select_family(models, policy)
    if result.chosen_family != selected[0]["model_family"]:
        raise VerificationError("Stored selection does not match the champion policy")
    if result.chosen_family != selection.get("chosen_family"):
        raise VerificationError("Selection block does not match the policy result")
    if result.chosen_family != manifest["model"]["name"]:
        raise VerificationError("Selected family does not match the manifest model")
    if not _close(selected[0]["validation_pr_auc"], selection.get("chosen_validation_pr_auc")):
        raise VerificationError("Selected PR-AUC does not match the selection block")
    best_primary = max(float(row["validation_pr_auc"]) for row in primary)
    if float(selected[0]["validation_pr_auc"]) + 1e-12 < best_primary:
        raise VerificationError("Selected primary model is not the validation PR-AUC leader")
    stored_rejections = list(selection.get("rejections") or [])
    if set(stored_rejections) != set(result.rejections):
        raise VerificationError("Stored policy rejections do not match the records")
    if selection.get("leader_family") != result.leader_family:
        raise VerificationError("Stored PR-AUC leader does not match the policy")
    if bool(selection.get("simpler_preference_applied")) != result.simpler_preference_applied:
        raise VerificationError("Simplicity preference flag does not match the policy")
    gap = abs(
        float(selection["leader_validation_pr_auc"]) - float(selection["chosen_validation_pr_auc"])
    )
    if not _close(gap, selection.get("absolute_gap")):
        raise VerificationError("Selection gap does not match the recorded PR-AUC values")
    for row in models:
        reason = row.get("ineligible_reason")
        if row.get("eligible") is True and reason not in (None, ""):
            raise VerificationError("Eligible model carries an ineligible reason")
        if row.get("eligible") is False and not reason:
            raise VerificationError("Ineligible model is missing a reason")


def _verify_curves(curves: dict[str, Any], comparison: dict[str, Any]) -> None:
    if curves.get("evidence_class") != "locally_generated" or curves.get("split") != "validation":
        raise VerificationError("Validation curves must be local validation evidence")
    selected = next(row for row in comparison["models"] if row.get("selected") is True)
    if not _close(curves.get("pr_auc"), selected.get("validation_pr_auc"), tol=1e-12):
        raise VerificationError("Curve PR-AUC does not match the selected model")
    if not _close(curves.get("roc_auc"), selected.get("validation_roc_auc"), tol=1e-12):
        raise VerificationError("Curve ROC-AUC does not match the selected model")
    roc = curves.get("roc") or {}
    fpr = roc.get("false_positive_rate") or []
    tpr = roc.get("true_positive_rate") or []
    if len(fpr) < 2 or abs(float(fpr[0])) > 1e-9 or abs(float(tpr[-1]) - 1.0) > 1e-9:
        raise VerificationError("ROC curve endpoints are not the unit square")
    pr = curves.get("precision_recall") or {}
    if len(pr.get("recall") or []) < 2:
        raise VerificationError("Precision-recall curve is missing")
    bins = curves.get("calibration_bins")
    if not isinstance(bins, list) or not bins:
        raise VerificationError("Calibration bins are missing")
    counted = sum(int(row["count"]) for row in bins)
    if counted != int(curves["n_rows"]):
        raise VerificationError("Calibration bin counts do not sum to the validation row count")
    if curves.get("score_is_calibrated") is not False:
        raise VerificationError("Validation curves must keep the score uncalibrated")


def _verify_threshold(
    grid: dict[str, Any],
    comparison: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if grid.get("evidence_class") != "locally_generated" or grid.get("split") != "validation":
        raise VerificationError("Threshold grid must be validation evidence")
    if grid.get("test_split_used") is not False:
        raise VerificationError("Threshold grid must record that the test split was not used")
    rows = grid.get("rows")
    if not isinstance(rows, list) or len(rows) != 37:
        raise VerificationError("Threshold grid must contain the documented 37 points")
    positives: set[int] = set()
    totals: set[int] = set()
    operating: dict[str, Any] | None = None
    for row in rows:
        metrics = metrics_from_counts(
            true_positives=int(row["true_positives"]),
            false_positives=int(row["false_positives"]),
            true_negatives=int(row["true_negatives"]),
            false_negatives=int(row["false_negatives"]),
            beta=float(row.get("beta", 2.0)),
        )
        if not counts_close(row, metrics):
            raise VerificationError(f"Threshold {row.get('threshold')} disagrees with its counts")
        positives.add(int(row["true_positives"]) + int(row["false_negatives"]))
        totals.add(int(metrics["n_rows"]))
        if row.get("selected_operating_point") is True:
            operating = row
    if len(positives) != 1 or len(totals) != 1:
        raise VerificationError("Threshold rows do not describe one fixed validation set")
    if operating is None:
        raise VerificationError("Threshold grid has no selected operating point")
    declared = grid.get("operating_point")
    if not isinstance(declared, dict):
        raise VerificationError("operating_point summary is missing")
    if not _close(declared.get("threshold"), operating.get("threshold"), tol=1e-12):
        raise VerificationError("Operating-point summary does not match the selected row")
    summary_metrics = metrics_from_counts(
        true_positives=int(declared["true_positives"]),
        false_positives=int(declared["false_positives"]),
        true_negatives=int(declared["true_negatives"]),
        false_negatives=int(declared["false_negatives"]),
    )
    if not counts_close(declared, summary_metrics):
        raise VerificationError("Operating-point summary disagrees with its counts")
    if not counts_close(
        declared,
        metrics_from_counts(
            true_positives=int(operating["true_positives"]),
            false_positives=int(operating["false_positives"]),
            true_negatives=int(operating["true_negatives"]),
            false_negatives=int(operating["false_negatives"]),
        ),
    ):
        raise VerificationError("Operating-point summary does not match the selected grid row")
    selected = next(row for row in comparison["models"] if row.get("selected") is True)
    if not _close(declared.get("precision"), selected.get("validation_precision_at_threshold")):
        raise VerificationError("Operating-point precision does not match the comparison table")
    if not _close(declared.get("recall"), selected.get("validation_recall_at_threshold")):
        raise VerificationError("Operating-point recall does not match the comparison table")
    if not _close(declared.get("threshold"), manifest["model"]["threshold"], tol=1e-12):
        raise VerificationError("Operating threshold does not match the manifest")
    if int(declared["n_rows"]) != int(grid["n_rows"]):
        raise VerificationError("Operating-point row count does not match the grid")


def _verify_historical(historical: dict[str, Any], curves: dict[str, Any]) -> None:
    if historical.get("evidence_class") != "documented_historical":
        raise VerificationError("Historical test evidence class is wrong")
    if (
        historical.get("reproduced_by_this_export") is not False
        or historical.get("curves_included") is not False
    ):
        raise VerificationError("Historical test must stay an unreproduced summary")
    if historical.get("split") != "test":
        raise VerificationError("Historical summary split must be test")
    if historical.get("validation_rows_included") is not False:
        raise VerificationError("Historical file must not include validation rows")
    for key in (
        "pr_curve",
        "roc_curve",
        "calibration_curve",
        "fpr",
        "tpr",
        "precision_recall",
        "roc",
    ):
        if key in historical:
            raise VerificationError("Historical summary contains curve coordinates")
    metrics = historical.get("metrics")
    if not isinstance(metrics, dict):
        raise VerificationError("Historical metrics are missing")
    intervals = historical.get("bootstrap_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise VerificationError("Historical bootstrap summary is missing")
    pr_interval = next(row for row in intervals if row.get("metric") == "pr_auc")
    if not _close(pr_interval.get("point"), metrics.get("pr_auc"), tol=1e-12):
        raise VerificationError("Bootstrap PR-AUC point does not match the historical metric")
    if _close(metrics.get("pr_auc"), curves.get("pr_auc"), tol=1e-12):
        raise VerificationError("Historical and validation PR-AUC were not distinguished")


def _verify_examples(examples: dict[str, Any], manifest: dict[str, Any]) -> None:
    if examples.get("evidence_class") != "precomputed_fictional":
        raise VerificationError("Examples evidence class is wrong")
    if examples.get("scored_live_in_browser") is not False:
        raise VerificationError("Examples must be offline precomputations")
    rows = examples.get("examples")
    if not isinstance(rows, list) or len(rows) < 2:
        raise VerificationError("At least two precomputed examples are required")
    threshold = float(manifest["model"]["threshold"])
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("example_id"))
        if example_id in seen:
            raise VerificationError("Duplicate example id")
        seen.add(example_id)
        if row.get("precomputed_offline") is not True:
            raise VerificationError(f"{example_id} is not marked precomputed")
        logit = float(row["logit"])
        intercept = float(row["intercept_log_odds"])
        total = float(row["contributions_log_odds_sum"])
        score = float(row["review_score"])
        if abs(intercept + total - logit) > 1e-8:
            raise VerificationError(f"{example_id} contributions do not rebuild the logit")
        if abs(sigmoid(logit) - score) > 1e-8:
            raise VerificationError(f"{example_id} score does not match the logit")
        if float(row["reconstruction_error"]) > 1e-8:
            raise VerificationError(f"{example_id} reconstruction error is too large")
        flagged = bool(row["flagged_for_review"])
        if flagged != (score >= threshold):
            raise VerificationError(f"{example_id} flag does not match the threshold")
        if "requires_review" in row.get("inputs", {}):
            raise VerificationError(f"{example_id} publishes a label as an input")
        if row.get("score_is_calibrated") is not False:
            raise VerificationError(f"{example_id} must keep the score uncalibrated")


def _verify_explanations(explanations: dict[str, Any], manifest: dict[str, Any]) -> None:
    if explanations.get("evidence_class") != "locally_generated":
        raise VerificationError("Explanation evidence class is wrong")
    if explanations.get("score_is_calibrated") is not False:
        raise VerificationError("Explanations must keep the score uncalibrated")
    coefficients = explanations.get("top_coefficients")
    if not isinstance(coefficients, list) or len(coefficients) < 5:
        raise VerificationError("Coefficient summary is missing")
    permutation = explanations.get("permutation_importance")
    if not isinstance(permutation, dict):
        raise VerificationError("Permutation block is missing")
    if permutation.get("evidence_class") not in {"locally_generated_artifact", "unavailable"}:
        raise VerificationError("Permutation evidence class is wrong")
    if permutation.get("recomputed_by_this_export") is not False:
        raise VerificationError("Permutation importance must not be presented as recomputed")
    if explanations["model"]["mlflow_run_id"] != manifest["model"]["mlflow_run_id"]:
        raise VerificationError("Explanation model identity drifted")


def check_static_site(site_dir: Path) -> None:
    """Reject root-absolute assets, CDN scripts and backend URLs."""
    html_path = site_dir / "index.html"
    js_path = site_dir / "app.js"
    css_path = site_dir / "styles.css"
    for path in (html_path, js_path, css_path):
        if not path.is_file():
            raise VerificationError(f"Missing {path.name}")
    html = html_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")
    combined = "\n".join((html, js, css_path.read_text(encoding="utf-8")))
    for token in REMOTE_TOKENS:
        if token in combined:
            raise VerificationError(f"Static site references {token}")
    if 'base href="/"' in html or "base href='/" in html:
        raise VerificationError("A root base href breaks the project subpath")
    for match in ASSET_ATTR.findall(html):
        if match.startswith("#"):
            continue
        if match.startswith(("http://", "https://")):
            if not match.startswith(GITHUB_REPOSITORY):
                raise VerificationError(f"Unexpected remote link: {match}")
            continue
        if match.startswith("/"):
            raise VerificationError(f"Root-absolute asset is not subpath-safe: {match}")
        if not (site_dir / match).is_file():
            raise VerificationError(f"Static asset is missing: {match}")
    if "not a calibrated probability" not in html:
        raise VerificationError(
            "The page must say the review score is not a calibrated probability"
        )
    if "data/manifest.json" not in js:
        raise VerificationError("The page script must load the evidence manifest")
    for name in FIGURE_FILES:
        if f"figures/{name}" not in html:
            raise VerificationError(f"Figure figures/{name} is not referenced")


def verify_site(site_dir: Path) -> dict[str, Any]:
    """Run bundle and static-site checks."""
    manifest = verify_bundle(site_dir)
    check_static_site(site_dir)
    if not math.isfinite(float(manifest["model"]["threshold"])):
        raise VerificationError("Champion threshold is not finite")
    return manifest
