"""Public Pages bundle, threshold arithmetic and static-site checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from src.config import PROJECT_ROOT
from src.mlops.champion import default_policy, select_champion
from src.pages.metrics import confusion_counts, counts_close, metrics_from_counts, sigmoid
from src.pages.scenarios import FICTIONAL_SCENARIOS, validate_scenario_specs
from src.pages.schema import unexpected_site_files
from src.pages.selection import select_family
from src.pages.verify import VerificationError, verify_bundle, verify_site

SITE = PROJECT_ROOT / "site"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pages" / "DEVELOPMENT_FIXTURE.json"


def _read(name: str) -> dict[str, object]:
    payload = json.loads((SITE / "data" / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_undefined_precision_is_not_reported_as_zero() -> None:
    counts = confusion_counts([1, 0, 0], [0.2, 0.1, 0.1], 0.9)
    metrics = metrics_from_counts(**counts)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] is None
    assert metrics["precision_status"] == "undefined_no_flagged_rows"
    assert metrics["recall"] == 0.0
    assert metrics["fbeta"] is None
    assert counts_close(metrics, metrics_from_counts(**counts))


def test_defined_rates_match_the_counts() -> None:
    counts = confusion_counts([1, 1, 0, 0], [0.8, 0.2, 0.7, 0.1], 0.5)
    metrics = metrics_from_counts(**counts, beta=2.0)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["flagged_for_review"] == 2
    expected = (1 + 4) * 0.5 * 0.5 / (4 * 0.5 + 0.5)
    assert metrics["fbeta"] == pytest.approx(expected)


def test_selection_helper_matches_the_champion_policy() -> None:
    policy = default_policy()
    shared = {
        "roundtrip_ok": True,
        "threshold": 0.5,
        "threshold_policy": "validation_fbeta",
        "calibration_status": "none",
        "artifact_uri": "runs:/abc/model",
        "run_id": "abc",
        "serialization": "mlflow.sklearn.cloudpickle",
        "dataset_fingerprint": "fixture",
    }
    candidates = [
        {
            **shared,
            "model_family": "dummy",
            "validation_metrics": {
                "val_pr_auc": 0.10,
                "val_recall": 1.0,
                "val_brier": 0.2,
                "latency_p99_ms": 5.0,
            },
        },
        {
            **shared,
            "model_family": "logreg",
            "run_id": "logreg",
            "validation_metrics": {
                "val_pr_auc": 0.200,
                "val_recall": 0.50,
                "val_brier": 0.22,
                "latency_p99_ms": 6.0,
            },
        },
        {
            **shared,
            "model_family": "logreg_unweighted",
            "run_id": "unweighted",
            "validation_metrics": {
                "val_pr_auc": 0.203,
                "val_recall": 0.50,
                "val_brier": 0.11,
                "latency_p99_ms": 6.0,
            },
        },
    ]
    result = select_family(candidates, policy)
    chosen = select_champion(candidates)
    assert result.chosen_family == chosen.model_name == "logreg"
    assert result.leader_family == "logreg_unweighted"
    assert result.simpler_preference_applied is True
    assert any(item.startswith("dummy:") for item in result.rejections)


def test_curve_payload_and_figures_use_review_score_labels(tmp_path: Path) -> None:
    import numpy as np
    from src.pages.curves import validation_curve_payload
    from src.pages.plots import (
        plot_model_comparison,
        plot_threshold_tradeoff,
        plot_validation_calibration,
        plot_validation_pr,
        plot_validation_roc,
    )

    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6])
    payload = validation_curve_payload(labels, scores)
    assert payload["split"] == "validation"
    assert payload["score_is_calibrated"] is False
    assert payload["n_rows"] == 6
    assert sum(int(row["count"]) for row in payload["calibration_bins"]) == 6
    plot_validation_pr(payload, tmp_path / "pr.png")
    plot_validation_roc(payload, tmp_path / "roc.png")
    plot_validation_calibration(payload, tmp_path / "calibration.png")
    plot_model_comparison(
        [
            {
                "display_name": "Dummy (class prior)",
                "validation_pr_auc": 0.2,
                "training_fold_cv_pr_auc_mean": 0.2,
            },
            {
                "display_name": "Logistic regression",
                "validation_pr_auc": 0.4,
                "training_fold_cv_pr_auc_mean": 0.3,
            },
        ],
        tmp_path / "comparison.png",
    )
    rows = []
    for threshold in (0.2, 0.5, 0.9):
        counts = confusion_counts(labels.tolist(), scores.tolist(), threshold)
        rows.append({"threshold": threshold, **metrics_from_counts(**counts)})
    plot_threshold_tradeoff(rows, 0.5, tmp_path / "threshold.png")
    for name in ("pr.png", "roc.png", "calibration.png", "comparison.png", "threshold.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_fictional_scenarios_are_not_labelled_rows() -> None:
    validate_scenario_specs()
    assert len(FICTIONAL_SCENARIOS) >= 3
    for scenario in FICTIONAL_SCENARIOS:
        assert "requires_review" not in scenario["inputs"]
        assert "shipment_id" not in scenario["inputs"]


def test_development_fixture_is_labelled_and_absent_from_the_site() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["development_fixture"] is True
    assert "DEVELOPMENT FIXTURE" in payload["label"]
    assert FIXTURE.relative_to(PROJECT_ROOT).as_posix().startswith("tests/fixtures/pages/")
    for path in SITE.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".html", ".js", ".css", ".md"}:
            assert "DEVELOPMENT FIXTURE" not in path.read_text(encoding="utf-8")
            assert '"development_fixture": true' not in path.read_text(encoding="utf-8")


def test_unexpected_site_file_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "secret.csv").write_text("a,b\n", encoding="utf-8")
    assert unexpected_site_files(tmp_path) == ["data/secret.csv"]


def test_malformed_bundle_is_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(VerificationError, match="not valid JSON"):
        verify_bundle(tmp_path)


def test_missing_bundle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    with pytest.raises(VerificationError, match="Missing public evidence file"):
        verify_bundle(tmp_path)


def test_export_script_does_not_evaluate_the_frozen_test() -> None:
    text = (PROJECT_ROOT / "scripts" / "export_pages_evidence.py").read_text(encoding="utf-8")
    assert "evaluate_model" not in text
    assert "metrics_test.json" in text
    assert "test_split_scored" in text


def test_published_bundle_keeps_validation_and_historical_test_apart() -> None:
    verify_site(SITE)
    manifest = _read("manifest.json")
    comparison = _read("model_comparison.json")
    grid = _read("threshold_grid.json")
    historical = _read("historical_test.json")
    curves = _read("validation_curves.json")
    examples = _read("examples.json")
    model = manifest["model"]
    assert isinstance(model, dict)
    assert model["mlflow_run_id"] == "8041c2e0afaf4ecea05399ae55a87816"
    assert model["score_is_calibrated"] is False
    assert manifest["historical_test_reproduced"] is False
    assert manifest["source_split_for_curves"] == "validation"
    operating = grid["operating_point"]
    assert isinstance(operating, dict)
    assert operating["threshold"] == pytest.approx(0.525)
    assert operating["true_positives"] == 104
    assert operating["false_positives"] == 383
    assert operating["true_negatives"] == 923
    assert operating["false_negatives"] == 90
    assert grid["split"] == "validation"
    assert grid["test_split_used"] is False
    selected_rows = [
        row
        for row in grid["rows"]  # type: ignore[union-attr]
        if isinstance(row, dict) and row.get("selected_operating_point") is True
    ]
    assert len(selected_rows) == 1
    assert selected_rows[0]["true_positives"] == operating["true_positives"]
    assert selected_rows[0]["precision"] == pytest.approx(operating["precision"])
    models = comparison["models"]
    assert isinstance(models, list)
    selected = next(row for row in models if isinstance(row, dict) and row.get("selected") is True)
    assert selected["model_family"] == "logreg"
    assert selected["validation_pr_auc"] == pytest.approx(0.2272784442866974)
    assert selected["validation_precision_at_threshold"] == pytest.approx(operating["precision"])
    assert curves["split"] == "validation"
    assert curves["pr_auc"] == pytest.approx(selected["validation_pr_auc"])
    assert historical["evidence_class"] == "documented_historical"
    assert historical["reproduced_by_this_export"] is False
    assert historical["curves_included"] is False
    assert historical["split"] == "test"
    metrics = historical["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["pr_auc"] == pytest.approx(0.20420446319791113)
    assert metrics["true_positives"] == 187
    assert metrics["false_positives"] == 746
    assert metrics["false_negatives"] == 201
    assert metrics["true_negatives"] == 1866
    for key in ("pr_curve", "roc_curve", "fpr", "tpr"):
        assert key not in historical
    example_rows = examples["examples"]
    assert isinstance(example_rows, list)
    assert [row["example_id"] for row in example_rows if isinstance(row, dict)] == [
        scenario["example_id"] for scenario in FICTIONAL_SCENARIOS
    ]
    assert examples["scored_live_in_browser"] is False
    collision = examples["collision_check"]
    assert isinstance(collision, dict)
    assert collision["test_split_scored"] is False
    assert collision["matches_found"] == 0
    expected_scores = {
        "complete-sea-foodstuffs": 0.141505,
        "expedited-air-pharmaceuticals": 0.073889,
        "incomplete-air-electronics": 0.824355,
        "ordinary-road-machinery": 0.293056,
    }
    for row in example_rows:
        assert isinstance(row, dict)
        logit = float(row["logit"])
        assert abs(sigmoid(logit) - float(row["review_score"])) <= 1e-8
        assert row["precomputed_offline"] is True
        assert float(row["review_score"]) == pytest.approx(
            expected_scores[str(row["example_id"])], abs=1e-5
        )


def test_tampered_confusion_counts_fail_verification(tmp_path: Path) -> None:
    copied = tmp_path / "site"
    shutil.copytree(SITE, copied)
    grid_path = copied / "data" / "threshold_grid.json"
    payload = json.loads(grid_path.read_text(encoding="utf-8"))
    payload["rows"][0]["true_positives"] = int(payload["rows"][0]["true_positives"]) + 5
    text = json.dumps(payload, indent=2) + "\n"
    grid_path.write_text(text, encoding="utf-8", newline="\n")
    manifest_path = copied / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["threshold_grid.json"]["sha256"] = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(VerificationError, match="disagrees with its counts"):
        verify_bundle(copied)


def test_preview_tree_uses_the_project_subpath(tmp_path: Path) -> None:
    from scripts.preview_pages import assemble_preview

    root = assemble_preview(SITE, tmp_path)
    index = (root / "Cargo-Risk-ML-Lab" / "index.html").read_text(encoding="utf-8")
    assert 'src="/' not in index
    assert 'href="/' not in index
    assert "styles.css" in index
    assert "app.js" in index
    assert not any(path.name == "DEVELOPMENT_FIXTURE.json" for path in root.rglob("*"))
    script = (root / "Cargo-Risk-ML-Lab" / "app.js").read_text(encoding="utf-8")
    assert "data/manifest.json" in script
    assert "localhost" not in script
    assert "127.0.0.1" not in script


def test_pages_verifier_imports_no_scientific_stack() -> None:
    import ast

    source = (PROJECT_ROOT / "scripts" / "verify_pages_evidence.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "src.config" not in imported
    assert "numpy" not in imported
    assert "src.pages.verify" in imported


def test_repository_policy_names_the_public_bundle() -> None:
    text = (PROJECT_ROOT / "docs" / "repository_policy.md").read_text(encoding="utf-8")
    for name in (
        "site/data/manifest.json",
        "site/data/threshold_grid.json",
        "site/figures/validation_pr_curve.png",
        "development_fixture",
    ):
        assert name in text
