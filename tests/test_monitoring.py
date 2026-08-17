"""Tests for monitoring metrics, profiles, scenarios and reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.config import AppConfig, get_config, load_yaml_config
from src.data.schema import ID_COLUMN, TARGET_COLUMN
from src.monitoring.metrics import (
    categorical_feature_metrics,
    default_thresholds,
    kolmogorov_smirnov,
    numeric_feature_metrics,
    overall_severity,
    population_stability_index,
    score_distribution_metrics,
    unseen_category_rate,
)
from src.monitoring.reference import build_reference_profile, validate_profile_payload
from src.monitoring.runner import create_reference_profile, run_monitoring
from src.monitoring.scenarios import generate_monitoring_batch


def _numeric_frame(n: int = 200, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "declared_value_eur": rng.normal(10000 + shift, 500, n),
            "shipment_weight_kg": rng.normal(80, 10, n),
            "value_to_weight_ratio": rng.normal(120, 20, n),
            "declaration_completeness_score": rng.uniform(0.4, 0.95, n),
            "documentation_count": rng.integers(1, 10, n),
            "previous_discrepancies": rng.integers(0, 3, n),
            "sender_history_length": rng.integers(1, 40, n),
            "route_rarity": rng.uniform(0, 1, n),
            "declared_vs_estimated_value_deviation": rng.normal(0.05, 0.2, n),
            "submission_hour": rng.integers(0, 23, n),
            "expedited_shipment": rng.integers(0, 1, n),
            "transport_mode": rng.choice(["road", "sea", "air", "rail"], n),
            "origin_region": rng.choice(["Asia", "Americas"], n),
            "destination_region": rng.choice(["Northern Europe", "Central Europe"], n),
            "commodity_category": rng.choice(["electronics", "textiles"], n),
        }
    )


def test_psi_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=1000)
    psi = population_stability_index(values, values.copy(), n_bins=10)
    assert psi < 0.05


def test_identical_batches_have_low_severity() -> None:
    frame = _numeric_frame()
    thresholds = default_thresholds()
    row = numeric_feature_metrics(
        frame["declared_value_eur"],
        frame["declared_value_eur"],
        thresholds=thresholds,
        n_bins=10,
    )
    assert row["severity"] == "none"


def test_moderate_and_major_numerical_drift() -> None:
    ref = pd.Series(np.linspace(1000.0, 20000.0, 400), name="declared_value_eur")
    moderate = ref * 1.35 + 500.0
    major = ref * 2.5 + 3000.0
    thresholds = default_thresholds()
    moderate_row = numeric_feature_metrics(ref, moderate, thresholds=thresholds, n_bins=10)
    major_row = numeric_feature_metrics(ref, major, thresholds=thresholds, n_bins=10)
    assert moderate_row["severity"] in {"warning", "critical"}
    assert major_row["severity"] == "critical"
    assert float(major_row["psi"]) > float(moderate_row["psi"])
    assert float(major_row["standardized_mean_difference"]) > float(
        moderate_row["standardized_mean_difference"]
    )


def test_categorical_drift_and_unseen_categories() -> None:
    ref = pd.Series(["road", "sea", "air"] * 100, name="transport_mode")
    cur = pd.Series(["air", "air", "sea"] * 100, name="transport_mode")
    unseen = pd.Series(["road", "drone", "air"] * 100, name="transport_mode")
    thresholds = default_thresholds()
    drift_row = categorical_feature_metrics(ref, cur, thresholds=thresholds)
    unseen_rate, unseen_values = unseen_category_rate(ref, unseen)
    assert drift_row["jensen_shannon_divergence"] > 0.0
    assert unseen_rate > 0.0
    assert "drone" in unseen_values


def test_missingness_drift() -> None:
    ref = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="documentation_count")
    cur = pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0], name="documentation_count")
    thresholds = default_thresholds()
    row = numeric_feature_metrics(ref, cur, thresholds=thresholds, n_bins=5)
    assert row["missing_rate_change"] > 0.0


def test_constant_column_safe() -> None:
    ref = pd.Series([5.0] * 50, name="route_rarity")
    cur = pd.Series([5.0] * 50, name="route_rarity")
    thresholds = default_thresholds()
    row = numeric_feature_metrics(ref, cur, thresholds=thresholds, n_bins=5)
    assert row["severity"] == "none"


def test_all_missing_column_safe() -> None:
    ref = pd.Series([np.nan] * 20, name="sender_history_length")
    cur = pd.Series([np.nan] * 20, name="sender_history_length")
    thresholds = default_thresholds()
    row = numeric_feature_metrics(ref, cur, thresholds=thresholds, n_bins=5)
    assert row["severity"] == "none"


def test_empty_batch_rejected(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "scenarios").mkdir()
    train = _numeric_frame(300)
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    empty = _numeric_frame(10)
    report = run_monitoring("none", config=cfg, current=empty)
    assert report["status"] == "insufficient_data"
    assert report["report_complete"] is False
    assert report["available"] is False


def test_run_drift_checks_smoke() -> None:
    from src.monitoring import prediction_drift_score, run_drift_checks

    ref = _numeric_frame(100)
    cur = ref.copy()
    result = run_drift_checks(ref, cur)
    assert "lightweight_summary" in result
    scores = prediction_drift_score(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
    assert scores["ks_statistic"] == 0.0


def test_score_and_review_rate_drift() -> None:
    ref_scores = np.linspace(0.2, 0.8, 300)
    cur_scores = np.linspace(0.5, 0.95, 300)
    payload = score_distribution_metrics(ref_scores, cur_scores, threshold=0.525, n_bins=10)
    assert payload["predicted_review_rate_change"] > 0.0
    assert payload["psi"] > 0.0
    assert kolmogorov_smirnov(ref_scores, cur_scores)["ks_statistic"] > 0.0


def test_create_reference_writes_only_configured_paths(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import PROJECT_ROOT

    production = PROJECT_ROOT / "artifacts" / "monitoring" / "reference_profile.json"
    before = production.read_text(encoding="utf-8") if production.exists() else None
    train = _numeric_frame(300)
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    assert (tmp_path / "reference_profile.json").exists()
    if before is not None:
        assert production.read_text(encoding="utf-8") == before


def test_reference_profile_excludes_forbidden_fields() -> None:
    frame = _numeric_frame(50)
    profile = build_reference_profile(
        frame,
        champion_metadata={"model_version": "x", "model_name": "logreg", "mlflow_run_id": "abc"},
        scores=np.linspace(0.1, 0.9, 50),
        threshold=0.525,
        seed=42,
        source="test",
    )
    validate_profile_payload(profile)
    blob = json.dumps(profile)
    assert TARGET_COLUMN not in profile
    assert ID_COLUMN not in profile
    assert "raw_records" not in blob


def _monitoring_config(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AppConfig:
    monkeypatch.setenv("CHAMPION_PATH", str(tiny_champion))
    get_config.cache_clear()
    base = load_yaml_config()
    monitoring = dict(base.monitoring)
    monitoring.update(
        {
            "reference_dataset_path": str(tmp_path / "train.csv"),
            "reference_sample_path": str(tmp_path / "reference_sample.csv"),
            "reference_profile_path": str(tmp_path / "reference_profile.json"),
            "report_dir": str(tmp_path / "reports"),
            "scenario_dir": str(tmp_path / "scenarios"),
            "min_batch_size": 50,
        }
    )
    return AppConfig(
        raw=base.raw,
        random_seed=base.random_seed,
        data=dict(base.data),
        features=dict(base.features),
        model=dict(base.model),
        training=dict(base.training),
        evaluation=dict(base.evaluation),
        explainability=dict(base.explainability),
        monitoring=monitoring,
        mlops=dict(base.mlops),
        api=dict(base.api),
        logging=dict(base.logging),
        project=dict(base.project),
    )


def test_scenarios_are_deterministic() -> None:
    first = generate_monitoring_batch("moderate", n_samples=120, seed=91002)
    second = generate_monitoring_batch("moderate", n_samples=120, seed=91002)
    pd.testing.assert_frame_equal(first, second)


def test_unlabelled_report_has_no_labels_or_raw_records(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _numeric_frame(300)
    (tmp_path / "scenarios").mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    current = generate_monitoring_batch("moderate", n_samples=200, config=cfg)
    current.to_csv(tmp_path / "scenarios" / "current_moderate.csv", index=False)
    report = run_monitoring("moderate", mode="unlabelled_monitoring", config=cfg, current=current)
    blob = json.dumps(report)
    assert TARGET_COLUMN not in blob
    assert "shipments" not in blob
    assert report["ground_truth_available"] is False
    assert "simulated_performance" not in report


def test_labelled_simulation_is_separate(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _numeric_frame(300)
    (tmp_path / "scenarios").mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    current = generate_monitoring_batch("none", n_samples=200, config=cfg)
    current.to_csv(tmp_path / "scenarios" / "current_none.csv", index=False)
    report = run_monitoring("none", mode="labelled_simulation", config=cfg, current=current)
    assert report["mode"] == "labelled_simulation"
    assert report["ground_truth_available"] is True
    assert "simulated_performance" in report


def test_monitoring_sources_do_not_use_test_csv() -> None:
    for relative in (
        "src/monitoring/runner.py",
        "src/monitoring/scenarios.py",
        "src/monitoring/audit.py",
        "src/monitoring/policy.py",
        "src/monitoring/report.py",
        "scripts/run_monitoring.py",
    ):
        text = Path(relative).read_text(encoding="utf-8")
        assert "test.csv" not in text
        assert "metrics_test" not in text


def test_overall_severity_aggregation() -> None:
    rows = [{"severity": "none"}, {"severity": "warning"}]
    score = {"severity": "critical"}
    assert overall_severity(rows, score) == "critical"
