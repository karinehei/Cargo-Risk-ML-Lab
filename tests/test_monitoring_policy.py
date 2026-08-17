"""Monitoring policy 1.1.0, null-audit helpers, and CSV exclusion tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from src.config import PROJECT_ROOT
from src.monitoring.metrics import default_thresholds
from src.monitoring.policy import (
    MONITORING_POLICY_V1,
    MONITORING_POLICY_VERSION,
    apply_policy,
    expected_union_alert_rate,
    extract_alert_reasons,
)
from src.monitoring.report import load_latest_status
from src.monitoring.runner import create_reference_profile, run_monitoring
from tests.test_monitoring import _monitoring_config, _numeric_frame


def _feature(
    name: str,
    *,
    kind: str = "numeric",
    severity: str = "warning",
    psi: float | None = None,
    ks: float | None = None,
    smd: float | None = None,
    missing: float = 0.0,
    js: float | None = None,
    tv: float | None = None,
    unseen: float | None = None,
) -> dict[str, object]:
    if kind == "numeric":
        return {
            "feature": name,
            "type": "numeric",
            "psi": 0.0 if psi is None else psi,
            "ks_statistic": 0.0 if ks is None else ks,
            "standardized_mean_difference": 0.0 if smd is None else smd,
            "missing_rate_change": missing,
            "severity": severity,
        }
    return {
        "feature": name,
        "type": "categorical",
        "jensen_shannon_divergence": 0.0 if js is None else js,
        "total_variation_distance": 0.0 if tv is None else tv,
        "unseen_category_rate": 0.0 if unseen is None else unseen,
        "missing_rate_change": missing,
        "severity": severity,
    }


def _score(
    *, severity: str = "none", psi: float = 0.01, rate: float = 0.01, ks: float = 0.02
) -> dict[str, object]:
    return {
        "psi": psi,
        "ks_statistic": ks,
        "predicted_review_rate_change": rate,
        "severity": severity,
    }


def test_expected_union_false_alert_rate() -> None:
    assert expected_union_alert_rate([]) == 0.0
    assert expected_union_alert_rate([0.0, 0.0]) == 0.0
    observed = expected_union_alert_rate([0.05, 0.05, 0.05])
    assert abs(observed - (1.0 - 0.95**3)) < 1e-12


def test_multiple_comparison_aggregation_isolated_vs_coordinated() -> None:
    isolated_rows = [
        _feature("declaration_completeness_score", ks=0.12, smd=0.23, severity="warning")
    ]
    score = _score()
    isolated = apply_policy(isolated_rows, score, policy_version=MONITORING_POLICY_VERSION)
    assert isolated["status"] == "no_material_drift"
    assert isolated["isolated_weak_warning"] is True
    assert isolated["alert_reasons"]
    assert isolated["alert_reasons"][0]["role"] == "isolated_weak_warning"

    coordinated_rows = isolated_rows + [
        _feature("route_rarity", ks=0.11, smd=0.21, severity="warning")
    ]
    coordinated = apply_policy(coordinated_rows, score, policy_version=MONITORING_POLICY_VERSION)
    assert coordinated["status"] == "warning"
    assert coordinated["isolated_weak_warning"] is False


def test_persistent_drift_raises_warning() -> None:
    rows = [_feature("declaration_completeness_score", ks=0.12, smd=0.23, severity="warning")]
    first = apply_policy(rows, _score(), policy_version=MONITORING_POLICY_VERSION)
    assert first["status"] == "no_material_drift"
    second = apply_policy(
        rows,
        _score(),
        policy_version=MONITORING_POLICY_VERSION,
        previous_status=first["status"],
        previous_warning_names=list(first["warning_feature_names"]),
    )
    assert second["status"] == "warning"


def test_immediate_critical_schema_violation() -> None:
    rows = [
        _feature(
            "transport_mode",
            kind="categorical",
            unseen=0.08,
            severity="critical",
        )
    ]
    result = apply_policy(rows, _score(), policy_version=MONITORING_POLICY_VERSION)
    assert result["status"] == "critical"
    assert any(item.get("immediate_critical") for item in result["alert_reasons"])
    assert any(item.get("role") == "immediate_critical" for item in result["alert_reasons"])


def test_v1_none_warning_was_isolated_completeness_shift() -> None:
    rows = [
        _feature(
            "declaration_completeness_score",
            psi=0.06742066210674215,
            ks=0.12254183465320156,
            smd=0.22670153662810774,
            missing=-0.004966666666666668,
            severity="warning",
        )
    ]
    score = _score(psi=0.040896844753508535, rate=-0.029766666666666663, ks=0.07673333333333332)
    reasons = extract_alert_reasons(rows, score, default_thresholds())
    names = {item["name"] for item in reasons}
    metrics = {item["metric"] for item in reasons}
    assert names == {"declaration_completeness_score"}
    assert "ks_statistic" in metrics
    assert "standardized_mean_difference" in metrics
    assert "psi" not in metrics
    assert all(item["name"] != "review_score" for item in reasons)
    legacy = apply_policy(rows, score, policy_version=MONITORING_POLICY_V1)
    current = apply_policy(rows, score, policy_version=MONITORING_POLICY_VERSION)
    assert legacy["status"] == "warning"
    assert current["status"] == "no_material_drift"
    assert current["isolated_weak_warning"] is True


def test_null_simulation_reproducible(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.monitoring.audit import run_null_monte_carlo

    train = _numeric_frame(300)
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    first = run_null_monte_carlo(n_replications=2, seed_base=92001, config=cfg, persist=False)
    second = run_null_monte_carlo(n_replications=2, seed_base=92001, config=cfg, persist=False)
    assert first["replications"] == second["replications"]
    assert first["false_alert_rate_overall_v1_1_0"]["any_warning_or_critical"] <= 1.0


def test_insufficient_sample_size_status(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _numeric_frame(300)
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    report = run_monitoring("none", config=cfg, current=_numeric_frame(10))
    assert report["status"] == "insufficient_data"
    assert report["report_complete"] is False
    assert report["available"] is False


def test_missing_report_is_insufficient_data(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    monkeypatch.setattr("src.monitoring.report.get_config", lambda: cfg)
    status = load_latest_status()
    assert status["available"] is False
    assert status["status"] == "insufficient_data"
    assert status["report_complete"] is False


def test_monitoring_computation_failure(
    tmp_path: Path,
    tiny_champion: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _numeric_frame(300)
    (tmp_path / "scenarios").mkdir()
    train.to_csv(tmp_path / "train.csv", index=False)
    cfg = _monitoring_config(tmp_path, tiny_champion, monkeypatch)
    create_reference_profile(cfg)
    current = _numeric_frame(80)
    with patch("src.monitoring.runner.evaluate_comparison", side_effect=RuntimeError("boom")):
        report = run_monitoring("none", config=cfg, current=current)
    assert report["status"] == "monitoring_error"
    assert report["report_complete"] is False
    assert report["available"] is False


def test_raw_monitoring_csv_git_and_docker_exclusion() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "data/monitoring" in gitignore
    assert "!data/monitoring/.gitkeep" in gitignore
    docker_lines = {line.strip() for line in dockerignore.splitlines() if line.strip()}
    assert "data" in docker_lines
    assert "data/monitoring/*.csv" in docker_lines
    assert "COPY data" not in dockerfile
    gitkeep = PROJECT_ROOT / "data" / "monitoring" / ".gitkeep"
    assert gitkeep.exists()
    listed = subprocess.run(
        ["git", "ls-files", "data/monitoring"],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    tracked = [line for line in listed.stdout.splitlines() if line.endswith(".csv")]
    assert tracked == []


def test_frozen_v1_hash_and_champion_unchanged() -> None:
    frozen = PROJECT_ROOT / "artifacts" / "frozen_v1" / "metrics_test.json"
    if not frozen.exists():
        pytest.skip("frozen-v1 artifacts are local and gitignored")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert digest.startswith("fc94af40")
    champion_path = PROJECT_ROOT / "artifacts" / "mlops" / "champion.json"
    if not champion_path.exists():
        pytest.skip("champion metadata is local")
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    assert champion["model_version"] == "logreg-none-1.0.0"
    assert float(champion["threshold"]) == 0.525
    assert champion["mlflow_run_id"] == "8041c2e0afaf4ecea05399ae55a87816"


def test_policy_sources_do_not_use_test_csv() -> None:
    for relative in (
        "src/monitoring/audit.py",
        "src/monitoring/policy.py",
        "src/monitoring/runner.py",
        "src/monitoring/scenarios.py",
        "src/monitoring/report.py",
        "src/monitoring/reference.py",
        "scripts/run_monitoring.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "test.csv" not in text
        assert "metrics_test" not in text
