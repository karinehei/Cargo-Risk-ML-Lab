"""MLflow tracking, round-trip, champion policy and frozen-v1 preservation tests."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import mlflow
import numpy as np
import pytest
from src.config import PROJECT_ROOT
from src.data import generate_synthetic_shipments, split_dataset
from src.evaluation.metrics import compute_classification_metrics
from src.features import prepare_xy
from src.mlops.calibration import fit_calibration_candidates
from src.mlops.champion import select_champion
from src.mlops.serialization import (
    SKLEARN_SERIALIZATION_FORMAT,
    assert_prediction_roundtrip,
    load_sklearn_pipeline,
    log_sklearn_pipeline,
)
from src.mlops.serving import ChampionLoadError, load_champion
from src.mlops.tracking import configure_tracking, init_tracking_store
from src.models import train_model


def test_frozen_v1_test_metrics_hash_preserved() -> None:
    path = PROJECT_ROOT / "artifacts" / "frozen_v1" / "metrics_test.json"
    if not path.exists():
        pytest.skip("frozen-v1 artifacts are local and gitignored")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest.startswith("fc94af40")


def test_champion_policy_rejects_test_metrics() -> None:
    candidate = {
        "model_family": "logreg",
        "run_id": "abc",
        "roundtrip_ok": True,
        "threshold": 0.5,
        "threshold_policy": "validation",
        "calibration_status": "none",
        "artifact_uri": "runs:/abc/model",
        "validation_metrics": {"val_pr_auc": 0.3, "val_recall": 0.5, "test_pr_auc": 0.9},
    }
    with pytest.raises(ValueError, match="Test metrics"):
        select_champion([candidate])


def test_champion_policy_prefers_eligible_logreg() -> None:
    dummy = {
        "model_family": "dummy",
        "run_id": "d1",
        "roundtrip_ok": True,
        "threshold": 0.5,
        "threshold_policy": "validation",
        "calibration_status": "none",
        "artifact_uri": "runs:/d1/model",
        "dataset_fingerprint": "x",
        "serialization": "mlflow.sklearn.cloudpickle",
        "validation_metrics": {
            "val_pr_auc": 0.13,
            "val_recall": 0.0,
            "val_brier": 0.12,
            "latency_p99_ms": 2.0,
        },
    }
    logreg = {
        "model_family": "logreg",
        "run_id": "l1",
        "roundtrip_ok": True,
        "threshold": 0.525,
        "threshold_policy": "validation_fbeta",
        "calibration_status": "none",
        "artifact_uri": "runs:/l1/model",
        "dataset_fingerprint": "x",
        "serialization": "mlflow.sklearn.cloudpickle",
        "validation_metrics": {
            "val_pr_auc": 0.227,
            "val_recall": 0.54,
            "val_brier": 0.24,
            "latency_p99_ms": 6.0,
        },
    }
    forest = {
        "model_family": "random_forest",
        "run_id": "r1",
        "roundtrip_ok": True,
        "threshold": 0.4,
        "threshold_policy": "validation_fbeta",
        "calibration_status": "none",
        "artifact_uri": "runs:/r1/model",
        "dataset_fingerprint": "x",
        "serialization": "mlflow.sklearn.cloudpickle",
        "validation_metrics": {
            "val_pr_auc": 0.220,
            "val_recall": 0.45,
            "val_brier": 0.19,
            "latency_p99_ms": 8.0,
        },
    }
    chosen = select_champion([dummy, logreg, forest])
    assert chosen.model_name == "logreg"
    assert chosen.threshold == 0.525
    assert "simple" in chosen.reason or "logistic" in chosen.reason


def test_mlflow_sklearn_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = "sqlite:///" + (tmp_path / "ml.db").resolve().as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    from src.config import get_settings

    get_settings.cache_clear()
    init_tracking_store(tracking_uri=uri)
    configure_tracking(
        tracking_uri=uri,
        experiment="pytest-roundtrip",
        artifact_location=str(tmp_path / "mlartifacts"),
    )
    df = generate_synthetic_shipments(n_samples=220, seed=3, validate=False)
    splits = split_dataset(df, seed=3, strategy="stratified")
    trained = train_model(splits.train, val_df=splits.val, estimator_name="logreg")
    x_val, _ = prepare_xy(splits.val, fit_derived_reference=splits.train)
    with mlflow.start_run(run_name="pytest-roundtrip"):
        logged = log_sklearn_pipeline(trained.pipeline, allow_joblib_fallback=False)
        assert logged.method == f"mlflow.sklearn.{SKLEARN_SERIALIZATION_FORMAT}"
        loaded = load_sklearn_pipeline(logged.artifact_uri)
        stats = assert_prediction_roundtrip(trained.pipeline, loaded, x_val.head(12))
        assert stats["max_abs_proba_delta"] < 1e-8


def test_champion_load_and_threshold(tiny_champion: Path) -> None:
    bundle = load_champion(str(tiny_champion))
    assert bundle.threshold == pytest.approx(float(bundle.metadata["threshold"]))
    assert bundle.metadata["mlflow_run_id"] in bundle.metadata["artifact_uri"]
    assert bundle.pipeline is not None


def test_missing_champion_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAMPION_PATH", str(tmp_path / "missing.json"))
    from src.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ChampionLoadError, match="unavailable"):
        load_champion()


def test_corrupted_champion_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "champion.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("CHAMPION_PATH", str(path))
    from src.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ChampionLoadError, match="unreadable"):
        load_champion()


def test_calibration_workflow_is_train_val_only() -> None:
    source = inspect.getsource(fit_calibration_candidates)
    assert "test.csv" not in source
    df = generate_synthetic_shipments(n_samples=260, seed=5, validate=False)
    splits = split_dataset(df, seed=5, strategy="stratified")
    from src.features import prepare_xy as _prepare

    x_train, y_train = _prepare(splits.train, fit_derived_reference=splits.train)
    x_val, y_val = _prepare(splits.val, fit_derived_reference=splits.train)
    fitted = fit_calibration_candidates(x_train, y_train, x_val, y_val)
    names = {item.name for item in fitted}
    assert "logreg_uncalibrated_weighted" in names
    assert "logreg_sigmoid" in names
    for item in fitted:
        assert "pr_auc" in item.val_metrics_at_selected_threshold
        assert item.val_metrics_at_frozen_threshold["ece"] >= 0.0


def test_run_mlops_script_does_not_load_test_csv() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_mlops.py").read_text(encoding="utf-8")
    assert "test.csv" not in source
    assert "metrics_test" not in source
    assert "artifacts/frozen_v1" not in source


def test_compute_metrics_includes_ece() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= metrics["ece"] <= 1.0


def test_experiment_metadata_is_logged(tiny_champion: Path) -> None:
    import json

    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    client = mlflow.MlflowClient()
    run = client.get_run(str(metadata["mlflow_run_id"]))
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags
    for key in ("random_seed", "class_weight", "threshold", "threshold_policy", "model_family"):
        assert key in params
    for key in (
        "val_pr_auc",
        "val_roc_auc",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_brier",
        "val_ece",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "roundtrip_ok",
    ):
        assert key in metrics
    assert tags.get("split") == "validation"
    assert tags.get("model_family") == "logreg"
    assert all(not str(key).startswith("test_") for key in metrics)
    assert all(not str(key).startswith("test_") for key in params)
    assert metadata["artifact_uri"].startswith("runs:/")
    assert metadata["mlflow_run_id"] in metadata["artifact_uri"]


def test_champion_threshold_mismatch_fails(tiny_champion: Path, tmp_path: Path) -> None:
    import json

    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    metadata["threshold"] = 0.99
    bad = tmp_path / "mismatch.json"
    bad.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ChampionLoadError, match="threshold"):
        load_champion(str(bad))


def test_champion_uri_mismatch_fails(tiny_champion: Path, tmp_path: Path) -> None:
    import json

    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    metadata["artifact_uri"] = "runs:/00000000000000000000000000000000/model"
    bad = tmp_path / "uri.json"
    bad.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ChampionLoadError, match="match"):
        load_champion(str(bad))


def test_champion_filesystem_uri_rejected(tiny_champion: Path, tmp_path: Path) -> None:
    import json

    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    metadata["artifact_uri"] = str(tmp_path / "model.pkl")
    bad = tmp_path / "path.json"
    bad.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ChampionLoadError, match="unsupported"):
        load_champion(str(bad))


def test_champion_policy_prefers_simpler_within_indifference() -> None:
    logreg = {
        "model_family": "logreg",
        "run_id": "l2",
        "roundtrip_ok": True,
        "threshold": 0.525,
        "threshold_policy": "validation_fbeta",
        "calibration_status": "none",
        "artifact_uri": "runs:/l2/model",
        "dataset_fingerprint": "x",
        "serialization": "mlflow.sklearn.cloudpickle",
        "validation_metrics": {
            "val_pr_auc": 0.227,
            "val_recall": 0.54,
            "val_brier": 0.238,
            "latency_p99_ms": 6.0,
        },
    }
    sigmoid = {
        "model_family": "logreg_sigmoid",
        "run_id": "s1",
        "roundtrip_ok": True,
        "threshold": 0.40,
        "threshold_policy": "validation_fbeta",
        "calibration_status": "sigmoid",
        "artifact_uri": "runs:/s1/model",
        "dataset_fingerprint": "x",
        "serialization": "mlflow.sklearn.cloudpickle",
        "validation_metrics": {
            "val_pr_auc": 0.229,
            "val_recall": 0.50,
            "val_brier": 0.110,
            "latency_p99_ms": 8.0,
        },
    }
    chosen = select_champion([logreg, sigmoid], calibrated_in_pool=True)
    assert chosen.model_name == "logreg"
    assert chosen.awaiting_authorized_v2_test is False


def test_mlops_scripts_do_not_write_frozen_v1() -> None:
    for rel in (
        "scripts/run_mlops.py",
        "scripts/verify_mlflow_roundtrip.py",
        "scripts/select_champion.py",
        "src/mlops/champion.py",
        "src/mlops/calibration.py",
        "src/mlops/logging.py",
    ):
        source = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert "metrics_test.json" not in source
        assert "evaluate_model" not in source
        if "frozen_v1" in source:
            lowered = source.lower()
            assert any(
                token in lowered
                for token in ("must not", "never", "does not write", "not be written")
            )


def test_log_candidate_run_rejects_test_metric_keys() -> None:
    import pandas as pd
    from src.mlops.logging import log_candidate_run

    with pytest.raises(ValueError, match="Test metrics"):
        log_candidate_run(
            run_name="bad",
            model_family="logreg",
            pipeline=object(),  # type: ignore[arg-type]
            hyperparameters={},
            val_metrics={"pr_auc": 0.2, "test_pr_auc": 0.9},
            x_val=pd.DataFrame({"a": [1.0]}),
            train_df=pd.DataFrame({"a": [1.0]}),
            val_df=pd.DataFrame({"a": [1.0]}),
            split_manifest_path=None,
            threshold=0.5,
            threshold_policy="validation",
            cv_mean=None,
            cv_std=None,
            class_weight="balanced",
            calibration_status="none",
            preprocess_config={},
        )
