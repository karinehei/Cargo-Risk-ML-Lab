"""Pytest fixtures for local MLflow champion tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.config import get_settings
from src.data import generate_synthetic_shipments, split_dataset
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.threshold import select_threshold
from src.features import prepare_xy
from src.mlops.champion import ChampionRecord, save_champion
from src.mlops.logging import log_candidate_run
from src.mlops.tracking import configure_tracking, init_tracking_store
from src.models import train_model


def sqlite_uri(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


@pytest.fixture()
def tiny_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Train a tiny logreg, log it to a temp MLflow store, and write champion.json."""
    db_path = tmp_path / "mlflow.db"
    champion_path = tmp_path / "champion.json"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", sqlite_uri(db_path))
    monkeypatch.setenv("CHAMPION_PATH", str(champion_path))
    get_settings.cache_clear()
    init_tracking_store(tracking_uri=sqlite_uri(db_path))
    configure_tracking(
        tracking_uri=sqlite_uri(db_path),
        experiment="pytest-cargo-risk",
        artifact_location=str(tmp_path / "mlartifacts"),
    )

    df = generate_synthetic_shipments(n_samples=280, seed=7, validate=False)
    splits = split_dataset(df, seed=7, strategy="stratified")
    trained = train_model(splits.train, val_df=splits.val, estimator_name="logreg")
    x_val, y_val = prepare_xy(splits.val, fit_derived_reference=splits.train)
    y_prob = trained.pipeline.predict_proba(x_val)[:, 1]
    threshold_info = select_threshold(y_val, y_prob, split_name="validation")
    metrics = compute_classification_metrics(
        y_val, y_prob, threshold=float(threshold_info["threshold"])
    )
    record = log_candidate_run(
        run_name="pytest_logreg",
        model_family="logreg",
        pipeline=trained.pipeline,
        hyperparameters={"C": 1.0},
        val_metrics=metrics,
        x_val=x_val,
        train_df=splits.train,
        val_df=splits.val,
        split_manifest_path=None,
        threshold=float(threshold_info["threshold"]),
        threshold_policy="validation_fbeta",
        cv_mean=0.2,
        cv_std=0.01,
        class_weight="balanced",
        calibration_status="none",
        preprocess_config={"scale_numeric": True},
        latency_repeats=8,
        fixture=x_val.head(8),
    )
    champion = ChampionRecord(
        model_name="logreg",
        model_version="logreg-none-1.0.0",
        mlflow_run_id=str(record["run_id"]),
        dataset_fingerprint=str(record["dataset_fingerprint"]),
        threshold=float(record["threshold"]),
        threshold_selection_method=str(record["threshold_policy"]),
        calibration_status="none",
        validation_metrics=record["validation_metrics"],
        artifact_uri=str(record["artifact_uri"]),
        created_at="2026-08-17T00:00:00+00:00",
        git_commit="unavailable",
        policy_version="1.0.0",
        reason="pytest fixture",
        awaiting_authorized_v2_test=False,
        test_evaluation_note="pytest",
        serialization=str(record["serialization"]),
        roundtrip_ok=bool(record["roundtrip_ok"]),
    )
    save_champion(champion, champion_path)
    return champion_path
