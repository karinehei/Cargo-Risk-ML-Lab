"""Unit and integration tests for model training and inference."""

from __future__ import annotations

from pathlib import Path

from src.data import generate_synthetic_shipments, split_dataset
from src.features import prepare_xy
from src.models import (
    load_model_bundle,
    predict_proba,
    prepare_inference_frame,
    save_model_bundle,
    train_model,
)


def test_train_predict_roundtrip(tmp_path: Path) -> None:
    df = generate_synthetic_shipments(n_samples=400, seed=42, validate=False)
    bundle_data = split_dataset(df, seed=42)
    trained = train_model(bundle_data.train, val_df=bundle_data.val)
    x_test, y_test = prepare_xy(bundle_data.test, fit_derived_reference=bundle_data.train)
    scores = predict_proba(trained.pipeline, x_test)
    assert len(scores) == len(y_test)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0

    out = save_model_bundle(trained, artifact_dir=tmp_path)
    loaded = load_model_bundle(out)
    scores2 = predict_proba(loaded.pipeline, x_test)
    assert abs(float(scores.mean()) - float(scores2.mean())) < 1e-9


def test_prepare_inference_frame_derived_columns() -> None:
    rows = [
        {
            "origin_region": "Asia",
            "destination_region": "Northern Europe",
            "commodity_category": "electronics",
            "transport_mode": "air",
            "declared_value_eur": 30000.0,
            "shipment_weight_kg": 40.0,
            "declaration_completeness_score": 0.4,
            "documentation_count": 2,
            "previous_discrepancies": 3,
            "sender_history_length": 4,
            "route_rarity": 0.8,
            "declared_vs_estimated_value_deviation": 0.6,
            "submission_hour": 2,
            "expedited_shipment": 1,
        }
    ]
    frame = prepare_inference_frame(rows)
    assert frame.loc[0, "is_off_hours"] == 1
    assert frame.loc[0, "log_declared_value"] > 0
    assert "requires_review" not in frame.columns
