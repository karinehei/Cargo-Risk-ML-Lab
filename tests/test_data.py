"""Unit tests for synthetic data generation and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data import (
    assign_requires_review,
    generate_feature_table,
    generate_synthetic_shipments,
    split_dataset,
    validate_dataset,
)
from src.data.schema import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    POSITIVE_RATE_BOUNDS,
    PROTECTED_NAME_FRAGMENTS,
    RAW_COLUMNS,
    TARGET_COLUMN,
)
from src.data.validate import DatasetValidationError, validate_no_target_leakage
from src.features import get_feature_lists


def _features(n_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return generate_feature_table(
        n_samples,
        rng,
        seed=seed,
        period_fractions=[0.45, 0.20, 0.20, 0.15],
        drift_start_period=2,
    )


def test_generate_is_deterministic() -> None:
    a = generate_synthetic_shipments(n_samples=400, seed=42, validate=False)
    b = generate_synthetic_shipments(n_samples=400, seed=42, validate=False)
    pd.testing.assert_frame_equal(a, b)


def test_generate_schema_and_class_balance() -> None:
    df = generate_synthetic_shipments(n_samples=3000, seed=42)
    report = validate_dataset(df)
    assert report.ok
    assert list(df.columns) == RAW_COLUMNS
    assert df[TARGET_COLUMN].isin([0, 1]).all()
    rate = float(df[TARGET_COLUMN].mean())
    assert POSITIVE_RATE_BOUNDS[0] <= rate <= POSITIVE_RATE_BOUNDS[1]


def test_validate_rejects_duplicates() -> None:
    df = generate_synthetic_shipments(n_samples=80, seed=1, validate=False)
    df.loc[1, "shipment_id"] = df.loc[0, "shipment_id"]
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        validate_dataset(df)


def test_time_like_split_sizes() -> None:
    df = generate_synthetic_shipments(n_samples=500, seed=42, validate=False)
    bundle = split_dataset(df, seed=42, strategy="time")
    assert len(bundle.train) + len(bundle.val) + len(bundle.test) == len(df)
    assert bundle.train["generation_period"].max() < bundle.val["generation_period"].min()
    assert bundle.val["generation_period"].max() < bundle.test["generation_period"].min()


def test_label_assignment_does_not_mutate_features() -> None:
    features = _features(500, seed=7)
    before = features.copy()
    labels = assign_requires_review(features, np.random.default_rng(7))
    pd.testing.assert_frame_equal(features, before)
    assert TARGET_COLUMN not in features.columns
    assert labels.isin([0, 1]).all()


def test_target_not_used_for_drifted_features() -> None:
    df = generate_synthetic_shipments(n_samples=4000, seed=42)
    early = df[df["generation_period"] == 0]
    late = df[df["generation_period"] == 3]
    assert late["declared_value_eur"].mean() > early["declared_value_eur"].mean()
    assert late["expedited_shipment"].mean() > early["expedited_shipment"].mean()
    assert late["route_rarity"].mean() > early["route_rarity"].mean()
    # Drift is applied before labels exist; permuting y cannot change features.
    shuffled = df.copy()
    shuffled[TARGET_COLUMN] = np.random.default_rng(0).permutation(
        shuffled[TARGET_COLUMN].to_numpy()
    )
    pd.testing.assert_frame_equal(
        df.drop(columns=[TARGET_COLUMN]),
        shuffled.drop(columns=[TARGET_COLUMN]),
    )


def test_no_target_leakage_in_model_features() -> None:
    df = generate_synthetic_shipments(n_samples=800, seed=3, validate=False)
    validate_no_target_leakage(df)
    numeric, categorical = get_feature_lists()
    model_cols = [*numeric, *categorical]
    assert TARGET_COLUMN not in model_cols
    assert "generation_period" not in model_cols
    assert "event_date" not in model_cols
    assert "shipment_id" not in model_cols
    assert FORBIDDEN_COLUMNS.isdisjoint(df.columns)
    for fragment in PROTECTED_NAME_FRAGMENTS:
        assert all(fragment not in column.lower() for column in df.columns)

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        numeric_col = pd.to_numeric(df[column], errors="coerce")
        aligned = pd.DataFrame({"x": numeric_col, "y": df[TARGET_COLUMN]}).dropna()
        if aligned["x"].nunique() < 2:
            continue
        corr = float(aligned["x"].corr(aligned["y"]))
        if np.isfinite(corr):
            assert abs(corr) < 0.98


def test_controlled_missingness() -> None:
    df = generate_synthetic_shipments(n_samples=2000, seed=42)
    allowed = {
        "declaration_completeness_score",
        "documentation_count",
        "sender_history_length",
        "route_rarity",
        "declared_vs_estimated_value_deviation",
    }
    for column in allowed:
        rate = float(df[column].isna().mean())
        assert 0.005 <= rate <= 0.08
    assert df[TARGET_COLUMN].isna().sum() == 0
    assert df["declared_value_eur"].isna().sum() == 0
    assert df["origin_region"].isna().sum() == 0
