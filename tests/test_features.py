"""Unit tests for feature preparation."""

from __future__ import annotations

from src.data import generate_synthetic_shipments
from src.features import build_preprocess_pipeline, get_feature_lists, prepare_xy


def test_prepare_xy_shapes() -> None:
    df = generate_synthetic_shipments(n_samples=120, seed=42, validate=False)
    x, y = prepare_xy(df)
    numeric, categorical = get_feature_lists()
    assert list(x.columns) == [*numeric, *categorical]
    assert len(y) == len(df)
    assert set(y.tolist()).issubset({0, 1})


def test_preprocess_pipeline_fit_transform() -> None:
    df = generate_synthetic_shipments(n_samples=150, seed=3, validate=False)
    x, _ = prepare_xy(df)
    pipe = build_preprocess_pipeline()
    matrix = pipe.fit_transform(x)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] > len(x.columns)  # one-hot expands categoricals
