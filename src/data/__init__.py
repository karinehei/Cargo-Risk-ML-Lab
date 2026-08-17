"""Synthetic data generation and validation."""

from src.data.generate import (
    DatasetBundle,
    assign_requires_review,
    build_and_persist_splits,
    generate_feature_table,
    generate_synthetic_shipments,
    introduce_missingness,
    load_dataset,
    save_dataset,
    save_split_manifest,
    split_dataset,
)
from src.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.data.validate import DatasetValidationError, ValidationReport, validate_dataset

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "TARGET_COLUMN",
    "DatasetBundle",
    "DatasetValidationError",
    "ValidationReport",
    "assign_requires_review",
    "build_and_persist_splits",
    "generate_feature_table",
    "generate_synthetic_shipments",
    "introduce_missingness",
    "load_dataset",
    "save_dataset",
    "save_split_manifest",
    "split_dataset",
    "validate_dataset",
]
