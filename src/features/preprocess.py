"""Feature engineering and preprocessing pipelines.

Preprocessing is intentionally separate from synthetic data generation.
Derived columns here are deterministic transforms of raw features and never
use ``requires_review``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import AppConfig, get_config, setup_logging
from src.data.schema import TARGET_COLUMN

logger = setup_logging(name="src.features")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic derived features used by the model pipeline."""
    out = df.copy()
    if "value_to_weight_ratio" not in out.columns:
        out["value_to_weight_ratio"] = out["declared_value_eur"] / out["shipment_weight_kg"].clip(
            lower=0.1
        )
    out["log_declared_value"] = np.log1p(out["declared_value_eur"])
    hour = pd.to_numeric(out["submission_hour"], errors="coerce")
    out["is_off_hours"] = ((hour < 6) | (hour >= 22)).astype(int)
    return out


def get_feature_lists(config: AppConfig | None = None) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature column names from config."""
    cfg = config or get_config()
    numeric = list(cfg.features.get("numeric", []))
    categorical = list(cfg.features.get("categorical", []))
    derived = list(cfg.features.get("derived", []))
    numeric_extended = list(dict.fromkeys([*numeric, *derived]))
    return numeric_extended, categorical


def build_preprocess_pipeline(
    config: AppConfig | None = None,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical features.

    Missing values are imputed inside the pipeline. Categorical unknowns are
    ignored at transform time. Numeric scaling is optional so tree models are
    not forced through a StandardScaler.
    """
    numeric_features, categorical_features = get_feature_lists(config)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    logger.info(
        "Built preprocess pipeline numeric=%s categorical=%s scale=%s",
        numeric_features,
        categorical_features,
        scale_numeric,
    )
    return transformer


def prepare_xy(
    df: pd.DataFrame,
    config: AppConfig | None = None,
    *,
    fit_derived_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create model-ready feature frame ``X`` and label vector ``y``.

    ``fit_derived_reference`` is accepted for API stability; current derived
    features do not depend on training quantiles.
    """
    _ = fit_derived_reference
    cfg = config or get_config()
    target = str(cfg.data.get("target_column", TARGET_COLUMN))
    working = add_derived_features(df)

    numeric_features, categorical_features = get_feature_lists(cfg)
    feature_cols = [*numeric_features, *categorical_features]
    missing = [col for col in feature_cols if col not in working.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if target in feature_cols:
        raise ValueError("Target column must not be included in model features")

    x = working[feature_cols].copy()
    y = working[target].to_numpy(dtype=int)
    return x, y


def get_feature_names(pipeline: ColumnTransformer) -> list[str]:
    """Extract transformed feature names from a fitted ColumnTransformer."""
    names: list[str] = []
    for name, transformer, columns in pipeline.transformers_:
        if name == "remainder":
            continue
        if isinstance(transformer, Pipeline) and hasattr(transformer[-1], "get_feature_names_out"):
            names.extend(list(transformer[-1].get_feature_names_out(columns)))
        elif hasattr(transformer, "get_feature_names_out"):
            names.extend(list(transformer.get_feature_names_out(columns)))
        else:
            names.extend([str(c) for c in columns])
    return names


def feature_metadata(config: AppConfig | None = None) -> dict[str, Any]:
    """Return a serialisable summary of configured features."""
    numeric, categorical = get_feature_lists(config)
    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "disclaimer": (config or get_config()).disclaimer,
    }
