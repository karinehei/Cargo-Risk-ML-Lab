"""Versioned reference profiles built from train-derived summaries only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config, resolve_path, setup_logging
from src.data.schema import (
    CATEGORY_VALUES,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    ID_COLUMN,
    PERIOD_COLUMN,
    TARGET_COLUMN,
)
from src.mlops.fingerprints import dataframe_fingerprint
from src.monitoring.policy import MONITORING_POLICY_VERSION

logger = setup_logging(name="src.monitoring.reference")

PROFILE_VERSION = "1.0.0"

FORBIDDEN_PROFILE_FIELDS = frozenset(
    {
        TARGET_COLUMN,
        ID_COLUMN,
        DATE_COLUMN,
        PERIOD_COLUMN,
        "review_probability_latent",
        "latent_score",
        "logit",
        "true_risk",
        "inspector_decision",
    }
)


def _numeric_summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return {
            "count": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "missing_rate": float(values.isna().mean()),
        }
    return {
        "count": float(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "q25": float(finite.quantile(0.25)),
        "q50": float(finite.quantile(0.50)),
        "q75": float(finite.quantile(0.75)),
        "missing_rate": float(values.isna().mean()),
    }


def _categorical_frequencies(series: pd.Series) -> dict[str, float]:
    counts = series.astype(str).replace("nan", "__MISSING__").value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.items()}


def validate_profile_payload(payload: dict[str, Any]) -> None:
    """Ensure the profile does not contain forbidden fields or raw records."""
    forbidden = FORBIDDEN_PROFILE_FIELDS.intersection(payload.keys())
    if forbidden:
        raise ValueError(
            f"Reference profile contains forbidden top-level fields: {sorted(forbidden)}"
        )
    for key in ("raw_records", "rows", "records", "shipments"):
        if key in payload:
            raise ValueError(f"Reference profile must not contain raw records under '{key}'.")
    nested_keys = (
        "numerical_summaries",
        "categorical_frequencies",
        "missing_value_rates",
        "expected_categories",
    )
    for section in nested_keys:
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            continue
        overlap = FORBIDDEN_PROFILE_FIELDS.intersection(section_payload.keys())
        if overlap:
            raise ValueError(f"Reference profile section '{section}' contains forbidden fields.")


def build_reference_profile(
    frame: pd.DataFrame,
    *,
    champion_metadata: dict[str, Any],
    scores: np.ndarray,
    threshold: float,
    seed: int,
    source: str,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Build a versioned reference profile from aggregate statistics only."""
    cfg = config or get_config()
    feature_cols = [col for col in FEATURE_COLUMNS if col in frame.columns]
    overlap = FORBIDDEN_PROFILE_FIELDS.intersection(frame.columns)
    if overlap:
        raise ValueError(f"Reference input frame contains forbidden columns: {sorted(overlap)}")

    numeric_cols = list(cfg.features.get("numeric", []))
    categorical_cols = list(cfg.features.get("categorical", []))
    numerical_summaries = {
        col: _numeric_summary(frame[col]) for col in numeric_cols if col in frame.columns
    }
    categorical_frequencies = {
        col: _categorical_frequencies(frame[col])
        for col in categorical_cols
        if col in frame.columns
    }
    missing_value_rates = {
        col: float(frame[col].isna().mean()) for col in feature_cols if col in frame.columns
    }
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    score_distribution = {
        "mean": float(np.mean(finite_scores)) if len(finite_scores) else None,
        "std": float(np.std(finite_scores)) if len(finite_scores) else None,
        "q10": float(np.quantile(finite_scores, 0.10)) if len(finite_scores) else None,
        "q25": float(np.quantile(finite_scores, 0.25)) if len(finite_scores) else None,
        "q50": float(np.quantile(finite_scores, 0.50)) if len(finite_scores) else None,
        "q75": float(np.quantile(finite_scores, 0.75)) if len(finite_scores) else None,
        "q90": float(np.quantile(finite_scores, 0.90)) if len(finite_scores) else None,
    }
    predicted_review_rate = (
        float(np.mean(finite_scores >= threshold)) if len(finite_scores) else None
    )
    profile = {
        "profile_version": PROFILE_VERSION,
        "monitoring_policy_version": MONITORING_POLICY_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_fingerprint": dataframe_fingerprint(frame[feature_cols]),
        "feature_schema": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "feature_columns": feature_cols,
        },
        "expected_categories": {key: list(value) for key, value in CATEGORY_VALUES.items()},
        "numerical_summaries": numerical_summaries,
        "categorical_frequencies": categorical_frequencies,
        "missing_value_rates": missing_value_rates,
        "champion_version": str(champion_metadata.get("model_version") or ""),
        "model_name": str(champion_metadata.get("model_name") or ""),
        "mlflow_run_id": str(champion_metadata.get("mlflow_run_id") or ""),
        "decision_threshold": float(threshold),
        "score_distribution": score_distribution,
        "predicted_review_rate": predicted_review_rate,
        "n_rows": int(len(frame)),
        "seed": int(seed),
        "source": source,
        "disclaimer": cfg.disclaimer,
    }
    validate_profile_payload(profile)
    return profile


def save_reference_profile(
    profile: dict[str, Any],
    path: str | None = None,
    *,
    config: AppConfig | None = None,
) -> str:
    cfg = config or get_config()
    output = resolve_path(
        path
        or str(
            cfg.monitoring.get(
                "reference_profile_path", "artifacts/monitoring/reference_profile.json"
            )
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_profile_payload(profile)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Saved reference profile (%s rows)", profile.get("n_rows"))
    return str(output)


def load_reference_profile(
    path: str | None = None, *, config: AppConfig | None = None
) -> dict[str, Any]:
    cfg = config or get_config()
    profile_path = resolve_path(
        path
        or str(
            cfg.monitoring.get(
                "reference_profile_path", "artifacts/monitoring/reference_profile.json"
            )
        )
    )
    if not profile_path.exists():
        raise FileNotFoundError("Reference profile is unavailable.")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_profile_payload(payload)
    return cast(dict[str, Any], payload)
