"""Deterministic synthetic monitoring batches for drift demonstrations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config, resolve_path, set_seed, setup_logging
from src.data.generate import (
    assign_requires_review,
    generate_feature_table,
    introduce_missingness,
    save_dataset,
)
from src.data.schema import (
    COMMODITY_CATEGORIES,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    ID_COLUMN,
    PERIOD_COLUMN,
    RAW_COLUMNS,
    TARGET_COLUMN,
)

logger = setup_logging(name="src.monitoring.scenarios")

ScenarioName = Literal[
    "none",
    "subtle",
    "moderate",
    "major",
    "missingness",
    "unseen_category",
]

MONITORING_PERIOD_BASE = 10
SCENARIO_SEEDS: dict[str, int] = {
    "none": 91001,
    "moderate": 91002,
    "major": 91003,
    "missingness": 91004,
    "unseen_category": 91005,
    "labelled_simulation": 91006,
    "subtle": 91007,
}
NULL_SEED_BASE = 92001
VALIDATION_SEEDS: dict[str, list[int]] = {
    "none": [93001, 93002, 93003, 93004, 93005],
    "subtle": [93101],
    "moderate": [93102],
    "major": [93103],
    "missingness": [93104],
    "unseen_category": [93105],
}


def _monitoring_period(scenario: str) -> int:
    mapping = {
        "none": MONITORING_PERIOD_BASE,
        "subtle": MONITORING_PERIOD_BASE + 6,
        "moderate": MONITORING_PERIOD_BASE + 1,
        "major": MONITORING_PERIOD_BASE + 2,
        "missingness": MONITORING_PERIOD_BASE + 3,
        "unseen_category": MONITORING_PERIOD_BASE + 4,
        "labelled_simulation": MONITORING_PERIOD_BASE + 5,
    }
    return mapping[scenario]


def _apply_scenario_drift(
    features: pd.DataFrame, scenario: ScenarioName, rng: np.random.Generator
) -> pd.DataFrame:
    """Apply covariate drift without manipulating the target directly."""
    out = features.copy()
    n_rows = len(out)
    if scenario == "none":
        return out

    if scenario == "subtle":
        transport = out["transport_mode"].to_numpy(copy=True)
        air_mask = transport == "road"
        transport[air_mask & (rng.random(n_rows) < 0.04)] = "air"
        out["transport_mode"] = transport
        out["declaration_completeness_score"] = np.clip(
            out["declaration_completeness_score"] - 0.02, 0.05, 1.0
        )
        out["declared_value_eur"] = np.round(
            out["declared_value_eur"] * rng.lognormal(0.0, 0.03, n_rows), 2
        )
        out["value_to_weight_ratio"] = np.round(
            out["declared_value_eur"] / np.maximum(out["shipment_weight_kg"], 0.1),
            4,
        )
        return out

    if scenario == "moderate":
        transport = out["transport_mode"].to_numpy(copy=True)
        air_mask = transport == "road"
        transport[air_mask & (rng.random(n_rows) < 0.18)] = "air"
        out["transport_mode"] = transport
        commodity = out["commodity_category"].to_numpy(copy=True)
        commodity[rng.random(n_rows) < 0.12] = "electronics"
        out["commodity_category"] = commodity
        out["declared_value_eur"] = np.round(
            out["declared_value_eur"] * rng.lognormal(0.0, 0.08, n_rows), 2
        )
        out["declaration_completeness_score"] = np.clip(
            out["declaration_completeness_score"] - 0.05, 0.05, 1.0
        )
        out["expedited_shipment"] = np.where(
            rng.random(n_rows) < 0.08, 1, out["expedited_shipment"]
        )
        out["value_to_weight_ratio"] = np.round(
            out["declared_value_eur"] / np.maximum(out["shipment_weight_kg"], 0.1),
            4,
        )
        return out

    if scenario == "major":
        transport = np.full(n_rows, "air", dtype=object)
        transport[rng.random(n_rows) < 0.15] = "sea"
        out["transport_mode"] = transport
        commodity = rng.choice(
            COMMODITY_CATEGORIES, size=n_rows, p=[0.35, 0.05, 0.10, 0.05, 0.15, 0.15, 0.05, 0.10]
        )
        out["commodity_category"] = commodity
        out["declared_value_eur"] = np.round(
            out["declared_value_eur"] * rng.lognormal(0.25, 0.18, n_rows), 2
        )
        out["shipment_weight_kg"] = np.round(
            out["shipment_weight_kg"] * rng.lognormal(-0.05, 0.12, n_rows), 2
        )
        out["declaration_completeness_score"] = np.clip(
            out["declaration_completeness_score"] - 0.12, 0.05, 1.0
        )
        out["declared_vs_estimated_value_deviation"] = np.clip(
            out["declared_vs_estimated_value_deviation"] + rng.normal(0.35, 0.15, n_rows),
            -1.5,
            6.0,
        )
        out["expedited_shipment"] = np.where(
            rng.random(n_rows) < 0.22, 1, out["expedited_shipment"]
        )
        out["value_to_weight_ratio"] = np.round(
            out["declared_value_eur"] / np.maximum(out["shipment_weight_kg"], 0.1),
            4,
        )
        return out

    if scenario == "missingness":
        return out

    if scenario == "unseen_category":
        transport = out["transport_mode"].to_numpy(copy=True)
        replace_mask = rng.random(n_rows) < 0.08
        transport[replace_mask] = "drone"
        out["transport_mode"] = transport
        commodity = out["commodity_category"].to_numpy(copy=True)
        commodity[rng.random(n_rows) < 0.05] = "luxury_goods"
        out["commodity_category"] = commodity
        return out

    raise ValueError(f"Unknown monitoring scenario: {scenario}")


def generate_monitoring_batch(
    scenario: ScenarioName,
    *,
    n_samples: int | None = None,
    seed: int | None = None,
    config: AppConfig | None = None,
    include_labels: bool = False,
    match_training_generator: bool | None = None,
) -> pd.DataFrame:
    """Generate a deterministic monitoring batch for the requested scenario."""
    cfg = config or get_config()
    scenario_seed = int(seed if seed is not None else SCENARIO_SEEDS[str(scenario)])
    resolved_seed = set_seed(scenario_seed)
    rng = np.random.default_rng(resolved_seed)
    batch_size = int(n_samples or cfg.monitoring.get("scenario_batch_size", 1200))
    if batch_size < 1:
        raise ValueError("Monitoring batch size must be >= 1")

    period = _monitoring_period(str(scenario))
    match_training = bool(
        match_training_generator if match_training_generator is not None else scenario == "none"
    )
    if match_training:
        period_fractions = list(cfg.data.get("period_fractions", [0.45, 0.20, 0.20, 0.15]))
        drift_start = int(cfg.data.get("drift_start_period", 2))
    else:
        period_fractions = [1.0]
        drift_start = 999
    features = generate_feature_table(
        batch_size,
        rng,
        seed=resolved_seed,
        period_fractions=period_fractions,
        drift_start_period=drift_start,
    )
    features[PERIOD_COLUMN] = period
    features[ID_COLUMN] = [
        f"MON-{scenario_seed}-{period:02d}-{idx:06d}" for idx in range(batch_size)
    ]
    features[DATE_COLUMN] = pd.to_datetime("2025-06-01") + pd.to_timedelta(
        np.arange(batch_size), unit="h"
    )
    features[DATE_COLUMN] = features[DATE_COLUMN].dt.strftime("%Y-%m-%d")

    features = _apply_scenario_drift(features, scenario, rng)
    missing_rates = dict(cfg.data.get("missing_rates", {}))
    if scenario == "missingness":
        missing_rates = {
            key: min(float(value) + 0.08, 0.35) for key, value in missing_rates.items()
        }
        missing_rates["declaration_completeness_score"] = min(
            float(missing_rates.get("declaration_completeness_score", 0.025)) + 0.12,
            0.35,
        )
    features = introduce_missingness(features, rng, missing_rates)

    frame = features.copy()
    if include_labels:
        frame[TARGET_COLUMN] = assign_requires_review(
            features,
            rng,
            target_positive_rate=float(cfg.data.get("target_positive_rate", 0.11)),
            label_flip_rate=float(cfg.data.get("label_flip_rate", 0.025)),
            logit_noise_std=float(cfg.data.get("logit_noise_std", 0.65)),
        )
        return frame.loc[:, RAW_COLUMNS].copy()

    monitoring_cols = [col for col in RAW_COLUMNS if col != TARGET_COLUMN]
    return frame.loc[:, monitoring_cols].copy()


def generate_null_batch(
    seed: int,
    *,
    n_samples: int | None = None,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Generate a no-intentional-shift batch using the training generator mix."""
    return generate_monitoring_batch(
        "none",
        n_samples=n_samples,
        seed=seed,
        config=config,
        match_training_generator=True,
    )


def save_monitoring_batch(
    frame: pd.DataFrame,
    scenario: str,
    *,
    config: AppConfig | None = None,
) -> str:
    cfg = config or get_config()
    directory = resolve_path(str(cfg.monitoring.get("scenario_dir", "data/monitoring")))
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"current_{scenario}.csv"
    save_dataset(frame, output)
    return str(output)


def load_monitoring_batch(scenario: str, *, config: AppConfig | None = None) -> pd.DataFrame:
    cfg = config or get_config()
    path = (
        resolve_path(str(cfg.monitoring.get("scenario_dir", "data/monitoring")))
        / f"current_{scenario}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Monitoring batch for scenario '{scenario}' is unavailable.")
    return pd.read_csv(path)


def scenario_metadata(scenario: str, frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    from src.mlops.fingerprints import dataframe_fingerprint

    feature_cols = [col for col in FEATURE_COLUMNS if col in frame.columns]
    return {
        "scenario": scenario,
        "seed": seed,
        "monitoring_period": _monitoring_period(scenario),
        "batch_size": int(len(frame)),
        "batch_fingerprint": dataframe_fingerprint(frame[feature_cols]),
        "includes_labels": TARGET_COLUMN in frame.columns,
    }
