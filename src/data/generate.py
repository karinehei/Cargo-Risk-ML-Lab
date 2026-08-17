"""Synthetic shipment generation for a fictional human-review classifier.

Generation is separated from preprocessing: this module writes a raw table
with features and a binary ``requires_review`` label. Feature values are
never created from the label. Later ``generation_period`` slices apply
controlled covariate drift without looking at the target.

All rules are fictional toys for education. They must not be used for
real customs, compliance or operational decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import AppConfig, get_config, resolve_path, set_seed, setup_logging
from src.data.schema import (
    COMMODITY_CATEGORIES,
    COMMODITY_UNIT_VALUE,
    DATE_COLUMN,
    DEFAULT_N_SAMPLES,
    DESTINATION_REGIONS,
    DISCLAIMER,
    FEATURE_COLUMNS,
    ID_COLUMN,
    ORIGIN_REGIONS,
    PERIOD_COLUMN,
    RAW_COLUMNS,
    TARGET_COLUMN,
    TRANSPORT_MODES,
)
from src.data.validate import DatasetValidationError, validate_dataset

logger = setup_logging(name="src.data.generate")


@dataclass(frozen=True)
class DatasetBundle:
    """Time-ordered train / validation / test splits plus the full frame."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=float)


def _calibrate_intercept(raw_scores: np.ndarray, target_rate: float) -> float:
    """Binary-search an intercept so mean sigmoid probability matches a rate."""
    low, high = -10.0, 6.0
    for _ in range(40):
        mid = (low + high) / 2.0
        rate = float(np.mean(_sigmoid(raw_scores + mid)))
        if rate > target_rate:
            high = mid
        else:
            low = mid
    return float((low + high) / 2.0)


def _period_from_index(n_samples: int, fractions: list[float]) -> np.ndarray:
    """Assign sequential generation periods so index order is time-like."""
    weights = np.asarray(fractions, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("period_fractions must sum to a positive value")
    weights = weights / weights.sum()
    counts = np.floor(weights * n_samples).astype(int)
    counts[-1] += n_samples - int(counts.sum())
    parts = [np.full(int(count), period, dtype=int) for period, count in enumerate(counts)]
    return np.concatenate(parts) if parts else np.zeros(n_samples, dtype=int)


def generate_feature_table(
    n_samples: int,
    rng: np.random.Generator,
    *,
    seed: int,
    period_fractions: list[float],
    drift_start_period: int,
) -> pd.DataFrame:
    """Generate raw features, including period-based covariate drift.

    The target is not created here and is never used to shift features.
    """
    size = int(n_samples)
    if size < 1:
        raise ValueError("n_samples must be >= 1")

    period = _period_from_index(size, period_fractions)
    day_offset = (np.arange(size) * 112 / max(size, 1)).astype(int)
    event_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(day_offset, unit="D")

    origin = rng.choice(ORIGIN_REGIONS, size=size, p=[0.18, 0.22, 0.15, 0.25, 0.12, 0.08])
    destination = rng.choice(DESTINATION_REGIONS, size=size, p=[0.40, 0.28, 0.20, 0.12])
    commodity = rng.choice(COMMODITY_CATEGORIES, size=size)
    transport = rng.choice(TRANSPORT_MODES, size=size, p=[0.46, 0.28, 0.16, 0.10])

    declared_value = np.round(rng.lognormal(mean=8.55, sigma=1.05, size=size), 2)
    weight_noise = rng.lognormal(mean=0.0, sigma=0.35, size=size)
    base_weight = np.exp(np.log(np.maximum(declared_value, 1.0)) * 0.35 + 1.8)
    shipment_weight = np.round(np.clip(base_weight * weight_noise, 0.2, 50_000.0), 2)

    unit = np.array([COMMODITY_UNIT_VALUE[str(cat)] for cat in commodity], dtype=float)
    estimated_value = np.maximum(
        shipment_weight * unit * rng.lognormal(0.0, 0.28, size=size),
        1.0,
    )
    deviation = np.clip((declared_value - estimated_value) / estimated_value, -1.5, 6.0)
    deviation = np.round(deviation, 4)

    completeness = np.clip(rng.beta(7.5, 2.2, size=size), 0.05, 1.0)
    documentation_count = rng.poisson(lam=6.0, size=size).clip(0, 30)
    previous_discrepancies = rng.poisson(lam=0.45, size=size).clip(0, 20)
    sender_history = rng.negative_binomial(n=4, p=0.18, size=size).clip(0, 180)
    route_rarity = np.clip(rng.beta(2.0, 5.5, size=size), 0.0, 1.0)

    hour_p = np.array([0.015] * 6 + [0.04] * 2 + [0.06] * 10 + [0.04] * 4 + [0.02] * 2, dtype=float)
    hour_p = hour_p / hour_p.sum()
    submission_hour = rng.choice(np.arange(24), size=size, p=hour_p)
    expedited = (rng.random(size) < 0.10).astype(int)

    # Covariate drift by period only — never conditioned on the label.
    mild = period >= int(drift_start_period)
    strong = period >= int(drift_start_period) + 1
    declared_value = np.round(declared_value * (1.0 + 0.12 * mild + 0.18 * strong), 2)
    completeness = np.clip(completeness * (1.0 - 0.06 * mild - 0.08 * strong), 0.05, 1.0)
    route_rarity = np.clip(route_rarity + 0.04 * mild + 0.08 * strong, 0.0, 1.0)
    previous_discrepancies = previous_discrepancies + (rng.poisson(0.35, size=size) * strong)
    previous_discrepancies = previous_discrepancies.clip(0, 20)
    extra_expedited = mild & (expedited == 0) & (rng.random(size) < np.where(strong, 0.10, 0.06))
    expedited = np.where(extra_expedited, 1, expedited).astype(int)
    recode_air = mild & (transport == "road") & (rng.random(size) < np.where(strong, 0.18, 0.10))
    transport = np.where(recode_air, "air", transport)

    value_to_weight = np.round(declared_value / np.maximum(shipment_weight, 0.1), 4)

    return pd.DataFrame(
        {
            ID_COLUMN: [f"SYN-{seed:04d}-{i:06d}" for i in range(size)],
            DATE_COLUMN: pd.to_datetime(event_date).strftime("%Y-%m-%d"),
            PERIOD_COLUMN: period,
            "declared_value_eur": declared_value,
            "shipment_weight_kg": shipment_weight,
            "value_to_weight_ratio": value_to_weight,
            "transport_mode": transport,
            "origin_region": origin,
            "destination_region": destination,
            "commodity_category": commodity,
            "declaration_completeness_score": np.round(completeness, 4),
            "documentation_count": documentation_count.astype(float),
            "previous_discrepancies": previous_discrepancies.astype(int),
            "sender_history_length": sender_history.astype(float),
            "route_rarity": np.round(route_rarity, 4),
            "declared_vs_estimated_value_deviation": deviation,
            "submission_hour": submission_hour.astype(int),
            "expedited_shipment": expedited,
        }
    )


def _raw_review_scores(features: pd.DataFrame) -> np.ndarray:
    """Fictional non-linear score used only to sample ``requires_review``.

    These interactions are invented for a learnable classroom problem.
    """
    value_ratio = features["value_to_weight_ratio"].to_numpy(dtype=float)
    completeness = features["declaration_completeness_score"].to_numpy(dtype=float)
    docs = features["documentation_count"].to_numpy(dtype=float)
    discrepancies = features["previous_discrepancies"].to_numpy(dtype=float)
    history = features["sender_history_length"].to_numpy(dtype=float)
    rarity = features["route_rarity"].to_numpy(dtype=float)
    deviation = features["declared_vs_estimated_value_deviation"].to_numpy(dtype=float)
    hour = features["submission_hour"].to_numpy(dtype=float)
    expedited = features["expedited_shipment"].to_numpy(dtype=float)
    declared = features["declared_value_eur"].to_numpy(dtype=float)
    weight = features["shipment_weight_kg"].to_numpy(dtype=float)
    transport = features["transport_mode"].to_numpy()
    commodity = features["commodity_category"].to_numpy()

    high_ratio = _sigmoid((value_ratio - 180.0) / 70.0)
    incomplete = _sigmoid((0.58 - completeness) / 0.12)
    sparse_docs = _sigmoid((3.0 - docs) / 1.2)
    new_sender = _sigmoid((8.0 - history) / 3.5)
    off_hours = ((hour < 6.0) | (hour >= 22.0)).astype(float)
    abs_dev = np.abs(deviation)
    air_sensitive = (
        (transport == "air") & np.isin(commodity, ["electronics", "pharmaceuticals", "chemicals"])
    ).astype(float)
    heavy_fast = ((expedited > 0) & (weight > 600.0)).astype(float)
    high_value = _sigmoid((np.log1p(declared) - 10.2) / 0.8)

    scores = (
        1.25 * high_ratio
        + 1.05 * incomplete * np.clip(abs_dev, 0.0, 2.5)
        + 0.85 * (discrepancies >= 2).astype(float) * new_sender
        + 0.70 * rarity * expedited
        + 0.55 * sparse_docs
        + 0.45 * off_hours
        + 0.60 * air_sensitive
        + 0.50 * heavy_fast
        + 0.35 * high_value * (transport == "air").astype(float)
        + 0.40 * ((deviation < -0.35) & (completeness < 0.7)).astype(float)
        + 0.25 * np.tanh(discrepancies / 2.0)
        - 0.35 * _sigmoid((history - 30.0) / 10.0)
    )
    return np.asarray(scores, dtype=float)


def assign_requires_review(
    features: pd.DataFrame,
    rng: np.random.Generator,
    *,
    target_positive_rate: float = 0.11,
    label_flip_rate: float = 0.025,
    logit_noise_std: float = 0.65,
) -> pd.Series:
    """Sample the binary label from features plus noise; do not mutate features."""
    filled = features.copy()
    numeric = filled.select_dtypes(include=["number"])
    filled[numeric.columns] = numeric.fillna(numeric.median())
    raw = _raw_review_scores(filled)
    noise = rng.normal(0.0, logit_noise_std, size=len(features))
    intercept = _calibrate_intercept(raw + noise, target_positive_rate)
    probability = _sigmoid(raw + noise + intercept)
    labels = (rng.random(len(features)) < probability).astype(int)
    flip = rng.random(len(features)) < label_flip_rate
    labels = np.where(flip, 1 - labels, labels).astype(int)
    return pd.Series(labels, index=features.index, name=TARGET_COLUMN)


def introduce_missingness(
    features: pd.DataFrame,
    rng: np.random.Generator,
    missing_rates: dict[str, float],
) -> pd.DataFrame:
    """Insert a small amount of missingness without using the target.

    Missingness is mostly MCAR, with a mild MAR boost on documentation when
    completeness is already low (still independent of the label).
    """
    out = features.copy()
    n_rows = len(out)
    completeness = out["declaration_completeness_score"].to_numpy(dtype=float)

    for column, rate in missing_rates.items():
        if column not in out.columns:
            raise KeyError(f"Unknown missingness column: {column}")
        base = np.full(n_rows, float(rate))
        if column == "documentation_count":
            base = np.clip(base + 0.03 * (completeness < 0.55).astype(float), 0.0, 0.12)
        mask = rng.random(n_rows) < base
        out.loc[mask, column] = np.nan
    return out


def generate_synthetic_shipments(
    n_samples: int | None = None,
    seed: int | None = None,
    config: AppConfig | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Generate a fully synthetic cargo shipment dataset.

    Args:
        n_samples: Number of rows to generate.
        seed: Random seed for reproducibility.
        config: Optional config override.
        validate: If True, run schema and distribution checks.

    Returns:
        DataFrame of synthetic shipments with binary ``requires_review``.
    """
    cfg = config or get_config()
    resolved_seed = set_seed(seed if seed is not None else cfg.random_seed)
    rng = np.random.default_rng(resolved_seed)
    size = int(n_samples if n_samples is not None else cfg.data.get("n_samples", DEFAULT_N_SAMPLES))
    period_fractions = list(cfg.data.get("period_fractions", [0.45, 0.20, 0.20, 0.15]))
    drift_start = int(cfg.data.get("drift_start_period", 2))
    target_rate = float(cfg.data.get("target_positive_rate", 0.11))
    flip_rate = float(cfg.data.get("label_flip_rate", 0.025))
    noise_std = float(cfg.data.get("logit_noise_std", 0.65))
    missing_rates = dict(
        cfg.data.get(
            "missing_rates",
            {
                "declaration_completeness_score": 0.025,
                "documentation_count": 0.02,
                "sender_history_length": 0.02,
                "route_rarity": 0.015,
                "declared_vs_estimated_value_deviation": 0.03,
            },
        )
    )

    logger.info("Generating %s synthetic shipments (seed=%s). %s", size, resolved_seed, DISCLAIMER)
    features = generate_feature_table(
        size,
        rng,
        seed=resolved_seed,
        period_fractions=period_fractions,
        drift_start_period=drift_start,
    )
    feature_snapshot = features[FEATURE_COLUMNS].copy()
    labelled = features.copy()
    labelled[TARGET_COLUMN] = assign_requires_review(
        features,
        rng,
        target_positive_rate=target_rate,
        label_flip_rate=flip_rate,
        logit_noise_std=noise_std,
    )
    pd.testing.assert_frame_equal(labelled[FEATURE_COLUMNS], feature_snapshot, check_dtype=False)

    labelled = introduce_missingness(labelled, rng, missing_rates)
    frame = labelled.loc[:, RAW_COLUMNS].copy()

    if validate:
        report = validate_dataset(frame, config=cfg)
        logger.info(
            "Dataset valid: n=%s positive_rate=%.4f",
            report.n_rows,
            report.positive_rate,
        )
    else:
        logger.info("Skipped validation for generated frame with shape %s", frame.shape)
    return frame


def split_dataset(
    df: pd.DataFrame,
    config: AppConfig | None = None,
    seed: int | None = None,
    strategy: str | None = None,
) -> DatasetBundle:
    """Split into train / validation / test.

    Default strategy is stratified on the target so class balance is preserved
    and the test fold can be held out until model selection is finished.
    Pass ``strategy="time"`` to split on ``generation_period`` instead.
    """
    cfg = config or get_config()
    resolved = str(strategy or cfg.data.get("split_strategy", "stratified")).lower()
    if resolved == "time":
        return _time_split(df, cfg)
    if resolved == "stratified":
        return _stratified_split(df, cfg, seed=seed)
    raise ValueError(f"Unknown split strategy: {resolved}")


def _time_split(df: pd.DataFrame, cfg: AppConfig) -> DatasetBundle:
    if PERIOD_COLUMN not in df.columns:
        raise DatasetValidationError(f"Missing {PERIOD_COLUMN} for time-like split")

    train_periods = set(cfg.data.get("train_periods", [0, 1]))
    val_periods = set(cfg.data.get("val_periods", [2]))
    test_periods = set(cfg.data.get("test_periods", [3]))

    train = df[df[PERIOD_COLUMN].isin(train_periods)].reset_index(drop=True)
    val = df[df[PERIOD_COLUMN].isin(val_periods)].reset_index(drop=True)
    test = df[df[PERIOD_COLUMN].isin(test_periods)].reset_index(drop=True)
    if train.empty or val.empty or test.empty:
        raise DatasetValidationError(
            f"Time split produced an empty fold: train={len(train)} val={len(val)} test={len(test)}"
        )
    logger.info("Time split sizes train=%s val=%s test=%s", len(train), len(val), len(test))
    return DatasetBundle(train=train, val=val, test=test, full=df.reset_index(drop=True))


def _stratified_split(
    df: pd.DataFrame,
    cfg: AppConfig,
    seed: int | None = None,
) -> DatasetBundle:
    target = str(cfg.data.get("target_column", TARGET_COLUMN))
    test_size = float(cfg.data.get("test_size", 0.2))
    val_size = float(cfg.data.get("val_size", 0.1))
    resolved_seed = int(seed if seed is not None else cfg.random_seed)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=resolved_seed,
        stratify=df[target],
    )
    relative_val = val_size / max(1.0 - test_size, 1e-6)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=resolved_seed,
        stratify=train_val[target],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    _assert_disjoint_ids(train, val, test, str(cfg.data.get("id_column", ID_COLUMN)))
    logger.info(
        "Stratified split sizes train=%s val=%s test=%s",
        len(train),
        len(val),
        len(test),
    )
    return DatasetBundle(train=train, val=val, test=test, full=df.reset_index(drop=True))


def _assert_disjoint_ids(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
) -> None:
    if id_column not in train.columns:
        return
    for fold_name, fold in (("train", train), ("val", val), ("test", test)):
        if fold[id_column].isna().any():
            raise DatasetValidationError(f"Missing {id_column} values in {fold_name} split")
        series = fold[id_column].astype(str)
        if series.duplicated().any():
            raise DatasetValidationError(f"Duplicate {id_column} values within {fold_name} split")
    train_ids = set(train[id_column].astype(str))
    val_ids = set(val[id_column].astype(str))
    test_ids = set(test[id_column].astype(str))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise DatasetValidationError("Split folds share shipment_id values")


def save_split_manifest(bundle: DatasetBundle, path: str | Path, strategy: str) -> Path:
    """Persist shipment IDs per fold so train/test separation can be audited."""
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": strategy,
        "train_ids": bundle.train[ID_COLUMN].astype(str).tolist(),
        "val_ids": bundle.val[ID_COLUMN].astype(str).tolist(),
        "test_ids": bundle.test[ID_COLUMN].astype(str).tolist(),
        "n_train": int(len(bundle.train)),
        "n_val": int(len(bundle.val)),
        "n_test": int(len(bundle.test)),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote split manifest to %s", output)
    return output


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a dataset as CSV and return the resolved path."""
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote dataset to %s (%s rows)", output, len(df))
    return output


def load_dataset(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a CSV dataset and optionally validate its schema."""
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    df = pd.read_csv(input_path)
    if validate:
        validate_dataset(df)
    return df


def build_and_persist_splits(config: AppConfig | None = None) -> DatasetBundle:
    """Generate synthetic data, validate, split by period, and persist artifacts."""
    cfg = config or get_config()
    df = generate_synthetic_shipments(config=cfg, validate=True)
    raw_path = save_dataset(df, str(cfg.data.get("raw_path", "data/raw/synthetic_shipments.csv")))

    report = validate_dataset(df, config=cfg)
    report_path = resolve_path(
        str(cfg.data.get("validation_report_path", "data/raw/validation_report.json"))
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_json(), encoding="utf-8")
    logger.info("Wrote validation report to %s", report_path)

    bundle = split_dataset(df, config=cfg)
    processed_dir = resolve_path(str(cfg.data.get("processed_dir", "data/processed")))
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(bundle.train, processed_dir / "train.csv")
    save_dataset(bundle.val, processed_dir / "val.csv")
    save_dataset(bundle.test, processed_dir / "test.csv")
    strategy = str(cfg.data.get("split_strategy", "stratified"))
    save_split_manifest(bundle, processed_dir / "split_manifest.json", strategy)
    logger.info("Raw data available at %s", raw_path)
    return bundle
