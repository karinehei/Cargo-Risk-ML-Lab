"""Schema, range, missingness and leakage validation for synthetic shipments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config, setup_logging
from src.data.schema import (
    ALLOWED_MISSING_COLUMNS,
    CATEGORY_VALUES,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    ID_COLUMN,
    NUMERIC_RANGES,
    PERIOD_COLUMN,
    POSITIVE_RATE_BOUNDS,
    PROTECTED_NAME_FRAGMENTS,
    RAW_COLUMNS,
    TARGET_COLUMN,
)

logger = setup_logging(name="src.data.validate")


class DatasetValidationError(ValueError):
    """Raised when a synthetic dataset fails integrity checks."""


@dataclass
class ValidationReport:
    """Machine-readable summary of dataset validation."""

    ok: bool
    n_rows: int
    n_columns: int
    positive_rate: float
    positive_rate_by_period: dict[str, float]
    missing_rates: dict[str, float]
    issues: list[str] = field(default_factory=list)
    disclaimer: str = (
        "Validation applies to fully synthetic educational data only. "
        "Fictional review rules must not be used for real decisions."
    )

    def to_json(self) -> str:
        """Serialize the report as indented JSON."""
        return json.dumps(asdict(self), indent=2, default=str)


def _missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def validate_dataset(
    df: pd.DataFrame,
    config: AppConfig | None = None,
    *,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate schema, ranges, categories, missingness, leakage and target rate.

    Args:
        df: Candidate shipment table.
        config: Optional config override for positive-rate bounds.
        raise_on_error: If True, raise ``DatasetValidationError`` on failures.

    Returns:
        ``ValidationReport`` describing the checks.
    """
    cfg = config or get_config()
    issues: list[str] = []
    data_cfg = cfg.data

    rate_min = float(data_cfg.get("positive_rate_min", POSITIVE_RATE_BOUNDS[0]))
    rate_max = float(data_cfg.get("positive_rate_max", POSITIVE_RATE_BOUNDS[1]))
    miss_min = float(data_cfg.get("missing_rate_min", 0.005))
    miss_max = float(data_cfg.get("missing_rate_max", 0.08))

    missing_cols = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    extra_forbidden = sorted(FORBIDDEN_COLUMNS.intersection(df.columns))
    if extra_forbidden:
        issues.append(f"Forbidden / leakage columns present: {extra_forbidden}")

    lower_names = {str(col).lower() for col in df.columns}
    protected_hits = [
        fragment
        for fragment in PROTECTED_NAME_FRAGMENTS
        if any(fragment in name for name in lower_names)
    ]
    if protected_hits:
        issues.append(f"Protected-characteristic column names detected: {protected_hits}")

    n_rows = int(len(df))
    if n_rows == 0:
        issues.append("Dataset is empty")

    positive_rate = float("nan")
    period_rates: dict[str, float] = {}
    missing_rates = {
        col: _missing_rate(df[col]) if col in df.columns else 1.0 for col in RAW_COLUMNS
    }

    if TARGET_COLUMN in df.columns and n_rows > 0:
        target = df[TARGET_COLUMN]
        if target.isna().any():
            issues.append(f"{TARGET_COLUMN} contains missing values")
        unique = set(pd.unique(target.dropna()))
        if not unique.issubset({0, 1}):
            issues.append(f"{TARGET_COLUMN} must be binary 0/1, found {sorted(unique, key=str)}")
        positive_rate = float(pd.to_numeric(target, errors="coerce").mean())
        if n_rows >= 2000 and not (rate_min <= positive_rate <= rate_max):
            issues.append(
                f"positive rate {positive_rate:.4f} outside [{rate_min:.2f}, {rate_max:.2f}]"
            )

    if ID_COLUMN in df.columns:
        if df[ID_COLUMN].isna().any():
            issues.append(f"{ID_COLUMN} contains missing values")
        if df[ID_COLUMN].duplicated().any():
            issues.append("Duplicate shipment_id values found")

    if PERIOD_COLUMN in df.columns:
        if df[PERIOD_COLUMN].isna().any():
            issues.append(f"{PERIOD_COLUMN} contains missing values")
        if n_rows > 0:
            period_rates = {
                str(int(period)): float(group[TARGET_COLUMN].mean())
                for period, group in df.groupby(PERIOD_COLUMN)
                if TARGET_COLUMN in group.columns and len(group) > 0
            }

    if DATE_COLUMN in df.columns:
        parsed = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        if parsed.isna().any():
            issues.append(f"{DATE_COLUMN} contains unparseable dates")
        if n_rows > 1 and not bool(parsed.is_monotonic_increasing):
            logger.info(
                "%s is not globally sorted; time-like split still uses %s",
                DATE_COLUMN,
                PERIOD_COLUMN,
            )

    for column, allowed in CATEGORY_VALUES.items():
        if column not in df.columns:
            continue
        observed = set(df[column].dropna().astype(str).unique())
        unexpected = sorted(observed - set(allowed))
        if unexpected:
            issues.append(f"{column} has unexpected categories: {unexpected}")

    for column, (low, high) in NUMERIC_RANGES.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        if float(finite.min()) < low or float(finite.max()) > high:
            issues.append(
                f"{column} out of range [{low}, {high}]: min={finite.min()}, max={finite.max()}"
            )

    for column in df.columns:
        if column in ALLOWED_MISSING_COLUMNS or column not in RAW_COLUMNS:
            continue
        rate = _missing_rate(df[column])
        if rate > 0:
            issues.append(f"{column} is not allowed to contain missing values (rate={rate:.4f})")

    observed_missing = {
        col: missing_rates[col]
        for col in ALLOWED_MISSING_COLUMNS
        if col in missing_rates and missing_rates[col] > 0
    }
    if n_rows >= 500:
        if not observed_missing:
            issues.append("Expected a small amount of missing data in allowed columns")
        for column, rate in observed_missing.items():
            if rate < miss_min or rate > miss_max:
                issues.append(
                    f"{column} missing rate {rate:.4f} outside [{miss_min:.3f}, {miss_max:.3f}]"
                )

    if TARGET_COLUMN in df.columns:
        for column in FEATURE_COLUMNS:
            if column not in df.columns:
                continue
            if df[column].nunique(dropna=True) <= 1:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.notna().sum() < 20:
                continue
            aligned = pd.DataFrame(
                {"x": numeric, "y": pd.to_numeric(df[TARGET_COLUMN], errors="coerce")}
            ).dropna()
            if aligned["x"].nunique() < 2:
                continue
            corr = float(aligned["x"].corr(aligned["y"]))
            if np.isfinite(corr) and abs(corr) >= 0.98:
                issues.append(
                    f"Possible target leakage: |corr({column}, {TARGET_COLUMN})|={corr:.3f}"
                )

        for column in FEATURE_COLUMNS:
            if column not in df.columns:
                continue
            if df[column].equals(df[TARGET_COLUMN]):
                issues.append(f"Feature {column} is identical to {TARGET_COLUMN}")

    report = ValidationReport(
        ok=not issues,
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        positive_rate=positive_rate,
        positive_rate_by_period=period_rates,
        missing_rates=missing_rates,
        issues=issues,
    )
    if issues:
        logger.warning("Dataset validation failed: %s", issues)
        if raise_on_error:
            raise DatasetValidationError("; ".join(issues))
    else:
        logger.info("Dataset validation passed for %s rows", n_rows)
    return report


def validate_no_target_leakage(df: pd.DataFrame) -> None:
    """Raise if modelling features include the target or forbidden columns."""
    overlap = set(FEATURE_COLUMNS).intersection({TARGET_COLUMN})
    if overlap:
        raise DatasetValidationError("Target is listed as a feature")
    present_forbidden = FORBIDDEN_COLUMNS.intersection(df.columns)
    if present_forbidden:
        raise DatasetValidationError(f"Forbidden columns present: {sorted(present_forbidden)}")
    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise DatasetValidationError("Target leakage in FEATURE_COLUMNS")
    extras = [col for col in df.columns if col in {"review_probability_latent", "logit"}]
    if extras:
        raise DatasetValidationError(f"Latent label columns present: {extras}")
