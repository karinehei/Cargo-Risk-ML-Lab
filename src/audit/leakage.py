"""Feature-by-feature leakage inventory and shipment-ID split checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import AppConfig, get_config
from src.data.schema import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    ID_COLUMN,
    PERIOD_COLUMN,
    TARGET_COLUMN,
)
from src.features import add_derived_features, get_feature_lists


def parse_shipment_row_index(shipment_id: str) -> int | None:
    """Extract the sequential row suffix from ``SYN-{seed}-{row}`` identifiers."""
    parts = str(shipment_id).split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def id_split_audit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_column: str = ID_COLUMN,
) -> dict[str, Any]:
    """Check uniqueness, disjointness, and whether IDs encode order."""
    folds = {"train": train, "val": val, "test": test}
    unique_ok = True
    duplicates: dict[str, int] = {}
    id_sets: dict[str, set[str]] = {}
    for name, frame in folds.items():
        series = frame[id_column].astype(str)
        duplicated = int(series.duplicated().sum())
        duplicates[name] = duplicated
        if duplicated:
            unique_ok = False
        id_sets[name] = set(series.tolist())

    overlap = {
        "train_val": sorted(id_sets["train"] & id_sets["val"])[:5],
        "train_test": sorted(id_sets["train"] & id_sets["test"])[:5],
        "val_test": sorted(id_sets["val"] & id_sets["test"])[:5],
        "n_train_val": len(id_sets["train"] & id_sets["val"]),
        "n_train_test": len(id_sets["train"] & id_sets["test"]),
        "n_val_test": len(id_sets["val"] & id_sets["test"]),
    }
    disjoint = (
        overlap["n_train_val"] == 0 and overlap["n_train_test"] == 0 and overlap["n_val_test"] == 0
    )

    full = pd.concat(
        [
            train[[id_column, PERIOD_COLUMN]],
            val[[id_column, PERIOD_COLUMN]],
            test[[id_column, PERIOD_COLUMN]],
        ],
        ignore_index=True,
    )
    indices = full[id_column].map(parse_shipment_row_index)
    order_corr = float("nan")
    if indices.notna().sum() > 20:
        aligned = pd.DataFrame(
            {"row_index": pd.to_numeric(indices, errors="coerce"), "period": full[PERIOD_COLUMN]}
        ).dropna()
        if aligned["row_index"].nunique() > 1 and aligned["period"].nunique() > 1:
            order_corr = float(aligned["row_index"].corr(aligned["period"]))

    numeric, categorical = get_feature_lists()
    model_features = set(numeric) | set(categorical)
    id_used_as_feature = id_column in model_features or any(
        "shipment_id" in name.lower() or name.endswith("_id") for name in model_features
    )

    return {
        "unique_within_folds": unique_ok,
        "duplicates_per_fold": duplicates,
        "disjoint_across_folds": disjoint,
        "overlap": overlap,
        "n_train_ids": len(id_sets["train"]),
        "n_val_ids": len(id_sets["val"]),
        "n_test_ids": len(id_sets["test"]),
        "id_row_index_period_correlation": order_corr,
        "shipment_id_is_model_feature": id_used_as_feature,
        "sender_or_entity_id_present": any(
            col in set(train.columns) | set(val.columns) | set(test.columns)
            for col in ("sender_id", "entity_id", "company_id", "person_id")
        ),
        "notes": (
            "IDs are unique synthetic keys (SYN-{seed}-{row}). The numeric suffix "
            "tracks generation order and therefore correlates with generation_period, "
            "but shipment_id is excluded from model features. There is no sender_id, "
            "so the same entity cannot straddle splits."
        ),
    }


def feature_leakage_inventory(
    train_df: pd.DataFrame,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Inspect every generated / derived feature for direct or indirect target leakage.

    Associations are measured on **training** rows only. A non-zero association with
    the label is expected for predictive features and is not by itself leakage.
    Leakage here means using the label, a latent score, a post-outcome field, or an
    identifier that could leak split membership.
    """
    cfg = config or get_config()
    working = add_derived_features(train_df)
    numeric, categorical = get_feature_lists(cfg)
    rows: list[dict[str, Any]] = []

    excluded = [ID_COLUMN, DATE_COLUMN, PERIOD_COLUMN, TARGET_COLUMN]
    for column in excluded:
        rows.append(
            {
                "feature": column,
                "used_in_model": False,
                "kind": "excluded",
                "direct_leakage": column == TARGET_COLUMN,
                "indirect_leakage": False,
                "abs_spearman_with_target": float("nan"),
                "rationale": _exclusion_rationale(column),
            }
        )

    y = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")
    for column in [*numeric, *categorical]:
        series = working[column]
        spearman = float("nan")
        numeric_col = pd.to_numeric(series, errors="coerce")
        aligned = pd.DataFrame({"x": numeric_col, "y": y}).dropna()
        if aligned["x"].nunique() > 1 and aligned["y"].nunique() > 1:
            spearman = float(aligned["x"].corr(aligned["y"], method="spearman"))
        rows.append(
            {
                "feature": column,
                "used_in_model": True,
                "kind": "numeric" if column in numeric else "categorical",
                "direct_leakage": column == TARGET_COLUMN or column in FORBIDDEN_COLUMNS,
                "indirect_leakage": False,
                "abs_spearman_with_target": abs(spearman)
                if np.isfinite(spearman)
                else float("nan"),
                "rationale": _feature_rationale(column),
            }
        )

    for column in FEATURE_COLUMNS:
        if column in numeric or column in categorical:
            continue
        rows.append(
            {
                "feature": column,
                "used_in_model": False,
                "kind": "raw_unused",
                "direct_leakage": False,
                "indirect_leakage": False,
                "abs_spearman_with_target": float("nan"),
                "rationale": "Present in the raw table but not in the configured model feature list.",
            }
        )
    return rows


def _exclusion_rationale(column: str) -> str:
    if column == TARGET_COLUMN:
        return "Label column. Using it as a feature would be direct target leakage."
    if column == ID_COLUMN:
        return (
            "Unique synthetic identifier. The suffix encodes row order (and therefore time) "
            "so it is excluded to prevent split/order leakage."
        )
    if column == DATE_COLUMN:
        return "Calendar date used only for documentation and drift simulation, not as a model feature."
    if column == PERIOD_COLUMN:
        return "Generation era used for optional time splits and drift, not as a model feature."
    return "Excluded identifier or time field."


def _feature_rationale(column: str) -> str:
    notes: dict[str, str] = {
        "declared_value_eur": (
            "Generated before labels. Used as a noisy input to the fictional score. Not computed from y."
        ),
        "shipment_weight_kg": "Generated from value plus noise before labels. Not computed from y.",
        "value_to_weight_ratio": (
            "Deterministic ratio of two raw features. Collinear with value and weight, not leakage."
        ),
        "declaration_completeness_score": "Generated before labels; later used in the fictional score.",
        "documentation_count": "Generated before labels. Missingness is MCAR/MAR on completeness, not on y.",
        "previous_discrepancies": "Per-row fictional count, not a shared sender key.",
        "sender_history_length": (
            "Per-row history length. There is no sender_id, so this cannot join other rows' labels."
        ),
        "route_rarity": "Generated before labels; drifted by period only.",
        "declared_vs_estimated_value_deviation": (
            "Uses an internal toy estimated value that is not saved. The deviation is a feature, "
            "not a second copy of the label."
        ),
        "submission_hour": "Generated before labels. Off-hours is a deterministic transform of this field.",
        "expedited_shipment": "Generated before labels; also used in interaction terms of the toy score.",
        "transport_mode": "Generated before labels. Air appears in some fictional interactions.",
        "origin_region": "Coarse corridor label generated before y. Not a personal attribute.",
        "destination_region": "Coarse corridor label generated before y.",
        "commodity_category": "Generated before labels. Some categories interact with air in the toy score.",
        "log_declared_value": "Deterministic log1p of declared_value_eur. Redundant transform, not leakage.",
        "is_off_hours": "Deterministic function of submission_hour. Redundant transform, not leakage.",
    }
    return notes.get(
        column,
        "Row-wise feature created before the label; not a function of requires_review.",
    )
