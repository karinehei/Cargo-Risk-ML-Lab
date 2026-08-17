"""Model-agnostic permutation importance on validation features only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.explainability.semantics import CAUSATION_DISCLAIMER
from src.features import get_feature_lists


def permutation_importance_table(
    pipeline: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Permute original input columns on validation data (never the test set)."""
    numeric, categorical = get_feature_lists()
    columns = [column for column in [*numeric, *categorical] if column in x_val.columns]
    features = x_val[columns]
    result = permutation_importance(
        pipeline,
        features,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    frame["abs_importance"] = frame["importance_mean"].abs()
    return frame.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def compare_coefficient_and_permutation(
    grouped_coefficients: pd.DataFrame,
    permutation: pd.DataFrame,
) -> dict[str, Any]:
    """Join grouped |coefficients| with permutation importance and explain disagreement."""
    left = grouped_coefficients.rename(
        columns={"source_feature": "feature", "l2_transformed_coefficient": "coefficient_l2"}
    )
    merged = left.merge(permutation, on="feature", how="outer")
    merged["coefficient_rank"] = merged["coefficient_l2"].rank(ascending=False, method="min")
    merged["permutation_rank"] = merged["importance_mean"].rank(ascending=False, method="min")
    merged["rank_gap"] = merged["coefficient_rank"] - merged["permutation_rank"]
    disagreement = (
        "Coefficient magnitude is the weight on a transformed feature (one standard "
        "deviation for scaled numerics; one dummy for a category). Permutation importance "
        "shuffles the original column, so all of that column's dummies move together. "
        "Correlated features such as declared_value_eur, log_declared_value and "
        "value_to_weight_ratio can share credit: a large coefficient may look weak under "
        "permutation if a correlated column still carries the same signal. Neither view "
        "is causal."
    )
    return {
        "rows": merged.sort_values("permutation_rank", na_position="last").to_dict(
            orient="records"
        ),
        "disagreement_note": disagreement,
        "causation_disclaimer": CAUSATION_DISCLAIMER,
        "scoring": "average_precision",
        "split": "validation",
    }
