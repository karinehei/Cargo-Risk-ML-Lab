"""Optional explanations for comparison models. These are not the champion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.explainability.permutation import permutation_importance_table
from src.explainability.semantics import CAUSATION_DISCLAIMER

COMPARISON_FAMILIES = ("random_forest", "xgboost")


def permutation_for_comparison(
    pipeline: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    model_family: str,
    n_repeats: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    table = permutation_importance_table(pipeline, x_val, y_val, n_repeats=n_repeats, seed=seed)
    return {
        "role": "comparison_model_not_champion",
        "model_family": model_family,
        "split": "validation",
        "method": "permutation_importance_average_precision",
        "rows": table.to_dict(orient="records"),
        "causation_disclaimer": CAUSATION_DISCLAIMER,
        "note": (
            f"{model_family} is a comparison candidate, not the deployed champion. "
            "Do not treat these importances as explanations of the logistic-regression champion."
        ),
    }


def try_tree_shap(
    pipeline: Any,
    x_val: pd.DataFrame,
    *,
    model_family: str,
    max_samples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Attempt TreeExplainer on a comparison model. Never relax deserialization safety."""
    payload: dict[str, Any] = {
        "role": "comparison_model_not_champion",
        "model_family": model_family,
        "method": "shap_tree_explainer",
        "available": False,
    }
    try:
        import shap

        from src.features import get_feature_names
    except Exception as exc:  # noqa: BLE001
        payload["reason"] = f"SHAP import failed: {type(exc).__name__}"
        return payload

    try:
        model = pipeline.named_steps["model"]
        sample = x_val.sample(n=min(max_samples, len(x_val)), random_state=seed)
        transformed = np.asarray(pipeline.named_steps["preprocess"].transform(sample), dtype=float)
        names = get_feature_names(pipeline.named_steps["preprocess"])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            matrix = np.asarray(shap_values[-1], dtype=float)
        else:
            matrix = np.asarray(shap_values, dtype=float)
        mean_abs = np.abs(matrix).mean(axis=0)
        importance = (
            pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        payload.update(
            {
                "available": True,
                "n_samples": int(len(sample)),
                "top_features": importance.head(15).to_dict(orient="records"),
                "note": (
                    f"SHAP TreeExplainer values for the {model_family} comparison model only. "
                    "They do not explain the logistic-regression champion."
                ),
                "causation_disclaimer": CAUSATION_DISCLAIMER,
            }
        )
        return payload
    except Exception as exc:  # noqa: BLE001
        payload["reason"] = (
            f"SHAP TreeExplainer is incompatible with this {model_family} estimator "
            f"({type(exc).__name__}: {exc}). Permutation importance is retained. "
            "Dependency trust checks were not disabled."
        )
        return payload
