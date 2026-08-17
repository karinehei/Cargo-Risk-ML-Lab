"""Optional TreeExplainer helpers for comparison tree models.

The deployed champion is logistic regression and uses exact logit
decomposition in ``src.explainability.linear``. Do not treat this module as
the champion explanation path.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import AppConfig, get_config, resolve_path, set_seed, setup_logging
from src.features import get_feature_names

logger = setup_logging(name="src.explainability")


def _transformed_matrix(pipeline: Pipeline, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Transform raw features and return dense matrix plus feature names."""
    preprocess = pipeline.named_steps["preprocess"]
    matrix = np.asarray(preprocess.transform(features), dtype=float)
    names = get_feature_names(preprocess)
    return matrix, names


def explain_global(
    pipeline: Pipeline,
    features: pd.DataFrame,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Compute global SHAP summary values and persist a beeswarm plot.

    Returns:
        Dictionary with mean absolute SHAP values and artifact paths.
    """
    cfg = config or get_config()
    set_seed(cfg.random_seed)
    max_samples = int(cfg.explainability.get("max_samples", 200))
    sample = features.sample(n=min(max_samples, len(features)), random_state=cfg.random_seed)

    x_trans, feature_names = _transformed_matrix(pipeline, sample)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_trans)
    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[1], dtype=float)
    else:
        shap_matrix = np.asarray(shap_values, dtype=float)

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_dir = resolve_path(str(cfg.explainability.get("output_dir", "artifacts/explanations")))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "global_shap_importance.csv"
    importance.to_csv(csv_path, index=False)

    top_n = int(cfg.explainability.get("top_features", 15))
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_matrix,
        features=x_trans,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
    )
    plot_path = out_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()

    payload = {
        "n_samples": int(len(sample)),
        "top_features": importance.head(top_n).to_dict(orient="records"),
        "importance_path": str(csv_path),
        "plot_path": str(plot_path),
        "disclaimer": cfg.disclaimer,
    }
    meta_path = out_dir / "global_explanation.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved global SHAP artifacts to %s", out_dir)
    return payload


def explain_local(
    pipeline: Pipeline,
    features: pd.DataFrame,
    row_index: int = 0,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Compute a local SHAP explanation for a single row.

    Args:
        pipeline: Fitted sklearn pipeline.
        features: Feature frame in raw (pre-transform) space.
        row_index: Integer position of the row to explain.
        config: Optional config override.

    Returns:
        Dictionary with local contributions and optional plot path.
    """
    cfg = config or get_config()
    if row_index < 0 or row_index >= len(features):
        raise IndexError(f"row_index {row_index} out of bounds for length {len(features)}")

    row = features.iloc[[row_index]]
    x_trans, feature_names = _transformed_matrix(pipeline, row)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_trans)
    if isinstance(shap_values, list):
        local_values = np.asarray(shap_values[1][0], dtype=float)
    else:
        local_values = np.asarray(shap_values[0], dtype=float)

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        base_value = float(np.asarray(expected).ravel()[-1])
    else:
        base_value = float(expected)

    contributions = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": local_values,
                "abs_shap": np.abs(local_values),
                "feature_value": x_trans[0],
            }
        )
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_dir = resolve_path(str(cfg.explainability.get("output_dir", "artifacts/explanations")))
    out_dir.mkdir(parents=True, exist_ok=True)
    top_n = int(cfg.explainability.get("top_features", 15))
    top = contributions.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in top["shap_value"]]
    ax.barh(top["feature"][::-1], top["shap_value"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on high-risk score)")
    ax.set_title(f"Local explanation for row {row_index} (synthetic)")
    plot_path = out_dir / f"shap_local_row_{row_index}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    probability = float(pipeline.predict_proba(row)[0, 1])
    payload = {
        "row_index": row_index,
        "base_value": base_value,
        "predicted_probability": probability,
        "top_contributions": top.to_dict(orient="records"),
        "plot_path": str(plot_path),
        "disclaimer": cfg.disclaimer,
    }
    meta_path = out_dir / f"local_explanation_row_{row_index}.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved local SHAP explanation for row %s", row_index)
    return payload
