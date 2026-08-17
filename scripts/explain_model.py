"""Generate champion explanations and validation subgroup analysis.

Loads train and validation only. Does not read the test set, does not retrain,
and does not replace the MLflow champion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.config import PROJECT_ROOT, get_config, resolve_path, set_seed, setup_logging
from src.data import load_dataset
from src.explainability.comparison import (
    COMPARISON_FAMILIES,
    permutation_for_comparison,
    try_tree_shap,
)
from src.explainability.linear import LinearExplanationModel, global_linear_explanation
from src.explainability.permutation import (
    compare_coefficient_and_permutation,
    permutation_importance_table,
)
from src.explainability.plots import (
    plot_coefficient_bars,
    plot_local_contributions,
    plot_permutation_bars,
    plot_subgroup_recall,
    plot_subgroup_review_rate,
)
from src.explainability.semantics import (
    SCORE_SEMANTICS_UNCALIBRATED,
    SCORE_WARNING,
    score_metadata_from_champion,
)
from src.explainability.subgroups import score_validation, subgroup_payload
from src.features import prepare_xy
from src.mlops.serialization import load_sklearn_pipeline
from src.mlops.serving import load_champion
from src.mlops.tracking import configure_tracking


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _annotate_champion_semantics(champion_path: Path, metadata: dict[str, Any]) -> None:
    """Add score-semantics fields without changing identity or selection fields."""
    identity = ("model_name", "mlflow_run_id", "threshold", "artifact_uri")
    current = json.loads(champion_path.read_text(encoding="utf-8"))
    semantics = score_metadata_from_champion(metadata)
    for key in identity:
        if str(current.get(key)) != str(metadata.get(key)):
            raise RuntimeError("Refusing to alter champion identity fields")
    current["score_is_calibrated"] = bool(semantics["score_is_calibrated"])
    current["score_semantics"] = (
        SCORE_SEMANTICS_UNCALIBRATED
        if not semantics["score_is_calibrated"]
        else semantics["score_semantics"]
    )
    current["calibration_method"] = semantics["calibration_method"]
    current["score_warning"] = SCORE_WARNING
    champion_path.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")


def main() -> None:
    logger = setup_logging(name="scripts.explain_model")
    cfg = get_config()
    set_seed(cfg.random_seed)
    logger.info("Disclaimer: %s", cfg.disclaimer)

    bundle = load_champion()
    metadata = dict(bundle.metadata)
    logger.info(
        "Explaining champion %s run_id=%s (no retraining)",
        metadata.get("model_name"),
        metadata.get("mlflow_run_id"),
    )

    processed_dir = resolve_path(str(cfg.data.get("processed_dir", "data/processed")))
    train_df = load_dataset(processed_dir / "train.csv")
    val_df = load_dataset(processed_dir / "val.csv")
    logger.info("Loaded train=%s val=%s. Test CSV is not read.", len(train_df), len(val_df))

    x_val, y_val = prepare_xy(val_df, cfg, fit_derived_reference=train_df)
    out_dir = resolve_path(str(cfg.explainability.get("output_dir", "artifacts/explanations")))
    out_dir.mkdir(parents=True, exist_ok=True)
    top_n = int(cfg.explainability.get("top_features", 15))
    n_repeats = int(cfg.explainability.get("permutation_repeats", 10))
    min_n = int(cfg.explainability.get("subgroup_min_n", 50))
    threshold = float(bundle.threshold)

    linear = LinearExplanationModel.from_pipeline(bundle.pipeline)
    global_payload = global_linear_explanation(bundle.pipeline, top_n=top_n)
    coef_table = linear.global_coefficients()
    grouped = linear.grouped_original_importance()
    coef_plot = plot_coefficient_bars(
        coef_table, out_dir / "champion_coefficients.png", top_n=top_n
    )
    global_payload["plot"] = _rel(coef_plot)
    global_payload["model_name"] = metadata.get("model_name")
    global_payload["model_version"] = metadata.get("model_version")
    global_payload["mlflow_run_id"] = metadata.get("mlflow_run_id")
    global_payload["split"] = "coefficients_from_fitted_champion"
    _write_json(out_dir / "global_coefficients.json", global_payload)
    coef_table.to_csv(out_dir / "global_coefficients.csv", index=False)
    grouped.to_csv(out_dir / "grouped_coefficients.csv", index=False)

    perm = permutation_importance_table(
        bundle.pipeline,
        x_val,
        y_val,
        n_repeats=n_repeats,
        seed=int(cfg.random_seed),
    )
    perm_plot = plot_permutation_bars(perm, out_dir / "permutation_importance.png", top_n=top_n)
    comparison = compare_coefficient_and_permutation(grouped, perm)
    perm_payload = {
        "role": "champion",
        "model_name": metadata.get("model_name"),
        "mlflow_run_id": metadata.get("mlflow_run_id"),
        "split": "validation",
        "n_repeats": n_repeats,
        "scoring": "average_precision",
        "rows": perm.to_dict(orient="records"),
        "plot": _rel(perm_plot),
        "comparison_with_coefficients": comparison,
    }
    _write_json(out_dir / "permutation_importance.json", perm_payload)
    perm.to_csv(out_dir / "permutation_importance.csv", index=False)
    pd.DataFrame(comparison["rows"]).to_csv(out_dir / "importance_comparison.csv", index=False)

    local_rows = int(cfg.explainability.get("local_example_rows", 3))
    local_summaries: list[dict[str, Any]] = []
    scores = score_validation(bundle.pipeline, x_val)
    example_indices = _example_indices(scores, local_rows)
    for position, row_index in enumerate(example_indices):
        row = x_val.iloc[[row_index]]
        local = linear.explain_row(row, threshold=threshold, metadata=metadata, top_n=top_n)
        local["validation_row_position"] = int(row_index)
        plot_path = plot_local_contributions(local, out_dir / f"local_explanation_{position}.png")
        local["plot"] = _rel(plot_path)
        _write_json(out_dir / f"local_explanation_{position}.json", local)
        local_summaries.append(
            {
                "validation_row_position": int(row_index),
                "review_score": local["review_score"],
                "requires_review": local["requires_review"],
                "reconstruction_error": local["reconstruction_error"],
                "plot": local["plot"],
            }
        )
    _write_json(
        out_dir / "local_explanations.json",
        {
            "method": "exact_logit_linear",
            "n_examples": len(local_summaries),
            "examples": local_summaries,
            "reconstruction_atol": global_payload["reconstruction_atol"],
        },
    )

    subgroup = subgroup_payload(val_df, y_val, scores, threshold=threshold, min_n=min_n)
    table = pd.DataFrame(subgroup["rows"])
    table.to_csv(out_dir / "subgroup_performance.csv", index=False)
    plot_names: dict[str, str] = {}
    for column in table["group_column"].unique():
        recall_path = plot_subgroup_recall(table, out_dir / f"subgroup_recall_{column}.png", column)
        rate_path = plot_subgroup_review_rate(
            table, out_dir / f"subgroup_rates_{column}.png", column
        )
        if recall_path is not None:
            plot_names[f"recall_{column}"] = _rel(recall_path)
        if rate_path is not None:
            plot_names[f"rates_{column}"] = _rel(rate_path)
    subgroup["plots"] = plot_names
    _write_json(out_dir / "subgroup_performance.json", subgroup)

    comparison_dir = out_dir / "comparison_models"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _explain_comparison_models(comparison_dir, x_val, y_val, cfg, logger)

    champion_path = resolve_path(
        str(cfg.mlops.get("champion_path", "artifacts/mlops/champion.json"))
    )
    _annotate_champion_semantics(champion_path, metadata)
    semantics = score_metadata_from_champion(json.loads(champion_path.read_text(encoding="utf-8")))
    _write_json(resolve_path("artifacts/mlops/score_semantics.json"), semantics)

    max_recon = max((item["reconstruction_error"] for item in local_summaries), default=0.0)
    logger.info("Wrote explanations to %s (max reconstruction error=%.3e)", out_dir, max_recon)


def _example_indices(scores: Any, n: int) -> list[int]:
    import numpy as np

    order = np.argsort(np.asarray(scores, dtype=float))
    if len(order) == 0:
        return []
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    unique: list[int] = []
    for index in picks:
        if index not in unique:
            unique.append(index)
        if len(unique) >= n:
            break
    return unique[:n]


def _explain_comparison_models(
    out_dir: Path,
    x_val: pd.DataFrame,
    y_val: Any,
    cfg: Any,
    logger: Any,
) -> None:
    records_path = resolve_path("artifacts/mlops/experiment_records.json")
    summary: dict[str, Any] = {"role": "comparison_models_not_champion", "models": []}
    if not records_path.exists():
        summary["reason"] = "No experiment records; comparison explanations skipped."
        _write_json(out_dir / "summary.json", summary)
        return
    records = json.loads(records_path.read_text(encoding="utf-8"))
    configure_tracking()
    for family in COMPARISON_FAMILIES:
        match = next(
            (
                item
                for item in records
                if item.get("model_family") == family and item.get("roundtrip_ok")
            ),
            None,
        )
        if match is None:
            summary["models"].append({"model_family": family, "available": False})
            continue
        try:
            pipeline = load_sklearn_pipeline(str(match["artifact_uri"]))
        except Exception as exc:  # noqa: BLE001
            summary["models"].append(
                {
                    "model_family": family,
                    "available": False,
                    "reason": type(exc).__name__,
                }
            )
            continue
        perm = permutation_for_comparison(
            pipeline,
            x_val,
            y_val,
            model_family=family,
            n_repeats=int(cfg.explainability.get("permutation_repeats", 10)),
            seed=int(cfg.random_seed),
        )
        shap_payload = try_tree_shap(
            pipeline,
            x_val,
            model_family=family,
            max_samples=int(cfg.explainability.get("max_samples", 200)),
            seed=int(cfg.random_seed),
        )
        _write_json(out_dir / f"{family}_permutation.json", perm)
        _write_json(out_dir / f"{family}_shap.json", shap_payload)
        pd.DataFrame(perm["rows"]).to_csv(out_dir / f"{family}_permutation.csv", index=False)
        summary["models"].append(
            {
                "model_family": family,
                "available": True,
                "mlflow_run_id": match.get("run_id"),
                "shap_available": bool(shap_payload.get("available")),
                "shap_reason": shap_payload.get("reason"),
            }
        )
        logger.info(
            "Comparison explanations for %s (shap=%s)", family, shap_payload.get("available")
        )
    _write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
