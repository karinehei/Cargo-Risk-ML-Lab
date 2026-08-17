"""Run train/validation experiments, calibration and champion selection.

The held-out test set is not loaded. Frozen v1 artifacts are never written.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from src.config import get_config, resolve_path, set_seed, setup_logging
from src.data import load_dataset
from src.evaluation.metrics import compute_classification_metrics, save_evaluation_plots
from src.evaluation.threshold import select_threshold
from src.features import get_feature_lists, prepare_xy
from src.mlops.calibration import FROZEN_OPERATING_THRESHOLD, fit_calibration_candidates
from src.mlops.champion import save_champion, select_champion
from src.mlops.logging import log_candidate_run
from src.mlops.tracking import configure_tracking, init_tracking_store
from src.models import compare_models, predict_proba
from src.models.estimators import NEEDS_SCALING
from src.models.train import TrainedModelBundle, save_model_bundle


def _preprocess_config(family: str) -> dict[str, Any]:
    numeric, categorical = get_feature_lists()
    key = "logreg" if "logreg" in family else family
    return {
        "scale_numeric": NEEDS_SCALING.get(key, True),
        "numeric": ",".join(numeric),
        "categorical": ",".join(categorical),
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "one_hot_unknown": "ignore",
    }


def _pipeline_for_champion(comparison: Any, calibrated: Any, model_name: str) -> Any:
    for item in comparison.candidates:
        if item.name == model_name:
            return item.pipeline
    for candidate in calibrated:
        if candidate.name == model_name:
            return candidate.pipeline
    raise RuntimeError("Champion estimator is not in the current experiment pool")


def _threshold_policy(info: dict[str, Any]) -> str:
    return f"validation_fbeta_beta{info['beta']}_min_precision_{info['min_precision']}"


def main() -> None:
    """Fit on train, score on validation, log to MLflow, select a champion."""
    logger = setup_logging(name="scripts.run_mlops")
    cfg = get_config()
    set_seed(cfg.random_seed)
    logger.info("Disclaimer: %s", cfg.disclaimer)
    init_tracking_store()
    configure_tracking()

    processed_dir = resolve_path(str(cfg.data.get("processed_dir", "data/processed")))
    train_df = load_dataset(processed_dir / "train.csv")
    val_df = load_dataset(processed_dir / "val.csv")
    logger.info("Loaded train=%s val=%s. Test CSV is not read.", len(train_df), len(val_df))

    split_manifest = processed_dir / "split_manifest.json"
    x_train, y_train = prepare_xy(train_df, cfg, fit_derived_reference=train_df)
    x_val, y_val = prepare_xy(val_df, cfg, fit_derived_reference=train_df)
    fixture = x_val.head(min(16, len(x_val)))
    out_dir = resolve_path("artifacts/mlops")
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_models(train_df, val_df, config=cfg)
    records: list[dict[str, Any]] = []
    comparison_plots = out_dir / "comparison_plots"
    for item in comparison.candidates:
        info = select_threshold(y_val, item.val_probability, cfg, split_name="validation")
        metrics = compute_classification_metrics(
            y_val, item.val_probability, threshold=float(info["threshold"])
        )
        plot_paths = save_evaluation_plots(
            y_val,
            item.val_probability,
            comparison_plots,
            threshold=float(info["threshold"]),
            prefix=item.name,
        )
        family = item.name
        record = log_candidate_run(
            run_name=f"compare_{family}",
            model_family=family,
            pipeline=item.pipeline,
            hyperparameters=item.best_params,
            val_metrics=metrics,
            x_val=x_val,
            train_df=train_df,
            val_df=val_df,
            split_manifest_path=split_manifest,
            threshold=float(info["threshold"]),
            threshold_policy=_threshold_policy(info),
            cv_mean=item.cv_mean,
            cv_std=item.cv_std,
            class_weight="balanced" if family != "dummy" else "prior",
            calibration_status="none",
            preprocess_config=_preprocess_config(family),
            extra_tags={"stage": "model_comparison"},
            artifact_files=list(plot_paths.values()),
            fixture=fixture,
            config=cfg,
        )
        records.append(record)

    logger.info("Fitting calibration candidates on training data only")
    logreg_base = next(
        (item.pipeline for item in comparison.candidates if item.name == "logreg"), None
    )
    calibrated = fit_calibration_candidates(
        x_train, y_train, x_val, y_val, cfg, base_logreg=logreg_base
    )
    cal_rows: list[dict[str, Any]] = []
    plots_dir = out_dir / "calibration_plots"
    for candidate in calibrated:
        y_prob = predict_proba(candidate.pipeline, x_val)
        save_evaluation_plots(
            y_val,
            y_prob,
            plots_dir,
            threshold=candidate.selected_threshold,
            prefix=candidate.name,
        )
        record = log_candidate_run(
            run_name=f"calibrate_{candidate.name}",
            model_family=candidate.name,
            pipeline=candidate.pipeline,
            hyperparameters=candidate.hyperparameters,
            val_metrics=candidate.val_metrics_at_selected_threshold,
            x_val=x_val,
            train_df=train_df,
            val_df=val_df,
            split_manifest_path=split_manifest,
            threshold=candidate.selected_threshold,
            threshold_policy=candidate.threshold_policy,
            cv_mean=None,
            cv_std=None,
            class_weight=candidate.class_weight,
            calibration_status=candidate.calibration_status,
            preprocess_config=_preprocess_config("logreg"),
            extra_tags={"stage": "calibration"},
            extra_metrics={
                "val_pr_auc_frozen_threshold": float(
                    candidate.val_metrics_at_frozen_threshold["pr_auc"]
                ),
                "val_brier_frozen_threshold": float(
                    candidate.val_metrics_at_frozen_threshold["brier_score"]
                ),
                "val_recall_frozen_threshold": float(
                    candidate.val_metrics_at_frozen_threshold["recall"]
                ),
                "val_precision_frozen_threshold": float(
                    candidate.val_metrics_at_frozen_threshold["precision"]
                ),
                "frozen_operating_threshold": float(FROZEN_OPERATING_THRESHOLD),
            },
            artifact_files=list(plots_dir.glob(f"{candidate.name}_*")),
            fixture=fixture,
            config=cfg,
        )
        records.append(record)
        cal_rows.append(
            {
                "name": candidate.name,
                "calibration_status": candidate.calibration_status,
                "val_pr_auc": candidate.val_metrics_at_selected_threshold["pr_auc"],
                "val_brier": candidate.val_metrics_at_selected_threshold["brier_score"],
                "val_ece": candidate.val_metrics_at_selected_threshold["ece"],
                "val_precision_selected_t": candidate.val_metrics_at_selected_threshold[
                    "precision"
                ],
                "val_recall_selected_t": candidate.val_metrics_at_selected_threshold["recall"],
                "selected_threshold": candidate.selected_threshold,
                "val_precision_frozen_t": candidate.val_metrics_at_frozen_threshold["precision"],
                "val_recall_frozen_t": candidate.val_metrics_at_frozen_threshold["recall"],
                "frozen_threshold": FROZEN_OPERATING_THRESHOLD,
                "run_id": record["run_id"],
            }
        )

    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_comparison.csv", index=False)
    (out_dir / "calibration_comparison.json").write_text(
        json.dumps(cal_rows, indent=2), encoding="utf-8"
    )

    champion = select_champion(records, cfg, calibrated_in_pool=True)
    save_champion(champion, out_dir / "champion.json")
    (out_dir / "experiment_records.json").write_text(
        json.dumps(records, indent=2, default=str), encoding="utf-8"
    )

    winner_pipeline = _pipeline_for_champion(comparison, calibrated, champion.model_name)
    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(winner_pipeline, Pipeline) and "preprocess" in winner_pipeline.named_steps:
        bundle = TrainedModelBundle(
            pipeline=winner_pipeline,
            feature_names=list(comparison.bundle.feature_names),
            metadata={
                **champion.to_dict(),
                "disclaimer": cfg.disclaimer,
                "joblib_role": "recovery_export_not_primary",
            },
        )
        save_model_bundle(bundle, artifact_dir=export_dir)
    else:
        import joblib

        joblib.dump(winner_pipeline, export_dir / "model.joblib")
        (export_dir / "model_metadata.json").write_text(
            json.dumps(
                {**champion.to_dict(), "joblib_role": "recovery_export_not_primary"},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    logger.info("Champion %s run_id=%s", champion.model_name, champion.mlflow_run_id)
    logger.info("Reason: %s", champion.reason)


if __name__ == "__main__":
    main()
