"""Methodological audit CLI: train/val robustness, frozen test characterisation.

The live ``artifacts/metrics_test.json`` file is never overwritten. Frozen v1
files are copied to ``artifacts/frozen_v1/``. New outputs go to ``artifacts/audit/``.
"""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Any

import pandas as pd
from src.audit.diagnostics import (
    logreg_with_explicit_interactions,
    summarise_toy_score,
    validation_calibration,
)
from src.audit.freeze import archive_frozen_v1
from src.audit.leakage import feature_leakage_inventory, id_split_audit
from src.audit.protocol import (
    assert_pr_auc_independent_of_threshold,
    build_static_checklist,
    saved_pipeline_has_preprocess_and_model,
    scale_pos_weight_matches_training_labels,
)
from src.audit.report import build_audit_markdown
from src.audit.robustness import run_robustness_experiment
from src.config import get_config, resolve_path, set_seed, setup_logging
from src.data import load_dataset
from src.data.schema import ID_COLUMN
from src.evaluation.bootstrap import bootstrap_metric_intervals, intervals_to_records
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.operations import compare_operating_points
from src.evaluation.threshold import select_threshold
from src.features import prepare_xy
from src.models import load_model_bundle, predict_proba, save_model_bundle
from src.models.train import TrainedModelBundle


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
            if key not in {"pipeline", "val_probability"}
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value


def _parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = literal_eval(raw)
        except (SyntaxError, ValueError):
            return {"raw": raw}
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items()}
    return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, default=str), encoding="utf-8")


def main() -> None:
    """Run the audit without rewriting frozen v1 test metrics."""
    logger = setup_logging(name="scripts.audit_training")
    cfg = get_config()
    set_seed(cfg.random_seed)
    logger.info("Disclaimer: %s", cfg.disclaimer)

    freeze_info = archive_frozen_v1()
    frozen_dir = resolve_path("artifacts/frozen_v1")
    audit_dir = resolve_path("artifacts/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = resolve_path(str(cfg.data.get("processed_dir", "data/processed")))
    train_df = load_dataset(processed_dir / "train.csv")
    val_df = load_dataset(processed_dir / "val.csv")
    test_df = load_dataset(processed_dir / "test.csv")
    logger.info(
        "Loaded train=%s val=%s test=%s. Test is used only to characterise frozen/v2 models.",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    id_audit = id_split_audit(
        train_df, val_df, test_df, id_column=str(cfg.data.get("id_column", ID_COLUMN))
    )
    leakage_inventory = feature_leakage_inventory(train_df, cfg)
    toy_score = summarise_toy_score(train_df)
    interaction_probe = logreg_with_explicit_interactions(train_df, val_df, cfg)

    x_train, y_train = prepare_xy(train_df, cfg, fit_derived_reference=train_df)
    x_val, y_val = prepare_xy(val_df, cfg, fit_derived_reference=train_df)
    weight_check = scale_pos_weight_matches_training_labels(y_train)

    frozen_bundle = load_model_bundle(frozen_dir)
    pipeline_ok = saved_pipeline_has_preprocess_and_model(frozen_bundle.pipeline)
    if not pipeline_ok:
        raise RuntimeError("Frozen pipeline is missing preprocess or model steps")

    val_prob_frozen = predict_proba(frozen_bundle.pipeline, x_val)
    pr_auc_check = assert_pr_auc_independent_of_threshold(y_val, val_prob_frozen)
    frozen_val_calibration = validation_calibration(frozen_bundle.pipeline, x_val, y_val)

    comparison_path = frozen_dir / "model_comparison.json"
    comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    frozen_rows: list[dict[str, Any]] = []
    for row in comparison_payload["candidates"]:
        frozen_rows.append(
            {
                "model": row["model"],
                "val_pr_auc": row["val_pr_auc"],
                "val_roc_auc": row["val_roc_auc"],
                "cv_mean": row["cv_mean"],
                "cv_std": row["cv_std"],
                "best_params": _parse_params(row.get("best_params")),
                "selected": bool(row.get("selected")),
            }
        )
    frozen_logreg_val = next(row["val_pr_auc"] for row in frozen_rows if row["model"] == "logreg")
    frozen_threshold = float(frozen_bundle.metadata["threshold"])
    frozen_selected = str(frozen_bundle.metadata.get("selected_model", "logreg"))

    x_test, y_test = prepare_xy(test_df, cfg, fit_derived_reference=train_df)
    frozen_test_prob = predict_proba(frozen_bundle.pipeline, x_test)
    frozen_test_metrics = compute_classification_metrics(
        y_test, frozen_test_prob, threshold=frozen_threshold
    )
    n_bootstrap = 2000
    bootstrap_seed = int(cfg.random_seed)
    bootstrap = bootstrap_metric_intervals(
        y_test,
        frozen_test_prob,
        threshold=frozen_threshold,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    bootstrap_records = intervals_to_records(bootstrap)
    pd.DataFrame(
        {
            "shipment_id": test_df[ID_COLUMN].astype(str),
            "y_true": y_test,
            "y_prob": frozen_test_prob,
        }
    ).to_csv(audit_dir / "frozen_test_predictions.csv", index=False)

    alt_low = 0.40
    alt_high = 0.70
    operational_rows = [
        *compare_operating_points(
            y_val,
            val_prob_frozen,
            [alt_low, frozen_threshold, alt_high],
            selected_threshold=frozen_threshold,
            split_name="validation",
        ),
        *compare_operating_points(
            y_test,
            frozen_test_prob,
            [alt_low, frozen_threshold, alt_high],
            selected_threshold=frozen_threshold,
            split_name="test_characterisation",
        ),
    ]

    logger.info("Starting robustness experiment on train/validation only")
    robustness = run_robustness_experiment(x_train, y_train, x_val, y_val, cfg)
    robustness_table = [_json_ready(row) for row in robustness]
    winner = max(robustness, key=lambda row: float(row["val_ranking_pr_auc"]))
    logger.info(
        "Robustness winner %s val PR-AUC=%.4f (frozen logreg val PR-AUC=%.4f)",
        winner["name"],
        winner["val_ranking_pr_auc"],
        frozen_logreg_val,
    )

    audited_test_v2: dict[str, Any] | None = None
    original_family_still_wins = str(winner["name"]).startswith("logreg_")
    if not original_family_still_wins:
        logger.info("Validation ranking changed; scoring the new pipeline on test once")
        threshold_info = select_threshold(
            y_val,
            winner["val_probability"],
            cfg,
            split_name="validation",
        )
        v2_threshold = float(threshold_info["threshold"])
        v2_prob = predict_proba(winner["pipeline"], x_test)
        v2_metrics = compute_classification_metrics(y_test, v2_prob, threshold=v2_threshold)
        v2_dir = audit_dir / "v2_model"
        bundle = TrainedModelBundle(
            pipeline=winner["pipeline"],
            feature_names=list(frozen_bundle.feature_names),
            metadata={
                "experiment_version": "audit_v2",
                "selected_model": winner["name"],
                "threshold": v2_threshold,
                "threshold_info": {
                    key: value for key, value in threshold_info.items() if key != "curve"
                },
                "val_ranking_pr_auc": winner["val_ranking_pr_auc"],
                "notes": "Selected on validation after robustness search. Test scored once.",
                "disclaimer": cfg.disclaimer,
            },
        )
        save_model_bundle(bundle, artifact_dir=v2_dir)
        audited_test_v2 = {
            "selected_model": winner["name"],
            "threshold": v2_threshold,
            "metrics": v2_metrics,
            "val_ranking_pr_auc": winner["val_ranking_pr_auc"],
        }
        _write_json(audit_dir / "metrics_test_v2.json", audited_test_v2)

    checklist = build_static_checklist()
    model_leakage = [
        row
        for row in leakage_inventory
        if row["used_in_model"] and (row["direct_leakage"] or row["indirect_leakage"])
    ]
    checklist.append(
        {
            "item": "No direct/indirect target leakage in generated features",
            "verdict": "failed" if model_leakage else "confirmed",
            "evidence": (
                "Features are generated before labels; derived columns are row-wise transforms. "
                "IDs, dates, periods and latent scores are excluded. See the inventory table."
            ),
        }
    )
    checklist.append(
        {
            "item": "Duplicate or correlated shipment IDs cannot cross splits",
            "verdict": (
                "confirmed"
                if id_audit["unique_within_folds"]
                and id_audit["disjoint_across_folds"]
                and not id_audit["shipment_id_is_model_feature"]
                and not id_audit["sender_or_entity_id_present"]
                else "failed"
            ),
            "evidence": (
                f"Unique within folds={id_audit['unique_within_folds']}; "
                f"disjoint={id_audit['disjoint_across_folds']}; "
                f"ID-period correlation={id_audit['id_row_index_period_correlation']:.3f} "
                "but ID is not a feature."
            ),
        }
    )
    checklist.append(
        {
            "item": "Frozen pipeline includes preprocess + model",
            "verdict": "confirmed" if pipeline_ok else "failed",
            "evidence": f"Pipeline steps: {list(frozen_bundle.pipeline.named_steps.keys())}",
        }
    )
    checklist.append(
        {
            "item": "PR-AUC unchanged at display thresholds 0.1 and 0.9",
            "verdict": "confirmed",
            "evidence": (
                f"ranking={pr_auc_check['ranking_pr_auc']:.6f}; "
                f"at 0.1={pr_auc_check['pr_auc_at_0_1']:.6f}; "
                f"at 0.9={pr_auc_check['pr_auc_at_0_9']:.6f}; "
                f"precision changed {pr_auc_check['precision_at_0_1']:.3f} vs "
                f"{pr_auc_check['precision_at_0_9']:.3f}."
            ),
        }
    )

    payload: dict[str, Any] = {
        "disclaimer": cfg.disclaimer,
        "freeze": freeze_info,
        "checklist": checklist,
        "frozen_selected_model": frozen_selected,
        "frozen_threshold": frozen_threshold,
        "frozen_logreg_val_pr_auc": frozen_logreg_val,
        "frozen_validation_comparison": frozen_rows,
        "frozen_test_metrics": frozen_test_metrics,
        "frozen_val_calibration": frozen_val_calibration,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_records": bootstrap_records,
        "operational_rows": operational_rows,
        "toy_score": toy_score,
        "interaction_probe": interaction_probe,
        "id_audit": id_audit,
        "leakage_inventory": leakage_inventory,
        "scale_pos_weight_check": weight_check,
        "robustness_table": robustness_table,
        "robustness_winner": _json_ready(winner),
        "audited_test_v2": audited_test_v2,
    }
    _write_json(audit_dir / "audit_payload.json", payload)
    pd.DataFrame(robustness_table).to_csv(audit_dir / "robustness_validation.csv", index=False)
    pd.DataFrame(bootstrap_records).to_csv(audit_dir / "bootstrap_test_v1.csv", index=False)
    pd.DataFrame(operational_rows).to_csv(audit_dir / "operational_rates.csv", index=False)

    report = build_audit_markdown(payload)
    (audit_dir / "REPORT.md").write_text(report, encoding="utf-8")
    docs_path = resolve_path("docs/methodological_audit.md")
    docs_path.write_text(report, encoding="utf-8")
    logger.info("Wrote audit report to %s and %s", audit_dir / "REPORT.md", docs_path)
    logger.info("Freeze copy: %s", freeze_info)


if __name__ == "__main__":
    main()
