"""Render the methodological audit Markdown report from measured results."""

from __future__ import annotations

from typing import Any

from src.audit.markdown import markdown_table


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np_isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _params(params: dict[str, Any] | None) -> str:
    if not params:
        return "—"
    parts = [f"{key}={value}" for key, value in params.items()]
    return ", ".join(parts)


def build_audit_markdown(payload: dict[str, Any]) -> str:
    """Assemble a standalone audit document with a header on every table column."""
    checklist_rows = [
        [str(item["item"]), str(item["verdict"]), str(item["evidence"])]
        for item in payload["checklist"]
    ]
    leakage_rows = [
        [
            str(row["feature"]),
            "yes" if row["used_in_model"] else "no",
            str(row["kind"]),
            "yes" if row["direct_leakage"] else "no",
            "yes" if row["indirect_leakage"] else "no",
            _fmt(row.get("abs_spearman_with_target"), 3),
            str(row["rationale"]),
        ]
        for row in payload["leakage_inventory"]
    ]
    frozen_cmp = [
        [
            str(row["model"]),
            _fmt(row["val_pr_auc"], 3),
            _fmt(row["val_roc_auc"], 3),
            _fmt(row.get("cv_mean"), 3),
            _fmt(row.get("cv_std"), 3),
            _params(row.get("best_params") if isinstance(row.get("best_params"), dict) else None)
            if isinstance(row.get("best_params"), dict)
            else str(row.get("best_params") or "—"),
            "yes" if row.get("selected") else "no",
        ]
        for row in payload["frozen_validation_comparison"]
    ]
    robustness_rows = [
        [
            str(row["name"]),
            _fmt(row["val_ranking_pr_auc"], 3),
            _fmt(row["val_roc_auc"], 3),
            _fmt(row["val_brier"], 3),
            _fmt(row["val_ece"], 3),
            _params(row.get("params")),
        ]
        for row in payload["robustness_table"]
    ]
    bootstrap_rows = [
        [
            str(row["metric"]),
            _fmt(row["point"], 3),
            _fmt(row["ci_low"], 3),
            _fmt(row["ci_high"], 3),
        ]
        for row in payload["bootstrap_records"]
    ]
    ops_rows = [
        [
            str(row["split"]),
            _fmt(row["threshold"], 3),
            "yes" if row.get("is_selected_threshold") else "no",
            _fmt(row["reviews_per_1000"], 1),
            _fmt(row["true_positives_per_1000"], 1),
            _fmt(row["false_positives_per_1000"], 1),
            _fmt(row["missed_positives_per_1000"], 1),
            _fmt(row["precision_positive"], 3),
            _fmt(row["recall_positive"], 3),
        ]
        for row in payload["operational_rows"]
    ]
    frozen_test = payload["frozen_test_metrics"]
    score = payload["toy_score"]
    interaction = payload["interaction_probe"]
    ids = payload["id_audit"]
    winner = payload["robustness_winner"]
    v2 = payload.get("audited_test_v2")
    es_row: dict[str, Any] = next(
        (
            row
            for row in payload["robustness_table"]
            if "early_stopping" in str(row.get("name", ""))
        ),
        {},
    )
    es_best = (es_row.get("params") or {}).get("best_iteration", "—")
    es_pr = _fmt(es_row.get("val_ranking_pr_auc"), 3) if es_row else "—"

    v2_section = (
        "_No second test evaluation. Validation ranking still selects logistic regression._"
    )
    if v2:
        v2_metrics = v2["metrics"]
        v2_section = "\n".join(
            [
                "Validation ranking changed the selected family. The new pipeline was scored on the test set **once**.",
                "",
                markdown_table(
                    [
                        "Experiment version",
                        "Model",
                        "Threshold",
                        "Test PR-AUC",
                        "Test ROC-AUC",
                        "Test precision",
                        "Test recall",
                        "Test F1",
                    ],
                    [
                        [
                            "frozen_v1",
                            str(payload["frozen_selected_model"]),
                            _fmt(frozen_test["threshold"], 3),
                            _fmt(frozen_test["pr_auc"], 3),
                            _fmt(frozen_test["roc_auc"], 3),
                            _fmt(frozen_test["precision"], 3),
                            _fmt(frozen_test["recall"], 3),
                            _fmt(frozen_test["f1"], 3),
                        ],
                        [
                            "audit_v2",
                            str(v2["selected_model"]),
                            _fmt(v2["threshold"], 3),
                            _fmt(v2_metrics["pr_auc"], 3),
                            _fmt(v2_metrics["roc_auc"], 3),
                            _fmt(v2_metrics["precision"], 3),
                            _fmt(v2_metrics["recall"], 3),
                            _fmt(v2_metrics["f1"], 3),
                        ],
                    ],
                ),
            ]
        )

    return f"""# Methodological audit – Cargo Risk ML Lab

> **Disclaimer:** This audit covers a fully synthetic educational pipeline. Figures are from the frozen v1 experiment and a train/validation robustness run. They are not operational customs metrics.

This document records a focused protocol audit **before** MLflow and Evidently work. Frozen v1 test artifacts were copied to `artifacts/frozen_v1/` and were not overwritten.

## Protocol checklist

{markdown_table(["Check", "Verdict", "Evidence"], checklist_rows)}

## Frozen v1 validation ranking

Ranking used **validation PR-AUC from predicted probabilities**. Precision, recall and F1 in the original comparison used a display threshold of 0.5 and **did not** enter the ranking statistic.

{
        markdown_table(
            [
                "Model",
                "Validation PR-AUC",
                "Validation ROC-AUC",
                "Training-fold CV PR-AUC mean",
                "Training-fold CV PR-AUC std",
                "Best hyperparameters",
                "Selected",
            ],
            frozen_cmp,
        )
    }

Selected frozen model: **{
        payload["frozen_selected_model"]
    }**. Validation threshold (F-beta, β=2, min precision 0.20): **{
        _fmt(payload["frozen_threshold"], 3)
    }**.

## Frozen v1 test characterisation (not used for selection)

These numbers repeat the already-produced held-out scores. They were not used to change preprocessing, grids, or the threshold.

{
        markdown_table(
            [
                "Metric",
                "Point estimate",
                "Notes",
            ],
            [
                ["PR-AUC", _fmt(frozen_test["pr_auc"], 3), "Threshold-free; from probabilities"],
                ["ROC-AUC", _fmt(frozen_test["roc_auc"], 3), "Threshold-free; from probabilities"],
                [
                    "Precision",
                    _fmt(frozen_test["precision"], 3),
                    f"At threshold {_fmt(frozen_test['threshold'], 3)}",
                ],
                [
                    "Recall",
                    _fmt(frozen_test["recall"], 3),
                    f"At threshold {_fmt(frozen_test['threshold'], 3)}",
                ],
                [
                    "F1",
                    _fmt(frozen_test["f1"], 3),
                    f"At threshold {_fmt(frozen_test['threshold'], 3)}",
                ],
                [
                    "True positives",
                    str(int(frozen_test["true_positives"])),
                    "Count on 3,000 test rows",
                ],
                [
                    "False positives",
                    str(int(frozen_test["false_positives"])),
                    "Count on 3,000 test rows",
                ],
                [
                    "False negatives",
                    str(int(frozen_test["false_negatives"])),
                    "Count on 3,000 test rows",
                ],
                [
                    "True negatives",
                    str(int(frozen_test["true_negatives"])),
                    "Count on 3,000 test rows",
                ],
            ],
        )
    }

### Bootstrap 95% confidence intervals

Percentile intervals from {payload["n_bootstrap"]} resamples of the frozen test predictions, seed {
        payload["bootstrap_seed"]
    }. **Overlapping intervals limit strong claims about differences** between models or thresholds.

{
        markdown_table(
            ["Metric", "Point estimate", "95% CI lower", "95% CI upper"],
            bootstrap_rows,
        )
    }

## Why logistic regression won on validation

This is treated as a plausible outcome, not an automatic bug.

{
        markdown_table(
            ["Quantity", "Value", "Interpretation"],
            [
                [
                    "Additive share of abs toy score",
                    _fmt(score["additive_share"], 3),
                    "Monotone single-feature terms dominate the fictional score",
                ],
                [
                    "Interaction share of abs toy score",
                    _fmt(score["interaction_share"], 3),
                    "AND / product terms exist but are smaller on average",
                ],
                [
                    "Logit noise σ (config)",
                    _fmt(score["logit_noise_std_config"], 2),
                    "Label noise shrinks the value of extra capacity",
                ],
                [
                    "Label flip rate (config)",
                    _fmt(score["label_flip_rate_config"], 3),
                    "Additional irreducible error",
                ],
                [
                    "LogReg validation PR-AUC",
                    _fmt(interaction["logreg_val_pr_auc"], 3),
                    "Main-effects logistic regression",
                ],
                [
                    "LogReg + explicit interactions validation PR-AUC",
                    _fmt(interaction["logreg_with_interactions_val_pr_auc"], 3),
                    "Same family with a few toy-score products added",
                ],
                [
                    "Interaction delta",
                    _fmt(interaction["delta_val_pr_auc"], 3),
                    "Small gains mean trees are not guaranteed to win",
                ],
            ],
        )
    }

Reconstruction error between the diagnostic term split and `_raw_review_scores` is {
        _fmt(score["max_abs_reconstruction_error"], 6)
    } (should be ~0).

Original tree grids were small (RF 80–120 trees, depth 6/12; XGBoost 80–120 trees, depth 3/5). Preprocessing is model-specific: logistic regression scales numerics; trees do not. Class weights use training labels (`class_weight='balanced'` per `fit` for LogReg/RF; XGBoost `scale_pos_weight` from the labels passed to `fit`). Calibration (Brier / ECE) is reported in the robustness table; a better-calibrated linear model can look stronger on PR-AUC even when trees fit noise.

## Robustness experiment (train fit, validation score)

The test set was not used to choose among these configurations. Early stopping monitored an inner split of **training** rows.

Best robustness candidate by validation PR-AUC: **{winner["name"]}** ({
        _fmt(winner["val_ranking_pr_auc"], 3)
    }). Frozen logistic regression validation PR-AUC was {
        _fmt(payload["frozen_logreg_val_pr_auc"], 3)
    }.

{
        markdown_table(
            [
                "Candidate",
                "Validation PR-AUC",
                "Validation ROC-AUC",
                "Validation Brier",
                "Validation ECE",
                "Hyperparameters",
            ],
            robustness_rows,
        )
    }

Larger or deeper trees did **not** overtake logistic regression on validation PR-AUC. Several expanded models have lower Brier scores (better looking calibration) while ranking worse. The unweighted Random Forest (`class_weight=None`) has the lowest Brier/ECE in this set, but its probabilities stay below 0.5 so F1 at the display threshold 0.5 is 0. Class weighting is configured from training labels as intended; it shifts scores upward and is not a coding error. XGBoost early stopping stopped at iteration {
        es_best
    } on an inner training split (validation PR-AUC {
        es_pr
    }) and still ranked below logistic regression.

## Audited test evaluation

{v2_section}

## Operational analysis (per 1,000 shipments)

Rates use the frozen logistic regression probabilities. The selected threshold remains the validation F-beta point. Alternative thresholds are shown only to illustrate workload trade-offs; they were **not** chosen from the test set.

Threshold selection depends on the real cost of missed cases versus unnecessary reviews. Those costs are not identified in this educational project, so 0.525 is a fictional operating point, not a policy recommendation.

{
        markdown_table(
            [
                "Split",
                "Threshold",
                "Selected operating point",
                "Reviews per 1,000",
                "True positives per 1,000",
                "False positives per 1,000",
                "Missed positives per 1,000",
                "Precision",
                "Recall",
            ],
            ops_rows,
        )
    }

## Shipment IDs and splits

{
        markdown_table(
            ["Check", "Result"],
            [
                ["Unique IDs within each fold", "yes" if ids["unique_within_folds"] else "no"],
                ["Disjoint IDs across folds", "yes" if ids["disjoint_across_folds"] else "no"],
                [
                    "shipment_id used as a model feature",
                    "yes" if ids["shipment_id_is_model_feature"] else "no",
                ],
                [
                    "Shared sender/entity ID present",
                    "yes" if ids["sender_or_entity_id_present"] else "no",
                ],
                [
                    "Corr(ID row suffix, generation_period)",
                    _fmt(ids["id_row_index_period_correlation"], 3),
                ],
            ],
        )
    }

{ids["notes"]}

## Feature leakage inventory

Spearman associations are computed on **training** rows. Association with the label is expected for predictive features; leakage requires using the label, a latent score, or an identifier that leaks membership.

{
        markdown_table(
            [
                "Feature",
                "Used in model",
                "Kind",
                "Direct leakage",
                "Indirect leakage",
                "Training abs Spearman with target",
                "Rationale",
            ],
            leakage_rows,
        )
    }

## Issues deferred to later phases

MLflow and Evidently are **not** fixed in this audit.

{
        markdown_table(
            ["Component", "Current issue", "Why it is deferred"],
            [
                [
                    "MLflow",
                    "File-store and sklearn `log_model` failures (untrusted `numpy.dtype` / skops) were observed previously; training still writes joblib artifacts.",
                    "Tracking is independent of the leakage/ranking protocol. Fix in the MLflow phase.",
                ],
                [
                    "Evidently",
                    "`ColumnMapping` import failed; the pipeline already falls back to PSI/KS.",
                    "Monitoring packaging is independent of train/validation/test isolation. Fix in the monitoring phase.",
                ],
            ],
        )
    }

## How to reproduce

```bash
source ~/.venvs/cargo-risk-ml-lab/bin/activate
cd "/mnt/d/Cargo Risk ML Lab"
python -m scripts.audit_training
```

Do not run `make train` or `make evaluate` if you need to keep `artifacts/metrics_test.json` bit-for-bit identical; those commands rewrite the live artifact directory. Use `artifacts/frozen_v1/` as the preserved copy.
"""
