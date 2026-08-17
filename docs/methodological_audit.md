# Methodological audit – Cargo Risk ML Lab

> **Disclaimer:** This audit covers a fully synthetic educational pipeline. Figures are from the frozen v1 experiment and a train/validation robustness run. They are not operational customs metrics.

This document records a focused protocol audit **before** MLflow and Evidently work. Frozen v1 test artifacts were copied to `artifacts/frozen_v1/` and were not overwritten.

## Protocol checklist

| Check | Verdict | Evidence |
| --- | --- | --- |
| Model ranking uses validation PR-AUC from probabilities | confirmed | compare_models ranks on ranking_pr_auc = average_precision_score(y_val, val_prob). compute_classification_metrics uses average_precision_score on probabilities; threshold 0.5 only affects precision/recall/F1. |
| Hyperparameter search uses training data only | confirmed | GridSearchCV/cross_val_score are called with x_train, y_train in src/models/compare.py. |
| Validation set is not included in cross-validation | confirmed | StratifiedKFold is created inside _fit_candidate and applied to the training arrays. x_val is used only after fit, via predict_proba. |
| Test set is loaded only after model and threshold selection | confirmed | scripts/train_model.py loads train.csv and val.csv only (it may write test.csv during the split). scripts/evaluate_model.py loads the saved bundle and then test.csv. compare_models has no test argument. |
| Preprocessing is fitted only on relevant training folds | confirmed | Imputer, optional scaler and one-hot encoder live inside the sklearn Pipeline. GridSearchCV clones that pipeline so each CV fit sees only the training indices of that fold; refit=True then refits on the full training fold. |
| scale_pos_weight and class weights use training labels only | confirmed | LogReg/RF use class_weight='balanced' inside fit (fold y). XGBTrainWeightedClassifier sets scale_pos_weight from the y vector passed to fit. Validation and test labels are not used. |
| Decision threshold is selected only from validation predictions | confirmed | select_threshold is called with split_name='validation' and raises if the name contains 'test'. |
| Saved pipeline contains preprocessing and the fitted model together | confirmed | joblib.dump writes the sklearn Pipeline with steps preprocess and model. |
| No direct/indirect target leakage in generated features | confirmed | Features are generated before labels; derived columns are row-wise transforms. IDs, dates, periods and latent scores are excluded. See the inventory table. |
| Duplicate or correlated shipment IDs cannot cross splits | confirmed | Unique within folds=True; disjoint=True; ID-period correlation=0.934 but ID is not a feature. |
| Frozen pipeline includes preprocess + model | confirmed | Pipeline steps: ['preprocess', 'model'] |
| PR-AUC unchanged at display thresholds 0.1 and 0.9 | confirmed | ranking=0.227278; at 0.1=0.227278; at 0.9=0.227278; precision changed 0.129 vs 0.000. |

## Frozen v1 validation ranking

Ranking used **validation PR-AUC from predicted probabilities**. Precision, recall and F1 in the original comparison used a display threshold of 0.5 and **did not** enter the ranking statistic.

| Model | Validation PR-AUC | Validation ROC-AUC | Training-fold CV PR-AUC mean | Training-fold CV PR-AUC std | Best hyperparameters | Selected |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.129 | 0.500 | 0.129 | 0.000 | — | no |
| logreg | 0.227 | 0.645 | 0.200 | 0.003 | model__C=4.0 | yes |
| random_forest | 0.220 | 0.618 | 0.185 | 0.004 | model__max_depth=6, model__n_estimators=80 | no |
| xgboost | 0.213 | 0.609 | 0.191 | 0.005 | model__learning_rate=0.05, model__max_depth=3, model__n_estimators=80 | no |

Selected frozen model: **logreg**. Validation threshold (F-beta, β=2, min precision 0.20): **0.525**.

## Frozen v1 test characterisation (not used for selection)

These numbers repeat the already-produced held-out scores. They were not used to change preprocessing, grids, or the threshold.

| Metric | Point estimate | Notes |
| --- | --- | --- |
| PR-AUC | 0.204 | Threshold-free; from probabilities |
| ROC-AUC | 0.628 | Threshold-free; from probabilities |
| Precision | 0.200 | At threshold 0.525 |
| Recall | 0.482 | At threshold 0.525 |
| F1 | 0.283 | At threshold 0.525 |
| True positives | 187 | Count on 3,000 test rows |
| False positives | 746 | Count on 3,000 test rows |
| False negatives | 201 | Count on 3,000 test rows |
| True negatives | 1866 | Count on 3,000 test rows |

### Bootstrap 95% confidence intervals

Percentile intervals from 2000 resamples of the frozen test predictions, seed 42. **Overlapping intervals limit strong claims about differences** between models or thresholds.

| Metric | Point estimate | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- |
| pr_auc | 0.204 | 0.178 | 0.239 |
| roc_auc | 0.628 | 0.598 | 0.657 |
| precision | 0.200 | 0.175 | 0.226 |
| recall | 0.482 | 0.433 | 0.530 |
| f1 | 0.283 | 0.252 | 0.313 |

## Why logistic regression won on validation

This is treated as a plausible outcome, not an automatic bug.

| Quantity | Value | Interpretation |
| --- | --- | --- |
| Additive share of abs toy score | 0.637 | Monotone single-feature terms dominate the fictional score |
| Interaction share of abs toy score | 0.363 | AND / product terms exist but are smaller on average |
| Logit noise σ (config) | 0.65 | Label noise shrinks the value of extra capacity |
| Label flip rate (config) | 0.025 | Additional irreducible error |
| LogReg validation PR-AUC | 0.227 | Main-effects logistic regression |
| LogReg + explicit interactions validation PR-AUC | 0.233 | Same family with a few toy-score products added |
| Interaction delta | 0.005 | Small gains mean trees are not guaranteed to win |

Reconstruction error between the diagnostic term split and `_raw_review_scores` is 0.000000 (should be ~0).

Original tree grids were small (RF 80–120 trees, depth 6/12; XGBoost 80–120 trees, depth 3/5). Preprocessing is model-specific: logistic regression scales numerics; trees do not. Class weights use training labels (`class_weight='balanced'` per `fit` for LogReg/RF; XGBoost `scale_pos_weight` from the labels passed to `fit`). Calibration (Brier / ECE) is reported in the robustness table; a better-calibrated linear model can look stronger on PR-AUC even when trees fit noise.

## Robustness experiment (train fit, validation score)

The test set was not used to choose among these configurations. Early stopping monitored an inner split of **training** rows.

Best robustness candidate by validation PR-AUC: **logreg_original_best** (0.227). Frozen logistic regression validation PR-AUC was 0.227.

| Candidate | Validation PR-AUC | Validation ROC-AUC | Validation Brier | Validation ECE | Hyperparameters |
| --- | --- | --- | --- | --- | --- |
| logreg_original_best | 0.227 | 0.645 | 0.238 | 0.355 | C=4.0 |
| random_forest_original_best | 0.220 | 0.618 | 0.231 | 0.349 | max_depth=6, n_estimators=80 |
| xgboost_original_best | 0.213 | 0.609 | 0.233 | 0.348 | learning_rate=0.05, max_depth=3, n_estimators=80 |
| random_forest_robust_1 | 0.212 | 0.611 | 0.219 | 0.329 | n_estimators=200, max_depth=8, min_samples_leaf=2, max_features=sqrt, class_weight=balanced |
| random_forest_robust_2 | 0.206 | 0.615 | 0.190 | 0.280 | n_estimators=200, max_depth=16, min_samples_leaf=8, max_features=sqrt, class_weight=balanced |
| random_forest_robust_3 | 0.203 | 0.612 | 0.160 | 0.218 | n_estimators=300, max_depth=None, min_samples_leaf=4, max_features=sqrt, class_weight=balanced |
| random_forest_robust_4 | 0.181 | 0.575 | 0.176 | 0.247 | n_estimators=400, max_depth=12, min_samples_leaf=2, max_features=0.5, class_weight=balanced |
| random_forest_robust_5 | 0.207 | 0.618 | 0.185 | 0.270 | n_estimators=400, max_depth=20, min_samples_leaf=8, max_features=log2, class_weight=balanced |
| random_forest_robust_6 | 0.219 | 0.621 | 0.109 | 0.008 | n_estimators=300, max_depth=8, min_samples_leaf=1, max_features=sqrt, class_weight=None |
| xgboost_robust_1 | 0.215 | 0.614 | 0.229 | 0.340 | n_estimators=150, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8 |
| xgboost_robust_2 | 0.192 | 0.602 | 0.199 | 0.280 | n_estimators=200, learning_rate=0.1, max_depth=4, min_child_weight=3, subsample=0.9, colsample_bytree=0.9 |
| xgboost_robust_3 | 0.192 | 0.604 | 0.186 | 0.255 | n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.7 |
| xgboost_robust_4 | 0.204 | 0.621 | 0.210 | 0.308 | n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=7, subsample=0.7, colsample_bytree=0.8 |
| xgboost_robust_5 | 0.177 | 0.595 | 0.162 | 0.184 | n_estimators=250, learning_rate=0.08, max_depth=6, min_child_weight=1, subsample=1.0, colsample_bytree=1.0 |
| xgboost_early_stopping_inner_train | 0.187 | 0.592 | 0.195 | 0.267 | n_estimators=400, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30, best_iteration=398, inner_train_frac=0.85, scale_pos_weight_from=inner_train_fit_labels |

Larger or deeper trees did **not** overtake logistic regression on validation PR-AUC. Several expanded models have lower Brier scores (better looking calibration) while ranking worse. The unweighted Random Forest (`class_weight=None`) has the lowest Brier/ECE in this set, but its probabilities stay below 0.5 so F1 at the display threshold 0.5 is 0. Class weighting is configured from training labels as intended; it shifts scores upward and is not a coding error. XGBoost early stopping stopped at iteration 398 on an inner training split (validation PR-AUC 0.187) and still ranked below logistic regression.

## Audited test evaluation

_No second test evaluation. Validation ranking still selects logistic regression._

## Operational analysis (per 1,000 shipments)

Rates use the frozen logistic regression probabilities. The selected threshold remains the validation F-beta point. Alternative thresholds are shown only to illustrate workload trade-offs; they were **not** chosen from the test set.

Threshold selection depends on the real cost of missed cases versus unnecessary reviews. Those costs are not identified in this educational project, so 0.525 is a fictional operating point, not a policy recommendation.

| Split | Threshold | Selected operating point | Reviews per 1,000 | True positives per 1,000 | False positives per 1,000 | Missed positives per 1,000 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 0.400 | no | 774.0 | 111.3 | 662.7 | 18.0 | 0.144 | 0.861 |
| validation | 0.525 | yes | 324.7 | 69.3 | 255.3 | 60.0 | 0.214 | 0.536 |
| validation | 0.700 | no | 38.7 | 11.3 | 27.3 | 118.0 | 0.293 | 0.088 |
| test_characterisation | 0.400 | no | 743.7 | 108.7 | 635.0 | 20.7 | 0.146 | 0.840 |
| test_characterisation | 0.525 | yes | 311.0 | 62.3 | 248.7 | 67.0 | 0.200 | 0.482 |
| test_characterisation | 0.700 | no | 24.0 | 8.3 | 15.7 | 121.0 | 0.347 | 0.064 |

## Shipment IDs and splits

| Check | Result |
| --- | --- |
| Unique IDs within each fold | yes |
| Disjoint IDs across folds | yes |
| shipment_id used as a model feature | no |
| Shared sender/entity ID present | no |
| Corr(ID row suffix, generation_period) | 0.934 |

IDs are unique synthetic keys (SYN-{seed}-{row}). The numeric suffix tracks generation order and therefore correlates with generation_period, but shipment_id is excluded from model features. There is no sender_id, so the same entity cannot straddle splits.

## Feature leakage inventory

Spearman associations are computed on **training** rows. Association with the label is expected for predictive features; leakage requires using the label, a latent score, or an identifier that leaks membership.

| Feature | Used in model | Kind | Direct leakage | Indirect leakage | Training abs Spearman with target | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| shipment_id | no | excluded | no | no | — | Unique synthetic identifier. The suffix encodes row order (and therefore time) so it is excluded to prevent split/order leakage. |
| event_date | no | excluded | no | no | — | Calendar date used only for documentation and drift simulation, not as a model feature. |
| generation_period | no | excluded | no | no | — | Generation era used for optional time splits and drift, not as a model feature. |
| requires_review | no | excluded | yes | no | — | Label column. Using it as a feature would be direct target leakage. |
| declared_value_eur | yes | numeric | no | no | 0.071 | Generated before labels. Used as a noisy input to the fictional score. Not computed from y. |
| shipment_weight_kg | yes | numeric | no | no | 0.032 | Generated from value plus noise before labels. Not computed from y. |
| value_to_weight_ratio | yes | numeric | no | no | 0.076 | Deterministic ratio of two raw features. Collinear with value and weight, not leakage. |
| declaration_completeness_score | yes | numeric | no | no | 0.079 | Generated before labels; later used in the fictional score. |
| documentation_count | yes | numeric | no | no | 0.039 | Generated before labels. Missingness is MCAR/MAR on completeness, not on y. |
| previous_discrepancies | yes | numeric | no | no | 0.023 | Per-row fictional count, not a shared sender key. |
| sender_history_length | yes | numeric | no | no | 0.018 | Per-row history length. There is no sender_id, so this cannot join other rows' labels. |
| route_rarity | yes | numeric | no | no | 0.001 | Generated before labels; drifted by period only. |
| declared_vs_estimated_value_deviation | yes | numeric | no | no | 0.046 | Uses an internal toy estimated value that is not saved. The deviation is a feature, not a second copy of the label. |
| submission_hour | yes | numeric | no | no | 0.004 | Generated before labels. Off-hours is a deterministic transform of this field. |
| expedited_shipment | yes | numeric | no | no | 0.024 | Generated before labels; also used in interaction terms of the toy score. |
| log_declared_value | yes | numeric | no | no | 0.071 | Deterministic log1p of declared_value_eur. Redundant transform, not leakage. |
| is_off_hours | yes | numeric | no | no | 0.054 | Deterministic function of submission_hour. Redundant transform, not leakage. |
| transport_mode | yes | categorical | no | no | — | Generated before labels. Air appears in some fictional interactions. |
| origin_region | yes | categorical | no | no | — | Coarse corridor label generated before y. Not a personal attribute. |
| destination_region | yes | categorical | no | no | — | Coarse corridor label generated before y. |
| commodity_category | yes | categorical | no | no | — | Generated before labels. Some categories interact with air in the toy score. |

## Issues deferred to later phases

MLflow and Evidently are **not** fixed in this audit.

| Component | Current issue | Why it is deferred |
| --- | --- | --- |
| MLflow | File-store and sklearn `log_model` failures (untrusted `numpy.dtype` / skops) were observed previously; training still writes joblib artifacts. | Tracking is independent of the leakage/ranking protocol. Fix in the MLflow phase. |
| Evidently | `ColumnMapping` import failed; the pipeline already falls back to PSI/KS. | Monitoring packaging is independent of train/validation/test isolation. Fix in the monitoring phase. |

## How to reproduce

```bash
source ~/.venvs/cargo-risk-ml-lab/bin/activate
cd "/mnt/d/Cargo Risk ML Lab"
python -m scripts.audit_training
```

Do not run `make train` or `make evaluate` if you need to keep `artifacts/metrics_test.json` bit-for-bit identical; those commands rewrite the live artifact directory. Use `artifacts/frozen_v1/` as the preserved copy.
