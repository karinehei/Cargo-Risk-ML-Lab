# Training and evaluation protocol

> **Disclaimer:** This protocol applies to fully synthetic educational data. The review-queue objective is fictional and must not be used for real customs or operational decisions.

## Splits

Modelling uses a **stratified** train / validation / test split on `requires_review` (default 70% / 10% / 20%). IDs are disjoint and written to `data/processed/split_manifest.json`.

`generation_period` remains in the raw table so later monitoring can simulate drift. It is **not** a model feature.

The **test set is not read during training**. Hyperparameter search, model ranking and threshold choice use train and validation only. `scripts/train_model.py` may write `test.csv` when creating splits, but it does not load that file into `compare_models`.

## Methodological audit

`python -m scripts.audit_training` archives frozen artifacts to `artifacts/frozen_v1/`, re-checks leakage and split isolation, runs a train/validation robustness grid for Random Forest and XGBoost, and writes bootstrap 95% CIs plus an operational per-1,000-shipment summary. It does **not** overwrite `artifacts/metrics_test.json`. The narrative report is `docs/methodological_audit.md`.

Local MLflow tracking, calibration and champion selection are documented in `docs/mlops.md`, `docs/calibration.md` and `docs/model_registry.md`. Explainability uses the registered champion and **validation** rows only (`docs/explainability.md`). MLflow packaging issues from the audit are addressed; Evidently remains deferred.

## Imbalance

No oversampling is applied before or after splitting. Logistic regression and random forests use `class_weight="balanced"`. XGBoost uses `scale_pos_weight = n_neg / n_pos` computed on **training** labels only. The dummy baseline uses the class prior.

## Models compared

1. `DummyClassifier(strategy="prior")` — sanity baseline  
2. `LogisticRegression` — linear, interpretable baseline (numeric features scaled)  
3. `RandomForestClassifier`  
4. `XGBClassifier`

All sit in a scikit-learn `Pipeline` with a `ColumnTransformer`: median imputation and optional scaling for numerics; most-frequent imputation and one-hot encoding with `handle_unknown="ignore"` for categoricals.

## Hyperparameter search

A small grid is searched with stratified 3-fold CV on the **training** fold, scoring **average precision (PR-AUC)**. Defaults live in `configs/default.yaml` under `training.search_grids`.

## Model selection

Candidates are ranked by **validation PR-AUC** (threshold-free). Accuracy is reported but not used for selection because the positive class is rare.

## Decision threshold

Fictional operating point: missed reviews (false negatives) cost more than extra human checks (false positives). On the **validation** set we maximise **F-beta with beta=2**, requiring precision of at least `training.threshold.min_precision` when any threshold meets that floor.

The chosen threshold is stored in `artifacts/model_metadata.json` and reused on the test set **without retuning**.

## Test evaluation

`python -m scripts.evaluate_model` scores the held-out test fold once, writes JSON/CSV metrics, confusion/ROC/PR/calibration plots, grouped FP/FN error analysis, and inference latency p50/p95/p99. Exact numbers are artifacts of a run — this document does not hard-code them.
