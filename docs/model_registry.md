# Model registry and champion policy

> **Disclaimer:** Champion selection uses synthetic validation evidence only. Frozen v1 is the only authorised test result.

## What is registered

Each MLflow run stores:

- experiment and run name
- model family and hyperparameters
- dataset fingerprints (train/val) and split-manifest hash
- git commit when available
- random seed
- preprocessing configuration
- class-weight / calibration status
- validation PR-AUC, ROC-AUC, precision, recall, F1, Brier, ECE
- training-fold CV mean and standard deviation when applicable
- inference latency p50 / p95 / p99 (measured on validation rows)
- selected threshold and the validation F-beta policy string
- evaluation plots when produced
- the fitted preprocessing-and-model object via `mlflow.sklearn`

Primary lookup is the MLflow run artifact URI (`runs:/<id>/model`). Joblib under `artifacts/mlops/export/` is recovery only.

## Champion policy (version 1.0.0)

Rules are sequential and listed in `configs/default.yaml` under `mlops.champion_policy`. There is **no** opaque weighted score.

1. Require a successful `mlflow.sklearn` round-trip (`require_roundtrip: true`).
2. Require validation recall at the candidate's **selected** threshold ≥ `min_validation_recall` (0.40).
3. Require validation PR-AUC ≥ `min_validation_pr_auc` (0.15).
4. Require p99 latency ≤ `max_latency_p99_ms` (100 ms).
5. Rank remaining candidates by validation PR-AUC (higher is better).
6. If another eligible model is simpler and within `pr_auc_indifference` (0.005), prefer the simpler family.
7. Remaining ties: lower Brier, then lower p99 latency.

Calibration (Brier/ECE) cannot override a material PR-AUC or recall gap. A simple logistic regression winning is a valid result.

Test metrics (`test_*` keys or `split=test`) raise an error if they appear in the candidate pool.

## Champion metadata

Written to `artifacts/mlops/champion.json` (never under `artifacts/frozen_v1/`):

- model name and version
- MLflow run ID and artifact URI
- dataset fingerprint
- threshold and threshold-selection method
- calibration status
- validation metrics
- creation timestamp and git commit
- policy version
- reason string
- `awaiting_authorized_v2_test` when a newly calibrated model is selected
- note that frozen v1 remains the only test characterisation

## Serving

The API loads **only** this champion through `src.mlops.serving.load_champion` (see `src/api/readiness.py`). It:

- verifies required metadata fields
- loads the MLflow sklearn artifact
- checks that the URI contains the recorded run ID and that the run's threshold param matches
- verifies that a linear explanation model can be constructed
- caches the verified bundle in-process after a successful load
- fails with a generic HTTP 503 payload if anything is missing or inconsistent
- never trains on startup
- never silently loads a different model or switches the threshold
- exposes name, version, calibration status, threshold and run ID on `GET /model` without filesystem paths

A token-gated `POST /reload` is available only when `API_RELOAD_TOKEN` is set.

## Commands

```bash
make experiments       # produce records + champion.json
make select-champion   # re-apply the policy to saved records
make champion-show
```
