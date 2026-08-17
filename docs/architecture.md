# Architecture

## Purpose

Cargo Risk ML Lab is an educational portfolio project that demonstrates an end-to-end supervised learning workflow on **fully synthetic** cargo shipment data. It is not connected to any customs authority or operational targeting system.

## High-level flow

```text
configs/default.yaml
        │
        ▼
 scripts/generate_data ──► data/raw + data/processed
        │
        ▼
 scripts/train_model ──► scripts/run_mlops ──► MLflow + artifacts/mlops/
        │
        ├──► scripts/evaluate_model ──► metrics, plots, drift report (authorised test only)
        ├──► scripts/explain_model ──► linear explanations + validation subgroups
        ├──► FastAPI (src/api) ──► /health /ready /model /predict /predict/batch /explain
        └──► Streamlit (app/) ──► recruiter-facing demonstration
```

## Package responsibilities

| Package | Responsibility |
|---|---|
| `src/data` | Synthetic data generation (`generate.py`), schema (`schema.py`), validation, time-like splits |
| `src/features` | Derived features, ColumnTransformer preprocessing |
| `src/models` | Estimators, comparison, threshold-aware persistence |
| `src/evaluation` | Metrics, plots, error analysis |
| `src/explainability` | Exact logistic logit explanations, validation permutation importance, subgroups |
| `src/monitoring` | Evidently (when available) + lightweight PSI/KS drift |
| `src/mlops` | Local MLflow tracking, calibration, champion policy, safe serving |
| `src/api` | FastAPI service with Pydantic validation |
| `app` | Streamlit dashboard |
| `scripts` | Thin CLI entrypoints for reproducible workflows |

## Design decisions

1. **Synthetic labels from explicit toy rules** – `requires_review` is sampled from a noisy, non-linear function of several features. Rules are fictional and must not be used for real decisions. Latent scores are not saved.
2. **Stratified modelling splits; time column kept for drift** – train/val/test for modelling are stratified on `requires_review` with disjoint IDs. `generation_period` remains available for later drift simulation and is not a model feature.
3. **Generation separated from preprocessing** – raw CSV is produced first; `src/features` only imputes, encodes, scales and adds deterministic transforms.
4. **Central YAML + env settings** – runtime knobs live in `configs/default.yaml`; environment overrides use `.env` / pydantic-settings.
5. **Deterministic seeds** – `set_seed()` covers Python, NumPy and model `random_state`.
6. **Sklearn pipeline + MLflow lineage** – Dummy, logistic regression, random forest and XGBoost are compared on validation PR-AUC. Thresholds are chosen on validation (F-beta, beta=2). Fitted pipelines are logged with `mlflow.sklearn` (`cloudpickle`). The test set is scored only in an independently authorised characterisation; frozen v1 is preserved under `artifacts/frozen_v1/`.
7. **Evidently optional** – if Evidently integrates cleanly it may supplement reports; PSI/KS/JS in `src/monitoring/metrics.py` remain the source of truth. Evidently repair is deferred when APIs are incompatible.
8. **Monitoring** – train-derived reference profiles, deterministic monitoring scenarios, unlabelled input/score drift by default, optional labelled simulation kept separate. Policy 1.1.0 aggregates isolated vs coordinated drift. Raw monitoring CSVs are gitignored. Never uses the frozen test set or triggers automatic retraining.
9. **No fabricated metrics in docs** – evaluation numbers are produced only by running the pipeline.

## Serving

- FastAPI loads **only** the registered champion from MLflow via `artifacts/mlops/champion.json`. It never trains on startup and never silently falls back to another model or threshold.
- `GET /health` is process liveness. `GET /ready` verifies champion metadata, MLflow load, threshold/run agreement and linear explanation metadata, and fails closed with HTTP 503.
- Streamlit prefers `STREAMLIT_API_URL` (Compose: `http://api:8000`; local default `http://127.0.0.1:8000`), then scores with the same champion loader. Missing artifacts produce a setup message rather than invented metrics.
- Docker Compose runs `api`, `app` and `mlflow` on a private network. `mlruns/` and `artifacts/` are mounted from the host; datasets and secrets are not baked into the image.

See `docs/api.md` and `docs/deployment.md`.
