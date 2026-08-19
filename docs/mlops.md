# MLflow and local experiment tracking

> **Disclaimer:** Tracking covers fully synthetic educational models. Frozen v1 remains the only authorised test characterisation.

## Why the original `log_model` call failed

Installed versions at diagnosis: MLflow **3.15.1**, skops **0.14.0**, NumPy **2.5.2**, scikit-learn **1.9.0**.

MLflow 3 defaults `mlflow.sklearn.log_model(..., serialization_format="skops")`. After dumping, MLflow immediately loads the file with skops trust checks. A fitted sklearn `Pipeline` (imputer, scaler, one-hot encoder, logistic regression) stores `numpy.dtype` objects. skops 0.14 reports:

```text
Untrusted types found: ['numpy.dtype']
```

This is **sklearn/skops serialization**, not a bad input example, not a global skops env var, and not specific to the custom XGBoost wrapper (it also failed for the frozen logistic-regression pipeline).

## Fix

Log with the official sklearn flavor and an **explicit** serializer:

```python
mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="model",
    serialization_format="cloudpickle",
)
```

Cloudpickle and pickle both round-tripped the frozen logistic-regression pipeline in diagnosis. We do **not** set `SKOPS_ALLOW_UNTRUSTED`, `MLFLOW_ALLOW_FILE_STORE` (tracking is SQLite), or a process-wide trusted-type list. skops remains the MLflow default; this project overrides it per call.

If `XGBTrainWeightedClassifier` cannot be logged that way, the run records a **documented joblib fallback** artifact and is ineligible for champion selection when `require_roundtrip` is true. Logistic regression must succeed via `mlflow.sklearn`.

Round-trip tolerance: relative `1e-7`, absolute `1e-10` (`src/mlops/serialization.py`). Class predictions must match exactly.

## Local backend

Default tracking URI: `sqlite:///mlruns/mlflow.db` (see `.env.example`). The SQLite file, `mlruns/` and `mlartifacts/` are gitignored. Joblib under `artifacts/mlops/export/` is a recovery export, not the primary lineage store.

Frozen v1 files under `artifacts/frozen_v1/` are never written by these commands.

## Commands

```bash
source ~/.venvs/cargo-risk-ml-lab/bin/activate
cd "/mnt/d/Cargo Risk ML Lab"

make mlflow-init          # create/migrate the SQLite store
make mlflow-verify        # frozen logreg joblib → MLflow → load → compare on val fixture
make experiments          # train/val comparison + calibration + champion (no test set)
make mlflow-list          # list runs
make champion-show        # print artifacts/mlops/champion.json
make mlflow-ui            # http://127.0.0.1:5000
```

`make train` now calls the same MLOps entrypoint as `make experiments`. When `artifacts/frozen_v1/` is present, `save_champion` keeps the existing serving `champion.json` (set `CARGO_RISK_REPLACE_CHAMPION=1` to replace it). `make evaluate` still exists for an independently authorised test characterisation; do not run it against frozen v1 unless you intend to rewrite `artifacts/metrics_test.json` (the frozen copy stays in `artifacts/frozen_v1/`).

## Evidently

Evidently is **not** repaired in this phase. Drift checks keep the PSI/KS fallback.

## Related docs

- `docs/model_registry.md` — champion policy and serving
- `docs/calibration.md` — train-only calibration experiment
- `docs/methodological_audit.md` — frozen protocol
