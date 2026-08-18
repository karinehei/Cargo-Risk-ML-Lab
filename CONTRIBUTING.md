# Contributing

This is an educational synthetic-data portfolio. Keep train/validation/test isolation intact. Do not retrain or re-evaluate the frozen-v1 test characterisation unless an independent test pass is explicitly authorised.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install --require-hashes -r requirements/dev.lock.txt
python -m pip install --no-deps -e .
cp .env.example .env
```

If lock files are being regenerated, see `docs/dependencies.md`.

## Checks before a pull request

```bash
make format
make lint
make typecheck
make test
make bandit
make pip-audit
make secrets-scan
make repo-audit
```

CI also runs a lightweight `make bootstrap-ci` that **must not** load `test.csv` for model selection and **must not** claim frozen-v1 metrics.

Mutation testing is optional and local-only. If used, scope it to `src/monitoring/policy.py` or `src/models/threshold.py` so CI stays practical. Do not add a full-suite mutmut job.

## Do not

- commit `.env`, `mlruns/`, raw CSVs, pickle/joblib models or personal files
- add request-controlled model paths or MLflow URIs to the API
- set `SKOPS_ALLOW_UNTRUSTED` or disable skops globally
- use `@main` floating GitHub Actions
- lower security suppressions without a recorded reason
