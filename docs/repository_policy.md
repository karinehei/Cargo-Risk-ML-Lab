# Repository policy

> **Disclaimer:** This repository is an educational demonstration on fully synthetic data. It is not affiliated with Finnish Customs or any other authority.

This document is the public-repository boundary. Paths below are classified as they should appear in a published Git repository. Local developer machines may contain additional generated files that must stay untracked.

## Classification

| Class | Paths | Commit? |
|---|---|---|
| Source code | `src/`, `app/`, `scripts/`, `tests/` | Yes |
| Configuration | `configs/`, `pyproject.toml`, `requirements/`, `.github/`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example`, `bandit.yaml`, `.secrets.baseline` | Yes |
| Documentation | `docs/`, `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, `LICENSE`, `artifacts/README.md` | Yes |
| Safe aggregate demonstration artifact | `artifacts/README.md` only. Champion JSON, plots, monitoring reports and frozen-v1 metrics are **local** | No (except README placeholders) |
| Reproducible generated artifact | `data/raw/*.csv`, `data/processed/*.csv`, `data/monitoring/*.csv`, bootstrap outputs under `artifacts/bootstrap/` and `.ci-work/` | No; recreate with documented commands |
| Local runtime state | `mlruns/`, `mlartifacts/`, `*.db`, `*.sqlite`, uvicorn/streamlit logs | No |
| Raw row-level synthetic data | any `*.csv` under `data/` | No |
| Model serialization | `*.pkl`, `*.joblib`, MLflow `model/` trees, cloudpickle artifacts | No. Treat as executable Python, not trustworthy downloads |
| Secret or sensitive file | `.env`, `.env.local`, tokens, credentials, IDE secrets | No |
| Cache/build output | `__pycache__/`, `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `htmlcov/`, `*.egg-info/`, Docker `*.tar` | No |
| Docker runtime volumes | Compose mounts of `./mlruns` and `./artifacts` | Not repository content |

Directory placeholders that **may** be committed: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/monitoring/.gitkeep`, `artifacts/.gitkeep`, `artifacts/bootstrap/.gitkeep`.

## Must not be committed

- `.env` files or API reload tokens
- credentials, cloud keys, personal documents or images
- local MLflow SQLite databases and raw `mlruns/` trees
- generated train/validation/test CSVs
- raw monitoring CSVs and reference samples
- Python caches and virtual environments
- local filesystem absolute paths in committed files
- pickle/joblib files presented as public model downloads
- Docker runtime volume contents

## Trusted local artifacts

Serialized MLflow models use **cloudpickle** and are executable. Load only from the local tracking store (`runs:/` or `models:/` URIs). The API never accepts a model path, MLflow URI, config path or reload target from a request body. Verify checksums when `artifact_sha256` is present on champion metadata.

Frozen-v1 test characterisation lives under `artifacts/frozen_v1/` (gitignored). It is not a CI input and is never used for model selection.

## Git worktree note

If this working copy is not a Git repository, do not run `git init` from automation. Filesystem ignore rules still apply. Real `git ls-files` and history scanning must be executed in the actual Git repository.

## Recreate locally

```bash
make bootstrap-demo          # isolated full demo under artifacts/bootstrap/
make bootstrap-ci            # lightweight architecture smoke under .ci-work/
make evaluate                # separate, explicit frozen-test characterisation
```
