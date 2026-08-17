# Deployment

> **Disclaimer:** Local containers serve a synthetic educational model. They are not a production customs system.

Expected URLs after a **successful** local startup (do not treat this list as a runtime verification):

- API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`
- MLflow UI: `http://127.0.0.1:5000`

## Prepare artifacts

The images do not bake datasets, MLflow runs or secrets. Mount local `mlruns/` and `artifacts/` at runtime.

```bash
source ~/.venvs/cargo-risk-ml-lab/bin/activate   # or your venv
cd "/mnt/d/Cargo Risk ML Lab"                    # or the project root
cp .env.example .env

make mlflow-init
make experiments    # train/val + champion; does not touch frozen v1 test
make explain        # validation explanations only
```

Confirm `artifacts/mlops/champion.json` exists and that `mlruns/mlflow.db` is present.

## Local processes (no Docker)

```bash
make serve-api      # uvicorn on 8000
make serve-app      # Streamlit on 8501
make mlflow-ui      # MLflow UI on 5000
```

Streamlit reads `STREAMLIT_API_URL` (default `http://127.0.0.1:8000`). If the API is down, the dashboard scores with the local champion loader and shows a setup message.

Health and readiness:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
```

## Docker Compose

Python **3.12-slim**, non-root `appuser`, runtime dependencies only, private `cargo-risk` network.

```bash
docker compose config
docker compose up --build -d
```

Inside Compose, Streamlit uses `STREAMLIT_API_URL=http://api:8000`. The API uses `sqlite:///mlruns/mlflow.db` on the mounted `mlruns` volume.

Read-only mounts: `configs`, `artifacts`, `docs`. `mlruns` is read-write because SQLite needs locks.

Stop:

```bash
docker compose down
# or
make docker-down
```

### Image notes

- No secrets are copied into the image.
- `data/` and generated datasets are not baked in.
- Health checks use Python `urllib` against `/ready`, Streamlit `/_stcore/health`, and the MLflow UI root.
- Rebuild after dependency changes: `docker compose build --no-cache`.

If Docker is unavailable, `tests/test_docker.py` still checks Dockerfile and Compose **configuration**. That is not the same as a runtime-verified stack.

## Environment

See `.env.example`. Important variables:

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | Local tracking store |
| `CHAMPION_PATH` | `artifacts/mlops/champion.json` | Registered champion metadata |
| `STREAMLIT_API_URL` | `http://127.0.0.1:8000` | Dashboard → API |
| `API_MAX_BATCH_SIZE` | `100` | Batch rejection limit |
| `API_MAX_REQUEST_BYTES` | `1048576` | Request `Content-Length` limit |
| `API_RELOAD_TOKEN` | empty | Enables token-gated `POST /reload` when set |

Do not commit `.env`.
