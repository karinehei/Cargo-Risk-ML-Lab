# Cargo Risk ML Lab

Educational machine learning portfolio project for **synthetic** cargo shipment review scoring.

> **Disclaimer:** All data, labels and review logic in this repository are fully synthetic and exist only for educational demonstration. This project does **not** represent the Finnish Customs Service, any other customs authority, real border processes, or operational risk-assessment systems. The output is a **review score** (ranking/threshold value), not a calibrated probability.

## Status

Core modelling, local MLflow tracking, exact linear explanations, FastAPI serving and a Streamlit demonstration are implemented. Frozen v1 test artefacts are preserved. Documentation does **not** hard-code fabricated live prediction numbers.

## Quick start

### WSL (recommended on Windows)

```bash
python3 -m venv ~/.venvs/cargo-risk-ml-lab
source ~/.venvs/cargo-risk-ml-lab/bin/activate
cd "/mnt/d/Cargo Risk ML Lab"
pip install -e ".[dev]"
cp .env.example .env

make generate-data
make validate-data
make mlflow-init
make experiments    # train/val + calibration + champion; does not touch frozen v1 test
make mlflow-verify
# make evaluate     # only for an independently authorised test characterisation
make explain

# Monitoring (reference = train sample; never uses test.csv)
make monitoring-reference
make monitoring-all
make monitoring-status
make monitoring-null-audit
make monitoring-validation
```

### Native / other platforms

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

make generate-data
make validate-data
make mlflow-init
make experiments
make mlflow-verify
make explain
```

## Run the local services

```bash
make serve-api      # FastAPI on http://127.0.0.1:8000  (docs: /docs)
make serve-app      # Streamlit on http://127.0.0.1:8501
make mlflow-ui      # MLflow on http://127.0.0.1:5000
```

Health and readiness (API process must already be running):

```bash
make api-health
make api-ready
```

Example requests use fictional inputs. Copy live JSON from the running service; this README does not invent scores.

```bash
curl -sS http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"origin_region":"Asia","destination_region":"Northern Europe","commodity_category":"electronics","transport_mode":"air","declared_value_eur":12500.0,"shipment_weight_kg":85.5,"previous_discrepancies":0,"submission_hour":10,"expedited_shipment":0}'

curl -sS http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"origin_region":"Asia","destination_region":"Northern Europe","commodity_category":"electronics","transport_mode":"air","declared_value_eur":12500.0,"shipment_weight_kg":85.5,"previous_discrepancies":0,"submission_hour":10,"expedited_shipment":0}'

curl -sS http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"shipments":[{"origin_region":"Asia","destination_region":"Northern Europe","commodity_category":"electronics","transport_mode":"air","declared_value_eur":12500.0,"shipment_weight_kg":85.5,"previous_discrepancies":0,"submission_hour":10,"expedited_shipment":0}]}'
```

## Docker Compose

Prepare artifacts first (`make experiments` and `make explain`). Images do not bake datasets or secrets.

```bash
make docker-config
make docker-up
make docker-down
```

Expected URLs after a successful stack start: API `8000`, Streamlit `8501`, MLflow `5000`. See `docs/deployment.md`.

## Project layout

See `docs/architecture.md` for design decisions, `docs/monitoring.md` and `docs/drift_metrics.md` for drift monitoring, `docs/operations.md` for the monitoring workflow, `docs/api.md` for the HTTP contract, `docs/deployment.md` for local and container runtime, `docs/data_dictionary.md` for the synthetic schema, `docs/training.md` for the modelling protocol, `docs/methodological_audit.md` for the train/validation protocol audit, `docs/mlops.md` for local MLflow tracking, and `docs/explainability.md` / `docs/responsible_ai.md` / `docs/limitations.md` for score semantics and responsible-AI notes.

## License

MIT
