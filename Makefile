.PHONY: help install install-dev format lint typecheck test check \
	generate-data validate-data train evaluate audit explain \
	mlflow-init mlflow-ui mlflow-list mlflow-verify experiments select-champion champion-show \
	serve-api serve-app docker-up docker-down docker-config docker-build \
	api-health api-ready \
	monitoring-reference monitoring-scenario monitoring-run monitoring-labelled monitoring-status monitoring-all \
	monitoring-null-audit monitoring-validation \
	clean

PYTHON ?= python
PIP ?= pip

help:
	@echo "Cargo Risk ML Lab – common targets"
	@echo "  install            Install package and runtime dependencies"
	@echo "  install-dev        Install package with dev dependencies"
	@echo "  format             Format code with Ruff"
	@echo "  lint               Lint with Ruff"
	@echo "  typecheck          Run mypy"
	@echo "  test               Run pytest"
	@echo "  check              format + lint + typecheck + test"
	@echo "  generate-data      Generate synthetic datasets"
	@echo "  validate-data      Validate generated datasets"
	@echo "  train              Train/val experiments, calibration, champion (no test)"
	@echo "  experiments        Alias for train / MLOps experiment run"
	@echo "  evaluate           Evaluate on the held-out test set (authorised characterisation only)"
	@echo "  audit              Methodological audit (preserves frozen test artifacts)"
	@echo "  mlflow-init        Create or migrate the local SQLite tracking store"
	@echo "  mlflow-ui          Open the MLflow UI"
	@echo "  mlflow-list        List MLflow runs"
	@echo "  mlflow-verify      Frozen logreg round-trip through MLflow"
	@echo "  select-champion    Re-apply the champion policy to saved records"
	@echo "  champion-show      Print registered champion metadata"
	@echo "  explain            Champion coefficients, permutation, local logit, subgroups (validation only)"
	@echo "  serve-api          Start FastAPI server"
	@echo "  serve-app          Start Streamlit dashboard"
	@echo "  api-health         GET /health on 127.0.0.1:8000"
	@echo "  api-ready          GET /ready on 127.0.0.1:8000"
	@echo "  monitoring-reference  Build train-derived reference profile"
	@echo "  monitoring-scenario   Generate monitoring current batch (SCENARIO=none|moderate|major|missingness|unseen_category)"
	@echo "  monitoring-run        Run unlabelled monitoring (SCENARIO=...)"
	@echo "  monitoring-labelled   Run labelled simulation (SCENARIO=...)"
	@echo "  monitoring-status     Show latest monitoring status"
	@echo "  monitoring-all        Reference + all standard scenarios"
	@echo "  monitoring-null-audit Null Monte Carlo false-alert audit"
	@echo "  monitoring-validation Independent detection validation"
	@echo "  docker-config      Validate Compose file syntax"
	@echo "  docker-build       Build the local runtime image"
	@echo "  docker-up          Build and start the Compose stack"
	@echo "  docker-down        Stop the Compose stack"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

format:
	ruff format src app scripts tests
	ruff check --fix src app scripts tests

lint:
	ruff check src app scripts tests

typecheck:
	mypy src app

test:
	pytest

check: format lint typecheck test

generate-data:
	$(PYTHON) -m scripts.generate_data

validate-data:
	$(PYTHON) -m scripts.validate_data

train:
	$(PYTHON) -m scripts.run_mlops

experiments: train

evaluate:
	$(PYTHON) -m scripts.evaluate_model

audit:
	$(PYTHON) -m scripts.audit_training

mlflow-init:
	$(PYTHON) -m scripts.init_mlflow

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

mlflow-list:
	$(PYTHON) -m scripts.list_mlflow_runs

mlflow-verify:
	$(PYTHON) -m scripts.verify_mlflow_roundtrip

select-champion:
	$(PYTHON) -m scripts.select_champion

champion-show:
	$(PYTHON) -m scripts.show_champion

explain:
	$(PYTHON) -m scripts.explain_model

serve-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-app:
	streamlit run app/streamlit_app.py

api-health:
	$(PYTHON) -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"

api-ready:
	$(PYTHON) -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/ready').read().decode())"

docker-config:
	docker compose config -q

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

monitoring-reference:
	$(PYTHON) -m scripts.run_monitoring create-reference

monitoring-scenario:
	$(PYTHON) -m scripts.run_monitoring generate-scenario $(SCENARIO)

monitoring-run:
	$(PYTHON) -m scripts.run_monitoring run-unlabelled $(SCENARIO)

monitoring-labelled:
	$(PYTHON) -m scripts.run_monitoring run-labelled-simulation $(SCENARIO)

monitoring-status:
	$(PYTHON) -m scripts.run_monitoring status

monitoring-all:
	$(PYTHON) -m scripts.run_monitoring run-all

monitoring-null-audit:
	$(PYTHON) -m scripts.run_monitoring run-null-audit

monitoring-validation:
	$(PYTHON) -m scripts.run_monitoring run-validation

clean:
	-$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	-rmdir /s /q .mypy_cache .ruff_cache .pytest_cache htmlcov 2>nul || true
