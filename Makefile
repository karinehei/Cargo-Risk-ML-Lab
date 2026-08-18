.PHONY: help install install-dev install-locked format lint typecheck test check \
	generate-data validate-data train evaluate audit explain \
	mlflow-init mlflow-ui mlflow-list mlflow-verify experiments select-champion champion-show \
	serve-api serve-app docker-up docker-down docker-config docker-build \
	api-health api-ready \
	monitoring-reference monitoring-scenario monitoring-run monitoring-labelled monitoring-status monitoring-all \
	monitoring-null-audit monitoring-validation \
	bootstrap-demo bootstrap-ci bandit pip-audit secrets-scan sbom repo-audit \
	clean

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

help:
	@echo "Cargo Risk ML Lab – common targets"
	@echo "  install-locked     Install from hashed lock files, then the local package"
	@echo "  bootstrap-demo     Isolated full clean-clone bootstrap (no frozen test eval)"
	@echo "  bootstrap-ci       Lightweight architecture smoke under .ci-work/"
	@echo "  bandit             Python security linter (src, app, scripts)"
	@echo "  pip-audit          Known-vulnerability scan of the lock file"
	@echo "  secrets-scan       detect-secrets baseline comparison"
	@echo "  sbom               CycloneDX SBOM from the current environment"
	@echo "  repo-audit         Filesystem repository-boundary checks"
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
	$(PIP) install -e ".[dev,security]"

install-locked:
	$(PIP) install --require-hashes -r requirements/dev.lock.txt
	$(PIP) install --no-deps -e .

format:
	$(PYTHON) -m ruff format src app scripts tests
	$(PYTHON) -m ruff check --fix src app scripts tests

lint:
	$(PYTHON) -m ruff check src app scripts tests

typecheck:
	$(PYTHON) -m mypy src app

test:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

bootstrap-demo:
	$(PYTHON) -m scripts.bootstrap_demo --mode full

bootstrap-ci:
	$(PYTHON) -m scripts.bootstrap_demo --mode ci

bandit:
	$(PYTHON) -m bandit -c bandit.yaml -r src app scripts

pip-audit:
	$(PYTHON) -m pip_audit -r requirements/dev.lock.txt

secrets-scan:
	$(PYTHON) -m detect_secrets scan --baseline .secrets.baseline --exclude-files '\.ci-work/.*|mlruns/.*|artifacts/.*|.*\.lock\.txt|sbom.*'

sbom:
	$(PYTHON) -c "import subprocess, sys; from pathlib import Path; script = Path(sys.executable).resolve().parent / 'cyclonedx-py'; subprocess.check_call([str(script), 'environment', '-o', 'sbom.cdx.json', '--pyproject', 'pyproject.toml'])"

repo-audit:
	$(PYTHON) -m scripts.audit_repository_boundary

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
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

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
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

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
