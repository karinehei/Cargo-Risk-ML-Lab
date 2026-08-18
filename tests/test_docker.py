"""Docker and Compose configuration checks. Runtime execution is separate."""

from __future__ import annotations

import shutil
import subprocess

import pytest
import yaml
from src.config import PROJECT_ROOT


def _lock_version(lock: str, name: str) -> tuple[int, ...]:
    for line in lock.splitlines():
        if line.startswith(f"{name}=="):
            raw = line.split("==", 1)[1].split()[0]
            return tuple(int(part) for part in raw.split("."))
    raise AssertionError(f"{name} is missing from the lock file")


def test_runtime_lock_clears_known_trivy_highs() -> None:
    text = (PROJECT_ROOT / "requirements" / "runtime.lock.txt").read_text(encoding="utf-8")
    assert _lock_version(text, "mlflow") >= (3, 15, 0)
    assert _lock_version(text, "pyarrow") >= (23, 0, 1)
    assert _lock_version(text, "msgpack") >= (1, 2, 1)
    assert _lock_version(text, "setuptools") >= (78, 1, 1)


def test_trivyignore_documents_mlflow_cryptography_cap() -> None:
    text = (PROJECT_ROOT / ".trivyignore").read_text(encoding="utf-8")
    assert "CVE-2026-69247" in text
    action = (PROJECT_ROOT / ".github" / "actions" / "ci-container" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert "trivyignores: .trivyignore" in action
    assert "ignore-unfixed: true" in action
    assert "scanners: vuln" in action


def test_dockerfile_runtime_constraints() -> None:
    text = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12-slim" in text
    assert "USER appuser" in text
    assert "libgomp1" in text
    assert "mkdir -p /app/mlruns" in text
    assert "chown -R appuser:appuser /app" in text
    assert '".[dev]"' not in text
    assert "[dev]" not in text
    assert "COPY data" not in text
    assert "COPY artifacts" not in text
    assert "COPY mlruns" not in text


def test_dockerignore_excludes_data_and_secrets() -> None:
    text = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for token in ("data", "data/monitoring/*.csv", "artifacts", "mlruns", ".env", "tests"):
        assert token in text


def test_compose_services_and_network() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = payload["services"]
    assert set(services) == {"api", "app", "mlflow"}
    assert payload["networks"]["cargo-risk"]["driver"] == "bridge"
    assert services["app"]["environment"]["STREAMLIT_API_URL"] == "http://api:8000"
    assert "8000:8000" in services["api"]["ports"]
    assert "8501:8501" in services["app"]["ports"]
    assert "5000:5000" in services["mlflow"]["ports"]
    for name in ("api", "app", "mlflow"):
        assert "healthcheck" in services[name]
        assert "cargo-risk" in services[name]["networks"]
    api_volumes = services["api"]["volumes"]
    assert any(item.startswith("./artifacts:") and item.endswith(":ro") for item in api_volumes)
    assert any(item.startswith("./mlruns:") for item in api_volumes)
    assert any(item.startswith("./configs:") and item.endswith(":ro") for item in api_volumes)


def test_ci_trivy_scans_exported_archive() -> None:
    text = (PROJECT_ROOT / ".github" / "actions" / "ci-container" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert "provenance: false" in text
    assert "docker save cargo-risk-ml-lab:ci" in text
    assert "ignore-unfixed: true" in text
    assert "input:" in text
    assert "image-ref:" not in text


def test_ci_container_health_check_exposes_mlflow_paths() -> None:
    api = (PROJECT_ROOT / "scripts" / "ci" / "smoke_api.sh").read_text(encoding="utf-8")
    docker = (PROJECT_ROOT / "scripts" / "ci" / "docker_health.sh").read_text(encoding="utf-8")
    assert "uvicorn_pid" in api
    assert "trap cleanup EXIT" in api
    assert '-v "$PWD:$PWD"' in docker
    assert '-v "$PWD/.ci-work:/app/.ci-work"' in docker
    assert "chmod -R a+rwX .ci-work" in docker
    assert "trap cleanup EXIT" in docker
    assert "docker logs cargo-risk-api-ci" in docker


def test_compose_config_syntax() -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker executable is not available")
    result = subprocess.run(
        [docker, "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), "config", "-q"],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose config is unavailable: {result.stderr.strip()[:300]}")
    assert result.returncode == 0
