"""Public-repository boundary, CI workflow and trusted-artifact tests."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml
from src.config import PROJECT_ROOT
from src.mlops.integrity import ArtifactIntegrityError, validate_artifact_uri

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
GITHUB_DIR = PROJECT_ROOT / ".github"


def _github_yaml_files() -> list[Path]:
    return sorted(path for path in GITHUB_DIR.rglob("*") if path.suffix in {".yml", ".yaml"})


def _github_yaml_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _github_yaml_files())


def test_gitignore_and_dockerignore_exclude_runtime_secrets() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for token in (".env", "mlruns/", "data/monitoring", "*.pkl", "*.joblib", ".venv", ".ci-work/"):
        assert token in gitignore
    for token in (".env", "data", "artifacts", "mlruns", ".venv"):
        assert token in dockerignore


def test_env_files_are_gitignored() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".env.local" in gitignore
    assert (PROJECT_ROOT / ".env.example").exists()


def test_bootstrap_sources_do_not_evaluate_frozen_test() -> None:
    for relative in ("scripts/bootstrap_demo.py", "scripts/run_mlops.py"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "scripts.evaluate_model" not in text
        assert "metrics_test.json" not in text or "FROZEN_FORBIDDEN" in text
        assert "processed/test.csv" not in text
    ci = (PROJECT_ROOT / "configs" / "ci.yaml").read_text(encoding="utf-8")
    assert "NOT frozen-v1" in ci
    assert "evaluate_model" not in ci


def test_api_request_schemas_cannot_set_model_location() -> None:
    source = (PROJECT_ROOT / "src" / "api" / "main.py").read_text(encoding="utf-8")
    reload_block = source.split("def reload_champion")[1][:1200]
    assert "artifact_uri" not in reload_block
    assert "champion_path" not in reload_block.lower()
    schemas = (PROJECT_ROOT / "src" / "api" / "schemas.py").read_text(encoding="utf-8")
    request_block = schemas.split("class ShipmentFeatures", 1)[1].split("\nclass ", 1)[0]
    for token in ("model_path", "artifact_uri", "mlflow_tracking_uri", "champion_path"):
        assert token not in request_block


def test_skops_untrusted_flag_is_not_set() -> None:
    assert os.environ.get("SKOPS_ALLOW_UNTRUSTED") in {None, ""}
    serialization = (PROJECT_ROOT / "src" / "mlops" / "serialization.py").read_text(
        encoding="utf-8"
    )
    assert "does **not** set" in serialization
    assert "SKOPS_ALLOW_UNTRUSTED" in serialization


def test_artifact_uri_validation_rejects_filesystem_and_http() -> None:
    with pytest.raises(ArtifactIntegrityError):
        validate_artifact_uri("file:///tmp/model", "abc")
    with pytest.raises(ArtifactIntegrityError):
        validate_artifact_uri("https://example.invalid/model", "abc")
    with pytest.raises(ArtifactIntegrityError):
        validate_artifact_uri("runs:/other/model", "abc")
    assert validate_artifact_uri("runs:/abc/model", "abc") == "runs:/abc/model"


def test_github_actions_minimum_permissions_and_pinned_shas() -> None:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    permissions = workflow.get("permissions") or {}
    assert permissions.get("contents") == "read"
    text = _github_yaml_text()
    assert "contents: write" not in text
    assert "@main" not in text
    assert "scripts.bootstrap_demo" in text
    assert "--mode ci" in text
    assert "evaluate_model" not in text
    assert "uses: ./.github/actions/ci-quality" in workflow_path.read_text(encoding="utf-8")
    assert "uses: ./.github/actions/ci-container" in workflow_path.read_text(encoding="utf-8")
    for path in _github_yaml_files():
        body = path.read_text(encoding="utf-8")
        for match in re.finditer(r"uses:\s+(\S+)", body):
            ref = match.group(1)
            if ref.startswith("./"):
                continue
            _name, _, version = ref.partition("@")
            assert version, f"{path}: {ref}"
            sha = version.split("#")[0].strip()
            assert SHA_RE.match(sha), f"{path}: {ref}"


def test_docker_user_is_non_root() -> None:
    text = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in text
    assert "USER root" not in text.split("USER appuser")[-1]


def test_no_raw_monitoring_csv_in_safe_upload_globs() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    upload = workflow.split("Upload safe reports")[-1]
    assert "*.csv" not in upload
    assert "data/monitoring" not in upload
    assert "mlruns" not in upload
    assert "champion.json" not in upload


def test_api_logs_do_not_include_feature_fields() -> None:
    source = (PROJECT_ROOT / "src" / "api" / "logging.py").read_text(encoding="utf-8")
    for token in ("declared_value_eur", "origin_region", "shipment_id", "password", "token"):
        assert token not in source


def test_workflow_does_not_require_cloud_credentials() -> None:
    text = _github_yaml_text()
    for token in ("AWS_", "AZURE_", "GCP_", "OPENAI_", "${{ secrets."):
        assert token not in text
