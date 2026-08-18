"""Trusted-artifact URI and checksum checks."""

from __future__ import annotations

import pytest
from src.mlops.integrity import (
    ArtifactIntegrityError,
    validate_artifact_uri,
    verify_champion_integrity,
)


def test_validate_artifact_uri_accepts_runs_and_models() -> None:
    assert validate_artifact_uri("runs:/abc123/model", "abc123") == "runs:/abc123/model"
    assert validate_artifact_uri("models:/cargo/1", "unused") == "models:/cargo/1"


def test_validate_artifact_uri_rejects_remote_and_path_tokens() -> None:
    for uri in (
        "file:///tmp/model",
        "https://example.invalid/model",
        "s3://bucket/model",
        "runs:/../model",
        r"C:\models\x",
    ):
        with pytest.raises(ArtifactIntegrityError):
            validate_artifact_uri(uri, "abc")


def test_checksum_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.mlops.integrity.compute_mlflow_artifact_sha256",
        lambda uri: "1" * 64,
    )
    metadata = {
        "mlflow_run_id": "abc",
        "artifact_uri": "runs:/abc/model",
        "artifact_sha256": "0" * 64,
        "serialization": "mlflow.sklearn:cloudpickle",
    }
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        verify_champion_integrity(metadata)


def test_missing_checksum_still_validates_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.mlops.integrity.compute_mlflow_artifact_sha256",
        lambda uri: pytest.fail("checksum must not be computed when absent"),
    )
    metadata = {
        "mlflow_run_id": "abc",
        "artifact_uri": "runs:/abc/model",
        "serialization": "mlflow.sklearn:cloudpickle",
    }
    result = verify_champion_integrity(metadata)
    assert result["artifact_uri"] == "runs:/abc/model"
    assert result["artifact_sha256"] == ""
