"""Trusted-artifact checksums and URI validation for local MLflow models.

Cloudpickle MLflow artifacts are executable Python. Load them only from the
local tracking store after URI and checksum checks. Request bodies must never
supply model paths or MLflow URIs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, resolve_path, setup_logging

logger = setup_logging(name="src.mlops.integrity")

TRUSTED_URI_PREFIXES = ("runs:/", "models:/")
FORBIDDEN_URI_TOKENS = (":\\", "..", "file:", "http://", "https://", "s3://", "ftp://")


class ArtifactIntegrityError(RuntimeError):
    """Raised when a local artifact fails provenance or checksum checks."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def config_fingerprint(config_path: str | Path | None = None) -> str:
    path = resolve_path(config_path or "configs/default.yaml")
    return sha256_file(path)


def hash_directory(directory: str | Path) -> str:
    """Deterministic SHA-256 over relative paths and file bytes."""
    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_artifact_uri(uri: str, run_id: str) -> str:
    """Reject filesystem, remote and mismatched MLflow URIs."""
    cleaned = str(uri or "").strip()
    if not cleaned:
        raise ArtifactIntegrityError("Champion artifact URI is missing.")
    lowered = cleaned.lower()
    for token in FORBIDDEN_URI_TOKENS:
        if token in cleaned or token in lowered:
            raise ArtifactIntegrityError("Champion artifact URI is unsupported.")
    if not cleaned.startswith(TRUSTED_URI_PREFIXES):
        raise ArtifactIntegrityError("Champion artifact URI is unsupported.")
    if cleaned.startswith("runs:/") and run_id not in cleaned:
        raise ArtifactIntegrityError("Champion artifact URI does not match the recorded run.")
    return cleaned


def compute_mlflow_artifact_sha256(artifact_uri: str) -> str:
    """Hash a trusted ``runs:/`` or ``models:/`` artifact tree from the local store."""
    import mlflow.artifacts

    local = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    return hash_directory(local)


def verify_champion_integrity(metadata: dict[str, Any]) -> dict[str, str]:
    """Validate URI, optional checksum, and trusted serialization."""
    run_id = str(metadata.get("mlflow_run_id") or "")
    uri = validate_artifact_uri(str(metadata.get("artifact_uri") or ""), run_id)
    serialization = str(metadata.get("serialization") or "")
    if serialization.startswith("joblib"):
        raise ArtifactIntegrityError("Joblib fallback artifacts are not trusted for serving.")
    expected = metadata.get("artifact_sha256")
    observed = None
    if expected:
        observed = compute_mlflow_artifact_sha256(uri)
        if str(expected) != observed:
            raise ArtifactIntegrityError("Champion artifact checksum mismatch.")
    return {
        "artifact_uri": uri,
        "artifact_sha256": str(observed or expected or ""),
        "config_sha256": str(metadata.get("config_sha256") or ""),
        "dataset_fingerprint": str(metadata.get("dataset_fingerprint") or ""),
    }


def write_reproducibility_manifest(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output


def assert_path_inside_project(path: str | Path) -> Path:
    resolved = resolve_path(path)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ArtifactIntegrityError("Path is outside the trusted project boundary.") from exc
    return resolved
