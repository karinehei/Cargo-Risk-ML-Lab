"""Copy frozen training artifacts so later audit writes cannot overwrite them."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.config import resolve_path, setup_logging

logger = setup_logging(name="src.audit.freeze")

FROZEN_V1_FILES: tuple[str, ...] = (
    "model.joblib",
    "preprocess_pipeline.joblib",
    "model_metadata.json",
    "metrics_test.json",
    "metrics_validation.json",
    "model_comparison.csv",
    "model_comparison.json",
    "threshold_sweep_validation.csv",
    "error_analysis_test.json",
    "error_analysis_validation.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_frozen_v1(
    artifact_dir: str | Path = "artifacts",
    dest_dir: str | Path = "artifacts/frozen_v1",
) -> dict[str, Any]:
    """Copy the original experiment artifacts once. Existing copies are left intact."""
    source = resolve_path(artifact_dir)
    dest = resolve_path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped_existing: list[str] = []
    missing: list[str] = []
    checksums: dict[str, str] = {}

    for name in FROZEN_V1_FILES:
        src_path = source / name
        dest_path = dest / name
        if not src_path.exists():
            missing.append(name)
            continue
        checksums[name] = _sha256(src_path)
        if dest_path.exists():
            skipped_existing.append(name)
            continue
        shutil.copy2(src_path, dest_path)
        copied.append(name)

    plots_src = source / "plots"
    plots_dest = dest / "plots"
    if plots_src.exists() and not plots_dest.exists():
        shutil.copytree(plots_src, plots_dest)
        copied.append("plots/")

    manifest = {
        "source": str(source),
        "destination": str(dest),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "missing": missing,
        "checksums_sha256": checksums,
        "note": (
            "Frozen v1 test metrics must not be overwritten. Audit outputs belong "
            "under artifacts/audit/."
        ),
    }
    manifest_path = dest / "FREEZE_MANIFEST.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Frozen v1 archive at %s (copied=%s missing=%s)", dest, copied, missing)
    return manifest
