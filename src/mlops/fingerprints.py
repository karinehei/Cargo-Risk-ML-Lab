"""Dataset and split fingerprints for experiment lineage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 of a table's column names, dtypes and hashed values."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("|".join(str(dtype) for dtype in frame.dtypes).encode("utf-8"))
    digest.update(str(len(frame)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def split_fingerprint(manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Hash the split manifest when present so fold membership is auditable."""
    if manifest_path is None:
        return {"available": False}
    path = resolve_path(manifest_path)
    if not path.exists():
        return {"available": False}
    return {
        "available": True,
        "sha256": sha256_file(path),
        "n_bytes": path.stat().st_size,
    }
