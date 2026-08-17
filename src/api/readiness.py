"""Readiness checks for the cached champion. Does not train."""

from __future__ import annotations

from typing import Any

from src.explainability.linear import LinearExplanationModel
from src.mlops.serving import ChampionLoadError, ServingBundle, load_champion


def verify_bundle(bundle: ServingBundle) -> None:
    """Confirm metadata, threshold, run ID and linear explanations are consistent."""
    meta = bundle.metadata
    version = str(meta.get("model_version") or "")
    run_id = str(meta.get("mlflow_run_id") or "")
    if not version or not run_id:
        raise ChampionLoadError("Champion metadata is incomplete.")
    uri = str(meta.get("artifact_uri") or "")
    if uri.startswith("runs:/") and run_id not in uri:
        raise ChampionLoadError("Champion artifact URI does not match the recorded run.")
    threshold = float(bundle.threshold)
    meta_threshold = float(meta.get("decision_threshold") or meta.get("threshold") or 0.0)
    if abs(threshold - meta_threshold) > 1e-9:
        raise ChampionLoadError("Champion threshold does not match the recorded metadata.")
    try:
        LinearExplanationModel.from_pipeline(bundle.pipeline)
    except Exception as exc:  # noqa: BLE001
        raise ChampionLoadError("Champion explanation metadata is unavailable.") from exc


def load_and_verify(champion_path: str | None = None) -> ServingBundle:
    """Load from MLflow and verify. Never trains and never returns a fallback model."""
    bundle = load_champion(champion_path)
    verify_bundle(bundle)
    return bundle


def readiness_payload(bundle: ServingBundle) -> dict[str, Any]:
    meta = bundle.metadata
    return {
        "status": "ready",
        "model_version": str(meta.get("model_version") or ""),
        "mlflow_run_id": str(meta.get("mlflow_run_id") or ""),
        "decision_threshold": float(bundle.threshold),
        "score_is_calibrated": bool(meta.get("score_is_calibrated", False)),
        "explanations_available": True,
    }
