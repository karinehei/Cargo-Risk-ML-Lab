"""Load the registered champion for serving. No training, no silent fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow
from sklearn.base import BaseEstimator

from src.config import get_config, get_settings, resolve_path, setup_logging
from src.explainability.semantics import SCORE_WARNING, score_metadata_from_champion
from src.mlops.serialization import load_sklearn_pipeline
from src.mlops.tracking import configure_tracking

logger = setup_logging(name="src.mlops.serving")

REQUIRED_CHAMPION_FIELDS = (
    "model_name",
    "model_version",
    "mlflow_run_id",
    "threshold",
    "threshold_selection_method",
    "calibration_status",
    "validation_metrics",
    "artifact_uri",
    "policy_version",
)


class ChampionLoadError(RuntimeError):
    """Raised when the champion cannot be loaded. Messages must not include paths."""


@dataclass
class ServingBundle:
    """In-memory champion used by the API."""

    pipeline: BaseEstimator
    metadata: dict[str, Any]
    threshold: float


def load_champion(champion_path: str | None = None) -> ServingBundle:
    """Load the registered champion and matching threshold from MLflow.

    Raises:
        ChampionLoadError: If metadata or the artifact is missing/inconsistent.
    """
    cfg = get_config()
    settings = get_settings()
    configured = (
        champion_path
        or settings.champion_path
        or str(cfg.mlops.get("champion_path", "artifacts/mlops/champion.json"))
    )
    path = resolve_path(configured)
    if not path.exists():
        raise ChampionLoadError("Champion metadata is unavailable.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Champion metadata could not be parsed: %s", type(exc).__name__)
        raise ChampionLoadError("Champion metadata is unreadable.") from exc

    missing = [field for field in REQUIRED_CHAMPION_FIELDS if field not in metadata]
    if missing:
        raise ChampionLoadError("Champion metadata is incomplete.")

    threshold = float(metadata["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ChampionLoadError("Champion threshold is inconsistent.")

    run_id = str(metadata["mlflow_run_id"])
    uri = str(metadata["artifact_uri"])
    if uri.startswith("file:") or ":\\" in uri or uri.startswith("/"):
        raise ChampionLoadError("Champion artifact URI is unsupported.")
    if uri.startswith("runs:/") and run_id not in uri:
        raise ChampionLoadError("Champion artifact URI does not match the recorded run.")
    if not (uri.startswith("runs:/") or uri.startswith("models:/")):
        raise ChampionLoadError("Champion artifact URI is unsupported.")

    try:
        configure_tracking()
        pipeline = load_sklearn_pipeline(uri)
    except ChampionLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Champion artifact load failed: %s", type(exc).__name__)
        raise ChampionLoadError("Champion model artifact could not be loaded.") from exc

    try:
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        run_threshold = run.data.params.get("threshold")
        if run_threshold is not None and abs(float(run_threshold) - threshold) > 1e-9:
            raise ChampionLoadError("Champion threshold does not match the MLflow run.")
    except ChampionLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Champion run verification failed: %s", type(exc).__name__)
        raise ChampionLoadError("Champion run could not be verified.") from exc

    logger.info(
        "Loaded champion %s version %s (calibration=%s)",
        metadata.get("model_name"),
        metadata.get("model_version"),
        metadata.get("calibration_status"),
    )
    semantics = score_metadata_from_champion(metadata)
    metadata = {**metadata, **semantics, "score_warning": SCORE_WARNING}
    return ServingBundle(pipeline=pipeline, metadata=metadata, threshold=threshold)
