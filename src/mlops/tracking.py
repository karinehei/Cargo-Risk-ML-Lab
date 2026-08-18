"""Local MLflow tracking configuration (SQLite backend, artifacts gitignored)."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
from pathlib import Path

import mlflow

from src.config import PROJECT_ROOT, get_config, get_settings, setup_logging

logger = setup_logging(name="src.mlops.tracking")

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"


def _sqlite_file(uri: str) -> Path | None:
    if not uri.startswith("sqlite:///"):
        return None
    rest = uri[len("sqlite:///") :]
    path = Path(rest)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def current_git_commit() -> str:
    """Return HEAD SHA when this directory is a git work tree, else 'unavailable'."""
    git = shutil.which("git")
    if not git:
        return "unavailable"
    try:
        # Fixed argv; git executable resolved from PATH. No user-controlled command.
        result = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def configure_tracking(
    tracking_uri: str | None = None,
    experiment: str | None = None,
    artifact_location: str | None = None,
) -> str:
    """Point MLflow at the local SQLite store and ensure the experiment exists."""
    settings = get_settings()
    cfg = get_config()
    uri = tracking_uri or settings.mlflow_tracking_uri or DEFAULT_TRACKING_URI
    sqlite_path = _sqlite_file(uri)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        (PROJECT_ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    name = experiment or str(
        cfg.mlops.get("experiment_name")
        or cfg.training.get("mlflow_experiment")
        or settings.mlflow_experiment_name
    )
    existing = mlflow.get_experiment_by_name(name)
    if existing is None:
        if artifact_location:
            mlflow.create_experiment(name, artifact_location=str(Path(artifact_location).resolve()))
        else:
            mlflow.create_experiment(name)
    mlflow.set_experiment(name)
    logger.info("MLflow tracking_uri=%s experiment=%s", uri, name)
    return uri


def init_tracking_store(tracking_uri: str | None = None) -> Path:
    """Create the SQLite tracking database by listing experiments."""
    uri = configure_tracking(tracking_uri=tracking_uri)
    client = mlflow.MlflowClient()
    experiments = client.search_experiments()
    logger.info("Tracking store ready with %s experiment(s)", len(experiments))
    return _sqlite_file(uri) or (PROJECT_ROOT / "mlruns" / "mlflow.db")
