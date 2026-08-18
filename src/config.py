"""Shared configuration, logging and seeding utilities."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    config_path: str = "configs/default.yaml"
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    # Bind address for local/Compose serving; not an internet-facing default.
    api_host: str = "0.0.0.0"  # nosec B104
    api_port: int = 8000
    api_model_path: str = "artifacts/model.joblib"
    api_pipeline_path: str = "artifacts/preprocess_pipeline.joblib"
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment_name: str = "cargo-risk-ml-lab"
    champion_path: str = "artifacts/mlops/champion.json"
    streamlit_api_url: str = "http://localhost:8000"
    api_max_batch_size: int = 100
    api_max_request_bytes: int = 1_048_576
    api_reload_token: str = ""


@dataclass(frozen=True)
class AppConfig:
    """Typed view over the YAML configuration file."""

    raw: dict[str, Any] = field(repr=False)
    random_seed: int = 42
    data: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    mlops: dict[str, Any] = field(default_factory=dict)
    api: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    project: dict[str, Any] = field(default_factory=dict)

    @property
    def disclaimer(self) -> str:
        return str(
            self.project.get(
                "disclaimer",
                "Educational synthetic demonstration only.",
            )
        ).strip()


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root when not absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = root or PROJECT_ROOT
    return (base / candidate).resolve()


def load_yaml_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration into an ``AppConfig`` instance."""
    settings = get_settings()
    config_file = resolve_path(path or settings.config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return AppConfig(
        raw=raw,
        random_seed=int(raw.get("random_seed", settings.random_seed)),
        data=dict(raw.get("data", {})),
        features=dict(raw.get("features", {})),
        model=dict(raw.get("model", {})),
        training=dict(raw.get("training", {})),
        evaluation=dict(raw.get("evaluation", {})),
        explainability=dict(raw.get("explainability", {})),
        monitoring=dict(raw.get("monitoring", {})),
        mlops=dict(raw.get("mlops", {})),
        api=dict(raw.get("api", {})),
        logging=dict(raw.get("logging", {})),
        project=dict(raw.get("project", {})),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached environment settings."""
    return Settings()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached application configuration."""
    return load_yaml_config()


def set_seed(seed: int | None = None) -> int:
    """Set Python, NumPy and related RNG seeds for reproducibility."""
    resolved = int(seed if seed is not None else get_config().random_seed)
    random.seed(resolved)
    np.random.seed(resolved)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    return resolved


def setup_logging(level: str | None = None, name: str | None = None) -> logging.Logger:
    """Configure structured console logging and return a logger."""
    config = get_config()
    log_level = (level or config.logging.get("level") or get_settings().log_level).upper()
    log_format = str(
        config.logging.get(
            "format",
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    )

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format)
    else:
        root.setLevel(getattr(logging, log_level, logging.INFO))

    logger_name = name or "cargo_risk_ml_lab"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger
