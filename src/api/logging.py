"""Request-ID context and JSON operational logs. Never log prediction inputs."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line with operational fields only."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get("-"),
        }
        for key in ("endpoint", "status_code", "latency_ms", "batch_size", "model_version"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str)


def configure_api_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("src.api")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not any(isinstance(handler.formatter, JsonLogFormatter) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        logger.handlers.clear()
        logger.addHandler(handler)
    return logger


def log_request(
    logger: logging.Logger,
    *,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    batch_size: int | None = None,
    model_version: str | None = None,
) -> None:
    extra = {
        "request_id": request_id_var.get("-"),
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "batch_size": batch_size,
        "model_version": model_version,
    }
    logger.info(
        "request_completed",
        extra={key: value for key, value in extra.items() if value is not None},
    )
