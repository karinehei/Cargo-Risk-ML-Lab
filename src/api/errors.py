"""Safe API error payloads. No paths, stack traces or storage URIs."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

GENERIC_UNAVAILABLE = "Model is not available."
GENERIC_NOT_READY = "Service is not ready."
GENERIC_INVALID = "Request is invalid."
GENERIC_TOO_LARGE = "Request is too large."
GENERIC_BATCH_SIZE = "Batch size exceeds the configured maximum."
GENERIC_INTERNAL = "Request could not be completed."


def error_payload(error_code: str, message: str, request_id: str) -> dict[str, str]:
    return {
        "error_code": error_code,
        "message": message,
        "request_id": request_id,
    }


def error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(error_code, message, request_id),
        headers={"X-Request-ID": request_id},
    )


def sanitize_text(value: Any) -> str:
    """Drop filesystem/URI fragments from any unexpected string."""
    text = str(value)
    lowered = text.lower()
    if any(
        token in lowered for token in ("mlruns", "artifacts/", "file:", "traceback", ":\\", "/mnt/")
    ):
        return GENERIC_INTERNAL
    return text
