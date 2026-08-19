"""API integration tests. Uses tiny train/val fixtures, never the frozen test CSV."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from src.api.logging import JsonLogFormatter
from src.api.state import clear_cached_bundle, get_cached_bundle
from src.config import get_settings
from src.explainability.semantics import SCORE_SEMANTICS_UNCALIBRATED, SCORE_WARNING

VALID_SHIPMENT = {
    "origin_region": "Asia",
    "destination_region": "Northern Europe",
    "commodity_category": "electronics",
    "transport_mode": "air",
    "declared_value_eur": 28000.0,
    "shipment_weight_kg": 55.0,
    "declaration_completeness_score": 0.45,
    "documentation_count": 2,
    "previous_discrepancies": 3,
    "sender_history_length": 4,
    "route_rarity": 0.7,
    "declared_vs_estimated_value_deviation": 0.5,
    "submission_hour": 2,
    "expedited_shipment": 1,
}


def _fresh_app() -> object:
    get_settings.cache_clear()
    clear_cached_bundle()
    import src.api.main as api_main

    return api_main.create_app()


def _error_shape(body: dict[str, object]) -> None:
    assert set(body) == {"error_code", "message", "request_id"}
    assert body["error_code"]
    assert body["message"]
    assert body["request_id"]
    blob = json.dumps(body).lower()
    assert "traceback" not in blob
    assert "mlruns" not in blob
    assert "artifacts/" not in blob
    assert "file:" not in blob


def test_health_ready_predict_explain(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code == 307
        assert root.headers["location"] == "/docs"

        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert "disclaimer" in body
        assert "model_loaded" not in body
        assert health.headers.get("X-Request-ID")

        ready = client.get("/ready")
        assert ready.status_code == 200
        ready_body = ready.json()
        assert ready_body["status"] == "ready"
        assert ready_body["explanations_available"] is True
        assert ready_body["decision_threshold"] > 0

        info = client.get("/model")
        assert info.status_code == 200
        payload = info.json()
        assert payload["model_name"] == "logreg"
        assert payload["calibration_status"] == "none"
        assert payload["score_is_calibrated"] is False
        assert payload["calibration_method"] is None
        assert payload["score_semantics"] == SCORE_SEMANTICS_UNCALIBRATED
        assert "requires_review_probability" not in payload
        assert "artifact" not in payload
        assert "path" not in str(payload).lower()

        schema = client.get("/openapi.json").json()
        properties = schema["components"]["schemas"]["PredictionResponse"]["properties"]
        assert "requires_review_probability" not in properties
        assert "review_score" in properties
        assert "/model-info" not in schema["paths"]

        response = client.post("/predict", json=VALID_SHIPMENT)
        assert response.status_code == 200
        scored = response.json()
        assert 0.0 <= scored["review_score"] <= 1.0
        assert "requires_review_probability" not in scored
        assert "threshold" not in scored
        assert scored["requires_review"] in (0, 1)
        assert scored["score_is_calibrated"] is False
        assert scored["score_semantics"] == SCORE_SEMANTICS_UNCALIBRATED
        assert scored["score_warning"] == SCORE_WARNING
        assert scored["human_review_notice"]
        assert scored["model_version"]
        assert scored["mlflow_run_id"]
        assert "disclaimer" in scored
        assert (
            "probability" not in scored["score_semantics"]
            or "not a literal probability" in scored["score_semantics"]
        )

        explained = client.post("/explain", json=VALID_SHIPMENT)
        assert explained.status_code == 200
        local = explained.json()
        assert local["score_is_calibrated"] is False
        assert local["reconstruction_error"] < 1e-6
        assert local["reconstruction_ok"] is True
        assert local["classification"]
        assert local["strongest_positive_contributions"]
        assert local["strongest_negative_contributions"]
        assert "causation" in local["causation_disclaimer"].lower()
        assert "shipment_id" not in local["original_input_values"]
        assert local["model_version"]


def test_batch_prediction(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"shipments": [VALID_SHIPMENT, {**VALID_SHIPMENT, "transport_mode": "road"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2
        assert body["model_version"]
        assert "requires_review_probability" not in body["predictions"][0]


def test_batch_size_rejection(tiny_champion: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_MAX_BATCH_SIZE", "2")
    app = _fresh_app()
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"shipments": [VALID_SHIPMENT, VALID_SHIPMENT, VALID_SHIPMENT]},
        )
        assert response.status_code == 422
        _error_shape(response.json())
        assert "batch" in response.json()["message"].lower()


def test_invalid_category(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID_SHIPMENT, "origin_region": "Moon"})
        assert response.status_code == 422
        _error_shape(response.json())


def test_non_finite_number(tiny_champion: Path) -> None:
    app = _fresh_app()
    payload = dict(VALID_SHIPMENT)
    payload["declared_value_eur"] = float("nan")
    body = json.dumps(payload)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        _error_shape(response.json())


def test_unexpected_field(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID_SHIPMENT, "unexpected_field": "nope"})
        assert response.status_code == 422
        _error_shape(response.json())


def test_optional_fields_may_be_omitted(tiny_champion: Path) -> None:
    app = _fresh_app()
    payload = {
        "origin_region": "Asia",
        "destination_region": "Northern Europe",
        "commodity_category": "electronics",
        "transport_mode": "air",
        "declared_value_eur": 12500.0,
        "shipment_weight_kg": 85.5,
        "previous_discrepancies": 0,
        "submission_hour": 10,
        "expedited_shipment": 0,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


def test_uncalibrated_score_terminology(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        scored = client.post("/predict", json=VALID_SHIPMENT).json()
        assert scored["score_is_calibrated"] is False
        assert scored["score_semantics"] == SCORE_SEMANTICS_UNCALIBRATED
        assert "calibrated probability" not in scored["score_semantics"]
        assert "requires_review_probability" not in scored


def test_request_id_on_success_and_error(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        ok = client.get("/health", headers={"X-Request-ID": "req-health-1"})
        assert ok.headers["X-Request-ID"] == "req-health-1"
        bad = client.post(
            "/predict",
            json={**VALID_SHIPMENT, "origin_region": "Moon"},
            headers={"X-Request-ID": "req-invalid-1"},
        )
        assert bad.headers["X-Request-ID"] == "req-invalid-1"
        assert bad.json()["request_id"] == "req-invalid-1"


def test_logs_omit_prediction_inputs(tiny_champion: Path) -> None:
    app = _fresh_app()
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(JsonLogFormatter().format(record))

    logger = logging.getLogger("src.api")
    handler = _Capture()
    logger.addHandler(handler)
    unique = 424242.25
    payload = {**VALID_SHIPMENT, "declared_value_eur": unique}
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
    finally:
        logger.removeHandler(handler)
    blob = "\n".join(records)
    assert "424242.25" not in blob
    assert "electronics" not in blob
    assert "request_completed" in blob


def test_predict_without_champion_returns_generic_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHAMPION_PATH", str(tmp_path / "missing-champion.json"))
    app = _fresh_app()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert "model_loaded" not in health.json()
        ready = client.get("/ready")
        assert ready.status_code == 503
        _error_shape(ready.json())
        response = client.post("/predict", json=VALID_SHIPMENT)
        assert response.status_code == 503
        _error_shape(response.json())
        detail = json.dumps(response.json())
        assert "not available" in detail.lower() or "not ready" in detail.lower()
        assert str(tmp_path) not in detail
        assert "traceback" not in detail.lower()


def test_corrupted_champion_metadata_is_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "champion.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("CHAMPION_PATH", str(path))
    app = _fresh_app()
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        _error_shape(ready.json())
        assert str(path) not in json.dumps(ready.json())


def test_mismatched_threshold_is_503(
    tiny_champion: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    metadata["threshold"] = 0.99
    bad = tmp_path / "mismatch.json"
    bad.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("CHAMPION_PATH", str(bad))
    app = _fresh_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        _error_shape(ready.json())


def test_mismatched_run_id_is_503(
    tiny_champion: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    metadata["artifact_uri"] = "runs:/00000000000000000000000000000000/model"
    bad = tmp_path / "uri.json"
    bad.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("CHAMPION_PATH", str(bad))
    app = _fresh_app()
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        _error_shape(ready.json())


def test_unavailable_mlflow_artifact_is_503(
    tiny_champion: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = json.loads(tiny_champion.read_text(encoding="utf-8"))
    run_id = str(metadata["mlflow_run_id"])
    metadata["artifact_uri"] = f"runs:/{run_id}/missing-artifact"
    bad = tmp_path / "missing-artifact.json"
    bad.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("CHAMPION_PATH", str(bad))
    app = _fresh_app()
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        _error_shape(ready.json())


def test_cached_loading_does_not_reload(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        first = client.get("/ready")
        assert first.status_code == 200
        cached = get_cached_bundle()
        assert cached is not None
        with patch("src.api.main.load_and_verify") as mocked:
            scored = client.post("/predict", json=VALID_SHIPMENT)
            mocked.assert_not_called()
            assert scored.status_code == 200
            assert get_cached_bundle() is cached


def test_api_does_not_train_on_startup(tiny_champion: Path) -> None:
    source = Path("src/api/main.py").read_text(encoding="utf-8")
    assert "train_model" not in source
    assert "run_mlops" not in source
    app = _fresh_app()
    with patch("src.models.train.train_model") as train:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
        train.assert_not_called()


def test_api_sources_do_not_reference_test_csv() -> None:
    for relative in ("src/api/main.py", "src/api/readiness.py", "src/api/schemas.py"):
        text = Path(relative).read_text(encoding="utf-8")
        assert "test.csv" not in text
        assert "metrics_test" not in text


def test_monitoring_status_unavailable_without_report(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        response = client.get("/monitoring/status")
        assert response.status_code == 200
        body = response.json()
        if body.get("available"):
            assert body.get("monitoring_run_id")
        else:
            assert body["available"] is False
        assert "artifacts/" not in json.dumps(body)
        assert response.headers.get("X-Request-ID")


def test_monitoring_latest_safe_payload(tiny_champion: Path) -> None:
    app = _fresh_app()
    with TestClient(app) as client:
        response = client.get("/monitoring/latest")
        assert response.status_code == 200
        body = response.json()
        assert "available" in body
        blob = json.dumps(body).lower()
        assert "traceback" not in blob
        assert "mlruns" not in blob
        assert "shipments" not in blob
