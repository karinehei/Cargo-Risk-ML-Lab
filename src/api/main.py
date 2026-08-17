"""FastAPI application for synthetic cargo review scoring."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import DISCLAIMER, __version__
from src.api.errors import (
    GENERIC_BATCH_SIZE,
    GENERIC_INTERNAL,
    GENERIC_INVALID,
    GENERIC_NOT_READY,
    GENERIC_TOO_LARGE,
    GENERIC_UNAVAILABLE,
    error_response,
)
from src.api.logging import configure_api_logging, log_request, request_id_var
from src.api.readiness import load_and_verify, readiness_payload, verify_bundle
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    LocalExplanationResponse,
    ModelInfoResponse,
    MonitoringLatestResponse,
    MonitoringStatusResponse,
    PredictionResponse,
    ReadyResponse,
    ReloadResponse,
    ShipmentFeatures,
)
from src.api.state import clear_cached_bundle, get_cached_bundle, set_cached_bundle
from src.config import get_config, get_settings
from src.explainability.linear import LinearExplanationModel
from src.explainability.semantics import (
    HUMAN_REVIEW_NOTICE,
    REVIEW_LABEL_NEGATIVE,
    REVIEW_LABEL_POSITIVE,
    SCORE_WARNING,
    score_metadata_from_champion,
)
from src.mlops.serving import ChampionLoadError, ServingBundle
from src.models import predict_proba, prepare_inference_frame
from src.monitoring.report import load_latest_report, load_latest_status

logger = configure_api_logging()


def _request_id(request: Request | None = None) -> str:
    if request is not None:
        header = request.headers.get("x-request-id")
        if header:
            return header[:128]
    current = request_id_var.get("-")
    if current and current != "-":
        return current
    return str(uuid.uuid4())


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail=GENERIC_UNAVAILABLE)


def get_serving() -> ServingBundle:
    bundle = get_cached_bundle()
    if bundle is None:
        raise _unavailable()
    return bundle


def _prediction_body(score: float, bundle: ServingBundle) -> PredictionResponse:
    semantics = score_metadata_from_champion(bundle.metadata)
    threshold = float(bundle.threshold)
    meta = bundle.metadata
    return PredictionResponse(
        review_score=float(score),
        decision_threshold=threshold,
        requires_review=int(score >= threshold),
        model_version=str(meta.get("model_version") or ""),
        mlflow_run_id=str(meta.get("mlflow_run_id") or ""),
        score_is_calibrated=bool(semantics["score_is_calibrated"]),
        score_semantics=str(semantics["score_semantics"]),
        human_review_notice=HUMAN_REVIEW_NOTICE,
        score_warning=SCORE_WARNING,
        disclaimer=DISCLAIMER,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the registered champion. Never trains and never falls back silently."""
    try:
        set_cached_bundle(load_and_verify())
    except ChampionLoadError:
        clear_cached_bundle()
        logger.warning("Champion is not available at startup")
    yield
    clear_cached_bundle()


def create_app() -> FastAPI:
    """Application factory for the FastAPI service."""
    cfg = get_config()
    settings = get_settings()
    max_bytes = int(settings.api_max_request_bytes)
    application = FastAPI(
        title=str(cfg.api.get("title", "Cargo Risk ML Lab API")),
        version=str(cfg.api.get("version", __version__)),
        description=(
            "Educational API for synthetic cargo review scoring. "
            "review_score is a ranking/threshold value, not a calibrated probability. "
            f"{DISCLAIMER}"
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def operational_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        response: Response
        try:
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    size = int(length)
                except ValueError:
                    size = 0
                if size > max_bytes:
                    response = error_response(
                        413, "REQUEST_TOO_LARGE", GENERIC_TOO_LARGE, request_id
                    )
                else:
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except Exception:  # noqa: BLE001
            logger.warning("unhandled_error")
            response = error_response(500, "INTERNAL_ERROR", GENERIC_INTERNAL, request_id)
        latency_ms = (time.perf_counter() - started) * 1000.0
        bundle = get_cached_bundle()
        version = None if bundle is None else str(bundle.metadata.get("model_version") or "")
        batch_size = getattr(request.state, "batch_size", None)
        log_request(
            logger,
            endpoint=request.url.path,
            status_code=int(response.status_code),
            latency_ms=latency_ms,
            batch_size=batch_size if isinstance(batch_size, int) else None,
            model_version=version,
        )
        response.headers["X-Request-ID"] = request_id
        request_id_var.reset(token)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        message = GENERIC_INVALID
        raw = str(exc)
        if "batch size exceeds" in raw:
            message = GENERIC_BATCH_SIZE
        return error_response(422, "VALIDATION_ERROR", message, _request_id(request))

    @application.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        status = int(exc.status_code)
        code = "HTTP_ERROR"
        message = GENERIC_INTERNAL
        if status == 503:
            code = "MODEL_UNAVAILABLE"
            detail = str(exc.detail or "")
            message = GENERIC_NOT_READY if detail == GENERIC_NOT_READY else GENERIC_UNAVAILABLE
        elif status == 400:
            code = "VALIDATION_ERROR"
            message = GENERIC_INVALID
        elif status == 401:
            code = "UNAUTHORIZED"
            message = "Reload is not authorised."
        elif status == 404:
            code = "NOT_FOUND"
            message = "Resource was not found."
        elif status == 413:
            code = "REQUEST_TOO_LARGE"
            message = GENERIC_TOO_LARGE
        return error_response(status, code, message, _request_id(request))

    @application.exception_handler(Exception)
    async def unhandled_handler(request: Request, _exc: Exception) -> JSONResponse:
        logger.warning("handler_error type=%s", type(_exc).__name__)
        return error_response(500, "INTERNAL_ERROR", GENERIC_INTERNAL, _request_id(request))

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", disclaimer=DISCLAIMER)

    @application.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        bundle = get_cached_bundle()
        if bundle is None:
            raise HTTPException(status_code=503, detail=GENERIC_NOT_READY)
        try:
            verify_bundle(bundle)
        except ChampionLoadError:
            clear_cached_bundle()
            raise HTTPException(status_code=503, detail=GENERIC_NOT_READY) from None
        return ReadyResponse(**readiness_payload(bundle))

    @application.get("/model", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        bundle = get_serving()
        meta = bundle.metadata
        semantics = score_metadata_from_champion(meta)
        return ModelInfoResponse(
            model_name=str(meta.get("model_name", "")),
            model_version=str(meta.get("model_version", "")),
            mlflow_run_id=str(meta.get("mlflow_run_id", "")),
            decision_threshold=float(bundle.threshold),
            calibration_status=str(meta.get("calibration_status", "")),
            calibration_method=semantics["calibration_method"],
            score_is_calibrated=bool(semantics["score_is_calibrated"]),
            score_semantics=str(semantics["score_semantics"]),
            score_warning=SCORE_WARNING,
            human_review_notice=HUMAN_REVIEW_NOTICE,
            policy_version=str(meta.get("policy_version", "")),
            disclaimer=DISCLAIMER,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(shipment: ShipmentFeatures, request: Request) -> PredictionResponse:
        request.state.batch_size = 1
        bundle = get_serving()
        x = prepare_inference_frame([shipment.model_dump()])
        score = float(predict_proba(bundle.pipeline, x)[0])
        return _prediction_body(score, bundle)

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
        request.state.batch_size = len(payload.shipments)
        bundle = get_serving()
        rows = [item.model_dump() for item in payload.shipments]
        x = prepare_inference_frame(rows)
        scores = predict_proba(bundle.pipeline, x)
        predictions = [_prediction_body(float(score), bundle) for score in scores]
        return BatchPredictionResponse(
            predictions=predictions,
            model_version=str(bundle.metadata.get("model_version") or ""),
            mlflow_run_id=str(bundle.metadata.get("mlflow_run_id") or ""),
            disclaimer=DISCLAIMER,
        )

    @application.post("/explain", response_model=LocalExplanationResponse)
    def explain(shipment: ShipmentFeatures, request: Request) -> LocalExplanationResponse:
        request.state.batch_size = 1
        bundle = get_serving()
        try:
            explainer = LinearExplanationModel.from_pipeline(bundle.pipeline)
        except TypeError:
            raise _unavailable() from None
        x = prepare_inference_frame([shipment.model_dump()])
        payload = explainer.explain_row(
            x,
            threshold=float(bundle.threshold),
            metadata=bundle.metadata,
        )
        reconstruction_error = float(payload["reconstruction_error"])
        reconstruction_atol = float(payload["reconstruction_atol"])
        requires_review = int(payload["requires_review"])
        return LocalExplanationResponse(
            review_score=float(payload["review_score"]),
            decision_threshold=float(payload["decision_threshold"]),
            requires_review=requires_review,
            classification=REVIEW_LABEL_POSITIVE if requires_review else REVIEW_LABEL_NEGATIVE,
            intercept_log_odds=float(payload["intercept_log_odds"]),
            logit=float(payload["logit"]),
            reconstruction_error=reconstruction_error,
            reconstruction_ok=reconstruction_error <= reconstruction_atol,
            reconstruction_atol=reconstruction_atol,
            strongest_positive_contributions=list(payload["strongest_increasing"][:8]),
            strongest_negative_contributions=list(payload["strongest_decreasing"][:8]),
            original_input_values=dict(payload["original_input_values"]),
            model_version=str(payload.get("model_version") or bundle.metadata.get("model_version")),
            mlflow_run_id=str(payload.get("mlflow_run_id") or bundle.metadata.get("mlflow_run_id")),
            score_is_calibrated=bool(payload["score_is_calibrated"]),
            score_semantics=str(payload["score_semantics"]),
            score_warning=SCORE_WARNING,
            human_review_notice=HUMAN_REVIEW_NOTICE,
            causation_disclaimer=str(payload["causation_disclaimer"]),
            relationship=str(payload["relationship"]),
            disclaimer=DISCLAIMER,
        )

    @application.post("/reload", response_model=ReloadResponse, include_in_schema=False)
    def reload_champion(
        x_reload_token: str | None = Header(default=None, alias="X-Reload-Token"),
    ) -> ReloadResponse:
        expected = str(get_settings().api_reload_token or "")
        if not expected:
            raise HTTPException(status_code=404, detail="Resource was not found.")
        if not x_reload_token or x_reload_token != expected:
            raise HTTPException(status_code=401, detail="Reload is not authorised.")
        try:
            bundle = load_and_verify()
        except ChampionLoadError:
            raise _unavailable() from None
        set_cached_bundle(bundle)
        return ReloadResponse(
            status="reloaded",
            model_version=str(bundle.metadata.get("model_version") or ""),
            mlflow_run_id=str(bundle.metadata.get("mlflow_run_id") or ""),
        )

    @application.get("/monitoring/status", response_model=MonitoringStatusResponse)
    def monitoring_status() -> MonitoringStatusResponse:
        payload = load_latest_status()
        return MonitoringStatusResponse(
            available=bool(payload.get("available")),
            message=str(payload.get("message") or "") or None,
            monitoring_run_id=payload.get("monitoring_run_id"),
            timestamp=payload.get("timestamp"),
            mode=payload.get("mode"),
            scenario=payload.get("scenario"),
            status=payload.get("status") or "insufficient_data",
            overall_severity=payload.get("overall_severity"),
            policy_version=payload.get("policy_version"),
            recommended_action=payload.get("recommended_action"),
            alert_reasons=list(payload.get("alert_reasons") or []),
            batch_size=payload.get("batch_size"),
            reference_batch_size=payload.get("reference_batch_size"),
            n_monitored_features=payload.get("n_monitored_features"),
            n_warnings=payload.get("n_warnings"),
            n_critical_findings=payload.get("n_critical_findings"),
            report_complete=payload.get("report_complete"),
            reference_fingerprint=payload.get("reference_fingerprint"),
            current_fingerprint=payload.get("current_fingerprint"),
            champion_version=payload.get("champion_version"),
            ground_truth_available=payload.get("ground_truth_available"),
        )

    @application.get("/monitoring/latest", response_model=MonitoringLatestResponse)
    def monitoring_latest() -> MonitoringLatestResponse:
        try:
            report = load_latest_report()
        except FileNotFoundError:
            return MonitoringLatestResponse(
                available=False,
                status="insufficient_data",
                message="No monitoring report is available.",
                report_complete=False,
                ground_truth_available=False,
            )
        return MonitoringLatestResponse(
            available=bool(report.get("available", True))
            and bool(report.get("report_complete", True)),
            monitoring_run_id=str(report.get("monitoring_run_id") or ""),
            timestamp=str(report.get("timestamp") or ""),
            mode=str(report.get("mode") or ""),
            scenario=str(report.get("scenario") or ""),
            status=str(report.get("status") or "insufficient_data"),
            overall_severity=(
                str(report["overall_severity"]) if report.get("overall_severity") else None
            ),
            policy_version=str(
                report.get("policy_version") or report.get("monitoring_policy_version") or ""
            )
            or None,
            recommended_action=str(report.get("recommended_action") or ""),
            alert_reasons=list(report.get("alert_reasons") or []),
            batch_size=int(report.get("batch_size") or 0) or None,
            reference_batch_size=int(report.get("reference_batch_size") or 0) or None,
            n_monitored_features=int(report.get("n_monitored_features") or 0) or None,
            n_warnings=report.get("n_warnings", report.get("n_warning_features")),
            n_critical_findings=report.get(
                "n_critical_findings", report.get("n_critical_features")
            ),
            report_complete=bool(report.get("report_complete", True)),
            reference_fingerprint=str(report.get("reference_fingerprint") or ""),
            current_fingerprint=str(report.get("current_fingerprint") or ""),
            champion=dict(report.get("champion") or {}),
            score_drift=dict(report.get("score_drift") or {}),
            input_data_drift=dict(report.get("input_data_drift") or {}),
            feature_metrics=list(report.get("feature_metrics") or []),
            simulated_performance=(
                dict(report["simulated_performance"])
                if report.get("simulated_performance")
                else None
            ),
            ground_truth_available=bool(report.get("ground_truth_available")),
            limitations=list(report.get("limitations") or []),
            disclaimer=str(report.get("disclaimer") or DISCLAIMER),
        )

    return application


app = create_app()
