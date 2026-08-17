"""Pydantic schemas for the Cargo Risk ML Lab API."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import get_settings
from src.data.schema import (
    COMMODITY_CATEGORIES,
    DESTINATION_REGIONS,
    NUMERIC_RANGES,
    ORIGIN_REGIONS,
    TRANSPORT_MODES,
)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("must be a finite number")
    return float(value)


def _finite_int(value: int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("must be a finite integer")
        return int(value)
    if isinstance(value, int):
        return value
    raise ValueError("must be an integer")


class ErrorBody(BaseModel):
    """Consistent client error payload."""

    error_code: str
    message: str
    request_id: str


class ShipmentFeatures(BaseModel):
    """Input features for a single synthetic shipment review score."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    origin_region: str = Field(..., examples=["Asia"])
    destination_region: str = Field(..., examples=["Northern Europe"])
    commodity_category: str = Field(..., examples=["electronics"])
    transport_mode: str = Field(..., examples=["air"])
    declared_value_eur: float = Field(..., examples=[12500.0], allow_inf_nan=False)
    shipment_weight_kg: float = Field(..., examples=[85.5], allow_inf_nan=False)
    declaration_completeness_score: float | None = Field(
        default=None, examples=[0.82], allow_inf_nan=False
    )
    documentation_count: int | None = Field(default=None, examples=[6])
    previous_discrepancies: int = Field(..., examples=[0])
    sender_history_length: int | None = Field(default=None, examples=[12])
    route_rarity: float | None = Field(default=None, examples=[0.25], allow_inf_nan=False)
    declared_vs_estimated_value_deviation: float | None = Field(
        default=None, examples=[0.05], allow_inf_nan=False
    )
    submission_hour: int = Field(..., examples=[10])
    expedited_shipment: int = Field(..., examples=[0])

    @field_validator("origin_region")
    @classmethod
    def _check_origin(cls, value: str) -> str:
        if value not in ORIGIN_REGIONS:
            raise ValueError("origin_region is not an allowed value")
        return value

    @field_validator("destination_region")
    @classmethod
    def _check_destination(cls, value: str) -> str:
        if value not in DESTINATION_REGIONS:
            raise ValueError("destination_region is not an allowed value")
        return value

    @field_validator("commodity_category")
    @classmethod
    def _check_commodity(cls, value: str) -> str:
        if value not in COMMODITY_CATEGORIES:
            raise ValueError("commodity_category is not an allowed value")
        return value

    @field_validator("transport_mode")
    @classmethod
    def _check_transport(cls, value: str) -> str:
        if value not in TRANSPORT_MODES:
            raise ValueError("transport_mode is not an allowed value")
        return value

    @field_validator(
        "declared_value_eur",
        "shipment_weight_kg",
        "declaration_completeness_score",
        "route_rarity",
        "declared_vs_estimated_value_deviation",
    )
    @classmethod
    def _check_finite_float(cls, value: float | None) -> float | None:
        return _finite(value)

    @field_validator(
        "documentation_count",
        "previous_discrepancies",
        "sender_history_length",
        "submission_hour",
        "expedited_shipment",
    )
    @classmethod
    def _check_finite_int(cls, value: int | float | None) -> int | None:
        return _finite_int(value)

    @model_validator(mode="after")
    def _ranges(self) -> ShipmentFeatures:
        mapping: dict[str, float | int | None] = {
            "declared_value_eur": self.declared_value_eur,
            "shipment_weight_kg": self.shipment_weight_kg,
            "declaration_completeness_score": self.declaration_completeness_score,
            "documentation_count": self.documentation_count,
            "previous_discrepancies": self.previous_discrepancies,
            "sender_history_length": self.sender_history_length,
            "route_rarity": self.route_rarity,
            "declared_vs_estimated_value_deviation": self.declared_vs_estimated_value_deviation,
            "submission_hour": self.submission_hour,
            "expedited_shipment": self.expedited_shipment,
        }
        for name, value in mapping.items():
            if value is None or name not in NUMERIC_RANGES:
                continue
            low, high = NUMERIC_RANGES[name]
            number = float(value)
            if number < low or number > high:
                raise ValueError(f"{name} is outside the allowed synthetic range")
        return self


class PredictionResponse(BaseModel):
    """Synthetic review score. ``review_score`` is not a calibrated probability."""

    model_config = ConfigDict(extra="forbid")

    review_score: float = Field(
        ...,
        description="Ranking and threshold score in [0, 1]. Not a calibrated probability.",
    )
    decision_threshold: float = Field(
        ..., description="Operating threshold applied to review_score."
    )
    requires_review: int = Field(..., description="1 if additional human review is suggested.")
    model_version: str
    mlflow_run_id: str
    score_is_calibrated: bool
    score_semantics: str
    human_review_notice: str
    score_warning: str
    disclaimer: str


class HealthResponse(BaseModel):
    """Liveness payload. Does not inspect the champion."""

    status: str
    disclaimer: str


class ReadyResponse(BaseModel):
    """Readiness payload after champion and explanation checks."""

    status: str
    model_version: str
    mlflow_run_id: str
    decision_threshold: float
    score_is_calibrated: bool
    explanations_available: bool


class ModelInfoResponse(BaseModel):
    """Public champion information. Does not include filesystem paths."""

    model_name: str
    model_version: str
    mlflow_run_id: str
    decision_threshold: float
    calibration_status: str
    calibration_method: str | None
    score_is_calibrated: bool
    score_semantics: str
    score_warning: str
    human_review_notice: str
    policy_version: str
    disclaimer: str


class LocalExplanationResponse(BaseModel):
    """Exact logit-space explanation for one scored shipment."""

    review_score: float
    decision_threshold: float
    requires_review: int
    classification: str
    intercept_log_odds: float
    logit: float
    reconstruction_error: float
    reconstruction_ok: bool
    reconstruction_atol: float
    strongest_positive_contributions: list[dict[str, Any]]
    strongest_negative_contributions: list[dict[str, Any]]
    original_input_values: dict[str, Any]
    model_version: str | None
    mlflow_run_id: str | None
    score_is_calibrated: bool
    score_semantics: str
    score_warning: str
    human_review_notice: str
    causation_disclaimer: str
    relationship: str
    disclaimer: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    shipments: list[ShipmentFeatures] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _batch_size(self) -> BatchPredictionRequest:
        maximum = int(get_settings().api_max_batch_size)
        if len(self.shipments) > maximum:
            raise ValueError(f"batch size exceeds maximum of {maximum}")
        return self


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
    model_version: str
    mlflow_run_id: str
    disclaimer: str


class ReloadResponse(BaseModel):
    """Result of a token-gated champion reload."""

    status: str
    model_version: str
    mlflow_run_id: str


class MonitoringStatusResponse(BaseModel):
    """Aggregated monitoring status. No raw records or filesystem paths."""

    available: bool
    message: str | None = None
    monitoring_run_id: str | None = None
    timestamp: str | None = None
    mode: str | None = None
    scenario: str | None = None
    status: str | None = None
    overall_severity: str | None = None
    policy_version: str | None = None
    recommended_action: str | None = None
    alert_reasons: list[dict[str, Any]] | None = None
    batch_size: int | None = None
    reference_batch_size: int | None = None
    n_monitored_features: int | None = None
    n_warnings: int | None = None
    n_critical_findings: int | None = None
    report_complete: bool | None = None
    reference_fingerprint: str | None = None
    current_fingerprint: str | None = None
    champion_version: str | None = None
    ground_truth_available: bool | None = None


class MonitoringLatestResponse(BaseModel):
    """Latest monitoring summary without raw records."""

    available: bool
    message: str | None = None
    monitoring_run_id: str | None = None
    timestamp: str | None = None
    mode: str | None = None
    scenario: str | None = None
    status: str | None = None
    overall_severity: str | None = None
    policy_version: str | None = None
    recommended_action: str | None = None
    alert_reasons: list[dict[str, Any]] | None = None
    batch_size: int | None = None
    reference_batch_size: int | None = None
    n_monitored_features: int | None = None
    n_warnings: int | None = None
    n_critical_findings: int | None = None
    report_complete: bool | None = None
    reference_fingerprint: str | None = None
    current_fingerprint: str | None = None
    champion: dict[str, Any] | None = None
    score_drift: dict[str, Any] | None = None
    input_data_drift: dict[str, Any] | None = None
    feature_metrics: list[dict[str, Any]] | None = None
    simulated_performance: dict[str, Any] | None = None
    ground_truth_available: bool | None = None
    limitations: list[str] | None = None
    disclaimer: str | None = None
