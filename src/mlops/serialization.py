"""Official MLflow sklearn logging with an explicit, documented serializer.

MLflow 3.x defaults to skops. A fitted sklearn Pipeline stores ``numpy.dtype``
objects (NumPy 2.x / scikit-learn 1.x). skops 0.14 treats ``numpy.dtype`` as
untrusted, so the default ``log_model`` path fails with:

    Untrusted types found: ['numpy.dtype']

This module uses ``serialization_format='cloudpickle'``, which is an official
MLflow sklearn flavor option. It does **not** disable skops trust checks globally
and does **not** set ``SKOPS_ALLOW_UNTRUSTED`` or similar environment flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, setup_logging
from src.models.estimators import XGBTrainWeightedClassifier

logger = setup_logging(name="src.mlops.serialization")

SKLEARN_SERIALIZATION_FORMAT = "cloudpickle"
ROUNDTRIP_RTOL = 1e-7
ROUNDTRIP_ATOL = 1e-10
MODEL_ARTIFACT_NAME = "model"


@dataclass
class LoggedModel:
    """Result of attempting to persist a pipeline to MLflow."""

    method: str
    artifact_uri: str
    run_id: str
    roundtrip_ok: bool
    fallback_reason: str | None = None


def _contains_custom_xgb(estimator: BaseEstimator) -> bool:
    model = estimator
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps.get("model")
    return isinstance(model, XGBTrainWeightedClassifier)


def log_sklearn_pipeline(
    estimator: BaseEstimator,
    *,
    artifact_name: str = MODEL_ARTIFACT_NAME,
    allow_joblib_fallback: bool = True,
) -> LoggedModel:
    """Log a fitted sklearn Pipeline with the official sklearn flavor.

    Custom XGBoost wrappers may fail skops and, rarely, cloudpickle code-path
    loading. Those runs may record a documented joblib artifact instead. The
    logistic-regression champion path must succeed via ``mlflow.sklearn``.
    """
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("log_sklearn_pipeline must be called inside an MLflow run")
    run_id = active.info.run_id
    kwargs: dict[str, Any] = {
        "sk_model": estimator,
        "serialization_format": SKLEARN_SERIALIZATION_FORMAT,
        "pip_requirements": [
            "scikit-learn",
            "numpy",
            "pandas",
            "xgboost",
            "cloudpickle",
        ],
    }
    try:
        try:
            model_info = mlflow.sklearn.log_model(name=artifact_name, **kwargs)
        except TypeError:
            model_info = mlflow.sklearn.log_model(artifact_path=artifact_name, **kwargs)
        # runs:/ embeds the run id and loads successfully on MLflow 3.15.
        # models:/m-... is stored as a tag for native lookup; serving never uses filesystem paths.
        uri = f"runs:/{run_id}/{artifact_name}"
        native_uri = getattr(model_info, "model_uri", None)
        if native_uri:
            mlflow.set_tag("mlflow_native_model_uri", str(native_uri))
        mlflow.set_tag("serialization_format", SKLEARN_SERIALIZATION_FORMAT)
        mlflow.set_tag("mlflow_sklearn_flavor", "true")
        return LoggedModel(
            method=f"mlflow.sklearn.{SKLEARN_SERIALIZATION_FORMAT}",
            artifact_uri=uri,
            run_id=run_id,
            roundtrip_ok=False,
        )
    except Exception as exc:  # noqa: BLE001
        if allow_joblib_fallback and _contains_custom_xgb(estimator):
            logger.warning(
                "mlflow.sklearn.log_model failed for XGBTrainWeightedClassifier (%s). "
                "Recording a joblib artifact as a documented fallback.",
                exc,
            )
            return _log_joblib_fallback(estimator, run_id, artifact_name, str(exc))
        raise


def _log_joblib_fallback(
    estimator: BaseEstimator,
    run_id: str,
    artifact_name: str,
    reason: str,
) -> LoggedModel:
    import joblib

    tmp_dir = PROJECT_ROOT / "mlruns" / "_tmp_joblib"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{run_id}_{artifact_name}.joblib"
    joblib.dump(estimator, path)  # nosec B301
    mlflow.log_artifact(str(path), artifact_path=f"{artifact_name}_joblib")
    mlflow.set_tag("serialization_format", "joblib_fallback")
    mlflow.set_tag("mlflow_sklearn_flavor", "false")
    mlflow.set_tag("joblib_fallback_reason", reason[:500])
    path.unlink(missing_ok=True)
    return LoggedModel(
        method="joblib_fallback",
        artifact_uri=f"runs:/{run_id}/{artifact_name}_joblib/{path.name}",
        run_id=run_id,
        roundtrip_ok=False,
        fallback_reason=reason,
    )


def load_sklearn_pipeline(artifact_uri: str) -> BaseEstimator:
    """Load an estimator logged with ``mlflow.sklearn`` from a trusted URI."""
    from src.mlops.integrity import validate_artifact_uri

    validate_artifact_uri(artifact_uri, run_id=_run_id_from_uri(artifact_uri))
    loaded = mlflow.sklearn.load_model(artifact_uri)
    if not hasattr(loaded, "predict_proba"):
        raise TypeError("Loaded MLflow artifact does not support predict_proba")
    return loaded


def _run_id_from_uri(artifact_uri: str) -> str:
    if artifact_uri.startswith("runs:/"):
        return artifact_uri.split("/", 2)[1]
    return ""


def assert_prediction_roundtrip(
    original: BaseEstimator,
    loaded: BaseEstimator,
    features: pd.DataFrame,
    *,
    rtol: float = ROUNDTRIP_RTOL,
    atol: float = ROUNDTRIP_ATOL,
) -> dict[str, float]:
    """Require equal class predictions and numerically equivalent probabilities."""
    original_proba = np.asarray(original.predict_proba(features), dtype=float)
    loaded_proba = np.asarray(loaded.predict_proba(features), dtype=float)
    original_pred = np.asarray(original.predict(features))
    loaded_pred = np.asarray(loaded.predict(features))
    if original_pred.shape != loaded_pred.shape or not np.array_equal(original_pred, loaded_pred):
        raise AssertionError("Round-trip class predictions differ")
    if not np.allclose(original_proba, loaded_proba, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(original_proba - loaded_proba)))
        raise AssertionError(f"Round-trip probabilities differ (max abs {max_abs})")
    return {
        "n_rows": float(len(features)),
        "max_abs_proba_delta": float(np.max(np.abs(original_proba - loaded_proba))),
        "rtol": float(rtol),
        "atol": float(atol),
    }


def verify_logged_pipeline(
    original: BaseEstimator,
    logged: LoggedModel,
    features: pd.DataFrame,
) -> LoggedModel:
    """Load the logged artifact and compare predictions to the in-memory pipeline."""
    if logged.method.startswith("joblib"):
        logged.roundtrip_ok = False
        return logged
    loaded = load_sklearn_pipeline(logged.artifact_uri)
    assert_prediction_roundtrip(original, loaded, features)
    logged.roundtrip_ok = True
    return logged
