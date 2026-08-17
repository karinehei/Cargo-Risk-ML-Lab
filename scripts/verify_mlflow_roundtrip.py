"""Verify MLflow sklearn round-trip on the frozen logistic-regression pipeline.

Uses validation rows as a fixed fixture. The test set is not loaded.
Does not write to artifacts/frozen_v1/.
"""

from __future__ import annotations

import json

import mlflow
from src.config import get_config, resolve_path, setup_logging
from src.data import load_dataset
from src.features import prepare_xy
from src.mlops.serialization import (
    ROUNDTRIP_ATOL,
    ROUNDTRIP_RTOL,
    assert_prediction_roundtrip,
    load_sklearn_pipeline,
    log_sklearn_pipeline,
)
from src.mlops.tracking import configure_tracking, init_tracking_store
from src.models import load_model_bundle


def main() -> None:
    logger = setup_logging(name="scripts.verify_mlflow_roundtrip")
    cfg = get_config()
    init_tracking_store()
    configure_tracking()

    frozen = load_model_bundle(resolve_path("artifacts/frozen_v1"))
    processed_dir = resolve_path(str(cfg.data.get("processed_dir", "data/processed")))
    train_df = load_dataset(processed_dir / "train.csv")
    val_df = load_dataset(processed_dir / "val.csv")
    x_val, _ = prepare_xy(val_df, cfg, fit_derived_reference=train_df)
    fixture = x_val.head(min(32, len(x_val)))

    with mlflow.start_run(run_name="roundtrip_frozen_logreg") as run:
        logged = log_sklearn_pipeline(frozen.pipeline, allow_joblib_fallback=False)
        loaded = load_sklearn_pipeline(logged.artifact_uri)
        stats = assert_prediction_roundtrip(
            frozen.pipeline,
            loaded,
            fixture,
            rtol=ROUNDTRIP_RTOL,
            atol=ROUNDTRIP_ATOL,
        )
        payload = {
            "run_id": run.info.run_id,
            "artifact_uri": logged.artifact_uri,
            "method": logged.method,
            "n_fixture_rows": int(stats["n_rows"]),
            "max_abs_proba_delta": stats["max_abs_proba_delta"],
            "rtol": ROUNDTRIP_RTOL,
            "atol": ROUNDTRIP_ATOL,
            "class_predictions_equal": True,
        }
        out = resolve_path("artifacts/mlops/roundtrip_verification.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Round-trip OK run_id=%s max_abs_delta=%.3e",
            run.info.run_id,
            stats["max_abs_proba_delta"],
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
