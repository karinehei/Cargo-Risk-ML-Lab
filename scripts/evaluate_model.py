"""CLI script: evaluate the selected model on the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.config import get_config, resolve_path, set_seed, setup_logging
from src.data import load_dataset
from src.data.schema import ID_COLUMN
from src.evaluation import evaluate_predictions, measure_inference_latency
from src.features import prepare_xy
from src.models import load_model_bundle, predict_proba
from src.monitoring import run_drift_checks


def _assert_test_held_out(processed_dir: Path, test_df: pd.DataFrame) -> None:
    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_ids = set(test_df[ID_COLUMN].astype(str))
    train_ids = set(manifest.get("train_ids", []))
    val_ids = set(manifest.get("val_ids", []))
    overlap = test_ids & (train_ids | val_ids)
    if overlap:
        raise RuntimeError(f"Test IDs overlap train/val ({len(overlap)} ids)")


def main() -> None:
    """Score the untouched test fold with the frozen validation threshold."""
    logger = setup_logging(name="scripts.evaluate_model")
    cfg = get_config()
    set_seed(cfg.random_seed)
    logger.info("Disclaimer: %s", cfg.disclaimer)

    processed_dir = resolve_path(str(cfg.data.get("processed_dir", "data/processed")))
    train_df = load_dataset(processed_dir / "train.csv")
    test_df = load_dataset(processed_dir / "test.csv")
    _assert_test_held_out(processed_dir, test_df)

    bundle = load_model_bundle()
    threshold = float(bundle.metadata.get("threshold", cfg.model.get("threshold", 0.5)))
    logger.info(
        "Evaluating test set with frozen threshold=%.4f from metadata (not re-tuned).",
        threshold,
    )

    x_test, y_test = prepare_xy(test_df, cfg, fit_derived_reference=train_df)
    y_prob = predict_proba(bundle.pipeline, x_test)
    result = evaluate_predictions(
        y_test,
        y_prob,
        features=x_test,
        config=cfg,
        split_name="test",
        threshold=threshold,
    )

    latency = measure_inference_latency(
        bundle.pipeline,
        x_test,
        repeats=int(cfg.training.get("latency_repeats", 200)),
        seed=cfg.random_seed,
    )
    result["latency_ms"] = latency
    metrics_path = resolve_path(str(result["metrics_path"]))
    metrics_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Test metrics: %s", result["metrics"])
    logger.info("Latency (ms): %s", latency)

    x_train, _ = prepare_xy(train_df, cfg, fit_derived_reference=train_df)
    train_prob = predict_proba(bundle.pipeline, x_train)
    drift = run_drift_checks(
        reference=train_df,
        current=test_df,
        reference_scores=train_prob,
        current_scores=y_prob,
        config=cfg,
    )
    logger.info(
        "Drift summary: method=%s drifted=%s",
        (drift.get("data_drift") or {}).get("method"),
        (drift.get("lightweight_summary") or {}).get("n_drifted_features"),
    )


if __name__ == "__main__":
    main()
