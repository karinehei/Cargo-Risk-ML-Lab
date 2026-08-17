"""Streamlit dashboard for Cargo Risk ML Lab (educational / synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from src import DISCLAIMER, __version__
from src.config import get_config, get_settings, resolve_path, setup_logging
from src.explainability.semantics import (
    HUMAN_REVIEW_NOTICE,
    REVIEW_LABEL_NEGATIVE,
    REVIEW_LABEL_POSITIVE,
    SCORE_WARNING,
    score_metadata_from_champion,
)

logger = setup_logging(name="app.streamlit")

st.set_page_config(
    page_title="Cargo Risk ML Lab",
    page_icon=None,
    layout="wide",
)

SECTIONS = [
    "1. Project overview",
    "2. Champion and selection evidence",
    "3. Single-shipment prediction",
    "4. Exact local explanation",
    "5. Global coefficient analysis",
    "6. Permutation importance",
    "7. Model comparison",
    "8. Calibration and score semantics",
    "9. Subgroup analysis",
    "10. Monitoring",
    "11. Model card and limitations",
]


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return cast(dict[str, Any] | list[Any], json.load(handle))


def _artifact_json(relative: str) -> dict[str, Any] | None:
    payload = _load_json(resolve_path(relative))
    return payload if isinstance(payload, dict) else None


def _artifact_list(relative: str) -> list[Any] | None:
    payload = _load_json(resolve_path(relative))
    return payload if isinstance(payload, list) else None


def _missing(message: str) -> None:
    st.info(message)


def _show_image(relative: str, caption: str) -> None:
    path = resolve_path(relative)
    if path.exists() and path.is_file():
        st.image(str(path), caption=caption)


def _markdown_file(relative: str) -> None:
    path = resolve_path(relative)
    if not path.exists():
        _missing("This document is not available in the local workspace.")
        return
    st.markdown(path.read_text(encoding="utf-8"))


def _calibration_warning() -> None:
    st.warning(SCORE_WARNING)


def _disclaimer_banner() -> None:
    st.warning(DISCLAIMER)
    st.caption(
        "Fully synthetic data and labels. Not affiliated with Finnish Customs or any authority. "
        "Do not use for operational, enforcement or automated adverse decisions."
    )


def _api_base() -> str:
    return str(get_settings().streamlit_api_url or "http://127.0.0.1:8000").rstrip("/")


def _api_status(base: str) -> str:
    try:
        import httpx

        response = httpx.get(f"{base}/ready", timeout=2.0)
        if response.status_code == 200:
            return "ready"
        return "not-ready"
    except Exception:  # noqa: BLE001
        return "unavailable"


def _score_via_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import httpx

        response = httpx.post(f"{_api_base()}/predict", json=payload, timeout=8.0)
        if response.status_code != 200:
            return None
        body = response.json()
        body["source"] = "api"
        return cast(dict[str, Any], body)
    except Exception:  # noqa: BLE001
        logger.info("API predict unavailable; using local champion scoring")
        return None


def _explain_via_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import httpx

        response = httpx.post(f"{_api_base()}/explain", json=payload, timeout=10.0)
        if response.status_code != 200:
            return None
        return cast(dict[str, Any], response.json())
    except Exception:  # noqa: BLE001
        logger.info("API explain unavailable; using local champion explanation")
        return None


def _score_locally(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from src.mlops.serving import ChampionLoadError, load_champion
        from src.models import predict_proba, prepare_inference_frame
    except Exception:  # noqa: BLE001
        return None
    try:
        bundle = load_champion()
    except ChampionLoadError:
        return None
    x = prepare_inference_frame([payload])
    score = float(predict_proba(bundle.pipeline, x)[0])
    semantics = score_metadata_from_champion(bundle.metadata)
    threshold = float(bundle.threshold)
    return {
        "review_score": score,
        "requires_review": int(score >= threshold),
        "decision_threshold": threshold,
        "score_is_calibrated": semantics["score_is_calibrated"],
        "score_semantics": semantics["score_semantics"],
        "score_warning": SCORE_WARNING,
        "human_review_notice": HUMAN_REVIEW_NOTICE,
        "disclaimer": DISCLAIMER,
        "model_version": bundle.metadata.get("model_version"),
        "mlflow_run_id": bundle.metadata.get("mlflow_run_id"),
        "source": "champion",
    }


def _explain_locally(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from src.explainability.linear import LinearExplanationModel
        from src.mlops.serving import load_champion
        from src.models import prepare_inference_frame
    except Exception:  # noqa: BLE001
        return None
    try:
        bundle = load_champion()
        explainer = LinearExplanationModel.from_pipeline(bundle.pipeline)
    except Exception:  # noqa: BLE001
        return None
    x = prepare_inference_frame([payload])
    return explainer.explain_row(x, threshold=float(bundle.threshold), metadata=bundle.metadata)


def _shipment_form(key_prefix: str) -> dict[str, Any]:
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox(
            "Origin region",
            ["Northern Europe", "Central Europe", "Southern Europe", "Asia", "Americas", "Africa"],
            key=f"{key_prefix}_origin",
        )
        destination = st.selectbox(
            "Destination region",
            ["Northern Europe", "Central Europe", "Southern Europe", "UK & Ireland"],
            key=f"{key_prefix}_destination",
        )
        commodity = st.selectbox(
            "Commodity category",
            [
                "electronics",
                "textiles",
                "machinery",
                "foodstuffs",
                "chemicals",
                "pharmaceuticals",
                "automotive",
                "other",
            ],
            key=f"{key_prefix}_commodity",
        )
        transport = st.selectbox(
            "Transport mode", ["road", "sea", "air", "rail"], key=f"{key_prefix}_transport"
        )
        expedited = st.selectbox("Expedited shipment", [0, 1], key=f"{key_prefix}_expedited")
        hour = st.slider(
            "Submission hour", min_value=0, max_value=23, value=10, key=f"{key_prefix}_hour"
        )
    with col2:
        value = st.number_input(
            "Declared value (EUR)", min_value=1.0, value=12500.0, key=f"{key_prefix}_value"
        )
        weight = st.number_input(
            "Shipment weight (kg)", min_value=0.1, value=85.5, key=f"{key_prefix}_weight"
        )
        completeness = st.slider(
            "Declaration completeness", 0.0, 1.0, 0.82, key=f"{key_prefix}_completeness"
        )
        docs = st.number_input(
            "Documentation count", min_value=0, value=6, key=f"{key_prefix}_docs"
        )
        discrepancies = st.number_input(
            "Previous discrepancies", min_value=0, value=0, key=f"{key_prefix}_discrepancies"
        )
        history = st.number_input(
            "Sender history length", min_value=0, value=12, key=f"{key_prefix}_history"
        )
        rarity = st.slider("Route rarity", 0.0, 1.0, 0.25, key=f"{key_prefix}_rarity")
        deviation = st.number_input(
            "Declared vs estimated value deviation", value=0.05, key=f"{key_prefix}_deviation"
        )
    return {
        "origin_region": origin,
        "destination_region": destination,
        "commodity_category": commodity,
        "transport_mode": transport,
        "declared_value_eur": float(value),
        "shipment_weight_kg": float(weight),
        "declaration_completeness_score": float(completeness),
        "documentation_count": int(docs),
        "previous_discrepancies": int(discrepancies),
        "sender_history_length": int(history),
        "route_rarity": float(rarity),
        "declared_vs_estimated_value_deviation": float(deviation),
        "submission_hour": int(hour),
        "expedited_shipment": int(expedited),
    }


def _operational_sketch(metrics: dict[str, Any]) -> dict[str, float] | None:
    n = float(metrics.get("n_samples") or 0.0)
    if n <= 0:
        return None
    tp = float(metrics.get("true_positives") or 0.0)
    fp = float(metrics.get("false_positives") or 0.0)
    fn = float(metrics.get("false_negatives") or 0.0)
    return {
        "reviews_per_1000": (tp + fp) / n * 1000.0,
        "true_positives_per_1000": tp / n * 1000.0,
        "additional_reviews_per_1000": fp / n * 1000.0,
        "missed_positives_per_1000": fn / n * 1000.0,
    }


def _render_threshold(score: float, threshold: float, flagged: int) -> None:
    label = REVIEW_LABEL_POSITIVE if flagged else REVIEW_LABEL_NEGATIVE
    margin = score - threshold
    st.subheader("Score versus decision threshold")
    left, right, status = st.columns(3)
    left.metric("Review score", f"{score:.3f}")
    right.metric("Decision threshold", f"{threshold:.3f}")
    status.metric("Human-review flag", "requires_review=1" if flagged else "requires_review=0")
    st.caption("The review score is a ranking/threshold value, not a calibrated probability.")
    chart = pd.DataFrame({"Review score": [score], "Decision threshold": [threshold]})
    st.bar_chart(chart)
    if margin >= 0:
        st.write(
            f"Status: **{label}**. The review score is {margin:.3f} at or above the "
            f"decision threshold of {threshold:.3f}."
        )
    else:
        st.write(
            f"Status: **{label}**. The review score is {abs(margin):.3f} below the "
            f"decision threshold of {threshold:.3f}."
        )
    st.write(HUMAN_REVIEW_NOTICE)
    _calibration_warning()


def _render_score_result(result: dict[str, Any]) -> None:
    score = float(result["review_score"])
    threshold = float(result["decision_threshold"])
    flagged = int(result.get("requires_review", 0))
    _render_threshold(score, threshold, flagged)
    st.write(
        {
            "requires_review": flagged,
            "decision": REVIEW_LABEL_POSITIVE if flagged else REVIEW_LABEL_NEGATIVE,
            "model_version": result.get("model_version"),
            "mlflow_run_id": result.get("mlflow_run_id"),
            "score_is_calibrated": result.get("score_is_calibrated", False),
            "score_semantics": result.get("score_semantics"),
            "source": result.get("source"),
        }
    )
    st.caption(str(result.get("disclaimer", DISCLAIMER)))


def _positive_rows(explanation: dict[str, Any]) -> list[Any]:
    return list(
        explanation.get("strongest_positive_contributions")
        or explanation.get("strongest_increasing")
        or []
    )


def _negative_rows(explanation: dict[str, Any]) -> list[Any]:
    return list(
        explanation.get("strongest_negative_contributions")
        or explanation.get("strongest_decreasing")
        or []
    )


def _render_explanation(explanation: dict[str, Any]) -> None:
    flagged = int(explanation.get("requires_review", 0))
    label = str(
        explanation.get("classification")
        or (REVIEW_LABEL_POSITIVE if flagged else REVIEW_LABEL_NEGATIVE)
    )
    st.write(
        {
            "review_score": explanation.get("review_score"),
            "decision_threshold": explanation.get("decision_threshold"),
            "classification": label,
            "intercept_log_odds": explanation.get("intercept_log_odds"),
            "logit": explanation.get("logit"),
            "reconstruction_error": explanation.get("reconstruction_error"),
            "reconstruction_ok": explanation.get("reconstruction_ok"),
            "model_version": explanation.get("model_version"),
            "mlflow_run_id": explanation.get("mlflow_run_id"),
        }
    )
    st.caption(str(explanation.get("causation_disclaimer", "")))
    st.markdown("Strongest contributions that increase the review score")
    increasing = pd.DataFrame(_positive_rows(explanation))
    if increasing.empty:
        st.write("No increasing contributions in this artifact.")
    else:
        st.dataframe(increasing.head(8), use_container_width=True)
    st.markdown("Strongest contributions that decrease the review score")
    decreasing = pd.DataFrame(_negative_rows(explanation))
    if decreasing.empty:
        st.write("No decreasing contributions in this artifact.")
    else:
        st.dataframe(decreasing.head(8), use_container_width=True)
    original = explanation.get("original_input_values") or {}
    if original:
        st.markdown("Original input values used for this explanation")
        st.write(original)
    st.caption(str(explanation.get("relationship", "")))
    _calibration_warning()


def _section_overview() -> None:
    st.header("Project overview")
    st.write(
        "Cargo Risk ML Lab is an educational machine-learning portfolio. "
        "It scores **synthetic** shipments for a fictional additional-human-review queue."
    )
    st.markdown(
        """
        - Champion: class-weighted logistic regression (`logreg-none-1.0.0`).
        - Decision threshold: **0.525** (chosen on validation; F-beta β=2, min precision 0.20).
        - The output is a **review score**, not a calibrated probability.
        - Frozen v1 test characterisation is preserved and is not reused to retune the champion.
        - This demonstration is not affiliated with Finnish Customs or any authority.
        """
    )
    _calibration_warning()
    cfg = get_config()
    st.caption(cfg.disclaimer)


def _section_champion() -> None:
    st.header("Champion and model-selection evidence")
    champion = _artifact_json("artifacts/mlops/champion.json")
    if not champion:
        _missing(
            "Champion metadata is not available. Prepare it with `make experiments` "
            "(train/validation only; this does not evaluate the frozen test set)."
        )
        return
    semantics = score_metadata_from_champion(champion)
    val = champion.get("validation_metrics") or {}
    st.write(
        {
            "model_name": champion.get("model_name"),
            "model_version": champion.get("model_version"),
            "mlflow_run_id": champion.get("mlflow_run_id"),
            "decision_threshold": champion.get("threshold"),
            "calibration_status": champion.get("calibration_status"),
            "score_is_calibrated": semantics.get("score_is_calibrated", False),
            "score_semantics": semantics.get("score_semantics"),
            "policy_version": champion.get("policy_version"),
            "serialization": champion.get("serialization"),
            "roundtrip_ok": champion.get("roundtrip_ok"),
        }
    )
    _calibration_warning()
    st.subheader("Validation metrics used for selection")
    st.caption(
        "These are validation figures from champion metadata. They are not a test evaluation."
    )
    st.write(
        {
            "PR-AUC": val.get("val_pr_auc"),
            "ROC-AUC": val.get("val_roc_auc"),
            "precision": val.get("val_precision"),
            "recall": val.get("val_recall"),
            "F1": val.get("val_f1"),
            "Brier": val.get("val_brier"),
            "ECE": val.get("val_ece"),
        }
    )
    st.write(str(champion.get("reason", "")))
    st.info(str(champion.get("test_evaluation_note", "")))
    extras = champion.get("extras") or {}
    rejections = extras.get("rejections")
    if rejections:
        st.subheader("Policy rejections")
        st.write(rejections)
    frozen = _artifact_json("artifacts/frozen_v1/metrics_test.json")
    if frozen:
        st.subheader("Frozen v1 synthetic test characterisation")
        st.caption(
            "Preserved frozen-v1 figures. They were not used to select this champion "
            "and are not real-world expected performance."
        )
        metrics = frozen.get("metrics") or {}
        st.write(
            {
                "PR-AUC": metrics.get("pr_auc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "threshold": metrics.get("threshold"),
                "n_samples": metrics.get("n_samples"),
            }
        )
        sketch = _operational_sketch(metrics if isinstance(metrics, dict) else {})
        if sketch:
            st.write({key: round(value, 1) for key, value in sketch.items()})
            st.caption(
                "Per-1,000 figures are derived from the frozen confusion counts, not invented."
            )


def _section_predict() -> None:
    st.header("Single-shipment prediction")
    api_url = _api_base()
    status = _api_status(api_url)
    st.caption(f"API base URL: `{api_url}` · status: **{status}**")
    if status != "ready":
        st.info(
            "The API is not ready. The dashboard will score with the local champion loader "
            "when you submit a shipment. Start the API or Docker stack if you want live HTTP scoring."
        )
    payload = _shipment_form("predict")
    if st.button("Score shipment", type="primary", key="predict_submit"):
        result = _score_via_api(payload) if status == "ready" else None
        if result is None:
            result = _score_locally(payload)
        if result is None:
            _missing(
                "No champion is available to score this shipment. Prepare artifacts with "
                "`make experiments` and confirm MLflow tracking is reachable."
            )
        else:
            _render_score_result(result)


def _section_explain() -> None:
    st.header("Exact local explanation")
    st.caption(
        "Logit-space decomposition of the logistic champion. Contributions describe the "
        "model score, not real-world causation."
    )
    api_url = _api_base()
    status = _api_status(api_url)
    payload = _shipment_form("explain")
    if st.button("Explain shipment", type="primary", key="explain_submit"):
        explanation = _explain_via_api(payload) if status == "ready" else None
        result = _score_via_api(payload) if status == "ready" else None
        if explanation is None:
            explanation = _explain_locally(payload)
        if result is None:
            result = _score_locally(payload)
        if result is None and explanation is None:
            _missing("No champion is available to explain this shipment.")
            return
        if result is not None:
            _render_score_result(result)
        if explanation is not None:
            _render_explanation(explanation)
        else:
            _missing("A score is available, but an exact linear explanation could not be built.")
    example = _artifact_json("artifacts/explanations/local_explanation_0.json")
    if example:
        st.divider()
        st.subheader("Saved validation example")
        st.caption("Generated artifact from `make explain` (validation rows only).")
        _render_explanation(example)
        _show_image(
            "artifacts/explanations/local_explanation_0.png",
            "Local explanation (text in the tables above is the primary status; colour is secondary)",
        )


def _section_coefficients() -> None:
    st.header("Global coefficient analysis")
    global_expl = _artifact_json("artifacts/explanations/global_coefficients.json")
    if not global_expl:
        _missing("Run `make explain` to generate coefficient artifacts.")
        return
    st.caption(str(global_expl.get("coefficient_caveat", "")))
    st.caption(str(global_expl.get("causation_disclaimer", "")))
    table = pd.DataFrame(global_expl.get("ranked_coefficients") or [])
    if table.empty:
        _missing("Coefficient table is empty.")
    else:
        st.dataframe(table, use_container_width=True)
    _show_image(
        "artifacts/explanations/champion_coefficients.png",
        "Coefficient magnitude (text labels in the table are required; colour is secondary)",
    )


def _section_permutation() -> None:
    st.header("Permutation importance")
    perm = _artifact_json("artifacts/explanations/permutation_importance.json")
    if not perm:
        _missing("Run `make explain` to generate permutation-importance artifacts.")
        return
    st.caption("Computed on validation rows only. Complementary to coefficients; not causal.")
    st.caption(str(perm.get("causation_disclaimer", "")))
    table = pd.DataFrame(perm.get("rows") or [])
    if table.empty:
        _missing("Permutation table is empty.")
    else:
        st.dataframe(table, use_container_width=True)
    _show_image(
        "artifacts/explanations/permutation_importance.png",
        "Permutation importance on validation PR-AUC",
    )
    note = (perm.get("comparison_with_coefficients") or {}).get("disagreement_note")
    if note:
        st.info(str(note))


def _section_comparison() -> None:
    st.header("Model comparison")
    st.caption("Random Forest and XGBoost are comparison models, not the champion.")
    comparison = _artifact_json("artifacts/explanations/comparison_models/summary.json")
    if not comparison:
        _missing("Run `make explain` to generate comparison-model artifacts.")
        return
    st.write(comparison.get("models"))
    records = _artifact_list("artifacts/mlops/experiment_records.json")
    if records:
        st.subheader("Experiment records (validation only)")
        frame = pd.DataFrame(records)
        if "validation_metrics" in frame.columns:
            metrics = pd.json_normalize(frame["validation_metrics"])
            identity = [
                column
                for column in ("model_family", "run_id", "threshold", "calibration_status")
                if column in frame.columns
            ]
            frame = pd.concat([frame[identity].reset_index(drop=True), metrics], axis=1)
        keep = [
            column
            for column in (
                "model_family",
                "run_id",
                "val_pr_auc",
                "val_recall",
                "val_precision",
                "threshold",
                "calibration_status",
            )
            if column in frame.columns
        ]
        st.dataframe(frame[keep] if keep else frame, use_container_width=True)
    frozen_cmp = _artifact_json("artifacts/frozen_v1/model_comparison.json")
    if frozen_cmp:
        st.subheader("Frozen v1 comparison snapshot")
        st.caption("Preserved frozen-v1 artifact. Not used to replace the current champion.")
        if "models" in frozen_cmp:
            st.write(frozen_cmp.get("models"))
        else:
            st.write({key: frozen_cmp[key] for key in list(frozen_cmp)[:12]})


def _section_calibration() -> None:
    st.header("Calibration and score-semantics warning")
    _calibration_warning()
    semantics = _artifact_json("artifacts/mlops/score_semantics.json")
    if semantics:
        st.write(semantics)
    else:
        _missing("Score-semantics artifact is not available.")
    st.markdown(
        """
        The registered champion is **uncalibrated**. A later train-only calibration experiment
        did not displace it. Do not read `review_score` as P(additional review) or as a
        real-world probability.
        """
    )
    calibration = _artifact_list("artifacts/mlops/calibration_comparison.json")
    if calibration:
        st.subheader("Train/validation calibration comparison")
        st.caption("These rows are validation evidence. They are not a test evaluation.")
        st.dataframe(pd.DataFrame(calibration), use_container_width=True)
    else:
        _missing("Calibration comparison artifact is not available.")


def _section_subgroups() -> None:
    st.header("Subgroup analysis")
    sub = _artifact_json("artifacts/explanations/subgroup_performance.json")
    if not sub:
        _missing("Run `make explain` to generate subgroup artifacts.")
        return
    for line in sub.get("limitations") or []:
        st.caption(str(line))
    table = pd.DataFrame(sub.get("rows") or [])
    if table.empty:
        _missing("Subgroup table is empty.")
        return
    st.dataframe(table, use_container_width=True)
    plots = sub.get("plots") or {}
    if isinstance(plots, dict):
        for column in table["group_column"].unique() if "group_column" in table.columns else []:
            recall = plots.get(f"recall_{column}")
            rates = plots.get(f"rates_{column}")
            if isinstance(recall, str) and not recall.startswith("/") and ":\\" not in recall:
                _show_image(recall, f"Recall by {column}")
            else:
                _show_image(
                    f"artifacts/explanations/subgroup_recall_{column}.png",
                    f"Recall by {column}",
                )
            if isinstance(rates, str) and not rates.startswith("/") and ":\\" not in rates:
                _show_image(rates, f"Prevalence vs predicted review rate ({column})")
            else:
                _show_image(
                    f"artifacts/explanations/subgroup_rates_{column}.png",
                    f"Prevalence vs predicted review rate ({column})",
                )


def _status_badge(status: str) -> str:
    mapping = {
        "insufficient_data": "Status: insufficient_data — no complete monitoring report (not healthy)",
        "no_material_drift": "Status: no_material_drift — no material drift under the current policy",
        "warning": "Status: warning — investigate pipeline and operational context",
        "critical": "Status: critical — investigate before relying on outputs",
        "monitoring_error": "Status: monitoring_error — monitoring did not complete (not healthy)",
        "none": "Legacy severity none — see status field for operational meaning",
    }
    return mapping.get(str(status).lower(), f"Status: {status}")


def _severity_badge(severity: str) -> str:
    mapping = {
        "none": "Severity: none",
        "warning": "Severity: warning — investigate pipeline/context",
        "critical": "Severity: critical — investigate before relying on outputs",
    }
    return mapping.get(str(severity).lower(), f"Severity: {severity}")


def _section_monitoring() -> None:
    st.header("Monitoring")
    st.caption(
        "Operational monitoring compares a current batch to a train-derived reference profile. "
        "Input drift, score drift and known performance are separate concepts. "
        "Overall status is explained by alert reasons, not by a bare warning/critical label."
    )
    status = _artifact_json("artifacts/monitoring/status.json")
    latest = _artifact_json("artifacts/monitoring/latest_report.json")
    status_value = "insufficient_data"
    if status:
        status_value = str(status.get("status") or "insufficient_data")
    st.subheader("Overall status")
    st.write(_status_badge(status_value))
    if status and status.get("overall_severity"):
        st.caption(_severity_badge(str(status.get("overall_severity"))))

    if (
        not status
        or not status.get("available")
        or status_value
        in {
            "insufficient_data",
            "monitoring_error",
        }
    ):
        if status_value == "monitoring_error":
            st.error(
                status.get("message")
                if status
                else "Monitoring did not complete. Do not treat this as healthy."
            )
        else:
            _missing(
                "No complete monitoring report found. Run `make monitoring-reference`, "
                "generate scenarios, and `make monitoring-run SCENARIO=none` "
                "(or `make monitoring-all`). Missing or failed monitoring is not healthy."
            )
        st.info("Ground truth is unavailable in default unlabelled monitoring.")
        if (
            not latest
            or not latest.get("report_complete", True)
            or not status
            or not status.get("available")
        ):
            return

    st.write(
        {
            "policy_version": status.get("policy_version") if status else None,
            "monitoring_run_id": status.get("monitoring_run_id") if status else None,
            "timestamp": status.get("timestamp") if status else None,
            "mode": status.get("mode") if status else None,
            "scenario": status.get("scenario") if status else None,
            "recommended_action": status.get("recommended_action") if status else None,
            "reference_batch_size": status.get("reference_batch_size") if status else None,
            "batch_size": status.get("batch_size") if status else None,
            "n_monitored_features": status.get("n_monitored_features") if status else None,
            "n_warnings": status.get("n_warnings") if status else None,
            "n_critical_findings": status.get("n_critical_findings") if status else None,
            "ground_truth_available": status.get("ground_truth_available") if status else None,
            "report_complete": status.get("report_complete") if status else None,
            "champion_version": status.get("champion_version") if status else None,
            "reference_fingerprint": status.get("reference_fingerprint") if status else None,
            "current_fingerprint": status.get("current_fingerprint") if status else None,
        }
    )
    st.caption(
        "Drift checks are educational. They do not authorise automatic retraining or shipment blocking."
    )

    reasons = (status or {}).get("alert_reasons") or (latest or {}).get("alert_reasons") or []
    st.subheader("Alert reasons")
    if reasons:
        st.dataframe(pd.DataFrame(reasons), use_container_width=True)
        isolated = any(item.get("role") == "isolated_weak_warning" for item in reasons)
        if isolated and status_value == "no_material_drift":
            st.info(
                "An isolated weak warning was recorded but did not raise overall status. "
                "Policy 1.1.0 requires coordinated features, score drift, persistence, "
                "or an immediate-critical exception."
            )
    else:
        st.write("No metric exceeded a warning or critical effect-size threshold.")

    if latest:
        score = latest.get("score_drift") or {}
        st.subheader("Review-score and decision drift")
        st.write(
            {
                "review_score_psi": score.get("psi"),
                "predicted_review_rate_reference": score.get("reference_predicted_review_rate"),
                "predicted_review_rate_current": score.get("current_predicted_review_rate"),
                "predicted_review_rate_change": score.get("predicted_review_rate_change"),
                "score_drift_severity": score.get("severity"),
            }
        )
        st.caption("Review score is a ranking/threshold value, not a calibrated probability.")

        st.subheader("Input data drift summary")
        features = latest.get("feature_metrics") or []
        if features:
            frame = pd.DataFrame(features)
            numeric = frame[frame["type"] == "numeric"]
            categorical = frame[frame["type"] == "categorical"]
            if not numeric.empty:
                st.markdown("Numerical features")
                keep = [
                    col
                    for col in (
                        "feature",
                        "severity",
                        "psi",
                        "standardized_mean_difference",
                        "missing_rate_change",
                        "ks_statistic",
                    )
                    if col in numeric.columns
                ]
                st.dataframe(numeric[keep], use_container_width=True)
            if not categorical.empty:
                st.markdown("Categorical features")
                keep = [
                    col
                    for col in (
                        "feature",
                        "severity",
                        "jensen_shannon_divergence",
                        "total_variation_distance",
                        "unseen_category_rate",
                        "missing_rate_change",
                    )
                    if col in categorical.columns
                ]
                st.dataframe(categorical[keep], use_container_width=True)
        else:
            st.write("No feature metrics in the latest report.")

        if latest.get("mode") == "labelled_simulation":
            st.subheader("Labelled simulation (synthetic only)")
            st.warning(
                "These performance figures are simulated outcomes, not production monitoring evidence."
            )
            st.write(latest.get("simulated_performance") or {})
        else:
            st.subheader("Ground truth")
            st.info(
                "Ground truth is unavailable in unlabelled monitoring. "
                "Performance degradation can only be measured when delayed labels arrive."
            )

        for line in latest.get("limitations") or []:
            st.caption(str(line))


def _section_card() -> None:
    st.header("Model card and limitations")
    _markdown_file("docs/model_card.md")
    st.subheader("Limitations")
    _markdown_file("docs/limitations.md")


SECTION_RENDERERS = {
    SECTIONS[0]: _section_overview,
    SECTIONS[1]: _section_champion,
    SECTIONS[2]: _section_predict,
    SECTIONS[3]: _section_explain,
    SECTIONS[4]: _section_coefficients,
    SECTIONS[5]: _section_permutation,
    SECTIONS[6]: _section_comparison,
    SECTIONS[7]: _section_calibration,
    SECTIONS[8]: _section_subgroups,
    SECTIONS[9]: _section_monitoring,
    SECTIONS[10]: _section_card,
}


def main() -> None:
    """Render the Streamlit dashboard."""
    st.title("Cargo Risk ML Lab")
    st.caption(f"v{__version__} · educational portfolio · synthetic data only")
    _disclaimer_banner()

    section = st.sidebar.radio("Demonstration sections", SECTIONS, index=0)
    st.sidebar.caption(DISCLAIMER)
    renderer = SECTION_RENDERERS[str(section)]
    renderer()
    st.divider()
    st.caption(
        "Educational synthetic demonstration. Filesystem paths, MLflow storage URIs and "
        "serialized artifact internals are not shown."
    )


if __name__ == "__main__":
    main()
