"""Explainability, score semantics, subgroups and frozen-artifact isolation."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from src.config import PROJECT_ROOT
from src.data import generate_synthetic_shipments, split_dataset
from src.explainability.linear import (
    EXCLUDED_EXPLANATION_FIELDS,
    RECONSTRUCTION_ATOL,
    LinearExplanationModel,
    global_linear_explanation,
)
from src.explainability.semantics import (
    SCORE_SEMANTICS_UNCALIBRATED,
    assert_uncalibrated_not_labelled_probability,
    score_metadata_from_champion,
)
from src.explainability.subgroups import subgroup_payload, wilson_interval
from src.features import prepare_xy
from src.models import train_model


def _tiny_logreg():
    df = generate_synthetic_shipments(n_samples=260, seed=11, validate=False)
    splits = split_dataset(df, seed=11, strategy="stratified")
    trained = train_model(splits.train, val_df=splits.val, estimator_name="logreg")
    x_val, y_val = prepare_xy(splits.val, fit_derived_reference=splits.train)
    return trained, splits, x_val, y_val


def test_uncalibrated_semantics_reject_probability_label() -> None:
    payload = score_metadata_from_champion({"calibration_status": "none", "threshold": 0.525})
    assert payload["score_is_calibrated"] is False
    assert payload["calibration_method"] is None
    assert payload["score_semantics"] == SCORE_SEMANTICS_UNCALIBRATED
    assert_uncalibrated_not_labelled_probability(payload)
    with pytest.raises(AssertionError):
        assert_uncalibrated_not_labelled_probability(
            {
                "score_is_calibrated": False,
                "score_semantics": "calibrated probability of review",
            }
        )


def test_linear_reconstruction_and_direction() -> None:
    trained, _, x_val, _ = _tiny_logreg()
    explainer = LinearExplanationModel.from_pipeline(trained.pipeline)
    row = x_val.iloc[[0]]
    payload = explainer.explain_row(row, threshold=0.5, metadata={"model_name": "logreg"})
    assert payload["reconstruction_error"] <= RECONSTRUCTION_ATOL
    contrib = pd.DataFrame(payload["contributions"])
    reconstructed = payload["intercept_log_odds"] + float(contrib["log_odds_contribution"].sum())
    assert abs(reconstructed - payload["logit"]) <= RECONSTRUCTION_ATOL
    sample = contrib.iloc[0]
    expected_sign = np.sign(
        sample["transformed_value"] * sample["coefficient_transformed"]
        if sample["transformed_value"] * sample["coefficient_transformed"] != 0
        else 0.0
    )
    actual_sign = np.sign(sample["log_odds_contribution"])
    assert expected_sign == actual_sign or sample["log_odds_contribution"] == 0.0
    increasing = pd.DataFrame(payload["strongest_increasing"])
    assert (
        increasing.iloc[0]["log_odds_contribution"] >= increasing.iloc[-1]["log_odds_contribution"]
    )


def test_onehot_names_missing_unseen_and_excluded_fields() -> None:
    trained, _, x_val, _ = _tiny_logreg()
    explainer = LinearExplanationModel.from_pipeline(trained.pipeline)
    global_payload = global_linear_explanation(trained.pipeline)
    names = [str(item["display_name"]) for item in global_payload["ranked_coefficients"]]
    assert any("transport_mode =" in name or "origin_region =" in name for name in names)

    missing = x_val.iloc[[1]].copy()
    missing.loc[missing.index[0], "declaration_completeness_score"] = np.nan
    missing_expl = explainer.explain_row(missing, threshold=0.5)
    missing_rows = [
        item
        for item in missing_expl["contributions"]
        if item["source_feature"] == "declaration_completeness_score"
    ]
    assert missing_rows and missing_rows[0]["original_missing"] is True
    assert missing_expl["reconstruction_error"] <= RECONSTRUCTION_ATOL

    unseen = x_val.iloc[[2]].copy()
    unseen.loc[unseen.index[0], "transport_mode"] = "teleport"
    unseen_expl = explainer.explain_row(unseen, threshold=0.5)
    transport_rows = [
        item for item in unseen_expl["contributions"] if item["source_feature"] == "transport_mode"
    ]
    assert transport_rows
    assert any(bool(item["unseen_category"]) for item in transport_rows)
    assert unseen_expl["reconstruction_error"] <= RECONSTRUCTION_ATOL

    dirty = x_val.iloc[[0]].copy()
    dirty["shipment_id"] = "SYN-leak"
    with pytest.raises(ValueError, match="Excluded"):
        explainer.explain_row(dirty, threshold=0.5)
    contrib_names = {item["source_feature"] for item in missing_expl["contributions"]}
    assert contrib_names.isdisjoint(EXCLUDED_EXPLANATION_FIELDS)


def test_local_explanation_is_deterministic() -> None:
    trained, _, x_val, _ = _tiny_logreg()
    explainer = LinearExplanationModel.from_pipeline(trained.pipeline)
    row = x_val.iloc[[3]]
    first = explainer.explain_row(row, threshold=0.525)
    second = explainer.explain_row(row, threshold=0.525)
    assert first["logit"] == second["logit"]
    assert first["review_score"] == second["review_score"]
    assert first["contributions"] == second["contributions"]


def test_subgroup_small_sample_warning_and_no_tuning_flag() -> None:
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    scores = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2])
    frame = pd.DataFrame({"transport_mode": ["air"] * 3 + ["sea"] * 5})
    payload = subgroup_payload(frame, y, scores, threshold=0.5, min_n=10)
    assert payload["used_for_tuning"] is False
    assert payload["protected_characteristics_present"] is False
    assert payload["split"] == "validation"
    assert all(row["small_sample"] for row in payload["rows"])
    low, high = wilson_interval(2, 10)
    assert 0.0 <= low <= high <= 1.0


def test_explain_scripts_do_not_touch_test_or_frozen() -> None:
    source = (PROJECT_ROOT / "scripts" / "explain_model.py").read_text(encoding="utf-8")
    assert "test.csv" not in source
    assert "metrics_test.json" not in source
    assert "evaluate_model" not in source
    assert "select_champion" not in source
    linear_src = inspect.getsource(LinearExplanationModel.explain_row)
    assert "test.csv" not in linear_src


def test_frozen_v1_hash_still_intact() -> None:
    import hashlib

    path = PROJECT_ROOT / "artifacts" / "frozen_v1" / "metrics_test.json"
    if not path.exists():
        pytest.skip("frozen-v1 artifacts are local and gitignored")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest.startswith("fc94af40")


def test_dashboard_uses_review_score_language() -> None:
    source = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "review_score" in source
    assert "Requires-review probability" not in source
    assert "requires_review_probability" not in source
    assert "dangerous" not in source.lower()
    assert "fraudulent" not in source.lower()
    assert "REVIEW_LABEL_POSITIVE" in source
    from src.explainability.semantics import REVIEW_LABEL_POSITIVE

    assert "additional human review" in REVIEW_LABEL_POSITIVE.lower()
    assert "1. Project overview" in source
    assert "11. Model card and limitations" in source
    import app.streamlit_app as dashboard

    assert callable(dashboard.main)


def test_dashboard_smoke_apptest() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=45)
    app.run()
    assert not app.exception
