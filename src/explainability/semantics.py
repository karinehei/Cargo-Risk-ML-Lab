"""Score terminology for uncalibrated champion outputs.

The registered champion is a class-weighted logistic regression that was not
calibrated. Its ``predict_proba`` values are ranking and threshold scores, not
literal probabilities of requiring review.
"""

from __future__ import annotations

from typing import Any

SCORE_SEMANTICS_UNCALIBRATED = "ranking and threshold score, not a literal probability"
SCORE_SEMANTICS_CALIBRATED = "calibrated score; still not a real-world probability"
SCORE_WARNING = (
    "This review score is not a calibrated probability. It is a ranking and "
    "threshold score from a class-weighted logistic regression. Use it only "
    "with the decision threshold to decide whether to send a shipment for "
    "additional human review."
)
HUMAN_REVIEW_NOTICE = (
    "requires_review=1 means additional human review is suggested. "
    "This is not an automated enforcement decision."
)
CAUSATION_DISCLAIMER = (
    "Feature contributions explain this model's internal score, not real-world "
    "causation, customs risk, or enforcement outcomes."
)
REVIEW_LABEL_POSITIVE = "Additional human review suggested"
REVIEW_LABEL_NEGATIVE = "Not flagged for additional human review"
CALIBRATED_STATUSES = frozenset({"sigmoid", "isotonic", "platt", "calibrated"})


def score_metadata_from_champion(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Derive public score-semantics fields from champion metadata."""
    payload = dict(metadata or {})
    status = str(payload.get("calibration_status") or "none").strip().lower()
    calibrated = status in CALIBRATED_STATUSES
    method = status if calibrated else None
    semantics = SCORE_SEMANTICS_CALIBRATED if calibrated else SCORE_SEMANTICS_UNCALIBRATED
    threshold = payload.get("threshold")
    return {
        "score_is_calibrated": calibrated,
        "score_semantics": str(payload.get("score_semantics") or semantics),
        "calibration_method": payload.get("calibration_method", method),
        "calibration_status": payload.get("calibration_status", "none"),
        "decision_threshold": None if threshold is None else float(threshold),
        "score_warning": SCORE_WARNING,
    }


def assert_uncalibrated_not_labelled_probability(payload: dict[str, Any]) -> None:
    """Raise if an uncalibrated model is presented as a calibrated probability."""
    if payload.get("score_is_calibrated"):
        return
    text = " ".join(str(value).lower() for value in payload.values() if isinstance(value, str))
    forbidden = (
        "calibrated probability",
        "literal probability",
        "true probability",
        "estimated probability of review",
    )
    for phrase in forbidden:
        if phrase in text and "not a literal probability" not in text:
            raise AssertionError(f"Uncalibrated score labelled as probability: {phrase}")
    if payload.get("score_semantics") != SCORE_SEMANTICS_UNCALIBRATED:
        raise AssertionError("Uncalibrated champion must declare ranking/threshold semantics")
