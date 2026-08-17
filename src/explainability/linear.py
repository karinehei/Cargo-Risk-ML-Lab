"""Exact logistic-regression explanations in log-odds space.

Coefficients apply to the **transformed** feature matrix (imputed, scaled
numerics and one-hot categoricals). They are associations inside the fitted
model, not causal effects. Correlated inputs (for example declared value,
log value and value-to-weight ratio) can make individual coefficients unstable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.schema import (
    DATE_COLUMN,
    FORBIDDEN_COLUMNS,
    ID_COLUMN,
    PERIOD_COLUMN,
    TARGET_COLUMN,
)
from src.explainability.semantics import (
    CAUSATION_DISCLAIMER,
    SCORE_WARNING,
    score_metadata_from_champion,
)
from src.features import get_feature_lists, get_feature_names

RECONSTRUCTION_ATOL = 1e-8
EXCLUDED_EXPLANATION_FIELDS = frozenset(
    {
        ID_COLUMN,
        DATE_COLUMN,
        PERIOD_COLUMN,
        TARGET_COLUMN,
        *FORBIDDEN_COLUMNS,
    }
)


def _l2(values: pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(np.square(array))))


def _as_pipeline(estimator: Any) -> Pipeline:
    if not isinstance(estimator, Pipeline) or "preprocess" not in estimator.named_steps:
        raise TypeError(
            "Exact linear explanations require a sklearn Pipeline with a preprocess step"
        )
    model = estimator.named_steps.get("model")
    if not isinstance(model, LogisticRegression):
        raise TypeError("Exact logit explanations are implemented for LogisticRegression only")
    return estimator


def transformed_design(pipeline: Pipeline, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return the dense transformed matrix and human-mappable feature names."""
    preprocess = pipeline.named_steps["preprocess"]
    matrix = np.asarray(preprocess.transform(features), dtype=float)
    names = get_feature_names(preprocess)
    if matrix.shape[1] != len(names):
        raise RuntimeError("Transformed width does not match feature names")
    return matrix, names


def _parse_name(name: str, numeric: list[str], categorical: list[str]) -> dict[str, str | None]:
    if name in numeric:
        return {
            "kind": "numeric",
            "source_feature": name,
            "category": None,
            "display_name": name,
        }
    for column in categorical:
        prefix = f"{column}_"
        if name.startswith(prefix):
            category = name[len(prefix) :]
            return {
                "kind": "categorical",
                "source_feature": column,
                "category": category,
                "display_name": f"{column} = {category}",
            }
    return {
        "kind": "unknown",
        "source_feature": name,
        "category": None,
        "display_name": name,
    }


def _numeric_scale(pipeline: Pipeline) -> tuple[np.ndarray | None, np.ndarray | None]:
    numeric_step = pipeline.named_steps["preprocess"].named_transformers_["num"]
    if "scaler" not in numeric_step.named_steps:
        return None, None
    scaler = numeric_step.named_steps["scaler"]
    return np.asarray(scaler.mean_, dtype=float), np.asarray(scaler.scale_, dtype=float)


def _sigmoid(logit: float) -> float:
    value = float(np.clip(logit, -500.0, 500.0))
    return float(1.0 / (1.0 + np.exp(-value)))


@dataclass
class LinearExplanationModel:
    """Fitted logistic regression plus preprocessing, ready for exact attributions."""

    pipeline: Pipeline
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    numeric_features: list[str]
    categorical_features: list[str]
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None

    @classmethod
    def from_pipeline(cls, estimator: Any) -> LinearExplanationModel:
        pipeline = _as_pipeline(estimator)
        model: LogisticRegression = pipeline.named_steps["model"]
        names = get_feature_names(pipeline.named_steps["preprocess"])
        coef = np.asarray(model.coef_, dtype=float).ravel()
        if coef.shape[0] != len(names):
            raise RuntimeError("Coefficient length does not match transformed features")
        numeric, categorical = get_feature_lists()
        mean, scale = _numeric_scale(pipeline)
        intercept = float(np.asarray(model.intercept_).ravel()[0])
        return cls(
            pipeline=pipeline,
            feature_names=names,
            coefficients=coef,
            intercept=intercept,
            numeric_features=numeric,
            categorical_features=categorical,
            scaler_mean=mean,
            scaler_scale=scale,
        )

    def global_coefficients(self) -> pd.DataFrame:
        """Rank transformed-feature coefficients by absolute magnitude."""
        rows: list[dict[str, Any]] = []
        n_numeric = len(self.numeric_features)
        for index, name in enumerate(self.feature_names):
            parsed = _parse_name(name, self.numeric_features, self.categorical_features)
            coef = float(self.coefficients[index])
            original_unit: float | None = None
            scaled = False
            if parsed["kind"] == "numeric" and self.scaler_scale is not None and index < n_numeric:
                scale = float(self.scaler_scale[index])
                scaled = True
                if scale != 0.0:
                    original_unit = coef / scale
            rows.append(
                {
                    "transformed_feature": name,
                    "display_name": parsed["display_name"],
                    "source_feature": parsed["source_feature"],
                    "kind": parsed["kind"],
                    "category": parsed["category"],
                    "coefficient_transformed": coef,
                    "coefficient_original_unit": original_unit,
                    "standardized_numeric": scaled and parsed["kind"] == "numeric",
                    "abs_coefficient": abs(coef),
                    "direction": "higher_review_score" if coef > 0 else "lower_review_score",
                    "interpretation": (
                        "Change in log-odds for +1 SD of the imputed numeric feature"
                        if scaled and parsed["kind"] == "numeric"
                        else (
                            "Change in log-odds when this category dummy is 1 versus 0"
                            if parsed["kind"] == "categorical"
                            else "Change in log-odds for +1 of the transformed feature"
                        )
                    ),
                }
            )
        frame = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)
        return frame.reset_index(drop=True)

    def grouped_original_importance(self) -> pd.DataFrame:
        """L2 of transformed coefficients grouped back to original input columns."""
        table = self.global_coefficients()
        grouped = (
            table.groupby("source_feature", dropna=False)
            .agg(
                l2_transformed_coefficient=("coefficient_transformed", _l2),
                signed_sum=("coefficient_transformed", "sum"),
                n_transformed_features=("transformed_feature", "count"),
            )
            .reset_index()
        )
        grouped["abs_grouped"] = grouped["l2_transformed_coefficient"].abs()
        return grouped.sort_values("abs_grouped", ascending=False).reset_index(drop=True)

    def explain_row(
        self,
        features: pd.DataFrame,
        *,
        threshold: float,
        metadata: dict[str, Any] | None = None,
        top_n: int = 15,
    ) -> dict[str, Any]:
        """Exact local decomposition for a single-row frame."""
        if len(features) != 1:
            raise ValueError("explain_row expects exactly one row")
        _reject_excluded_columns(features)
        transformed, names = transformed_design(self.pipeline, features)
        vector = transformed[0]
        contributions = vector * self.coefficients
        logit = float(self.intercept + float(np.sum(contributions)))
        review_score = float(_sigmoid(logit))
        model_score = float(self.pipeline.predict_proba(features)[0, 1])
        reconstruction_error = abs(review_score - model_score)
        if reconstruction_error > RECONSTRUCTION_ATOL:
            raise AssertionError(
                f"Logit reconstruction missed the model score by {reconstruction_error}"
            )
        logit_from_decision = float(self.pipeline.decision_function(features)[0])
        if abs(logit - logit_from_decision) > RECONSTRUCTION_ATOL:
            raise AssertionError("Intercept plus contributions does not match decision_function")

        raw = features.iloc[0]
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            parsed = _parse_name(name, self.numeric_features, self.categorical_features)
            source = str(parsed["source_feature"])
            raw_value: Any = raw[source] if source in raw.index else None
            rows.append(
                {
                    "transformed_feature": name,
                    "display_name": parsed["display_name"],
                    "source_feature": source,
                    "kind": parsed["kind"],
                    "category": parsed["category"],
                    "transformed_value": float(vector[index]),
                    "coefficient_transformed": float(self.coefficients[index]),
                    "log_odds_contribution": float(contributions[index]),
                    "original_value": _json_value(raw_value),
                    "original_missing": bool(pd.isna(raw_value)) if source in raw.index else False,
                    "unseen_category": False,
                }
            )
        contrib_frame = pd.DataFrame(rows)
        contrib_frame = _mark_unseen_categories(contrib_frame, raw)
        increasing = contrib_frame.sort_values("log_odds_contribution", ascending=False)
        decreasing = contrib_frame.sort_values("log_odds_contribution", ascending=True)
        semantics = score_metadata_from_champion(metadata)
        original_inputs = {
            column: _json_value(raw[column])
            for column in features.columns
            if column not in EXCLUDED_EXPLANATION_FIELDS
        }
        return {
            "method": "exact_logit_linear",
            "review_score": review_score,
            "decision_threshold": float(threshold),
            "requires_review": int(review_score >= float(threshold)),
            "intercept_log_odds": self.intercept,
            "logit": logit,
            "reconstruction_error": reconstruction_error,
            "reconstruction_atol": RECONSTRUCTION_ATOL,
            "relationship": (
                "logit = intercept + sum(transformed_value * coefficient); "
                "review_score = sigmoid(logit). The review score is not a calibrated probability."
            ),
            "strongest_increasing": increasing.head(top_n).to_dict(orient="records"),
            "strongest_decreasing": decreasing.head(top_n).to_dict(orient="records"),
            "contributions": contrib_frame.to_dict(orient="records"),
            "original_input_values": original_inputs,
            "model_name": None if metadata is None else metadata.get("model_name"),
            "model_version": None if metadata is None else metadata.get("model_version"),
            "mlflow_run_id": None if metadata is None else metadata.get("mlflow_run_id"),
            "score_is_calibrated": semantics["score_is_calibrated"],
            "score_semantics": semantics["score_semantics"],
            "score_warning": SCORE_WARNING,
            "causation_disclaimer": CAUSATION_DISCLAIMER,
            "coefficient_caveat": (
                "Coefficients are for transformed features. Correlated inputs can make "
                "individual signs and magnitudes unstable. Do not treat them as causal effects."
            ),
        }


def _mark_unseen_categories(frame: pd.DataFrame, raw: pd.Series) -> pd.DataFrame:
    """Flag one-hot groups that are all zero because the category was unknown."""
    out = frame.copy()
    cat_mask = out["kind"] == "categorical"
    for source, group in out.loc[cat_mask].groupby("source_feature"):
        raw_value = raw[source] if source in raw.index else None
        if raw_value is None or pd.isna(raw_value):
            continue
        if float(group["transformed_value"].sum()) == 0.0:
            out.loc[group.index, "unseen_category"] = True
    return out


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _reject_excluded_columns(features: pd.DataFrame) -> None:
    present = [column for column in features.columns if column in EXCLUDED_EXPLANATION_FIELDS]
    if present:
        raise ValueError(f"Excluded fields must not enter explanations: {present}")


def global_linear_explanation(pipeline: Any, *, top_n: int = 15) -> dict[str, Any]:
    """Serializable global coefficient ranking for the logistic champion."""
    model = LinearExplanationModel.from_pipeline(pipeline)
    table = model.global_coefficients()
    grouped = model.grouped_original_importance()
    higher = table[table["coefficient_transformed"] > 0].head(top_n)
    lower = table[table["coefficient_transformed"] < 0].head(top_n)
    return {
        "method": "logistic_regression_coefficients",
        "intercept_log_odds": model.intercept,
        "n_transformed_features": len(model.feature_names),
        "reconstruction_atol": RECONSTRUCTION_ATOL,
        "ranked_coefficients": table.to_dict(orient="records"),
        "higher_review_score": higher.to_dict(orient="records"),
        "lower_review_score": lower.to_dict(orient="records"),
        "grouped_original_features": grouped.to_dict(orient="records"),
        "coefficient_caveat": (
            "Raw coefficients are on the transformed scale (standardised numerics and "
            "one-hot categories). coefficient_original_unit divides by the scaler scale "
            "for numerics. Correlated features can make coefficients unstable. "
            "These are model associations, not causal effects."
        ),
        "causation_disclaimer": CAUSATION_DISCLAIMER,
    }
