"""Train/validation diagnostics for why a linear model may beat tree ensembles.

None of these routines load the test set.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import AppConfig, get_config
from src.data.generate import _raw_review_scores, _sigmoid
from src.features import prepare_xy
from src.models.estimators import build_model_pipeline
from src.models.train import predict_proba


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Weighted absolute gap between mean predicted probability and empirical rate."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = max(len(y_true_arr), 1)
    for index in range(n_bins):
        left = bins[index]
        right = bins[index + 1]
        if index == n_bins - 1:
            mask = (y_prob_arr >= left) & (y_prob_arr <= right)
        else:
            mask = (y_prob_arr >= left) & (y_prob_arr < right)
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n_samples) * abs(
            float(y_true_arr[mask].mean()) - float(y_prob_arr[mask].mean())
        )
    return float(ece)


def toy_score_term_contributions(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Decompose the fictional label score into additive vs interaction terms.

    Arithmetic is a mirror of ``src.data.generate._raw_review_scores`` so we can
    measure contribution shares without changing data generation.
    """
    value_ratio = features["value_to_weight_ratio"].to_numpy(dtype=float)
    completeness = features["declaration_completeness_score"].to_numpy(dtype=float)
    docs = features["documentation_count"].to_numpy(dtype=float)
    discrepancies = features["previous_discrepancies"].to_numpy(dtype=float)
    history = features["sender_history_length"].to_numpy(dtype=float)
    rarity = features["route_rarity"].to_numpy(dtype=float)
    deviation = features["declared_vs_estimated_value_deviation"].to_numpy(dtype=float)
    hour = features["submission_hour"].to_numpy(dtype=float)
    expedited = features["expedited_shipment"].to_numpy(dtype=float)
    declared = features["declared_value_eur"].to_numpy(dtype=float)
    weight = features["shipment_weight_kg"].to_numpy(dtype=float)
    transport = features["transport_mode"].to_numpy()
    commodity = features["commodity_category"].to_numpy()

    high_ratio = _sigmoid((value_ratio - 180.0) / 70.0)
    incomplete = _sigmoid((0.58 - completeness) / 0.12)
    sparse_docs = _sigmoid((3.0 - docs) / 1.2)
    new_sender = _sigmoid((8.0 - history) / 3.5)
    off_hours = ((hour < 6.0) | (hour >= 22.0)).astype(float)
    abs_dev = np.abs(deviation)
    air_sensitive = (
        (transport == "air") & np.isin(commodity, ["electronics", "pharmaceuticals", "chemicals"])
    ).astype(float)
    heavy_fast = ((expedited > 0) & (weight > 600.0)).astype(float)
    high_value = _sigmoid((np.log1p(declared) - 10.2) / 0.8)

    return {
        "additive_high_ratio": 1.25 * high_ratio,
        "interaction_incomplete_absdev": 1.05 * incomplete * np.clip(abs_dev, 0.0, 2.5),
        "interaction_discrep_new_sender": 0.85 * (discrepancies >= 2).astype(float) * new_sender,
        "interaction_rarity_expedited": 0.70 * rarity * expedited,
        "additive_sparse_docs": 0.55 * sparse_docs,
        "additive_off_hours": 0.45 * off_hours,
        "interaction_air_sensitive": 0.60 * air_sensitive,
        "interaction_heavy_fast": 0.50 * heavy_fast,
        "interaction_high_value_air": 0.35 * high_value * (transport == "air").astype(float),
        "interaction_underdeclare_incomplete": 0.40
        * ((deviation < -0.35) & (completeness < 0.7)).astype(float),
        "additive_discrep_tanh": 0.25 * np.tanh(discrepancies / 2.0),
        "additive_long_history": -0.35 * _sigmoid((history - 30.0) / 10.0),
    }


def summarise_toy_score(train_df: pd.DataFrame) -> dict[str, Any]:
    """Mean absolute contribution of additive vs interaction terms on training rows."""
    filled = train_df.copy()
    numeric = filled.select_dtypes(include=["number"])
    filled[numeric.columns] = numeric.fillna(numeric.median())
    terms = toy_score_term_contributions(filled)
    stacked = np.vstack([np.abs(term) for term in terms.values()])
    total_abs = stacked.sum(axis=0)
    additive = np.vstack(
        [np.abs(value) for key, value in terms.items() if key.startswith("additive_")]
    ).sum(axis=0)
    interaction = np.vstack(
        [np.abs(value) for key, value in terms.items() if key.startswith("interaction_")]
    ).sum(axis=0)
    reconstructed = np.sum(list(terms.values()), axis=0)
    original = _raw_review_scores(filled)
    per_term = {name: float(np.mean(np.abs(value))) for name, value in terms.items()}
    return {
        "mean_abs_additive": float(np.mean(additive)),
        "mean_abs_interaction": float(np.mean(interaction)),
        "additive_share": float(np.mean(additive / np.clip(total_abs, 1e-12, None))),
        "interaction_share": float(np.mean(interaction / np.clip(total_abs, 1e-12, None))),
        "max_abs_reconstruction_error": float(np.max(np.abs(reconstructed - original))),
        "logit_noise_std_config": 0.65,
        "label_flip_rate_config": 0.025,
        "per_term_mean_abs": per_term,
        "interpretation": (
            "Shares describe the fictional score before noise and label flips. "
            "A large additive share plus logit noise can make a linear model competitive "
            "even when some interaction terms exist."
        ),
    }


def add_explicit_interaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few toy-score interactions as extra numeric columns (train/val diagnostics)."""
    out = df.copy()
    completeness = pd.to_numeric(out["declaration_completeness_score"], errors="coerce")
    deviation = pd.to_numeric(out["declared_vs_estimated_value_deviation"], errors="coerce")
    history = pd.to_numeric(out["sender_history_length"], errors="coerce")
    rarity = pd.to_numeric(out["route_rarity"], errors="coerce")
    expedited = pd.to_numeric(out["expedited_shipment"], errors="coerce")
    discrepancies = pd.to_numeric(out["previous_discrepancies"], errors="coerce")
    declared = pd.to_numeric(out["declared_value_eur"], errors="coerce")
    weight = pd.to_numeric(out["shipment_weight_kg"], errors="coerce")
    air = (out["transport_mode"].astype(str) == "air").astype(float)
    sensitive = (
        out["commodity_category"]
        .astype(str)
        .isin(["electronics", "pharmaceuticals", "chemicals"])
        .astype(float)
    )
    out["ix_incomplete_absdev"] = (1.0 - completeness.clip(0.0, 1.0)) * deviation.abs()
    out["ix_discrep_new_sender"] = ((discrepancies >= 2).astype(float)) * (
        (history < 8).astype(float)
    )
    out["ix_rarity_expedited"] = rarity * expedited
    out["ix_high_value_air"] = np.log1p(declared.clip(lower=0.0)) * air
    out["ix_underdeclare_incomplete"] = ((deviation < -0.35) & (completeness < 0.7)).astype(float)
    out["ix_air_sensitive"] = air * sensitive
    out["ix_heavy_fast"] = ((expedited > 0) & (weight > 600.0)).astype(float)
    return out


def logreg_with_explicit_interactions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppConfig | None = None,
) -> dict[str, float]:
    """Fit logistic regression with and without explicit interaction columns on train/val."""
    cfg = config or get_config()
    extra = [
        "ix_incomplete_absdev",
        "ix_discrep_new_sender",
        "ix_rarity_expedited",
        "ix_high_value_air",
        "ix_underdeclare_incomplete",
        "ix_air_sensitive",
        "ix_heavy_fast",
    ]
    features = dict(cfg.features)
    derived = list(features.get("derived", []))
    features["derived"] = list(dict.fromkeys([*derived, *extra]))
    cfg_ix = replace(cfg, features=features)

    x_train, y_train = prepare_xy(train_df, cfg, fit_derived_reference=train_df)
    x_val, y_val = prepare_xy(val_df, cfg, fit_derived_reference=train_df)
    base = build_model_pipeline("logreg", y_train, cfg)
    base.fit(x_train, y_train)
    base_pr = float(average_precision_score(y_val, predict_proba(base, x_val)))

    train_ix = add_explicit_interaction_columns(train_df)
    val_ix = add_explicit_interaction_columns(val_df)
    x_train_ix, y_train_ix = prepare_xy(train_ix, cfg_ix, fit_derived_reference=train_ix)
    x_val_ix, y_val_ix = prepare_xy(val_ix, cfg_ix, fit_derived_reference=train_ix)
    interacted = build_model_pipeline("logreg", y_train_ix, cfg_ix)
    interacted.fit(x_train_ix, y_train_ix)
    ix_pr = float(average_precision_score(y_val_ix, predict_proba(interacted, x_val_ix)))
    return {
        "logreg_val_pr_auc": base_pr,
        "logreg_with_interactions_val_pr_auc": ix_pr,
        "delta_val_pr_auc": ix_pr - base_pr,
    }


def validation_calibration(
    pipeline: Any,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Brier score and ECE on validation probabilities."""
    y_prob = predict_proba(pipeline, x_val)
    frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=8, strategy="quantile")
    return {
        "val_pr_auc": float(average_precision_score(y_val, y_prob)),
        "val_brier": float(brier_score_loss(y_val, y_prob)),
        "val_ece": expected_calibration_error(y_val, y_prob, n_bins=8),
        "calibration_slope_proxy": float(np.corrcoef(mean_pred, frac_pos)[0, 1])
        if len(mean_pred) > 1
        else float("nan"),
    }
