"""Error analysis comparing false positives and false negatives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def label_errors(
    features: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Attach prediction and error-type columns to a feature frame."""
    y_pred = (y_prob >= threshold).astype(int)
    frame = features.copy()
    frame["y_true"] = y_true
    frame["y_prob"] = y_prob
    frame["y_pred"] = y_pred
    frame["error_type"] = "correct"
    frame.loc[(frame["y_true"] == 0) & (frame["y_pred"] == 1), "error_type"] = "false_positive"
    frame.loc[(frame["y_true"] == 1) & (frame["y_pred"] == 0), "error_type"] = "false_negative"
    frame["confidence"] = np.where(
        frame["error_type"] == "false_positive",
        frame["y_prob"],
        np.where(frame["error_type"] == "false_negative", 1.0 - frame["y_prob"], np.nan),
    )
    return frame


def _band(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce"), bins=edges, labels=labels, include_lowest=True
    )


def summarise_errors_by_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Counts and rates of FP/FN for one grouping column."""
    working = frame.copy()
    working["_group"] = working[group_column].astype(str)
    rows: list[dict[str, Any]] = []
    for group_value, part in working.groupby("_group", dropna=False):
        n = int(len(part))
        fp = int((part["error_type"] == "false_positive").sum())
        fn = int((part["error_type"] == "false_negative").sum())
        tp = int(((part["y_true"] == 1) & (part["y_pred"] == 1)).sum())
        tn = int(((part["y_true"] == 0) & (part["y_pred"] == 0)).sum())
        rows.append(
            {
                "group_column": group_column,
                "group_value": str(group_value),
                "n": n,
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "fp_rate": fp / n if n else 0.0,
                "fn_rate": fn / n if n else 0.0,
                "share_of_all_fp": 0.0,
                "share_of_all_fn": 0.0,
            }
        )
    out = pd.DataFrame(rows)
    total_fp = float(out["false_positives"].sum()) or 1.0
    total_fn = float(out["false_negatives"].sum()) or 1.0
    out["share_of_all_fp"] = out["false_positives"] / total_fp
    out["share_of_all_fn"] = out["false_negatives"] / total_fn
    return out.sort_values(["false_negatives", "false_positives"], ascending=False)


def build_error_analysis_report(
    features: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compare FP vs FN across major synthetic feature groups."""
    labelled = label_errors(features, y_true, y_prob, threshold)
    working = labelled.copy()
    if "declaration_completeness_score" in working.columns:
        working["completeness_band"] = _band(
            working["declaration_completeness_score"],
            [0.0, 0.6, 0.8, 1.01],
            ["low", "medium", "high"],
        )
    if "previous_discrepancies" in working.columns:
        working["discrepancy_band"] = _band(
            working["previous_discrepancies"],
            [-0.1, 0.0, 1.0, 100.0],
            ["none", "one", "two_or_more"],
        )

    group_cols = [
        column
        for column in (
            "transport_mode",
            "commodity_category",
            "origin_region",
            "expedited_shipment",
            "completeness_band",
            "discrepancy_band",
        )
        if column in working.columns
    ]
    tables = [summarise_errors_by_group(working, column) for column in group_cols]
    grouped = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    fp = labelled[labelled["error_type"] == "false_positive"]
    fn = labelled[labelled["error_type"] == "false_negative"]
    summary = {
        "n_false_positives": int(len(fp)),
        "n_false_negatives": int(len(fn)),
        "n_correct": int((labelled["error_type"] == "correct").sum()),
        "threshold": float(threshold),
        "mean_score_fp": float(fp["y_prob"].mean()) if len(fp) else None,
        "mean_score_fn": float(fn["y_prob"].mean()) if len(fn) else None,
        "disclaimer": (
            "Error groups are from synthetic features only. Patterns are not real "
            "customs risk factors."
        ),
    }
    return {
        "summary": summary,
        "by_group": grouped.to_dict(orient="records"),
        "labelled": labelled,
        "grouped_table": grouped,
    }
