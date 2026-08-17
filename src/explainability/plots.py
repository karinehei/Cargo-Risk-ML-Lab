"""Accessible plots for explanations and subgroup tables.

Colour encoding never relies on red/green alone. Increase vs decrease uses
blue versus orange (Okabe–Ito).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

COLOR_INCREASE = "#0072B2"
COLOR_DECREASE = "#E69F00"
COLOR_NEUTRAL = "#5C6B73"
COLOR_SMALL = "#999999"


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coefficient_bars(table: pd.DataFrame, path: Path, *, top_n: int = 15) -> Path:
    subset = table.head(top_n).iloc[::-1]
    colors = [
        COLOR_INCREASE if value > 0 else COLOR_DECREASE
        for value in subset["coefficient_transformed"]
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["display_name"], subset["coefficient_transformed"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient on transformed feature (log-odds)")
    ax.set_title("Champion logistic coefficients (not causal)")
    return _save(fig, path)


def plot_permutation_bars(table: pd.DataFrame, path: Path, *, top_n: int = 15) -> Path:
    subset = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        subset["feature"],
        subset["importance_mean"],
        color=COLOR_NEUTRAL,
        xerr=subset["importance_std"],
    )
    ax.set_xlabel("Permutation importance (validation PR-AUC drop)")
    ax.set_title("Validation permutation importance (not causal)")
    return _save(fig, path)


def plot_local_contributions(payload: dict[str, Any], path: Path, *, top_n: int = 12) -> Path:
    contrib = pd.DataFrame(payload["contributions"])
    ordered = contrib.reindex(
        contrib["log_odds_contribution"].abs().sort_values(ascending=False).index
    )
    subset = ordered.head(top_n).iloc[::-1]
    colors = [
        COLOR_INCREASE if value > 0 else COLOR_DECREASE for value in subset["log_odds_contribution"]
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["display_name"], subset["log_odds_contribution"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-odds contribution (blue = higher score, orange = lower)")
    ax.set_title("Local linear explanation (model behaviour, not causation)")
    return _save(fig, path)


def plot_subgroup_recall(table: pd.DataFrame, path: Path, column: str) -> Path | None:
    subset = table[table["group_column"] == column].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("n", ascending=True)
    colors = [COLOR_SMALL if flag else COLOR_NEUTRAL for flag in subset["small_sample"]]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(subset))))
    ax.barh(subset["group_value"].astype(str), subset["recall"].fillna(0.0), color=colors)
    ax.set_xlabel("Recall on validation (grey = n below minimum)")
    ax.set_title(f"Validation recall by {column} (descriptive, not a fairness test)")
    return _save(fig, path)


def plot_subgroup_review_rate(table: pd.DataFrame, path: Path, column: str) -> Path | None:
    subset = table[table["group_column"] == column].copy()
    if subset.empty:
        return None
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        x - 0.2,
        subset["target_prevalence"],
        width=0.4,
        color=COLOR_DECREASE,
        label="Label prevalence",
    )
    ax.bar(
        x + 0.2,
        subset["predicted_review_rate"],
        width=0.4,
        color=COLOR_INCREASE,
        label="Predicted review rate",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["group_value"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(f"Validation prevalence vs predicted review rate ({column})")
    ax.legend()
    return _save(fig, path)
