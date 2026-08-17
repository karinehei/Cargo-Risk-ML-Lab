"""Shared helpers to prepare inference feature frames."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.preprocess import add_derived_features, get_feature_lists


def prepare_inference_frame(
    rows: list[dict[str, Any]],
    derived_stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a feature frame aligned with the training feature contract.

    Args:
        rows: Raw shipment feature dictionaries.
        derived_stats: Unused placeholder kept for backward compatibility.
    """
    _ = derived_stats
    frame = add_derived_features(pd.DataFrame(rows).copy())
    numeric, categorical = get_feature_lists()
    return frame[[*numeric, *categorical]]
