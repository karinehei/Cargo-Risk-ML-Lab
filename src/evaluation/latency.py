"""Per-row inference latency measurement."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def measure_inference_latency(
    pipeline: Any,
    features: pd.DataFrame,
    *,
    repeats: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Time single-row ``predict_proba`` calls and return millisecond percentiles."""
    if features.empty:
        raise ValueError("Cannot measure latency on an empty feature frame")

    sample = features.sample(n=min(repeats, len(features)), replace=True, random_state=seed)
    pipeline.predict_proba(sample.iloc[:1])  # warmup

    times_ms: list[float] = []
    for index in range(len(sample)):
        row = sample.iloc[index : index + 1]
        start = perf_counter()
        pipeline.predict_proba(row)
        times_ms.append((perf_counter() - start) * 1000.0)

    values = np.asarray(times_ms, dtype=float)
    return {
        "n_repeats": float(len(values)),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
        "latency_mean_ms": float(np.mean(values)),
    }
