"""Champion-policy facts for the public comparison table.

The ranking rules match ``src.mlops.champion.select_champion`` for policy
1.0.0. This module does not write a champion file and does not import MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def family_rank(name: str, order: list[str]) -> int:
    """Return the simplicity rank used by the champion policy."""
    if name in order:
        return order.index(name)
    for index, item in enumerate(order):
        if name.startswith(item):
            return index
    return len(order) + 1


def gate_reason(candidate: dict[str, Any], policy: dict[str, Any]) -> str | None:
    """Return the policy rejection string, or None when the candidate is eligible."""
    name = str(candidate["model_family"])
    metrics = candidate["validation_metrics"]
    if policy["require_roundtrip"] and not candidate.get("roundtrip_ok"):
        return f"{name}: failed MLflow sklearn round-trip"
    if float(metrics.get("val_recall", 0.0)) < float(policy["min_validation_recall"]):
        return f"{name}: recall below {policy['min_validation_recall']}"
    if float(metrics.get("val_pr_auc", 0.0)) < float(policy["min_validation_pr_auc"]):
        return f"{name}: PR-AUC below {policy['min_validation_pr_auc']}"
    if float(metrics.get("latency_p99_ms", 0.0)) > float(policy["max_latency_p99_ms"]):
        return f"{name}: p99 latency above {policy['max_latency_p99_ms']} ms"
    return None


@dataclass(frozen=True)
class SelectionResult:
    """Chosen family and the eligible leader it was compared with."""

    chosen_family: str
    leader_family: str
    leader_validation_pr_auc: float
    chosen_validation_pr_auc: float
    indifference: float
    simpler_preference_applied: bool
    eligible_families: tuple[str, ...]
    rejections: tuple[str, ...]


def select_family(candidates: list[dict[str, Any]], policy: dict[str, Any]) -> SelectionResult:
    """Apply the saved champion gates and simplicity rule. Does not refit models."""
    if not candidates:
        raise ValueError("No candidates supplied for champion selection")
    eligible: list[dict[str, Any]] = []
    rejections: list[str] = []
    for candidate in candidates:
        reason = gate_reason(candidate, policy)
        if reason is None:
            eligible.append(candidate)
        else:
            rejections.append(reason)
    if not eligible:
        raise ValueError("No candidate met the champion policy gates")

    order = [str(item) for item in policy["simplicity_order"]]
    eligible.sort(
        key=lambda item: (
            -float(item["validation_metrics"]["val_pr_auc"]),
            family_rank(str(item["model_family"]), order),
            float(item["validation_metrics"].get("val_brier", 1.0)),
            float(item["validation_metrics"].get("latency_p99_ms", 0.0)),
        )
    )
    leader = eligible[0]
    leader_pr = float(leader["validation_metrics"]["val_pr_auc"])
    indifference = float(policy["pr_auc_indifference"])
    simpler = [
        item
        for item in eligible
        if family_rank(str(item["model_family"]), order)
        < family_rank(str(leader["model_family"]), order)
        and abs(float(item["validation_metrics"]["val_pr_auc"]) - leader_pr) <= indifference
    ]
    chosen = (
        min(simpler, key=lambda item: family_rank(str(item["model_family"]), order))
        if simpler
        else leader
    )
    return SelectionResult(
        chosen_family=str(chosen["model_family"]),
        leader_family=str(leader["model_family"]),
        leader_validation_pr_auc=leader_pr,
        chosen_validation_pr_auc=float(chosen["validation_metrics"]["val_pr_auc"]),
        indifference=indifference,
        simpler_preference_applied=chosen is not leader,
        eligible_families=tuple(str(item["model_family"]) for item in eligible),
        rejections=tuple(rejections),
    )
