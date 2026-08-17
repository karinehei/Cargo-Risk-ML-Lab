"""Explicit, versioned champion selection from validation evidence only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig, get_config, resolve_path, setup_logging
from src.mlops.tracking import current_git_commit

logger = setup_logging(name="src.mlops.champion")

POLICY_VERSION = "1.0.0"
DEFAULT_SIMPLICITY_ORDER = [
    "dummy",
    "logreg",
    "logreg_uncalibrated_weighted",
    "logreg_unweighted",
    "logreg_sigmoid",
    "logreg_isotonic",
    "random_forest",
    "xgboost",
]


@dataclass
class ChampionRecord:
    """Machine-readable champion metadata. Contains no test-set metrics."""

    model_name: str
    model_version: str
    mlflow_run_id: str
    dataset_fingerprint: str
    threshold: float
    threshold_selection_method: str
    calibration_status: str
    validation_metrics: dict[str, float]
    artifact_uri: str
    created_at: str
    git_commit: str
    policy_version: str
    reason: str
    awaiting_authorized_v2_test: bool
    test_evaluation_note: str
    serialization: str
    roundtrip_ok: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_policy(config: AppConfig | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    raw = dict(cfg.mlops.get("champion_policy") or {})
    return {
        "version": str(raw.get("version", POLICY_VERSION)),
        "min_validation_recall": float(raw.get("min_validation_recall", 0.40)),
        "min_validation_pr_auc": float(raw.get("min_validation_pr_auc", 0.15)),
        "pr_auc_indifference": float(raw.get("pr_auc_indifference", 0.005)),
        "max_latency_p99_ms": float(raw.get("max_latency_p99_ms", 100.0)),
        "require_roundtrip": bool(raw.get("require_roundtrip", True)),
        "simplicity_order": list(raw.get("simplicity_order") or DEFAULT_SIMPLICITY_ORDER),
    }


def _reject_test_metrics(candidate: dict[str, Any]) -> None:
    metrics = candidate.get("validation_metrics") or {}
    if candidate.get("split") == "test":
        raise ValueError("Test-split records cannot enter champion selection")
    keys = list(metrics.keys()) + list(candidate.keys())
    if any(str(key).startswith("test_") for key in keys):
        raise ValueError("Test metrics cannot enter champion selection")


def _family_rank(name: str, order: list[str]) -> int:
    if name in order:
        return order.index(name)
    for index, item in enumerate(order):
        if name.startswith(item):
            return index
    return len(order) + 1


def select_champion(
    candidates: list[dict[str, Any]],
    config: AppConfig | None = None,
    *,
    calibrated_in_pool: bool = False,
) -> ChampionRecord:
    """Pick a champion with explicit sequential rules (no opaque score).

    1. Drop runs that failed serialization round-trip when required.
    2. Drop runs below minimum validation recall at their selected threshold.
    3. Drop runs below minimum validation PR-AUC.
    4. Drop runs above the p99 latency ceiling.
    5. Prefer higher validation PR-AUC.
    6. If PR-AUC is within the indifference band, prefer the simpler family.
    7. If still tied, prefer lower Brier then lower p99 latency.

    Calibration (Brier/ECE) never overrides a material PR-AUC or recall gap.
    """
    if not candidates:
        raise ValueError("No candidates supplied for champion selection")
    for candidate in candidates:
        _reject_test_metrics(candidate)

    policy = default_policy(config)
    eligible: list[dict[str, Any]] = []
    rejections: list[str] = []
    for candidate in candidates:
        name = str(candidate["model_family"])
        metrics = candidate["validation_metrics"]
        if policy["require_roundtrip"] and not candidate.get("roundtrip_ok"):
            rejections.append(f"{name}: failed MLflow sklearn round-trip")
            continue
        if float(metrics.get("val_recall", 0.0)) < policy["min_validation_recall"]:
            rejections.append(f"{name}: recall below {policy['min_validation_recall']}")
            continue
        if float(metrics.get("val_pr_auc", 0.0)) < policy["min_validation_pr_auc"]:
            rejections.append(f"{name}: PR-AUC below {policy['min_validation_pr_auc']}")
            continue
        if float(metrics.get("latency_p99_ms", 0.0)) > policy["max_latency_p99_ms"]:
            rejections.append(f"{name}: p99 latency above {policy['max_latency_p99_ms']} ms")
            continue
        eligible.append(candidate)

    if not eligible:
        raise ValueError("No candidate met the champion policy gates: " + "; ".join(rejections))

    order = list(policy["simplicity_order"])
    eligible.sort(
        key=lambda item: (
            -float(item["validation_metrics"]["val_pr_auc"]),
            _family_rank(str(item["model_family"]), order),
            float(item["validation_metrics"].get("val_brier", 1.0)),
            float(item["validation_metrics"].get("latency_p99_ms", 0.0)),
        )
    )
    leader = eligible[0]
    leader_pr = float(leader["validation_metrics"]["val_pr_auc"])
    simpler = [
        item
        for item in eligible
        if _family_rank(str(item["model_family"]), order)
        < _family_rank(str(leader["model_family"]), order)
        and abs(float(item["validation_metrics"]["val_pr_auc"]) - leader_pr)
        <= policy["pr_auc_indifference"]
    ]
    chosen = (
        min(simpler, key=lambda item: _family_rank(str(item["model_family"]), order))
        if simpler
        else leader
    )

    reason_parts = [
        f"policy {policy['version']}",
        f"val PR-AUC={chosen['validation_metrics']['val_pr_auc']:.4f}",
        f"val recall={chosen['validation_metrics']['val_recall']:.4f}",
        f"val Brier={chosen['validation_metrics'].get('val_brier', float('nan')):.4f}",
        f"serialization={chosen.get('serialization')}",
    ]
    if chosen is not leader:
        reason_parts.append(
            f"preferred simpler family within PR-AUC indifference {policy['pr_auc_indifference']}"
        )
    if (
        str(chosen["model_family"]).startswith("logreg")
        and chosen.get("calibration_status") == "none"
    ):
        reason_parts.append(
            "simple weighted logistic regression remains eligible; a simpler model winning is valid"
        )
    reason = "; ".join(reason_parts)
    if rejections:
        logger.info("Champion rejections: %s", rejections)

    version = (
        f"{chosen['model_family']}-{chosen.get('calibration_status', 'none')}-{policy['version']}"
    )
    awaiting = bool(calibrated_in_pool and chosen.get("calibration_status") not in {"none", None})
    note = (
        "Awaiting independently authorized v2 test characterisation. Frozen v1 remains the only test result."
        if awaiting
        else "Frozen v1 remains the only authorised test characterisation. This selection used validation only."
    )
    return ChampionRecord(
        model_name=str(chosen["model_family"]),
        model_version=version,
        mlflow_run_id=str(chosen["run_id"]),
        dataset_fingerprint=str(chosen.get("dataset_fingerprint", "")),
        threshold=float(chosen["threshold"]),
        threshold_selection_method=str(chosen.get("threshold_policy", "validation")),
        calibration_status=str(chosen.get("calibration_status", "none")),
        validation_metrics={
            key: float(value)
            for key, value in chosen["validation_metrics"].items()
            if isinstance(value, (int, float))
        },
        artifact_uri=str(chosen["artifact_uri"]),
        created_at=datetime.now(UTC).isoformat(),
        git_commit=str(chosen.get("git_commit") or current_git_commit()),
        policy_version=str(policy["version"]),
        reason=reason,
        awaiting_authorized_v2_test=awaiting,
        test_evaluation_note=note,
        serialization=str(chosen.get("serialization", "")),
        roundtrip_ok=bool(chosen.get("roundtrip_ok")),
        extras={"rejections": rejections, "policy": policy},
    )


def save_champion(record: ChampionRecord, path: str | Path | None = None) -> Path:
    """Persist champion metadata under artifacts/mlops (never frozen_v1)."""
    import json

    out = resolve_path(path or "artifacts/mlops/champion.json")
    if "frozen_v1" in str(out):
        raise ValueError("Champion metadata must not be written under artifacts/frozen_v1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")
    logger.info("Wrote champion metadata to %s", out)
    return out
