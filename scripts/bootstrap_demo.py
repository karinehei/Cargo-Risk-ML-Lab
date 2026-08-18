"""Isolated clean-clone bootstrap. Never evaluates frozen test metrics.

Full mode (`make bootstrap-demo`) writes to artifacts/bootstrap/ and does not
replace the portfolio champion under artifacts/mlops/. CI mode uses configs/ci.yaml
and .ci-work/; those metrics are not frozen-v1 results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from src.config import PROJECT_ROOT
from src.mlops.integrity import (
    config_fingerprint,
    sha256_file,
    write_reproducibility_manifest,
)
from src.mlops.tracking import current_git_commit

FROZEN_FORBIDDEN = ("metrics_test.json",)
PORTFOLIO_CHAMPION = PROJECT_ROOT / "artifacts" / "mlops" / "champion.json"


def _run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    blob = " ".join(command)
    if "evaluate_model" in blob:
        raise RuntimeError("Bootstrap must not evaluate the held-out test set")
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)  # nosec B603
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {blob}")


def _python() -> str:
    return sys.executable


def _validate_environment() -> dict[str, str]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, found {sys.version}")
    import mlflow
    import numpy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }


def _assert_no_frozen_access(source_paths: list[Path]) -> None:
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        for token in FROZEN_FORBIDDEN:
            if token in text:
                raise RuntimeError(f"Bootstrap source {path} must not reference {token}")


def _write_isolated_full_config(work: Path) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    prefix = "artifacts/bootstrap"
    raw["data"]["raw_path"] = f"{prefix}/data/raw/synthetic_shipments.csv"
    raw["data"]["processed_dir"] = f"{prefix}/data/processed"
    raw["data"]["validation_report_path"] = f"{prefix}/data/raw/validation_report.json"
    raw["training"]["artifact_dir"] = f"{prefix}/artifacts"
    raw["training"]["mlflow_experiment"] = "cargo-risk-ml-lab-bootstrap"
    raw["mlops"]["experiment_name"] = "cargo-risk-ml-lab-bootstrap"
    raw["mlops"]["champion_path"] = f"{prefix}/artifacts/mlops/champion.json"
    raw["mlops"]["artifact_dir"] = f"{prefix}/artifacts/mlops"
    raw["evaluation"]["plots_dir"] = f"{prefix}/artifacts/plots"
    raw["explainability"]["output_dir"] = f"{prefix}/artifacts/explanations"
    raw["monitoring"]["reference_dataset_path"] = f"{prefix}/data/processed/train.csv"
    raw["monitoring"]["reference_sample_path"] = f"{prefix}/data/monitoring/reference_sample.csv"
    raw["monitoring"]["reference_profile_path"] = (
        f"{prefix}/artifacts/monitoring/reference_profile.json"
    )
    raw["monitoring"]["scenario_dir"] = f"{prefix}/data/monitoring"
    raw["monitoring"]["report_dir"] = f"{prefix}/artifacts/monitoring"
    raw["api"]["model_path"] = f"{prefix}/artifacts/model.joblib"
    raw["api"]["pipeline_path"] = f"{prefix}/artifacts/preprocess_pipeline.joblib"
    raw["api"]["metadata_path"] = f"{prefix}/artifacts/model_metadata.json"
    path = work / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _verify_champion_roundtrip(env: dict[str, str]) -> None:
    code = (
        "from src.config import get_settings, get_config; "
        "get_settings.cache_clear(); get_config.cache_clear(); "
        "from src.mlops.serving import load_champion; "
        "bundle = load_champion(); "
        "assert bundle.threshold > 0; print(bundle.metadata.get('mlflow_run_id'))"
    )
    _run([_python(), "-c", code], env, PROJECT_ROOT)


def _lock_checksum() -> str | None:
    lock = PROJECT_ROOT / "requirements" / "dev.lock.txt"
    runtime = PROJECT_ROOT / "requirements" / "runtime.lock.txt"
    target = lock if lock.exists() else runtime
    if not target.exists():
        return None
    return sha256_file(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated clean bootstrap (no frozen test eval)")
    parser.add_argument("--mode", choices=("full", "ci"), default="full")
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    versions = _validate_environment()
    _assert_no_frozen_access(
        [
            PROJECT_ROOT / "scripts" / "run_mlops.py",
            PROJECT_ROOT / "scripts" / "explain_model.py",
            PROJECT_ROOT / "scripts" / "run_monitoring.py",
        ]
    )

    if args.mode == "ci":
        work = PROJECT_ROOT / ".ci-work"
        config_path = PROJECT_ROOT / "configs" / "ci.yaml"
        tracking = (
            "sqlite:///" + (work.relative_to(PROJECT_ROOT) / "mlruns" / "mlflow.db").as_posix()
        )
        champion = work / "artifacts" / "mlops" / "champion.json"
        experiment = "cargo-risk-ml-lab-ci"
    else:
        work = PROJECT_ROOT / "artifacts" / "bootstrap"
        config_path = _write_isolated_full_config(work)
        tracking = (
            "sqlite:///" + (work.relative_to(PROJECT_ROOT) / "mlruns" / "mlflow.db").as_posix()
        )
        champion = work / "artifacts" / "mlops" / "champion.json"
        experiment = "cargo-risk-ml-lab-bootstrap"

    if work.exists() and args.mode == "ci":
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "mlruns").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(config_path)
    env["MLFLOW_TRACKING_URI"] = tracking
    env["MLFLOW_EXPERIMENT_NAME"] = experiment
    env["CHAMPION_PATH"] = str(champion)
    env["ARTIFACTS_DIR"] = str(work / "artifacts") if args.mode == "ci" else str(work)
    env.pop("SKOPS_ALLOW_UNTRUSTED", None)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    commands = [
        [_python(), "-m", "scripts.generate_data"],
        [_python(), "-m", "scripts.validate_data"],
        [_python(), "-m", "scripts.init_mlflow"],
        [_python(), "-m", "scripts.run_mlops"],
        [_python(), "-m", "scripts.explain_model"],
        [_python(), "-m", "scripts.run_monitoring", "create-reference"],
        [_python(), "-m", "scripts.run_monitoring", "generate-scenario", "none"],
        [_python(), "-m", "scripts.run_monitoring", "run-unlabelled", "none"],
    ]

    started = time.perf_counter()
    executed: list[str] = []
    for command in commands:
        if args.mode == "full" and command[-2:] == ["generate-data"]:
            # Full demo uses default n_samples from configs/default.yaml.
            pass
        _run(command, env, PROJECT_ROOT)
        executed.append(" ".join(command))
    _verify_champion_roundtrip(env)
    executed.append("load_champion round-trip")

    if PORTFOLIO_CHAMPION.exists() and champion.resolve() == PORTFOLIO_CHAMPION.resolve():
        raise RuntimeError("Bootstrap refused to overwrite the portfolio champion path")

    api_ok = False
    if not args.skip_api:
        api_ok = _check_api(env)
        if not api_ok:
            raise RuntimeError("API readiness check failed")

    elapsed = time.perf_counter() - started
    champion_meta: dict[str, Any] = {}
    if champion.exists():
        champion_meta = json.loads(champion.read_text(encoding="utf-8"))
        if champion_meta.get("threshold") is None:
            raise RuntimeError("Bootstrap champion is missing a threshold")

    train_csv = work / "data" / "processed" / "train.csv"
    dataset_fp = sha256_file(train_csv) if train_csv.exists() else None
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "note": (
            "CI smoke metrics are not frozen-v1 results."
            if args.mode == "ci"
            else "Isolated bootstrap; portfolio frozen-v1 and champion were not replaced."
        ),
        "python_version": versions["python"],
        "dependency_versions": versions,
        "dependency_lock_sha256": _lock_checksum(),
        "random_seed": 42,
        "dataset_fingerprint": dataset_fp,
        "configuration_fingerprint": config_fingerprint(config_path),
        "source_revision": current_git_commit(),
        "champion_run_id": champion_meta.get("mlflow_run_id"),
        "champion_version": champion_meta.get("model_version"),
        "champion_threshold": champion_meta.get("threshold"),
        "commands": executed,
        "elapsed_seconds": round(elapsed, 3),
        "api_ready": api_ok,
        "evaluated_frozen_test": False,
    }
    manifest_path = work / "reproducibility_manifest.json"
    write_reproducibility_manifest(manifest, manifest_path)
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), **manifest}, indent=2))


def _check_api(env: dict[str, str]) -> bool:
    import urllib.error
    import urllib.request

    proc = subprocess.Popen(  # nosec B603
        [
            _python(),
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8010",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(  # nosec B310
                    "http://127.0.0.1:8010/ready", timeout=2
                ) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.5)
        return False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
