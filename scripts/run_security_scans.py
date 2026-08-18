#!/usr/bin/env python3
"""Local verification helpers. Does not print secret values."""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)  # nosec B603


def main() -> None:
    py = sys.executable
    print("=== BANDIT ===")
    bandit = run([py, "-m", "bandit", "-c", "bandit.yaml", "-r", "src", "app", "scripts", "-q"])
    print(bandit.stdout)
    if bandit.returncode != 0:
        print((bandit.stderr or "")[-2000:])
        raise SystemExit(f"bandit failed: {bandit.returncode}")

    print("=== PIP-AUDIT ===")
    audit = run(
        [py, "-m", "pip_audit", "-r", "requirements/dev.lock.txt", "--progress-spinner", "off"]
    )
    print(audit.stdout[-4000:] if audit.stdout else audit.stderr[-4000:])
    print("pip-audit_exit", audit.returncode)

    print("=== DETECT-SECRETS ===")
    scan = run(
        [
            py,
            "-m",
            "detect_secrets",
            "scan",
            "--exclude-files",
            r"\.ci-work/.*|mlruns/.*|artifacts/.*|.*\.lock\.txt|sbom.*|coverage.xml|pip-licenses.json",
        ]
    )
    if scan.returncode not in {0, 1}:
        print(scan.stderr[-1500:])
        raise SystemExit(f"detect-secrets failed: {scan.returncode}")
    baseline = json.loads(scan.stdout)
    (ROOT / ".secrets.baseline").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print("plugins", len(baseline.get("plugins_used", [])))
    print("result_files", len(baseline.get("results", {})))
    print("generated_at", baseline.get("generated_at"))

    print("=== SBOM ===")
    sbom = run(
        [
            str(Path(py).parent / "cyclonedx-py"),
            "environment",
            "-o",
            "sbom.cdx.json",
            "--pyproject",
            "pyproject.toml",
        ]
    )
    if sbom.returncode != 0:
        print((sbom.stderr or sbom.stdout)[-2000:])
        print("sbom_blocked")
    else:
        print("sbom", (ROOT / "sbom.cdx.json").stat().st_size, "bytes")

    print("=== FROZEN HASH ===")
    frozen = ROOT / "artifacts" / "frozen_v1" / "metrics_test.json"
    print("exists", frozen.exists())
    if frozen.exists():
        print("sha256", hashlib.sha256(frozen.read_bytes()).hexdigest())

    print("=== CHAMPION ===")
    champion = ROOT / "artifacts" / "mlops" / "champion.json"
    if champion.exists():
        meta = json.loads(champion.read_text(encoding="utf-8"))
        print("model_version", meta.get("model_version"))
        print("threshold", meta.get("threshold"))
        print("mlflow_run_id", meta.get("mlflow_run_id"))
        print("calibration_status", meta.get("calibration_status"))
    else:
        print("champion_missing")

    print("=== REPO AUDIT ===")
    audit_repo = run([py, "-m", "scripts.audit_repository_boundary"])
    print(audit_repo.stdout)
    if audit_repo.returncode != 0:
        print(audit_repo.stderr)
        raise SystemExit("repository audit failed")

    print("=== GIT HISTORY NAMES ===")
    names = run(["git", "log", "--all", "--pretty=format:", "--name-only"])
    hits = sorted(
        {
            line
            for line in names.stdout.splitlines()
            if line.endswith((".pkl", ".joblib", ".sqlite", ".env"))
            or line.endswith("mlflow.db")
            or line.endswith(".env.local")
        }
    )
    print("forbidden_history_paths", hits)


if __name__ == "__main__":
    main()
