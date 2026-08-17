"""Print the registered champion metadata (no test evaluation)."""

from __future__ import annotations

import json

from src.config import resolve_path, setup_logging


def main() -> None:
    logger = setup_logging(name="scripts.show_champion")
    path = resolve_path("artifacts/mlops/champion.json")
    if not path.exists():
        raise SystemExit("Champion metadata is unavailable. Run python -m scripts.run_mlops first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Champion %s (%s)", payload.get("model_name"), payload.get("mlflow_run_id"))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
