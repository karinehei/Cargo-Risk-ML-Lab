"""Select a champion from previously logged train/validation experiment records."""

from __future__ import annotations

import json

from src.config import resolve_path, setup_logging
from src.mlops.champion import save_champion, select_champion


def main() -> None:
    logger = setup_logging(name="scripts.select_champion")
    records_path = resolve_path("artifacts/mlops/experiment_records.json")
    if not records_path.exists():
        raise SystemExit("No experiment records. Run python -m scripts.run_mlops first.")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    calibrated = any(
        str(item.get("calibration_status")) not in {"none", "", "None"} for item in records
    )
    champion = select_champion(records, calibrated_in_pool=calibrated)
    path = save_champion(champion)
    logger.info("Champion %s run_id=%s -> %s", champion.model_name, champion.mlflow_run_id, path)
    print(json.dumps(champion.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
