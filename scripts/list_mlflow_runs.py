"""List local MLflow runs for the default experiment."""

from __future__ import annotations

import mlflow
from src.config import get_settings, setup_logging
from src.mlops.tracking import configure_tracking


def main() -> None:
    logger = setup_logging(name="scripts.list_mlflow_runs")
    configure_tracking()
    settings = get_settings()
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        logger.info("Experiment %s does not exist", settings.mlflow_experiment_name)
        return
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if runs.empty:
        logger.info("No runs found")
        return
    preferred = [
        "run_id",
        "tags.mlflow.runName",
        "params.model_family",
        "metrics.val_pr_auc",
        "status",
    ]
    cols = [col for col in preferred if col in runs.columns]
    print(runs[cols].to_string(index=False))


if __name__ == "__main__":
    main()
