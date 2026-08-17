"""Initialize or migrate the local SQLite MLflow tracking store."""

from __future__ import annotations

from src.config import setup_logging
from src.mlops.tracking import init_tracking_store


def main() -> None:
    logger = setup_logging(name="scripts.init_mlflow")
    path = init_tracking_store()
    logger.info("MLflow tracking store ready at %s", path)


if __name__ == "__main__":
    main()
