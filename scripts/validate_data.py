"""CLI script: validate a generated synthetic dataset."""

from __future__ import annotations

import argparse
import sys

from src.config import get_config, resolve_path, setup_logging
from src.data import load_dataset, validate_dataset
from src.data.schema import DISCLAIMER


def main() -> None:
    """Load a CSV, run validation, and print a JSON report."""
    parser = argparse.ArgumentParser(description="Validate synthetic cargo review data.")
    parser.add_argument(
        "--path",
        default=None,
        help="CSV path (defaults to configs data.raw_path)",
    )
    args = parser.parse_args()

    logger = setup_logging(name="scripts.validate_data")
    cfg = get_config()
    path = resolve_path(
        args.path or str(cfg.data.get("raw_path", "data/raw/synthetic_shipments.csv"))
    )
    logger.info("%s", DISCLAIMER)
    logger.info("Validating %s", path)

    df = load_dataset(path, validate=False)
    report = validate_dataset(df, config=cfg, raise_on_error=False)
    print(report.to_json())
    if not report.ok:
        logger.error("Validation failed: %s", report.issues)
        sys.exit(1)
    logger.info(
        "Validation passed: n=%s positive_rate=%.4f",
        report.n_rows,
        report.positive_rate,
    )


if __name__ == "__main__":
    main()
