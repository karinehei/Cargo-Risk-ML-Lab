"""CLI script: generate synthetic cargo shipment data."""

from __future__ import annotations

import argparse
from dataclasses import replace

from src.config import get_config, set_seed, setup_logging
from src.data import build_and_persist_splits
from src.data.schema import DISCLAIMER, TARGET_COLUMN


def main() -> None:
    """Generate, validate and persist synthetic train/val/test splits."""
    parser = argparse.ArgumentParser(
        description="Generate fully synthetic cargo review data (educational only)."
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Override row count")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    logger = setup_logging(name="scripts.generate_data")
    cfg = get_config()
    if args.n_samples is not None:
        cfg.data["n_samples"] = args.n_samples
    if args.seed is not None:
        cfg = replace(cfg, random_seed=args.seed)

    set_seed(cfg.random_seed)
    logger.info("%s", DISCLAIMER)
    bundle = build_and_persist_splits(cfg)
    positive = float(bundle.full[TARGET_COLUMN].mean())
    logger.info(
        "Done. rows=%s train=%s val=%s test=%s positive_rate=%.4f",
        len(bundle.full),
        len(bundle.train),
        len(bundle.val),
        len(bundle.test),
        positive,
    )


if __name__ == "__main__":
    main()
