"""Verify the public Pages bundle without loading models or datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pages.verify import verify_site

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the public Pages evidence bundle")
    parser.add_argument(
        "--site",
        type=Path,
        default=PROJECT_ROOT / "site",
        help="Path to the static site directory",
    )
    args = parser.parse_args()
    verify_site(args.site)
    print("pages_evidence_verified")


if __name__ == "__main__":
    main()
