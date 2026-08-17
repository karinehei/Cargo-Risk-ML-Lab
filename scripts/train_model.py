"""CLI script: train/validation MLOps experiments (no test evaluation).

This entrypoint now runs comparison, calibration and champion selection via
``scripts.run_mlops``. Frozen v1 artifacts are not modified.
"""

from __future__ import annotations

from scripts.run_mlops import main

if __name__ == "__main__":
    main()
