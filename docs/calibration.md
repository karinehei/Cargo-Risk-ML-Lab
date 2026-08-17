# Probability calibration (train / validation only)

> **Disclaimer:** These comparisons use synthetic labels. Frozen v1 test metrics are not updated here.

## Why this experiment exists

The frozen weighted logistic regression ranked first on validation PR-AUC (~0.227) but was poorly calibrated (validation Brier ~0.238, ECE ~0.355). Class weighting shifts scores upward, which can help ranking while hurting probability reliability. Serving and the dashboard therefore expose a **review score**, not a literal probability.

Monotonic calibration (Platt scaling / isotonic regression) can improve Brier and ECE **without materially changing ranking metrics**, because it is a monotone map of scores. That is not automatic permission to promote a calibrated model.

## Leakage-safe procedure

`CalibratedClassifierCV` is fitted on **training rows only**, with stratified 3-fold CV inside the training fold. Validation is used afterwards to compare already-fitted candidates. The frozen test set is not read.

Candidates:

1. Uncalibrated weighted logistic regression (current champion family)
2. Sigmoid (Platt) calibration of that estimator
3. Isotonic calibration, only if train n ≥ 1000 and positives ≥ 50 (the 10,500-row training fold qualifies)
4. Unweighted logistic regression as a calibration-oriented reference

## What is reported

For each candidate, on **validation**:

- PR-AUC (threshold-free)
- Brier score and ECE
- calibration curve plots under `artifacts/mlops/calibration_plots/`
- precision and recall at the **frozen** operational threshold 0.525
- precision and recall at a **new** validation-only F-beta threshold (β=2, min precision 0.20)

Do **not** promote a model only because Brier improved. The champion policy still requires acceptable PR-AUC and recall. If a calibrated model is selected on validation, it is labelled `awaiting_authorized_v2_test`. Frozen v1 remains the only test result until an independent test characterisation is authorised.

## Command

Calibration runs as part of:

```bash
make experiments
```

Results: `artifacts/mlops/calibration_comparison.csv` and MLflow runs named `calibrate_*`.
