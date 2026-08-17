Generated model artifacts, plots, explanations and monitoring reports are written here after running the training and evaluation scripts.

`frozen_v1/` is the preserved original test characterisation and must not be overwritten.
`mlops/` holds champion metadata, calibration tables and joblib recovery exports.
`explanations/` holds validation-only coefficient tables, permutation importance, local logit explanations and subgroup reports.

This directory's contents (except placeholders) are gitignored. MLflow databases live in `mlruns/` at the project root, also gitignored.
