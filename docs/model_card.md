# Model Card – Cargo Risk ML Lab

> **Disclaimer:** This model card describes an educational model trained exclusively on synthetic data with synthetic review labels. It does not represent any customs authority, real border process, or operational risk system. It is **not affiliated with or endorsed by Finnish Customs or any other authority**.

## Model details

| Field | Value |
|---|---|
| Name | Cargo Risk ML Lab review-score classifier |
| Champion version | `logreg-none-1.0.0` |
| Type | Binary flag for **additional human review** (`requires_review`) |
| Algorithm | Class-weighted logistic regression in a scikit-learn `Pipeline` (impute, scale, one-hot, `LogisticRegression`) |
| Operating threshold | `0.525` (validation F-beta, β=2, min precision 0.20) |
| Score semantics | Ranking and threshold score, **not a literal probability** |
| Calibrated | `false` (`calibration_method`: none) |
| Frameworks | scikit-learn, MLflow, optional SHAP for comparison models only |
| Intended users | Recruiters, hiring managers, students reviewing an ML portfolio |
| Out of scope | Any real customs, compliance, law-enforcement, underwriting or automated adverse decision |

MLflow run identity and validation metrics live in `artifacts/mlops/champion.json`. This card does not replace that file.

## Intended use

Demonstrate data generation, leakage-aware training, local MLOps, **transparent linear explainability**, subgroup reporting and API/UI packaging on a fully synthetic problem.

## Prohibited use

- Operational targeting, inspection selection or enforcement
- Automated denial, delay, seizure or other adverse action
- Any use on real personal or shipment data
- Treating the review score as a calibrated probability of “risk”, “fraud” or “danger”

## Training data

- Fully synthetic shipment records from `src/data/generate.py` (default 15,000 rows).
- No protected personal characteristics are generated (no gender, age, nationality, ethnicity, race, religion, disability, name or contact identifiers).
- Labels are sampled from fictional toy rules plus noise. **They are not domain-accurate review logic.**
- See `docs/data_dictionary.md`.

## Evaluation data

- **Validation** is used for threshold choice, calibration comparison, champion policy, explanations and subgroup tables.
- **Frozen v1 test** remains the only authorised held-out characterisation. It was not reused to tune or replace the champion in the explainability phase.

### Validation (champion, not a test pass)

Approximate figures from the registered champion metadata:

- PR-AUC ≈ 0.227
- Brier ≈ 0.238
- ECE ≈ 0.355
- Precision / recall at 0.525 ≈ 0.214 / 0.536

### Frozen v1 synthetic test characterisation (not used for this phase)

Point estimates from `artifacts/frozen_v1/metrics_test.json` (threshold 0.525):

- PR-AUC ≈ 0.204; ROC-AUC ≈ 0.628
- Precision ≈ 0.200; recall ≈ 0.482; F1 ≈ 0.283
- Confusion: TP 187, FP 746, FN 201, TN 1866 on 3,000 rows

**Operational sketch per 1,000 shipments (synthetic frozen test, not expected live performance):** about **311 reviews**, **62 true positives**, **249 additional reviews**, **67 missed positives**.

These numbers describe the toy generator and the frozen pipeline. They are not a forecast for any real corridor.

## Metrics tracked

ROC-AUC, PR-AUC, precision, recall, F1, Brier, ECE, latency percentiles, validation-only threshold selection, linear coefficients, permutation importance, subgroup tables with small-sample warnings.

## Ethical / responsible AI considerations

See `docs/responsible_ai.md` and `docs/limitations.md`.

## Serving

Local FastAPI endpoints: `GET /health`, `GET /ready`, `GET /model`, `POST /predict`, `POST /predict/batch`, `POST /explain`. The review score remains uncalibrated. See `docs/api.md`.

## Monitoring

Unlabelled monitoring compares deterministic monitoring batches to a train-derived reference profile (`make monitoring-reference`, `make monitoring-run`). Reports include policy version, status (`insufficient_data` / `no_material_drift` / `warning` / `critical` / `monitoring_error`) and machine-readable alert reasons. Raw monitoring CSVs are local-only. Performance degradation requires delayed labels and is available only in a separate labelled simulation mode. See `docs/monitoring.md`.

## Caveats

- Modest ranking quality and **low positive precision** mean most flags are extra human reviews.
- Class weighting improves recall ranking while **worsening calibration**; do not read the score as P(review).
- Coefficients and permutation ranks can disagree because of correlated transformed features.
- Explanations are not causal. This system must not take automated adverse decisions.
