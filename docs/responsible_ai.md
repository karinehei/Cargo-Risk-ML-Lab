# Responsible AI Analysis

> **Disclaimer:** Cargo Risk ML Lab is an educational demonstration using fully synthetic data and synthetic review logic. It is **not affiliated with the Finnish Customs Service or any other authority**, and it must **not** be used for operational decision-making or automated adverse actions.

## Purpose

Record how responsible-AI themes are handled in a portfolio setting where the problem, labels and “review” outcome are intentionally fictional.

## Intended vs prohibited use

| Allowed | Prohibited |
|---|---|
| Portfolio demonstration, teaching, code review | Real inspection targeting or enforcement |
| Synthetic scoring with a human-review metaphor | Automated denial, delay or penalty |
| Research on calibration, explanation and subgroup reporting | Claiming fairness certification or operational readiness |

Human oversight is mandatory in any analogue of this task: a score can only suggest **additional human review**, with an appeal or second-look path. This repository does not implement casework, appeals or legal process.

## Key risks if this were mistaken for a real system

1. **Authority confusion** – Repeated disclaimers in README, config, API, Streamlit, model card and metadata.
2. **Score–probability confusion** – The champion is uncalibrated (validation Brier ≈ 0.238, ECE ≈ 0.355). APIs and the UI expose `review_score` plus an explicit calibration warning.
3. **False confidence** – Validation PR-AUC ≈ 0.227 is modest. Precision at the operating point is about 0.20, so most flags are extra reviews.
4. **Automation bias** – Responses include a disclaimer; the UI never labels a shipment as dangerous or fraudulent.
5. **Proxy variables** – Origin/destination, commodity, route rarity and sender history could proxy sensitive attributes in real data. Here they are toy generator fields, not personal characteristics.
6. **Explanation overreach** – Local logit contributions explain the model, not the world.

## Frozen v1 operational interpretation (synthetic test only)

At threshold 0.525, the frozen characterisation implies roughly:

- 311 reviews per 1,000 shipments
- 62 true positives
- 249 additional reviews (false positives)
- 67 missed positives

These are **not** real-world expected performance.

## Fairness and subgroups

Validation subgroup tables report sample size, prevalence, predicted-review rate, precision, recall, F1, FPR, FNR and mean review score, with Wilson intervals and a minimum-n warning.

- Differences are **observed descriptive gaps**, not proof of unfairness.
- Passing subgroup checks **does not prove fairness**.
- No protected personal characteristics are in the data.
- Subgroup metrics are **not** used to retune the champion.

## Privacy and data minimisation

Only synthetic shipment attributes required for the toy score are modelled. Identifiers, dates and generation period are excluded from the model. Forbidden columns include latent scores and personal attributes (`src/data/schema.py`).

## Monitoring and drift

Default operational monitoring is **unlabelled**: input data drift and review-score drift only. Performance degradation is reported only in a separate **labelled simulation** mode with synthetic delayed labels.

- Reference profiles are built from train-derived samples; the frozen test set is never used.
- Raw monitoring CSVs (`data/monitoring/*.csv`) are local caches, gitignored and excluded from Docker images. Recreate the reference sample with `make monitoring-reference`.
- PSI/KS/JS metrics in `src/monitoring/metrics.py` are the source of truth; policy aggregation lives in `src/monitoring/policy.py`.
- Drift severity uses effect-size thresholds; p-values alone do not trigger alerts.
- Policy 1.1.0 records isolated weak warnings without automatically raising overall warning, to limit multiple-comparison false alerts.
- Input drift does not automatically imply model failure.
- Missing or failed monitoring is reported as `insufficient_data` or `monitoring_error`, never as healthy.
- Monitoring never retrains, changes the threshold or blocks shipments.

See `docs/monitoring.md`, `docs/drift_metrics.md` and `docs/operations.md`.

## Human oversight and uncertainty

Threshold 0.525 is a fictional F-beta operating point. Changing it trades missed positives against review burden. Costs are not identified, so the threshold is not a policy recommendation. A human must remain in the loop.

## Mitigations in this repository

| Theme | Implementation |
|---|---|
| Transparency | Model card, limitations, explainability doc, API `/model` score semantics |
| Reproducibility | Seeds, YAML, MLflow run IDs |
| Explainability | Exact logistic logit decomposition + validation permutation importance |
| Evaluation honesty | Frozen v1 preserved; no new test pass in this phase |
| Scope limitation | Persistent disclaimers; neutral “additional human review” language |
