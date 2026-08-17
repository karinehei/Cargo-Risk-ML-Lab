# Limitations

> **Disclaimer:** Limitations below apply to a synthetic educational pipeline. They are not an operational risk register for any authority.

## Problem and data

- Labels come from toy rules plus noise. Patterns are properties of the generator, not of real trade.
- There is **no affiliation** with Finnish Customs or another agency.
- Synthetic volume (15,000 rows) and a single generator family understate real missingness, fraud adversarial behaviour and concept drift.

## Discriminatory performance

The champion’s validation PR-AUC is about **0.227** (ROC-AUC about 0.645). Ranking is better than a dummy prior but far from a high-separation detector. Modest performance is an expected outcome, not a hidden success.

## Precision, review burden and missed cases

At the fictional threshold 0.525:

- Positive precision is low (about 0.21 on validation; about 0.20 on frozen test).
- Most suggested reviews are **additional human checks**, not confirmed positives.
- Frozen test characterisation: about **311 reviews**, **62 true positives**, **249 additional reviews** and **67 missed positives** per 1,000 shipments.

Missed positives still occur. There is no identified cost ratio that would justify a different threshold.

## Calibration and score semantics

Class-weighted logistic regression is **poorly calibrated** (validation Brier ≈ 0.238, ECE ≈ 0.355). The output is a **review score** for ranking and thresholding, not a literal probability. Sigmoid/isotonic calibration improved Brier/ECE on validation without becoming the champion under the existing policy. Those candidates were **not** scored on the test set in this phase.

## Explanations

- Coefficients apply to **transformed** features (scaled numerics, one-hot categories).
- Correlated features (value, log value, value-to-weight) make coefficients unstable.
- Permutation importance can disagree with |coefficient| ranks.
- Local contributions reconstruct the model score; they **do not** establish causation or real-world risk.
- SHAP is optional for comparison trees only and is not required for the champion.

## Subgroups

Small cells are noisy. Geographic and commodity gaps can replay the toy label rules. This is **not** a fairness audit of a real population.

## Drift, monitoring and retraining

Evidently remains deferred. PSI/KS can flag distribution shift on synthetic periods but cannot certify production health. Policy 1.1.0 reports `insufficient_data` / `monitoring_error` instead of treating missing jobs as healthy. This phase does not retrain or replace the champion.

## Why this must not automate adverse decisions

Low precision, poor calibration, synthetic labels, proxy-prone corridor features and non-causal explanations are each sufficient to forbid automated harm. A human review queue metaphor is the only intended decision surface, and even that is fictional.
