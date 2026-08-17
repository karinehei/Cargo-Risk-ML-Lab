# Drift metrics

> **Disclaimer:** Metrics describe distributional change on synthetic educational data. They are not proof of performance failure.

## Numerical features

| Metric | What it measures | Severity driver |
|---|---|---|
| **PSI** | Population Stability Index using reference quantile bins | Effect size (default warning ≥ 0.10, critical ≥ 0.25) |
| **KS statistic / p-value** | Maximum CDF separation between reference and current | KS statistic thresholds; p-value reported for context only |
| **Standardized mean difference** | Absolute mean shift normalised by pooled std | Effect size (warning ≥ 0.20, critical ≥ 0.50) |
| **Missing-rate change** | Current minus reference missing proportion | Absolute delta (warning ≥ 0.05, critical ≥ 0.15; critical is immediate) |

## Categorical features

| Metric | What it measures | Default thresholds |
|---|---|---|
| **Jensen–Shannon divergence** | Symmetric divergence between category distributions | warning ≥ 0.05, critical ≥ 0.15 |
| **Total-variation distance** | Half the L1 distance between distributions | warning ≥ 0.10, critical ≥ 0.25 |
| **Unseen-category rate** | Share of current rows with categories absent from reference | warning ≥ 0.01, critical ≥ 0.05 (critical is immediate) |
| **Missing-rate change** | Same as numerical missingness | Same missing-delta thresholds |

## Model outputs

| Metric | What it measures | Default thresholds |
|---|---|---|
| **Review-score PSI** | Distribution shift in champion review scores | warning ≥ 0.10, critical ≥ 0.25 |
| **Predicted review-rate change** | Change in share of rows with `review_score >= threshold` | warning ≥ 0.05, critical ≥ 0.08 (critical is immediate) |
| **KS / quantile shifts** | Score distribution movement | same KS thresholds as features |

The review score is a ranking/threshold value, **not** a calibrated probability.

## Why p-values alone are insufficient

With large batches, even tiny shifts can yield small p-values. At `n_ref=5000` and `n_cur=1200`, a conventional two-sample KS critical value near 5% is about 0.044, so a KS warning of 0.10 is already an effect-size rule, not a p-value rule. This project therefore prioritises **effect-size thresholds** for `warning` and `critical`. KS p-values are included as context only.

## Multiple comparisons

Monitoring many features increases the chance that **at least one** metric crosses a threshold by random variation even when no individual comparison is unreliable. Policy 1.1.0 therefore distinguishes:

- isolated weak warning (recorded, overall `no_material_drift`)
- repeated warning on the same feature across consecutive windows
- coordinated multi-feature drift
- immediate-critical schema, missingness or review-rate exceptions

Per-comparison false-alert rates and the overall union rate are estimated with null Monte Carlo (`make monitoring-null-audit`). Do not require a p-value threshold to raise severity.

## Safe handling

- Constant columns → zero drift severity
- All-missing columns → safe `none` severity
- Sparse/unseen categories → surfaced explicitly, not silently dropped
- Batches below `min_batch_size` → `insufficient_data` (not healthy)

## Source of truth

The implementation in `src/monitoring/metrics.py` plus aggregation in `src/monitoring/policy.py` is the source of truth. Evidently may add a supplementary report when its installed API works, but it never replaces the lightweight PSI/KS/JS path.

## Limitations

- Drift on synthetic period shifts does not forecast real corridor behaviour.
- Input drift can occur without performance degradation when labels are unavailable.
- Unseen categories may reflect pipeline/schema issues rather than model staleness.

See also `docs/monitoring.md` and `docs/operations.md`.
