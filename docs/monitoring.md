# Monitoring

> **Disclaimer:** Monitoring in this repository is an educational demonstration on fully synthetic data. It must not authorise automatic retraining, threshold changes, shipment blocking or operational alerting.

## Three concepts (kept separate)

1. **Input data drift** — changes in feature distributions relative to a train-derived reference profile.
2. **Review-score and decision drift** — changes in model outputs and predicted review rates at the fixed champion threshold.
3. **Performance degradation** — measurable only when delayed ground-truth labels are available.

Input drift does **not** automatically mean model failure. Default operational monitoring is **unlabelled** and does not report production performance.

## Status vocabulary

| Status | Meaning |
|---|---|
| `insufficient_data` | No complete current batch, reference, or report. **Not healthy.** |
| `no_material_drift` | No material drift under the active policy. Isolated weak findings may still be listed. |
| `warning` | Investigate pipeline and operational context. |
| `critical` | Investigate before relying on outputs. Immediate schema/missingness/review-rate exceptions apply. |
| `monitoring_error` | The monitoring job failed. **Not healthy.** |

`overall_severity` remains `none` / `warning` / `critical` for compatibility. Dashboards must display **status** and **alert reasons**, not a bare severity label.

## Policy versions

| Version | Aggregation |
|---|---|
| `1.0.0` | Max of any single feature or score metric (union rule). Preserved as evidence. |
| `1.1.0` (current) | Same PSI/KS/SMD/JS/TV/missingness thresholds as 1.0.0. Isolated single-feature warnings do not raise overall warning. Overall warning requires coordinated features, score-level warning, schema unseen-category warning, or persistence across consecutive windows of the same scenario. Immediate critical for high unseen-category rate, extreme missingness, or major review-rate change. Predicted-review-rate **warning** moved from 0.03 to 0.05 after null Monte Carlo showed 0.03 sitting at the sampling p95 for these batch sizes. |

P-values never raise severity. Thresholds were **not** tuned so the original `none` scenario becomes green, and were **not** fitted on moderate/major demonstration outcomes.

The original policy-1.0.0 `none` warning was an isolated `declaration_completeness_score` KS/SMD finding caused by a **generator mix mismatch** (train reference used period drift; the first `none` batch did not). Score PSI 0.0409 and review-rate change −0.0298 were below warning thresholds.

## Reference profile and raw CSVs

Built from a train-derived sample. Recreate deterministically with `make monitoring-reference` (samples `data/processed/train.csv`; never `test.csv`).

Publishable artifacts are **aggregates only**:

- `artifacts/monitoring/reference_profile.json` (schema, summaries, frequencies, fingerprints, champion identity)

Raw files are **local-only** and gitignored / Docker-excluded:

- `data/monitoring/reference_sample.csv`
- `data/monitoring/current_<scenario>.csv`

Do not commit monitoring CSVs. The sample is synthetic train-derived data and can be rebuilt.

## Synthetic monitoring scenarios

| Scenario | Intent |
|---|---|
| `none` | No intentional drift; uses the same generator mix as training |
| `subtle` | Small covariate shifts |
| `moderate` | Moderate covariate shifts |
| `major` | Large covariate shifts |
| `missingness` | Increased missing-value rates |
| `unseen_category` | Schema/category violations |

Scenarios use dedicated seeds (91001–91007 for demos; 92001+ for null Monte Carlo; 93001+ for independent validation). They do **not** reuse the frozen test dataset.

## Commands

```bash
make monitoring-reference
make monitoring-scenario SCENARIO=moderate
make monitoring-run SCENARIO=moderate
make monitoring-labelled SCENARIO=moderate
make monitoring-status
make monitoring-all
make monitoring-null-audit
make monitoring-validation
```

See `docs/operations.md` for the operational workflow and `docs/drift_metrics.md` for metric definitions.

## API (read-only)

- `GET /monitoring/status` — aggregated latest status, including `status`, policy version, alert reasons and sample-size fields
- `GET /monitoring/latest` — latest report summary without raw records

Missing reports return `status=insufficient_data` and `available=false`. Monitoring execution remains a CLI/offline workflow.

## Conservative actions

| Status | Recommended action |
|---|---|
| `no_material_drift` | Continue routine monitoring |
| `warning` | Investigate data pipeline and operational context |
| `critical` | Investigate before relying on outputs; do not automatically retrain |
| `insufficient_data` / `monitoring_error` | Do not treat as healthy; collect data or fix the job |

Retraining and promotion require a separate validated process.
