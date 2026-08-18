# Operations

> **Disclaimer:** Operational guidance here applies to an educational synthetic pipeline only.

## Default workflow (unlabelled)

```bash
make generate-data          # creates train.csv (never use test.csv for monitoring)
make monitoring-reference   # train-derived aggregate profile; local CSV cache is gitignored
make monitoring-scenario SCENARIO=moderate
make monitoring-run SCENARIO=moderate
make monitoring-status
```

Expected artifacts:

- `artifacts/monitoring/reference_profile.json` (publishable aggregates and fingerprints)
- `data/monitoring/reference_sample.csv` (local cache only; recreate with `make monitoring-reference`)
- `data/monitoring/current_<scenario>.csv` (local cache only; gitignored)
- `artifacts/monitoring/latest_report.json`
- `artifacts/monitoring/status.json`
- `artifacts/monitoring/unlabelled_monitoring_<scenario>_<id>.{json,md,csv}`

Policy 1.0.0 demonstration reports are retained under `artifacts/monitoring/policy_v1.0.0/` as evidence.

## Policy audit (optional)

```bash
make monitoring-null-audit      # matched no-shift Monte Carlo; seeds 92001+
make monitoring-validation      # hold-out seeds 93001+; not used to set thresholds
```

Null simulations use the synthetic generator with the training period mix. They never load `test.csv`. An individual no-drift batch may still warn; document the measured rate instead of requiring every null batch to be `no_material_drift`. After policy 1.1.0, 150 matched null windows produced 0 overall warnings (binomial 95% upper bound about 2%). Under the previous 0.03 review-rate warning the same seeds produced a 4.7% warning rate driven only by predicted review-rate sampling noise.

## Labelled simulation (optional, separate)

```bash
make monitoring-labelled SCENARIO=moderate
```

Outputs include `simulated_performance` in the report. These figures are **synthetic simulation only** and must not be treated as production monitoring evidence.

## Interpreting results

1. Read **status** (`insufficient_data`, `no_material_drift`, `warning`, `critical`, `monitoring_error`).
2. Read **alert reasons** (feature/output, metric, observed value, thresholds, severity, interpretation).
3. **Input data drift** — investigate upstream data generation, integrations or feature pipelines.
4. **Score drift** — investigate whether operational mix changed; the champion and threshold are unchanged.
5. **Performance** — unavailable in unlabelled monitoring; wait for delayed labels before measuring degradation.

Missing or failed monitoring is **not** healthy. Drift does not automatically trigger retraining, threshold changes or shipment blocking.

## When to investigate vs retrain

| Signal | Suggested response |
|---|---|
| Isolated weak warning | Recorded; overall `no_material_drift` unless it persists |
| Warning status | Investigate pipeline/context; continue monitoring |
| Critical status | Investigate before relying on outputs; do not auto-retrain |
| `insufficient_data` / `monitoring_error` | Do not treat as healthy |
| Sustained performance drop (with labels) | Open a separate validated retraining/evaluation process |

Retraining and promotion require champion policy, validation evidence and explicit authorisation. They are out of scope for monitoring CLI runs.

## API read-only checks

```bash
curl -sS http://127.0.0.1:8000/monitoring/status
curl -sS http://127.0.0.1:8000/monitoring/latest
```

Returns aggregated summaries only. Returns `status=insufficient_data` and `available: false` when no report exists.

## Ground-truth latency

In real systems, review outcomes arrive after human processing. Until labels exist, monitoring stays unlabelled. This is expected and not a tooling failure.

## Raw CSV handling

Do not commit `data/monitoring/*.csv`. `.gitignore` and `.dockerignore` exclude them. Docker images must not copy `data/`. Rebuild the reference sample from `data/processed/train.csv` when needed.

Do not commit MLflow SQLite files or generated train/validation/test tables. See `docs/repository_policy.md`. Serialized champion artifacts are local and trusted; the API does not download models from arbitrary URIs.
