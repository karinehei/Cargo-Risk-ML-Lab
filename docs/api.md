# Cargo Risk ML Lab API

> **Disclaimer:** This API scores fully synthetic shipments for a fictional additional-human-review queue. It is not affiliated with Finnish Customs or any authority. Do not use it for operational, enforcement or automated adverse decisions.

`review_score` is a ranking and threshold score, **not** a calibrated probability.

## Endpoints

| Method | Path | Role |
|---|---|---|
| `GET` | `/health` | Process liveness only. Does not inspect the champion. |
| `GET` | `/ready` | Champion metadata, MLflow load, threshold/run agreement, linear explanation metadata. Returns **503** if any check fails. |
| `GET` | `/model` | Public champion identity. No filesystem paths or storage URIs. |
| `POST` | `/predict` | Single-shipment review score. |
| `POST` | `/predict/batch` | Batch review scores. Default maximum 100 records (`API_MAX_BATCH_SIZE`). |
| `POST` | `/explain` | Exact logit-space explanation for one shipment. |
| `GET` | `/docs` | OpenAPI UI. |

- `GET /monitoring/status` — aggregated monitoring status (`status`, policy version, alert reasons; no raw records)
- `GET /monitoring/latest` — latest monitoring summary without raw records. Missing reports use `status=insufficient_data`.

The deprecated `requires_review_probability` alias has been **removed**. Use `review_score`.

## Prediction response fields

- `review_score`
- `decision_threshold`
- `requires_review`
- `model_version`
- `mlflow_run_id`
- `score_is_calibrated` (champion: `false`)
- `score_semantics`
- `human_review_notice`
- `score_warning`
- `disclaimer`

`requires_review=1` means additional human review is suggested. It is not an automated enforcement decision.

## Explanation response fields

- `review_score` and `decision_threshold`
- `classification` (human-review label)
- `intercept_log_odds` (baseline intercept)
- `strongest_positive_contributions` / `strongest_negative_contributions`
- `original_input_values`
- `reconstruction_ok`, `reconstruction_error`, `reconstruction_atol`
- `model_version` / `mlflow_run_id`
- `causation_disclaimer`

## Validation

Requests use strict Pydantic models (`extra=forbid`):

- allowed categorical values from `src/data/schema.py`
- synthetic numeric ranges
- optional fields that the training imputer already handles may be omitted or null
- finite numeric values only
- configurable maximum batch size (default 100)
- `Content-Length` rejected above `API_MAX_REQUEST_BYTES` (default 1 MiB)

Error bodies always look like:

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request is invalid.",
  "request_id": "..."
}
```

Clients never receive stack traces, local paths, MLflow storage URIs, or serialized artifact internals.

## Local commands

Prepare verified artifacts first (train/validation only; do not load the frozen test CSV):

```bash
make mlflow-init
make experiments
make explain
```

Start the API:

```bash
make serve-api
```

Check liveness and readiness:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
```

Single prediction (fictional input; copy the live response, do not hard-code scores):

```bash
curl -sS http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin_region": "Asia",
    "destination_region": "Northern Europe",
    "commodity_category": "electronics",
    "transport_mode": "air",
    "declared_value_eur": 12500.0,
    "shipment_weight_kg": 85.5,
    "declaration_completeness_score": 0.82,
    "documentation_count": 6,
    "previous_discrepancies": 0,
    "sender_history_length": 12,
    "route_rarity": 0.25,
    "declared_vs_estimated_value_deviation": 0.05,
    "submission_hour": 10,
    "expedited_shipment": 0
  }'
```

Batch prediction:

```bash
curl -sS http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"shipments":[{"origin_region":"Asia","destination_region":"Northern Europe","commodity_category":"electronics","transport_mode":"air","declared_value_eur":12500.0,"shipment_weight_kg":85.5,"previous_discrepancies":0,"submission_hour":10,"expedited_shipment":0}]}'
```

Explanation:

```bash
curl -sS http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"origin_region":"Asia","destination_region":"Northern Europe","commodity_category":"electronics","transport_mode":"air","declared_value_eur":12500.0,"shipment_weight_kg":85.5,"previous_discrepancies":0,"submission_hour":10,"expedited_shipment":0}'
```

Response numbers depend on the loaded champion. A local process smoke test on 2026-08-17 against the registered champion (`logreg-none-1.0.0`, run `8041c2e0afaf4ecea05399ae55a87816`, threshold `0.525`) returned `review_score` ≈ `0.5145` and `requires_review=0` for the fictional single-shipment example above. Reconstruction for `/explain` succeeded with error `0.0`. Do not treat these figures as a fixed contract.

## Logging

The service emits one JSON object per request with request ID, endpoint, status, latency, optional batch size and model version. Prediction feature payloads are not logged.
