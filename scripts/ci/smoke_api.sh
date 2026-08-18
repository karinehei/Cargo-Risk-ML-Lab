#!/usr/bin/env bash
# Host API smoke against CI bootstrap artifacts. Not frozen-v1.
set -euo pipefail

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
uvicorn_pid=$!
cleanup() { kill "$uvicorn_pid" 2>/dev/null || true; wait "$uvicorn_pid" 2>/dev/null || true; }
trap cleanup EXIT

payload='{"origin_region":"Asia","destination_region":"Northern Europe","commodity_category":"electronics","transport_mode":"air","declared_value_eur":12500.0,"shipment_weight_kg":85.5,"previous_discrepancies":0,"submission_hour":10,"expedited_shipment":0}'

for _ in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "$payload"
curl -fsS http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d "$payload"
