#!/usr/bin/env bash
# Non-root image smoke: champion must load so /ready can pass.
set -euo pipefail

chmod -R a+rX .ci-work
chmod -R a+rwX .ci-work/mlruns
if [ -d mlruns ]; then
  chmod -R a+rX mlruns
  chmod -R a+rwX mlruns
fi

docker run -d --name cargo-risk-api-ci --user 1000:1000 \
  -e CONFIG_PATH=configs/ci.yaml \
  -e CHAMPION_PATH=.ci-work/artifacts/mlops/champion.json \
  -e MLFLOW_TRACKING_URI=sqlite:///.ci-work/mlruns/mlflow.db \
  -e MLFLOW_EXPERIMENT_NAME=cargo-risk-ml-lab-ci \
  -v "$PWD:$PWD" \
  -v "$PWD/.ci-work:/app/.ci-work" \
  -w /app \
  -p 8011:8000 \
  cargo-risk-ml-lab:ci

cleanup() {
  docker logs cargo-risk-api-ci || true
  docker inspect --format '{{json .State}}' cargo-risk-api-ci || true
  docker rm -f cargo-risk-api-ci || true
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  status=$(docker inspect --format '{{.State.Health.Status}}' cargo-risk-api-ci || true)
  if [ "$status" = "healthy" ]; then
    break
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' cargo-risk-api-ci || true)" != "true" ]; then
    echo "Container exited before becoming healthy"
    exit 1
  fi
  sleep 2
done

docker inspect --format '{{.State.Health.Status}}' cargo-risk-api-ci | grep healthy
curl -fsS http://127.0.0.1:8011/health
curl -fsS http://127.0.0.1:8011/ready
