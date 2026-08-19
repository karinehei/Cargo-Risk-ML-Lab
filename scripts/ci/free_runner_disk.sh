#!/usr/bin/env bash
# Reclaim GitHub-hosted runner disk before the Docker build.
# Quality already installed a multi-GB Python env; Buildx then needs another
# copy of the runtime image (llvmlite/XGBoost/PyArrow). Without this step the
# load/import fails with "no space left on device".
set -euo pipefail

df -h /

if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
  echo "Skipping runner disk cleanup outside GitHub Actions"
  exit 0
fi

sudo rm -rf \
  /usr/share/dotnet \
  /usr/local/lib/android \
  /opt/ghc \
  /usr/local/.ghcup \
  /usr/share/swift \
  /usr/local/share/powershell \
  /opt/hostedtoolcache/CodeQL \
  /opt/hostedtoolcache/Java_Temurin-Hotspot_jdk \
  /opt/hostedtoolcache/go \
  /opt/hostedtoolcache/Ruby \
  /opt/hostedtoolcache/PyPy \
  /opt/hostedtoolcache/Python \
  /opt/pipx \
  /usr/local/.cargo \
  "${HOME}/.rustup" \
  "${HOME}/.cargo" \
  "${HOME}/.dotnet" \
  "${HOME}/.cache/pip" \
  "${HOME}/.cache/ms-playwright"

docker image prune -af || true
docker builder prune -af || true

df -h /
