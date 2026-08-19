# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# hadolint ignore=DL3008 -- slim image; OS packages are build-only and discarded
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY requirements/runtime.lock.txt ./requirements/runtime.lock.txt
COPY src ./src
COPY app ./app
COPY scripts ./scripts

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip uninstall -y setuptools wheel \
    && /opt/venv/bin/pip install --require-hashes --no-cache-dir -r requirements/runtime.lock.txt \
    && /opt/venv/bin/pip install --no-deps --no-cache-dir . \
    && /opt/venv/bin/pip uninstall -y nvidia-nccl-cu13 \
    && rm -rf /opt/venv/lib/python3.12/ensurepip

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    HOME=/home/appuser \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib \
    XDG_CACHE_HOME=/home/appuser/.cache

WORKDIR /app

# hadolint ignore=DL3008 -- runtime image needs libgomp for XGBoost; no pinned Debian patch versions
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get install -y --no-install-recommends --only-upgrade util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y setuptools wheel \
    && rm -rf /usr/local/lib/python3.12/ensurepip \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser docs ./docs
COPY --chown=appuser:appuser pyproject.toml README.md ./

# pip ships pip/_vendor/bom.cdx.json listing its vendored wheels (setuptools 70.x,
# msgpack 1.1.x). Trivy treats that CycloneDX file as installed Python packages.
# The hashed lock installs msgpack 1.2.1 and setuptools 84.x; METADATA rows are clean.
# Also drop ensurepip leftovers and any other SPDX/CycloneDX inventories.
RUN rm -rf /opt/venv/lib/python3.12/ensurepip /usr/local/lib/python3.12/ensurepip \
    && find /opt/venv /usr/local -depth \( \
         -name 'setuptools-70*' -o \
         -name 'msgpack-1.1*' -o \
         -name '*.spdx.json' -o \
         -name '*sbom*.json' -o \
         -name '*.cdx.json' -o \
         -name 'bom.json' \
       \) -exec rm -rf {} + \
    && mkdir -p /app/mlruns /app/.cache /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser /opt/venv

USER appuser

EXPOSE 8000 8501 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=8 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
