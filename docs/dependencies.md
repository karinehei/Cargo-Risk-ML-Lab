# Dependencies

Python **3.12** is required.

## Layout

| File | Role |
|---|---|
| `pyproject.toml` | Project definition and direct dependencies |
| `requirements/runtime.lock.txt` | Fully resolved, hashed runtime set (`pip-compile --generate-hashes`) |
| `requirements/dev.lock.txt` | Runtime + development + security tools, hashed |

Direct runtime dependencies are listed under `[project.dependencies]` in `pyproject.toml`. Development tools are `[project.optional-dependencies] dev` and `security`. Transitive packages appear only in the lock files.

## Install from the lock

```bash
python -m pip install --require-hashes -r requirements/dev.lock.txt
python -m pip install --no-deps -e .
```

CI and Docker runtime installs use the lock. Do not `pip install -e ".[dev]"` in CI once the lock exists.

## Update procedure

1. Change versions in `pyproject.toml` only when needed.
2. Recreate locks from Python 3.12 (Linux/WSL recommended so hashes match CI):

```bash
uv pip compile pyproject.toml --python 3.12 --generate-hashes -o requirements/runtime.lock.txt
uv pip compile pyproject.toml --python 3.12 --extra dev --extra security --generate-hashes -o requirements/dev.lock.txt
```

`pip-compile --generate-hashes` from pip-tools is an alternative; hash generation over HTTPS can time out on large wheels.

3. Run `pip-audit -r requirements/dev.lock.txt`.
4. Do not apply breaking upgrades solely to clear an advisory. Record unreachable or accepted findings in the verification notes.

`nvidia-nccl-cu13` is a transitive Linux extra of XGBoost 3.x. It is not used for training in this demo. Do not treat it as a CUDA runtime requirement.

## Known advisories retained

`pip-audit` on the 2026-08-17 lock reported `cryptography==49.0.0` / PYSEC-2026-3552 (CVE-2026-69247), fixed in 50.0.0. This project does not decrypt attacker-supplied PKCS#7 EnvelopedData. The finding is transitive (MLflow/Evidently/google-auth). It was **not** auto-upgraded.

## Serialization

MLflow sklearn models are logged with official `serialization_format=cloudpickle`. Do not enable global unsafe skops trust settings. Compatibility to retain: MLflow, NumPy, scikit-learn, skops (transitive), XGBoost, SHAP.
