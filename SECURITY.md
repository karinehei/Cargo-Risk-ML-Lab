# Security policy

This project is an **educational demonstration**. It is not a supported production product and is not affiliated with Finnish Customs or any other authority.

## Supported status

- Demonstration code is provided as-is under the repository license.
- There is no SLA, patch cadence or production support channel.
- Serialized model artifacts are **not** a public model registry. They must be generated locally and checksum-verified before loading.

## Scope

In scope for reports about this repository:

- accidental secret leakage in tracked files
- dependency vulnerabilities that are reachable in the local demo
- path injection that could load untrusted models
- CI workflow permission or supply-chain issues in this repo

Out of scope:

- using this software for operational enforcement or automated adverse decisions
- vulnerabilities that require treating cloudpickle models from untrusted sources as safe
- issues in upstream MLflow/skops/scikit-learn except as they affect this demo's documented usage

## Reporting

Do **not** open a public issue that includes secret values, credentials or production data (this project has none, but do not paste secrets anyway).

If you believe a tracked file contains a credential:

1. Open a GitHub issue **without** the secret value, describing the file class and approximate location.
2. If this repository is a fork, notify the owner through GitHub's private vulnerability reporting if that feature is enabled on the origin repository.

No email address is published here. Do not invent one.

## Known limitations

- MLflow sklearn logging uses official `serialization_format=cloudpickle`. That payload can execute Python on load.
- skops trust checks are **not** globally disabled.
- The API reload token, when set, is a local operator control. Leave `API_RELOAD_TOKEN` empty in `.env` for public demos.
- Monitoring CSVs and train/val/test tables are local synthetic data and must stay untracked.

Bandit suppressions are line-local only (`# nosec Bxxx`) for:

- `B301` joblib dumps of recovery exports (not the serving champion)
- `B404`/`B603`/`B607` fixed-argv `git` and bootstrap subprocesses (`shell=False`, no user command)
- `B104` default API bind `0.0.0.0` for Compose
- `B310` hardcoded `http://127.0.0.1` readiness probe in bootstrap

No security rule category is skipped in `bandit.yaml`.

## Model loading rule

Only load champion artifacts from the local MLflow store after URI validation. Fail closed on checksum mismatch when a checksum is recorded. Never download models from arbitrary URIs.
