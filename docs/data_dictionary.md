# Data dictionary

> **Disclaimer:** Every column, category, range and label in this project is **fully synthetic**. The process that creates `requires_review` is a fictional, non-linear toy. It does **not** represent the Finnish Customs Service, any other authority, real border processes, or operational risk logic. **Do not use these rules or this data for real decisions.**

Generation lives in `src/data/generate.py`. Preprocessing lives in `src/features/` and must not recreate the label. Empirical counts, missing rates and class balance are written at generation time to `data/raw/validation_report.json` (gitignored). This document does not hard-code those statistics.

## Target

| Column | Type | Description |
|---|---|---|
| `requires_review` | binary `{0,1}` | Whether a **fictional** shipment should be sent for additional human review. Sampled from a noisy logistic function of several features and interactions, then lightly label-flipped. Never missing. |

The positive class is calibrated to a realistic minority rate (about 8–15% on the default 15,000-row draw). Exact prevalence is an output of the generator, not a claim about the real world.

## Identifiers and time-like split

| Column | Type | Description |
|---|---|---|
| `shipment_id` | string | Synthetic identifier `SYN-{seed}-{row}`. Unique, never missing. |
| `event_date` | date (`YYYY-MM-DD`) | Synthetic calendar date spanning a 16-week window starting 2024-01-01. Rows are ordered in time. |
| `generation_period` | integer `0–3` | Sequential era used for a time-like train/validation/test split and for later drift simulation. Periods `2+` apply **feature** shifts only; the target is not used when those shifts are applied. |

Default split: periods `0–1` train, `2` validation, `3` test.

## Model features

These columns are eligible for modelling. The target, identifier and time columns are excluded.

| Column | Type | Allowed missing? | Notes |
|---|---|---|---|
| `declared_value_eur` | float `> 0` | no | Synthetic declared value in euro. |
| `shipment_weight_kg` | float `> 0` | no | Synthetic gross weight. Mildly related to value, not to the label. |
| `value_to_weight_ratio` | float `≥ 0` | no | `declared_value_eur / shipment_weight_kg`. Stored at generation time so preprocessing does not need the label. |
| `transport_mode` | category | no | `road`, `sea`, `air`, `rail`. |
| `origin_region` | category | no | Coarse trade-corridor labels, not personal origin. |
| `destination_region` | category | no | Coarse destination corridor. |
| `commodity_category` | category | no | Broad goods grouping. |
| `declaration_completeness_score` | float `0–1` | yes (~2–3%) | Toy completeness score. |
| `documentation_count` | integer `≥ 0` | yes (~2%) | Count of synthetic supporting documents. |
| `previous_discrepancies` | integer `≥ 0` | no | Count of prior fictional discrepancies for the sender history. |
| `sender_history_length` | integer `≥ 0` | yes (~2%) | How many prior synthetic shipments are attributed to the sender. |
| `route_rarity` | float `0–1` | yes (~1–2%) | Higher means a less common origin–destination–mode combination in the toy generator. |
| `declared_vs_estimated_value_deviation` | float | yes (~3%) | `(declared − toy_estimated) / toy_estimated`. The estimated value is an internal generator quantity and is **not** saved, to avoid leaking a second price oracle into the model table. |
| `submission_hour` | integer `0–23` | no | Hour of fictional electronic submission. |
| `expedited_shipment` | binary `{0,1}` | no | Expedited handling flag. |

## Derived in preprocessing (not stored in raw CSV)

| Column | Source |
|---|---|
| `log_declared_value` | `log1p(declared_value_eur)` |
| `is_off_hours` | `submission_hour < 6` or `≥ 22` |

These transforms do not use `requires_review`.

## What is intentionally absent

- No names, contact details, or personal identifiers.
- No gender, age, nationality, ethnicity, race, religion, or other protected personal characteristics.
- No latent probability / logit column in the saved table (that would be target leakage).
- No inspector outcome, seizure flag, or post-decision field.

## Fictional label process (educational only)

`requires_review` is **not** a deterministic rule on one column. The generator builds a non-linear score from interactions such as:

- high value-to-weight ratio
- incomplete declarations combined with large declared-vs-estimated deviation
- short sender history with prior discrepancies
- rare routes that are also expedited
- sparse documentation
- off-hours submission
- air transport with selected commodity categories

Gaussian noise is added on the logit scale, the intercept is calibrated so the expected positive rate is near 11%, labels are sampled Bernoulli, and a small fraction of labels are flipped. **These interactions are invented so the classification task is learnable. They are not real review policy.**

## Missingness

A small, controlled fraction of values is removed after features and labels are created. Missingness does **not** depend on `requires_review`. Most holes are MCAR; `documentation_count` has a mild MAR boost when completeness is already low.

## Drift

Later `generation_period` values shift covariates (higher declared values, more air/expedited traffic, slightly lower completeness, rarer routes). Shifts use only period membership and feature noise — never the label. Because the label is sampled from features afterwards, the positive rate may also move over time; that is a consequence of feature drift, not of leaking `y` into `X`.

## How to generate and validate

```bash
python -m scripts.generate_data
python -m scripts.validate_data
# or: make generate-data && make validate-data
```
