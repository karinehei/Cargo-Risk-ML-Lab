# Explainability

> **Disclaimer:** Explanations describe a synthetic logistic-regression champion. They are not causal claims and not explanations of real customs processes.

## Champion (unchanged in this phase)

Class-weighted, uncalibrated logistic regression (`C=4.0`), MLflow run recorded in `artifacts/mlops/champion.json`, decision threshold **0.525**. Output is a **review score**, not a calibrated probability.

This phase does **not** retrain, re-select or test-evaluate the champion. Frozen v1 artifacts are not modified.

## Methods

### 1. Exact logit decomposition (champion)

For a fitted `Pipeline` with `LogisticRegression`:

```text
logit = intercept + Σ (transformed_x_i × coefficient_i)
review_score = sigmoid(logit)
```

- Numeric inputs are imputed then **standardised**; coefficients are per +1 standard deviation unless `coefficient_original_unit` is shown (coefficient / scaler scale).
- Categorical inputs are **one-hot** with `handle_unknown="ignore"`. A coefficient is the log-odds change when that dummy is 1 versus 0, not versus a dropped reference level.
- Reconstruction: intercept plus contributions must match `decision_function` and the sigmoid must match `predict_proba` within absolute tolerance **1e-8** (`src/explainability/linear.py`).

Coefficients are **associations inside the model**. Correlated features can share or flip credit.

### 2. Permutation importance (validation only)

Original columns are shuffled on **validation** rows; scoring is average precision (PR-AUC). The test set is not used.

Permutation and |coefficient| ranks may disagree because:

- coefficients live in transformed space (each dummy separately; each scaled numeric separately);
- permutation moves a whole raw column (all of that column’s dummies together);
- correlated columns keep the signal when one of them is shuffled.

### 3. Comparison models

Random Forest and XGBoost, if logged in MLflow experiment records, receive permutation importance and an optional SHAP `TreeExplainer` attempt. They are labelled **comparison models, not the champion**. SHAP is not required; if TreeExplainer fails, the reason is stored and permutation is kept. Trust checks are not disabled.

## Local explanation payload

Each explained row includes review score, decision threshold, `requires_review`, intercept, strongest increasing and decreasing log-odds contributions, original input values, model version, MLflow run ID, reconstruction error, and an explicit non-causation statement. Identifiers and excluded fields never enter the contribution table.

## Commands

```bash
make explain    # train/val artifacts only; does not load test.csv
```

Outputs: `artifacts/explanations/` (coefficients, permutation, local examples, subgroups, comparison_models/).

## Dashboard and API

- API: `review_score`, `decision_threshold`, `requires_review`, `model_version`, `mlflow_run_id`, `score_is_calibrated`, `score_semantics`. See `docs/api.md`.
- The `requires_review_probability` alias has been removed. Use `review_score`.
- Streamlit shows the calibration warning next to every displayed score and never labels the score as a probability.
