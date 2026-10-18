# Pages evidence review

This note records the public evidence behind `site/`, the commands run while building it, and the checks completed on this machine. It is not a GitHub Actions log and not a record of a live Pages deployment.

## What the site is allowed to claim

The bundle was exported on 2026-10-18T10:19:38.549283+00:00 from the local champion and the saved experiment file. Schema version is `1.0.0`.

| Result | Class | Source | Identity and split |
| --- | --- | --- | --- |
| Four-model comparison and logistic variants | Locally generated artifact, copied into the bundle | `artifacts/mlops/experiment_records.json` | Experiment `cargo-risk-ml-lab`. Shared dataset fingerprint `b2a2e13599edd841447e6fbb0673057d369bf247c33cd6667d2ea329d8dd8a0e`. |
| Validation curves, calibration bins, threshold grid, fictional examples | Computed during this export | Registered champion scored on `data/processed/val.csv` | Model `logreg-none-1.0.0`, MLflow run `8041c2e0afaf4ecea05399ae55a87816`, run name `compare_logreg`. Split: validation, 1,500 rows. |
| Coefficient chart | Saved artifact, checked against the loaded champion | `artifacts/explanations/global_coefficients.json` | `saved_coefficient_artifact_matches_champion` is true. |
| Permutation ranks | Saved artifact, not recomputed | `artifacts/explanations/permutation_importance.json` | Run `bb0f045e02784e289bee4b0911e97e9a`, validation split, scoring `average_precision`, 10 repeats. |
| Historical test summary | Documented historical summary | `artifacts/frozen_v1/metrics_test.json`, `artifacts/audit/bootstrap_test_v1.csv`, the selected-threshold row of `artifacts/audit/operational_rates.csv` | `reproduced_by_this_export` is false. `curves_included` is false. Split: test, 3,000 rows. |

The logistic-regression row in the comparison table is run `bb0f045e02784e289bee4b0911e97e9a`, also named `compare_logreg`. That run id differs from the registered champion. Validation PR-AUC and the dataset fingerprint match. Curves, the threshold grid and the examples come from the registered champion.

`.ci-work/` smoke-test metrics were not used. `make evaluate` was not run. Frozen test predictions were not regenerated. `artifacts/frozen_v1/` was not overwritten. The export hashes `artifacts/frozen_v1/metrics_test.json` and checks that the hash is unchanged.

`export_git_commit` in the manifest is `64135b4aaf0fc3aedab0c28c7fe5b4abdacbe2a1`, the HEAD at export time. The site files were still uncommitted, so that hash does not identify a commit that contains the site. `training_git_commit` is `93de709982b24daba9031cafe28e7e86276d9d8f` from the champion metadata.

## Selection

Policy `1.0.0` drops a candidate when round-trip fails, validation recall is below 0.40, validation PR-AUC is below 0.15, or p99 latency is above 100 ms. Remaining candidates are ranked by validation PR-AUC. A simpler family wins when it is within 0.005 PR-AUC of the leader.

Among Dummy, logistic regression, random forest and XGBoost, logistic regression has the highest validation PR-AUC, 0.2272784442866974. Random forest is 0.21993413790194793 and is ineligible because recall is below 0.40. XGBoost is eligible at 0.21305432451192216. Dummy is ineligible because PR-AUC is below 0.15.

In the wider recorded pool the leader is unweighted logistic regression at 0.22828771115194793. The gap to class-weighted logistic regression is 0.0010092668652505188, inside the indifference band, so `simpler_preference_applied` is true and class-weighted logistic regression is the selected model.

Training-fold cross-validation PR-AUC is the search score. Precision and recall in the comparison table are at each model's own validation threshold.

## Numerical checks

Validation operating point, threshold 0.525, grid index 19, 1,500 rows and 194 positive labels:

- TP 104, FP 383, TN 923, FN 90
- 487 flagged
- precision 0.2135523613963039
- recall 0.5360824742268041

The public grid is `numpy.linspace(0.05, 0.95, 37)`. Precision is null at 0.95 because no row is flagged (`undefined_no_flagged_rows`). F-beta is null at 0.90, 0.925 and 0.95. The verifier recomputes precision, recall and F-beta from the counts.

Historical test summary at the same recorded threshold, not recomputed here:

- PR-AUC 0.20420446319791113
- ROC-AUC 0.6276878325255364
- precision 0.20042872454448016
- recall 0.48195876288659795
- TP 187, FP 746, FN 201, TN 1866, n = 3,000
- bootstrap intervals are the five aggregate rows from `bootstrap_test_v1.csv` (2,000 resamples, seed 42)

Precomputed examples are hand-specified inputs. An exact feature-signature comparison against train, validation and test found 0 matches. The test split was not scored. Offline review scores:

- `complete-sea-foodstuffs` 0.1415053544566813, not flagged
- `expedited-air-pharmaceuticals` 0.07388914765518706, not flagged
- `incomplete-air-electronics` 0.8243547957268974, flagged
- `ordinary-road-machinery` 0.2930559319115034, not flagged

The largest coefficient in the published summary is `commodity_category = pharmaceuticals`, transformed coefficient 0.3885424661728732. That is a change in the model's log-odds, not a calibrated probability and not a causal effect.

The review score is labelled as a ranking and threshold score. `score_is_calibrated` is false and `calibration_status` is `none`.

## Commands run

From the project virtual environment in WSL (`~/.venvs/cargo-risk-ml-lab`):

- `python -m scripts.export_pages_evidence` completed with `pages_evidence_ok` at the timestamp above.
- `python -m pytest --cov=src --cov-report=term-missing -q` completed with exit code 0. Coverage was 71.28 percent. The configured floor is 68 percent.
- `ruff check` on the Pages modules, scripts and tests, and `mypy src/pages`, completed before the export.
- After the sticky-navigation CSS change: `ruff check` on the same Pages paths passed, `python -m scripts.verify_pages_evidence` printed `pages_evidence_verified`, and `python -m detect_secrets scan --baseline .secrets.baseline` with the Makefile exclude pattern exited 0.
- `bandit -c bandit.yaml -r src/pages scripts/export_pages_evidence.py scripts/verify_pages_evidence.py scripts/preview_pages.py scripts/audit_repository_boundary.py` exited 0.

`python -m scripts.preview_pages` is serving `http://127.0.0.1:8765/Cargo-Risk-ML-Lab/`. The first request returned HTTP 200. That address is a local preview.

## Browser checks

The preview was opened at `http://127.0.0.1:8765/Cargo-Risk-ML-Lab/`.

- Provenance text shows schema 1.0.0, champion `logreg-none-1.0.0`, run `8041c2e0afaf4ecea05399ae55a87816`, and validation threshold 0.525.
- Resource requests stayed on `127.0.0.1:8765` under `/Cargo-Risk-ML-Lab/` for the stylesheet, script, seven JSON files and six figures. No CDN host and no application backend appeared.
- Moving the exported grid to index 36 showed threshold 0.950, precision described as undefined because no validation row is flagged, recall 0.000, and counts TP 0, FP 0, TN 1306, FN 194. Precision was not shown as zero.
- "Show the selected operating point" restored threshold 0.525 with TP 104, FP 383, TN 923, FN 90 and 487 flagged. The confusion table at that point showed the same four counts.
- Choosing "Incomplete air shipment of electronics" showed review score 0.824, flagged, with the stored-offline wording.
- The section link updated the URL to `/Cargo-Risk-ML-Lab/#examples`. The examples heading sat below the sticky navigation.
- Opening "Other logistic variants" revealed the calibration and class-weight table, including unweighted logistic regression at validation PR-AUC 0.228.
- At a 390px viewport the navigation is static, the page text fits, and `scrollWidth` equals the viewport width.
- In-page links use the project subpath. External links point at `https://github.com/karinehei/Cargo-Risk-ML-Lab` or a `blob/main` document in that repository.

The embedded browser did not deliver Tab or arrow keys to the page, so keyboard focus movement was not observed in the live control. The stylesheet defines a 3px solid `:focus-visible` outline. Pointer use of the operating-point button, example selector, disclosure and section links was observed.

The first preview process was started before `scroll-padding-top` was added. The updated stylesheet was copied into that preview and loaded. A later `python -m scripts.preview_pages` reads `site/` directly.

## Remaining limitations

- The labels come from the toy generator.
- Positive precision at the selected threshold is about 0.214 on validation, so most flags are extra human reviews.
- The champion score is not a calibrated probability. Class weights push it upward. Validation Brier score is about 0.238 and expected calibration error is about 0.355.
- Permutation ranks and coefficients can disagree because they group features differently. Permutation importance was not recomputed for the registered champion run.
- Explanations reconstruct the model score. They do not establish cause.
- Nothing on the page authorises an automated adverse decision.
- GitHub Actions has not run `.github/workflows/pages.yml` for this branch. Pages hosting has not been enabled. The site is not published at `https://karinehei.github.io/Cargo-Risk-ML-Lab/`.
