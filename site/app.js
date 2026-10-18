"use strict";

const DATA_FILES = [
  "model_comparison.json",
  "validation_curves.json",
  "threshold_grid.json",
  "explanations.json",
  "examples.json",
  "historical_test.json",
];

function byId(id) {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error("Missing element " + id);
  }
  return node;
}

function textCell(value, undefinedMetric) {
  const cell = document.createElement("td");
  if (value === null || value === undefined) {
    cell.textContent = "Undefined";
    cell.className = "undefined";
    if (undefinedMetric) {
      cell.title = undefinedMetric;
    }
    return cell;
  }
  cell.textContent = String(value);
  return valueCell(cell);
}

function valueCell(cell) {
  return cell;
}

function cell(value) {
  const node = document.createElement("td");
  node.textContent = value === null || value === undefined ? "—" : String(value);
  return node;
}

function headerCell(value) {
  const node = document.createElement("th");
  node.scope = "row";
  node.textContent = String(value);
  return node;
}

function fixed(value, digits) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }
  return Number(value).toFixed(digits);
}

function metricText(value, status) {
  if (value === null || value === undefined || status === "undefined" || (typeof status === "string" && status.startsWith("undefined"))) {
    return null;
  }
  return fixed(value, 3);
}

function clear(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function addRow(body, cells, options) {
  const row = document.createElement("tr");
  if (options && options.selected) {
    row.className = "is-selected";
  }
  if (options && options.id) {
    row.id = options.id;
  }
  cells.forEach(function (item) {
    row.appendChild(item);
  });
  body.appendChild(row);
  return row;
}

async function sha256Hex(text) {
  const encoded = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest))
    .map(function (byte) {
      return byte.toString(16).padStart(2, "0");
    })
    .join("");
}

async function loadJson(name, manifest) {
  const response = await fetch(new URL("data/" + name, document.baseURI));
  if (!response.ok) {
    throw new Error("Could not load data/" + name);
  }
  const text = await response.text();
  const actual = await sha256Hex(text);
  const expected = manifest.files[name] && manifest.files[name].sha256;
  if (actual !== expected) {
    throw new Error("Evidence hash mismatch for " + name);
  }
  return JSON.parse(text);
}

function renderComparison(comparison) {
  const body = byId("comparison-body");
  clear(body);
  comparison.models
    .filter(function (model) {
      return model.role === "primary";
    })
    .forEach(function (model) {
      const cv = model.training_fold_cv_pr_auc_mean === null
        ? null
        : fixed(model.training_fold_cv_pr_auc_mean, 3) + " ± " + fixed(model.training_fold_cv_pr_auc_std, 3);
      addRow(body, [
        headerCell(model.display_name),
        cell(cv),
        cell(fixed(model.validation_pr_auc, 3)),
        cell(fixed(model.validation_roc_auc, 3)),
        cell(fixed(model.decision_threshold, 3)),
        cell(fixed(model.validation_precision_at_threshold, 3)),
        cell(fixed(model.validation_recall_at_threshold, 3)),
        cell(model.eligible ? "Yes" : model.ineligible_reason),
      ], { selected: model.selected });
    });

  const variants = byId("variant-body");
  clear(variants);
  comparison.models
    .filter(function (model) {
      return model.role === "variant";
    })
    .forEach(function (model) {
      addRow(variants, [
        headerCell(model.display_name),
        cell(fixed(model.validation_pr_auc, 3)),
        cell(fixed(model.validation_brier, 3)),
        cell(fixed(model.validation_ece, 3)),
        cell(fixed(model.validation_recall_at_threshold, 3)),
        cell(model.eligible ? "Yes" : "No"),
      ]);
    });

  const selection = comparison.selection;
  const chosen = comparison.models.find(function (model) {
    return model.selected;
  });
  const primary = comparison.models.filter(function (model) {
    return model.role === "primary" && !model.selected;
  });
  const runnerUp = primary.reduce(function (best, model) {
    return model.validation_pr_auc > best.validation_pr_auc ? model : best;
  });
  const story = [
    chosen.display_name + " is the selected model.",
    "Its validation PR-AUC is " + fixed(chosen.validation_pr_auc, 3) + ",",
    "ahead of " + runnerUp.display_name + " at " + fixed(runnerUp.validation_pr_auc, 3) + " among the four primary candidates.",
    "Policy rejections: " + selection.rejections.join("; ") + ".",
  ];
  if (selection.simpler_preference_applied) {
    const leader = comparison.models.find(function (model) {
      return model.model_family === selection.leader_family;
    });
    story.push(
      "In the wider recorded pool, " +
        (leader ? leader.display_name : selection.leader_family) +
        " has validation PR-AUC " +
        fixed(selection.leader_validation_pr_auc, 3) +
        ". The gap to the selected model is " +
        fixed(selection.absolute_gap, 3) +
        ", inside the indifference band of " +
        fixed(selection.indifference, 3) +
        ", so the policy keeps the simpler family."
    );
  }
  byId("selection-story").textContent = story.join(" ");
  byId("run-identity").textContent = comparison.run_identity_note || "";
}

function renderCurves(curves) {
  const body = byId("ranking-body");
  clear(body);
  addRow(body, [
    cell(fixed(curves.pr_auc, 3)),
    cell(fixed(curves.roc_auc, 3)),
    cell(fixed(curves.brier, 3)),
    cell(fixed(curves.ece, 3)),
    cell(fixed(curves.positive_rate, 3)),
    cell(String(curves.n_rows)),
  ]);
  byId("pr-caption").textContent =
    "Validation precision–recall for run " +
    curves.model.mlflow_run_id +
    ". PR-AUC " +
    fixed(curves.pr_auc, 3) +
    ". The horizontal line is the validation positive rate " +
    fixed(curves.positive_rate, 3) +
    ".";
  byId("roc-caption").textContent =
    "Validation ROC for the same champion run. ROC-AUC " +
    fixed(curves.roc_auc, 3) +
    ". The dashed line is chance.";
  byId("calibration-caption").textContent =
    "Ten equal-width bins of the uncalibrated review score. ECE " +
    fixed(curves.ece, 3) +
    ". The diagonal is a calibration reference, not a claim that the score is a probability.";
  const bins = byId("calibration-body");
  clear(bins);
  curves.calibration_bins.forEach(function (bin) {
    addRow(bins, [
      cell(fixed(bin.score_left, 1)),
      cell(fixed(bin.score_right, 1)),
      cell(String(bin.count)),
      textCell(bin.mean_review_score === null ? null : fixed(bin.mean_review_score, 3), "Empty bin"),
      textCell(bin.fraction_positive === null ? null : fixed(bin.fraction_positive, 3), "Empty bin"),
    ]);
  });
}

function describeMetric(row, key, statusKey, undefinedText) {
  const status = row[statusKey];
  if (status && String(status).indexOf("undefined") === 0) {
    return undefinedText;
  }
  return fixed(row[key], 3);
}

function renderThreshold(grid) {
  const rows = grid.rows;
  const input = byId("threshold-index");
  input.min = "0";
  input.max = String(rows.length - 1);
  input.step = "1";
  const operating = rows.findIndex(function (row) {
    return row.selected_operating_point;
  });
  input.value = String(operating);
  const body = byId("threshold-body");
  clear(body);
  rows.forEach(function (row) {
    addRow(body, [
      headerCell(fixed(row.threshold, 3)),
      textCell(metricText(row.precision, row.precision_status), "Undefined when no row is flagged"),
      textCell(metricText(row.recall, row.recall_status), "Undefined when there are no positive labels"),
      textCell(metricText(row.fbeta, row.fbeta_status), "Undefined when precision or recall cannot form F-beta"),
      cell(String(row.true_positives)),
      cell(String(row.false_positives)),
      cell(String(row.true_negatives)),
      cell(String(row.false_negatives)),
      cell(String(row.flagged_for_review)),
    ], { id: "threshold-row-" + row.grid_index, selected: row.selected_operating_point });
  });

  function show(index, shouldScroll) {
    const row = rows[index];
    body.querySelectorAll("tr").forEach(function (tr, rowIndex) {
      tr.classList.toggle("is-selected", rowIndex === index);
    });
    const precision = describeMetric(row, "precision", "precision_status", "undefined");
    const recall = describeMetric(row, "recall", "recall_status", "undefined");
    let message = "Threshold " + fixed(row.threshold, 3) + " on the validation grid. ";
    message += precision === "undefined"
      ? "Precision is undefined because no validation row is flagged. "
      : "Precision " + precision + ". ";
    message += recall === "undefined"
      ? "Recall is undefined because the split has no positive labels. "
      : "Recall " + recall + ". ";
    message +=
      row.flagged_for_review +
      " of " +
      row.n_rows +
      " validation rows are flagged. TP " +
      row.true_positives +
      ", FP " +
      row.false_positives +
      ", TN " +
      row.true_negatives +
      ", FN " +
      row.false_negatives +
      ".";
    if (row.selected_operating_point) {
      message += " This is the selected operating point.";
    }
    byId("threshold-live").textContent = message;
    byId("count-tn").textContent = String(row.true_negatives);
    byId("count-fp").textContent = String(row.false_positives);
    byId("count-fn").textContent = String(row.false_negatives);
    byId("count-tp").textContent = String(row.true_positives);
    const selectedRow = byId("threshold-row-" + row.grid_index);
    if (shouldScroll && selectedRow) {
      selectedRow.scrollIntoView({ block: "nearest" });
    }
  }

  input.addEventListener("input", function () {
    show(Number(input.value), true);
  });
  byId("use-operating-point").addEventListener("click", function () {
    input.value = String(operating);
    show(operating, true);
    input.focus();
  });
  show(operating, false);
}

function renderExplanations(explanations) {
  const body = byId("coefficient-body");
  clear(body);
  explanations.top_coefficients.slice(0, 8).forEach(function (row) {
    addRow(body, [
      headerCell(row.display_name),
      cell(fixed(row.coefficient_transformed, 3)),
      cell(row.direction === "higher_review_score" ? "Higher review score" : "Lower review score"),
    ]);
  });
  byId("coefficient-caption").textContent = explanations.coefficient_caveat;
  const permutation = byId("permutation-body");
  clear(permutation);
  const block = explanations.permutation_importance;
  if (block.evidence_class === "unavailable") {
    byId("permutation-note").textContent = "Permutation importance is unavailable.";
    return;
  }
  block.rows.forEach(function (row) {
    addRow(permutation, [
      headerCell(row.feature),
      cell(fixed(row.importance_mean, 3)),
      cell(fixed(row.importance_std, 3)),
    ]);
  });
  byId("permutation-note").textContent =
    block.note +
    " Source run " +
    block.source_mlflow_run_id +
    " on the " +
    block.split +
    " split, scoring " +
    block.scoring +
    ". The registered champion run is " +
    explanations.model.mlflow_run_id +
    ".";
}

function inputList(inputs) {
  const list = document.createElement("dl");
  Object.keys(inputs).forEach(function (key) {
    const term = document.createElement("dt");
    term.textContent = key;
    const value = document.createElement("dd");
    value.textContent = String(inputs[key]);
    list.appendChild(term);
    list.appendChild(value);
  });
  return list;
}

function contributionTable(title, rows) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const caption = document.createElement("caption");
  caption.textContent = title;
  table.appendChild(caption);
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Contribution", "Log-odds"].forEach(function (label) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  rows.forEach(function (row) {
    addRow(body, [headerCell(row.display_name), cell(fixed(row.log_odds_contribution, 3))]);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function renderExamples(payload) {
  const select = byId("example-select");
  clear(select);
  payload.examples.forEach(function (example, index) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = example.title;
    select.appendChild(option);
  });

  function show(index) {
    const example = payload.examples[index];
    const detail = byId("example-detail");
    clear(detail);
    const title = document.createElement("h3");
    title.textContent = example.title;
    const description = document.createElement("p");
    description.textContent = example.description + " This is a stored offline result, not a live score.";
    const warning = document.createElement("p");
    warning.className = "note";
    warning.textContent = example.score_warning;
    const score = document.createElement("p");
    const flagged = example.flagged_for_review
      ? "Flagged for additional human review at the recorded threshold."
      : "Not flagged at the recorded threshold.";
    score.textContent =
      "Review score " +
      fixed(example.review_score, 3) +
      " at threshold " +
      fixed(example.decision_threshold, 3) +
      ". " +
      flagged;
    detail.appendChild(title);
    detail.appendChild(description);
    detail.appendChild(warning);
    detail.appendChild(score);
    detail.appendChild(inputList(example.inputs));
    detail.appendChild(contributionTable("Largest increases in the review score", example.strongest_increasing));
    detail.appendChild(contributionTable("Largest decreases in the review score", example.strongest_decreasing));
    const cause = document.createElement("p");
    cause.className = "note";
    cause.textContent = example.causation_disclaimer;
    detail.appendChild(cause);
  }

  select.addEventListener("change", function () {
    show(Number(select.value));
  });
  show(0);
}

function renderHistorical(historical) {
  const metrics = historical.metrics;
  const body = byId("historical-body");
  clear(body);
  addRow(body, [
    cell(fixed(metrics.pr_auc, 3)),
    cell(fixed(metrics.roc_auc, 3)),
    cell(fixed(metrics.precision, 3)),
    cell(fixed(metrics.recall, 3)),
    cell(String(metrics.true_positives)),
    cell(String(metrics.false_positives)),
    cell(String(metrics.false_negatives)),
    cell(String(metrics.true_negatives)),
    cell(String(metrics.n_samples)),
  ]);
  const intervals = byId("bootstrap-body");
  clear(intervals);
  historical.bootstrap_intervals.forEach(function (row) {
    addRow(intervals, [
      headerCell(row.metric),
      cell(fixed(row.point, 3)),
      cell(fixed(row.ci_low, 3) + " to " + fixed(row.ci_high, 3)),
    ]);
  });
  const perThousand = historical.selected_threshold_per_1000;
  byId("historical-note").textContent =
    historical.note +
    " At the recorded threshold, the frozen summary is about " +
    fixed(perThousand.reviews_per_1000, 0) +
    " reviews, " +
    fixed(perThousand.true_positives_per_1000, 0) +
    " true positives and " +
    fixed(perThousand.missed_positives_per_1000, 0) +
    " missed positives per 1,000 shipments. Bootstrap uses " +
    historical.bootstrap_intervals[0].n_bootstrap +
    " resamples, seed " +
    historical.bootstrap_intervals[0].seed +
    ".";
}

function renderProvenance(manifest) {
  const model = manifest.model;
  byId("provenance").textContent =
    "Evidence schema " +
    manifest.schema_version +
    ", generated " +
    manifest.generated_at +
    ". Champion " +
    model.version +
    ", MLflow run " +
    model.mlflow_run_id +
    ", validation threshold " +
    fixed(model.threshold, 3) +
    ". Curves and the threshold grid use the validation split. The historical test summary was not reproduced. " +
    model.score_warning;
}

function showError(error) {
  const note = byId("provenance");
  note.textContent = "The evidence bundle could not be verified in the browser: " + error.message;
}

async function main() {
  const response = await fetch(new URL("data/manifest.json", document.baseURI));
  if (!response.ok) {
    throw new Error("Could not load data/manifest.json");
  }
  const manifest = JSON.parse(await response.text());
  const documents = {};
  for (const name of DATA_FILES) {
    documents[name] = await loadJson(name, manifest);
  }
  renderProvenance(manifest);
  renderComparison(documents["model_comparison.json"]);
  renderCurves(documents["validation_curves.json"]);
  renderThreshold(documents["threshold_grid.json"]);
  renderExplanations(documents["explanations.json"]);
  renderExamples(documents["examples.json"]);
  renderHistorical(documents["historical_test.json"]);
}

main().catch(showError);
