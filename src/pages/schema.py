"""Allowlist for the public Pages bundle.

Only these paths may be committed under ``site/``. The allowlist is the
repository-boundary exception for aggregate portfolio evidence.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = "1.0.0"
PROJECT_SUBPATH = "Cargo-Risk-ML-Lab"
GITHUB_REPOSITORY = "https://github.com/karinehei/Cargo-Risk-ML-Lab"

SITE_ROOT_FILES: tuple[str, ...] = ("index.html", "styles.css", "app.js")

DATA_FILES: dict[str, str] = {
    "manifest.json": "pages.manifest",
    "model_comparison.json": "pages.model_comparison",
    "validation_curves.json": "pages.validation_curves",
    "threshold_grid.json": "pages.threshold_grid",
    "explanations.json": "pages.explanations",
    "examples.json": "pages.examples",
    "historical_test.json": "pages.historical_test",
}

FIGURE_FILES: tuple[str, ...] = (
    "validation_pr_curve.png",
    "validation_roc_curve.png",
    "validation_calibration_curve.png",
    "model_comparison_pr_auc.png",
    "champion_coefficients.png",
    "validation_threshold_tradeoff.png",
)

FORBIDDEN_SITE_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".pkl", ".joblib", ".sqlite", ".db", ".parquet"}
)

_WINDOWS_DRIVE = ":" + chr(92)

FORBIDDEN_PUBLIC_SUBSTRINGS: tuple[str, ...] = (
    "/mnt/",
    _WINDOWS_DRIVE,
    "mlruns/",
    ".joblib",
    ".pkl",
    "sqlite://",
    "runs:/",
    "models:/",
    "127.0.0.1",
    "localhost",
    "password",
    "api_key",
    "secret_key",
)

ROW_LEVEL_KEYS: frozenset[str] = frozenset(
    {"shipment_id", "y_true", "y_prob", "y_score", "probabilities", "prediction_row"}
)

PRIMARY_FAMILIES: tuple[str, ...] = ("dummy", "logreg", "random_forest", "xgboost")

DISPLAY_NAMES: dict[str, str] = {
    "dummy": "Dummy (class prior)",
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "logreg_uncalibrated_weighted": "Logistic regression, class-weighted, uncalibrated",
    "logreg_unweighted": "Logistic regression, unweighted",
    "logreg_sigmoid": "Logistic regression, sigmoid calibration",
    "logreg_isotonic": "Logistic regression, isotonic calibration",
}


def allowed_relative_files() -> tuple[str, ...]:
    """Repo-relative paths inside ``site/`` that the public site may contain."""
    data = tuple(f"data/{name}" for name in DATA_FILES)
    figures = tuple(f"figures/{name}" for name in FIGURE_FILES)
    return SITE_ROOT_FILES + data + figures


def unexpected_site_files(site_dir: Path) -> list[str]:
    """Return committed-style paths under ``site/`` that the allowlist rejects."""
    if not site_dir.exists():
        return []
    allowed = set(allowed_relative_files())
    unexpected: list[str] = []
    for path in sorted(site_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(site_dir).as_posix()
        if relative not in allowed or path.suffix.lower() in FORBIDDEN_SITE_SUFFIXES:
            unexpected.append(relative)
    return unexpected


def find_forbidden_substrings(text: str) -> list[str]:
    """Return forbidden public-bundle tokens present in ``text``."""
    return [token for token in FORBIDDEN_PUBLIC_SUBSTRINGS if token in text]


def find_row_level_keys(payload: object) -> list[str]:
    """Return nested object keys that would publish row-level records."""
    found: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ROW_LEVEL_KEYS:
                    found.append(str(key))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found
