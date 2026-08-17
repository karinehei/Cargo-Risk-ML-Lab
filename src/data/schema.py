"""Schema, vocabularies and column contracts for synthetic shipments.

All categories, ranges and labels are fictional educational constructs.
They must not be used for real customs, compliance or operational decisions.
"""

from __future__ import annotations

from typing import Final

DISCLAIMER: Final[str] = (
    "Educational demonstration using fully synthetic data and fictional review logic. "
    "Not affiliated with any customs authority. Do not use for real decisions."
)

TARGET_COLUMN: Final[str] = "requires_review"
ID_COLUMN: Final[str] = "shipment_id"
PERIOD_COLUMN: Final[str] = "generation_period"
DATE_COLUMN: Final[str] = "event_date"

ORIGIN_REGIONS: Final[list[str]] = [
    "Northern Europe",
    "Central Europe",
    "Southern Europe",
    "Asia",
    "Americas",
    "Africa",
]
DESTINATION_REGIONS: Final[list[str]] = [
    "Northern Europe",
    "Central Europe",
    "Southern Europe",
    "UK & Ireland",
]
COMMODITY_CATEGORIES: Final[list[str]] = [
    "electronics",
    "textiles",
    "machinery",
    "foodstuffs",
    "chemicals",
    "pharmaceuticals",
    "automotive",
    "other",
]
TRANSPORT_MODES: Final[list[str]] = ["road", "sea", "air", "rail"]

# Toy unit-value priors used only to build declared-vs-estimated deviation.
# These are not market prices and have no operational meaning.
COMMODITY_UNIT_VALUE: Final[dict[str, float]] = {
    "electronics": 160.0,
    "textiles": 18.0,
    "machinery": 28.0,
    "foodstuffs": 9.0,
    "chemicals": 35.0,
    "pharmaceuticals": 210.0,
    "automotive": 32.0,
    "other": 22.0,
}

# Columns persisted in the generated dataset (no latent scores).
RAW_COLUMNS: Final[list[str]] = [
    ID_COLUMN,
    DATE_COLUMN,
    PERIOD_COLUMN,
    "declared_value_eur",
    "shipment_weight_kg",
    "value_to_weight_ratio",
    "transport_mode",
    "origin_region",
    "destination_region",
    "commodity_category",
    "declaration_completeness_score",
    "documentation_count",
    "previous_discrepancies",
    "sender_history_length",
    "route_rarity",
    "declared_vs_estimated_value_deviation",
    "submission_hour",
    "expedited_shipment",
    TARGET_COLUMN,
]

FEATURE_COLUMNS: Final[list[str]] = [
    "declared_value_eur",
    "shipment_weight_kg",
    "value_to_weight_ratio",
    "transport_mode",
    "origin_region",
    "destination_region",
    "commodity_category",
    "declaration_completeness_score",
    "documentation_count",
    "previous_discrepancies",
    "sender_history_length",
    "route_rarity",
    "declared_vs_estimated_value_deviation",
    "submission_hour",
    "expedited_shipment",
]

NUMERIC_FEATURES: Final[list[str]] = [
    "declared_value_eur",
    "shipment_weight_kg",
    "value_to_weight_ratio",
    "declaration_completeness_score",
    "documentation_count",
    "previous_discrepancies",
    "sender_history_length",
    "route_rarity",
    "declared_vs_estimated_value_deviation",
    "submission_hour",
    "expedited_shipment",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "transport_mode",
    "origin_region",
    "destination_region",
    "commodity_category",
]

# Completeness is realistic for a subset of fields; identifiers and the target are never missing.
ALLOWED_MISSING_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "declaration_completeness_score",
        "documentation_count",
        "sender_history_length",
        "route_rarity",
        "declared_vs_estimated_value_deviation",
    }
)

# Must never appear in modelling tables (latent scores, post-outcome fields, identity proxies).
FORBIDDEN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "review_probability_latent",
        "latent_score",
        "logit",
        "true_risk",
        "inspector_decision",
        "gender",
        "age",
        "nationality",
        "ethnicity",
        "race",
        "religion",
        "disability",
        "name",
        "email",
        "phone",
        "person_id",
    }
)

PROTECTED_NAME_FRAGMENTS: Final[tuple[str, ...]] = (
    "gender",
    "sex",
    "age",
    "nationality",
    "ethnicity",
    "race",
    "religion",
    "disability",
    "passport",
    "ssn",
)

NUMERIC_RANGES: Final[dict[str, tuple[float, float]]] = {
    "declared_value_eur": (1.0, 5_000_000.0),
    "shipment_weight_kg": (0.05, 80_000.0),
    "value_to_weight_ratio": (0.0, 1_000_000.0),
    "declaration_completeness_score": (0.0, 1.0),
    "documentation_count": (0.0, 40.0),
    "previous_discrepancies": (0.0, 50.0),
    "sender_history_length": (0.0, 240.0),
    "route_rarity": (0.0, 1.0),
    "declared_vs_estimated_value_deviation": (-2.0, 8.0),
    "submission_hour": (0.0, 23.0),
    "expedited_shipment": (0.0, 1.0),
    PERIOD_COLUMN: (0.0, 20.0),
}

CATEGORY_VALUES: Final[dict[str, list[str]]] = {
    "origin_region": ORIGIN_REGIONS,
    "destination_region": DESTINATION_REGIONS,
    "commodity_category": COMMODITY_CATEGORIES,
    "transport_mode": TRANSPORT_MODES,
}

POSITIVE_RATE_BOUNDS: Final[tuple[float, float]] = (0.08, 0.15)
DEFAULT_N_SAMPLES: Final[int] = 15_000
DEFAULT_MISSING_RATE_BOUNDS: Final[tuple[float, float]] = (0.005, 0.08)
