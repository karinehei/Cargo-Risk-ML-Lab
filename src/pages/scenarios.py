"""Hand-specified fictional shipments for offline champion scoring.

These inputs are not copied from dataset rows. The export scores them and
checks that the resulting feature values do not match a stored row.
"""

from __future__ import annotations

from typing import Any

REQUIRED_INPUTS: tuple[str, ...] = (
    "origin_region",
    "destination_region",
    "commodity_category",
    "transport_mode",
    "declared_value_eur",
    "shipment_weight_kg",
    "declaration_completeness_score",
    "documentation_count",
    "previous_discrepancies",
    "sender_history_length",
    "route_rarity",
    "declared_vs_estimated_value_deviation",
    "submission_hour",
    "expedited_shipment",
)

FICTIONAL_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "example_id": "complete-sea-foodstuffs",
        "title": "Complete sea shipment of foodstuffs",
        "description": (
            "Hand-specified fictional input: a complete foodstuffs declaration "
            "moving by sea during ordinary hours."
        ),
        "inputs": {
            "origin_region": "Northern Europe",
            "destination_region": "Northern Europe",
            "commodity_category": "foodstuffs",
            "transport_mode": "sea",
            "declared_value_eur": 2400.0,
            "shipment_weight_kg": 800.0,
            "declaration_completeness_score": 0.96,
            "documentation_count": 8,
            "previous_discrepancies": 0,
            "sender_history_length": 42,
            "route_rarity": 0.18,
            "declared_vs_estimated_value_deviation": -0.04,
            "submission_hour": 11,
            "expedited_shipment": 0,
        },
    },
    {
        "example_id": "expedited-air-pharmaceuticals",
        "title": "Expedited air shipment of pharmaceuticals",
        "description": (
            "Hand-specified fictional input: an expedited air pharmaceutical "
            "shipment with a large declared-value deviation and an off-hours submission."
        ),
        "inputs": {
            "origin_region": "Asia",
            "destination_region": "Northern Europe",
            "commodity_category": "pharmaceuticals",
            "transport_mode": "air",
            "declared_value_eur": 48750.0,
            "shipment_weight_kg": 12.5,
            "declaration_completeness_score": 0.41,
            "documentation_count": 1,
            "previous_discrepancies": 3,
            "sender_history_length": 2,
            "route_rarity": 0.91,
            "declared_vs_estimated_value_deviation": 1.75,
            "submission_hour": 23,
            "expedited_shipment": 1,
        },
    },
    {
        "example_id": "incomplete-air-electronics",
        "title": "Incomplete air shipment of electronics",
        "description": (
            "Hand-specified fictional input: electronics sent by air with a "
            "sparse document count and a night-time submission."
        ),
        "inputs": {
            "origin_region": "Americas",
            "destination_region": "UK & Ireland",
            "commodity_category": "electronics",
            "transport_mode": "air",
            "declared_value_eur": 16340.0,
            "shipment_weight_kg": 36.0,
            "declaration_completeness_score": 0.28,
            "documentation_count": 0,
            "previous_discrepancies": 1,
            "sender_history_length": 6,
            "route_rarity": 0.57,
            "declared_vs_estimated_value_deviation": 0.82,
            "submission_hour": 2,
            "expedited_shipment": 0,
        },
    },
    {
        "example_id": "ordinary-road-machinery",
        "title": "Ordinary road shipment of machinery",
        "description": (
            "Hand-specified fictional input: a road machinery shipment with "
            "middling completeness and no previous discrepancies."
        ),
        "inputs": {
            "origin_region": "Central Europe",
            "destination_region": "Central Europe",
            "commodity_category": "machinery",
            "transport_mode": "road",
            "declared_value_eur": 6150.0,
            "shipment_weight_kg": 940.0,
            "declaration_completeness_score": 0.81,
            "documentation_count": 4,
            "previous_discrepancies": 0,
            "sender_history_length": 17,
            "route_rarity": 0.12,
            "declared_vs_estimated_value_deviation": 0.06,
            "submission_hour": 14,
            "expedited_shipment": 0,
        },
    },
)


def validate_scenario_specs(scenarios: tuple[dict[str, Any], ...] = FICTIONAL_SCENARIOS) -> None:
    """Reject specs that carry labels, identifiers, or a missing model field."""
    seen: set[str] = set()
    for scenario in scenarios:
        example_id = str(scenario["example_id"])
        if example_id in seen:
            raise ValueError(f"Duplicate fictional example id: {example_id}")
        seen.add(example_id)
        inputs = scenario["inputs"]
        if not isinstance(inputs, dict):
            raise ValueError(f"{example_id} inputs must be an object")
        missing = [key for key in REQUIRED_INPUTS if key not in inputs]
        if missing:
            raise ValueError(f"{example_id} is missing {missing}")
        forbidden = [
            key for key in inputs if key in {"shipment_id", "requires_review", "event_date"}
        ]
        if forbidden:
            raise ValueError(f"{example_id} must not include {forbidden}")
