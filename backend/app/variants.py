"""Named scoring models. Model P is the sole production embedding/scoring model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .taxonomies import DEFAULT_PAIR_WEIGHTS, POOLED_SIGNALS


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    signals: tuple[str, ...]
    default_weights: dict[str, float]
    fixed_weights: bool = False
    notes: str = ""


VARIANTS: dict[str, Variant] = {
    "P": Variant(
        name="P",
        description="Production: title max/median + description/skills/knowledge pools",
        signals=POOLED_SIGNALS,
        default_weights=dict(DEFAULT_PAIR_WEIGHTS),
        notes="Uses config/active_weights.json when applied from Optimize",
    ),
}


def get_variant(name: str) -> Variant:
    key = name.strip().upper() if name.strip().upper() in VARIANTS else name.strip()
    if key not in VARIANTS:
        if name.strip().lower() == "p":
            return VARIANTS["P"]
        raise KeyError(f"Unknown model: {name}. Only model P is supported.")
    return VARIANTS[key]


def list_variants() -> list[dict[str, Any]]:
    return [
        {
            "name": v.name,
            "description": v.description,
            "signals": list(v.signals),
            "default_weights": dict(v.default_weights),
            "fixed_weights": v.fixed_weights,
            "notes": v.notes,
        }
        for v in VARIANTS.values()
    ]


# Public aliases aligning with the "model" API surface.
get_model = get_variant
list_models = list_variants
