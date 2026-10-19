"""Taxonomy-agnostic registry for pillars, taxonomies, and crosswalk pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Pillar:
    key: str
    collection_suffix: str
    aggregate: str | None = None  # None | "max" | "median"


@dataclass(frozen=True)
class Taxonomy:
    key: str
    label: str
    pillars: tuple[Pillar, ...]
    id_field: str
    label_collection: str | None = None  # multi-label title collection, e.g. esco_label


@dataclass(frozen=True)
class TaxonomyPair:
    input_taxonomy: str
    output_taxonomy: str
    direction: str
    ground_truth_path: str
    signals: tuple[str, ...]
    default_weights: dict[str, float]
    label: str = ""


# Shared facet pillars used by both O*NET and ESCO today.
_SHARED_FACET_PILLARS = (
    Pillar(key="preferred_title", collection_suffix="title"),
    Pillar(key="description", collection_suffix="description"),
    Pillar(key="skills", collection_suffix="skills"),
    Pillar(key="knowledge", collection_suffix="knowledge"),
)

TAXONOMIES: dict[str, Taxonomy] = {
    "onet": Taxonomy(
        key="onet",
        label="O*NET",
        pillars=_SHARED_FACET_PILLARS,
        id_field="onet_soc_code",
        label_collection=None,
    ),
    "esco": Taxonomy(
        key="esco",
        label="ESCO",
        pillars=(
            Pillar(key="preferred_title", collection_suffix="title"),
            Pillar(key="alternate_title", collection_suffix="label", aggregate="max"),
            Pillar(key="description", collection_suffix="description"),
            Pillar(key="skills", collection_suffix="skills"),
            Pillar(key="knowledge", collection_suffix="knowledge"),
        ),
        id_field="esco_uri",
        label_collection="esco_label",
    ),
}

POOLED_SIGNALS = (
    "label_max",
    "label_median",
    "description_pool",
    "skills_pool",
    "knowledge_pool",
)

# Optimized-P defaults from requirements (Tmx=0.32, Tmd=0.12, D=0.26, S=0.24, K=0.06)
DEFAULT_PAIR_WEIGHTS: dict[str, float] = {
    "label_max": 0.32,
    "label_median": 0.12,
    "description_pool": 0.26,
    "skills_pool": 0.24,
    "knowledge_pool": 0.06,
}

TAXONOMY_PAIRS: dict[str, TaxonomyPair] = {
    "onet_to_esco": TaxonomyPair(
        input_taxonomy="onet",
        output_taxonomy="esco",
        direction="onet_to_esco",
        ground_truth_path="./Ground truth Onet to Esco/ONET_(Occupations)_0_updated.csv",
        signals=POOLED_SIGNALS,
        default_weights=dict(DEFAULT_PAIR_WEIGHTS),
        label="O*NET → ESCO",
    ),
}

DEFAULT_MODEL = "P"
DEFAULT_INPUT_TAXONOMY = "onet"
DEFAULT_OUTPUT_TAXONOMY = "esco"
DEFAULT_DIRECTION = "onet_to_esco"


def list_taxonomies() -> list[dict[str, Any]]:
    return [
        {
            "key": t.key,
            "label": t.label,
            "pillars": [
                {
                    "key": p.key,
                    "collection_suffix": p.collection_suffix,
                    "aggregate": p.aggregate,
                }
                for p in t.pillars
            ],
            "id_field": t.id_field,
            "label_collection": t.label_collection,
        }
        for t in TAXONOMIES.values()
    ]


def get_taxonomy(key: str) -> Taxonomy:
    if key not in TAXONOMIES:
        raise KeyError(f"Unknown taxonomy: {key}. Choose from {list(TAXONOMIES)}")
    return TAXONOMIES[key]


def get_pair(
    input_taxonomy: str | None = None,
    output_taxonomy: str | None = None,
    *,
    direction: str | None = None,
) -> TaxonomyPair:
    if direction:
        if direction not in TAXONOMY_PAIRS:
            raise KeyError(f"Unknown direction: {direction}")
        return TAXONOMY_PAIRS[direction]
    inp = (input_taxonomy or DEFAULT_INPUT_TAXONOMY).strip().lower()
    out = (output_taxonomy or DEFAULT_OUTPUT_TAXONOMY).strip().lower()
    key = f"{inp}_to_{out}"
    if key not in TAXONOMY_PAIRS:
        raise KeyError(f"Unknown taxonomy pair: {inp} → {out}")
    return TAXONOMY_PAIRS[key]


def list_pairs() -> list[dict[str, Any]]:
    return [
        {
            "direction": p.direction,
            "input_taxonomy": p.input_taxonomy,
            "output_taxonomy": p.output_taxonomy,
            "label": p.label or f"{p.input_taxonomy} → {p.output_taxonomy}",
            "signals": list(p.signals),
            "default_weights": dict(p.default_weights),
            "ground_truth_path": p.ground_truth_path,
        }
        for p in TAXONOMY_PAIRS.values()
    ]


def datasets() -> tuple[str, ...]:
    return tuple(TAXONOMIES.keys())


def facets() -> tuple[str, ...]:
    """Facet collection suffixes used for {dataset}_{facet} Chroma collections."""
    seen: list[str] = []
    for tax in TAXONOMIES.values():
        for pillar in tax.pillars:
            if pillar.aggregate is not None:
                continue
            if pillar.collection_suffix not in seen:
                seen.append(pillar.collection_suffix)
    return tuple(seen)


def label_collections() -> tuple[str, ...]:
    return tuple(
        t.label_collection for t in TAXONOMIES.values() if t.label_collection
    )


def pair_direction_map() -> dict[str, tuple[str, str]]:
    return {
        p.direction: (p.input_taxonomy, p.output_taxonomy)
        for p in TAXONOMY_PAIRS.values()
    }


def api_payload() -> dict[str, Any]:
    """Payload for GET /api/taxonomies."""
    return {
        "default_model": DEFAULT_MODEL,
        "default_input_taxonomy": DEFAULT_INPUT_TAXONOMY,
        "default_output_taxonomy": DEFAULT_OUTPUT_TAXONOMY,
        "default_direction": DEFAULT_DIRECTION,
        "taxonomies": list_taxonomies(),
        "pairs": list_pairs(),
        "models": [
            {
                "key": DEFAULT_MODEL,
                "label": "Model P",
                "description": "Production: title max/median + description/skills/knowledge pools",
            }
        ],
    }
