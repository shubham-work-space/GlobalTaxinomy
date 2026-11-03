"""Official ESCO–O*NET crosswalk ground truth (graded match types)."""

from __future__ import annotations

import csv
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .config import get_settings

MatchType = Literal["exactMatch", "closeMatch", "broadMatch", "narrowMatch"]

MATCH_TYPES: frozenset[str] = frozenset(
    {"exactMatch", "closeMatch", "broadMatch", "narrowMatch"}
)

# Swap broad/narrow when inverting O*NET-perspective labels to ESCO-perspective
INVERTED_MATCH_TYPE: dict[str, str] = {
    "broadMatch": "narrowMatch",
    "narrowMatch": "broadMatch",
    "exactMatch": "exactMatch",
    "closeMatch": "closeMatch",
}

# Graded relevance gains for NDCG
MATCH_GAIN: dict[str, float] = {
    "exactMatch": 3.0,
    "closeMatch": 2.0,
    "broadMatch": 1.0,
    "narrowMatch": 1.0,
}

STRICT_MATCH_TYPES: frozenset[str] = frozenset({"exactMatch", "closeMatch"})

HEADER_ROW_INDEX = 16
DATA_START_INDEX = 17

# Official report benchmarks (Model 3, batch 3) — O*NET → ESCO only
REPORT_BENCHMARK_EXACT_TOP1 = 0.8475
REPORT_BENCHMARK_EXACT_TOP10 = 1.0
REPORT_BENCHMARK_ANY_TOP1 = 0.9096
REPORT_BENCHMARK_ANY_TOP5 = 0.9945


def _uri_to_esco_uuid(uri: str) -> str | None:
    uri = (uri or "").strip()
    if "/esco/occupation/" not in uri:
        return None
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _esco_canonical_id(esco_id: str) -> str:
    """Normalize ESCO id to full occupation URI (Chroma storage format)."""
    esco_id = (esco_id or "").strip()
    if esco_id.startswith("http://") or esco_id.startswith("https://"):
        return esco_id.rstrip("/")
    return f"http://data.europa.eu/esco/occupation/{esco_id}"


def _esco_lookup_keys(esco_id: str) -> set[str]:
    """Keys that may appear in stored embeddings or ground-truth CSV."""
    esco_id = (esco_id or "").strip()
    uuid = _uri_to_esco_uuid(esco_id) or esco_id
    return {uuid, _esco_canonical_id(uuid)}


def _invert_match_type(match_type: str) -> str:
    return INVERTED_MATCH_TYPE.get(match_type, match_type)


def _parse_crosswalk_csv(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    Returns:
        onet_to_esco: onet_id -> {esco_uri: match_type} (O*NET perspective)
        esco_to_onet: esco_uri -> {onet_id: match_type} (ESCO perspective, broad/narrow swapped)
    """
    onet_to_esco: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return onet_to_esco, {}

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))

    if len(rows) <= DATA_START_INDEX:
        return onet_to_esco, {}

    for row in rows[DATA_START_INDEX:]:
        if len(row) < 7:
            continue
        onet_id = row[0].strip()
        esco_uri = row[3].strip()
        match_type = row[6].strip()
        if not onet_id or not esco_uri or match_type not in MATCH_TYPES:
            continue
        esco_uuid = _uri_to_esco_uuid(esco_uri)
        if not esco_uuid:
            continue
        esco_key = _esco_canonical_id(esco_uuid)
        onet_to_esco.setdefault(onet_id, {})[esco_key] = match_type

    esco_to_onet: dict[str, dict[str, str]] = {}
    for onet_id, targets in onet_to_esco.items():
        for esco_key, match_type in targets.items():
            esco_to_onet.setdefault(esco_key, {})[onet_id] = _invert_match_type(match_type)

    return onet_to_esco, esco_to_onet


@lru_cache
def _load_maps() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    settings = get_settings()
    path = settings.ground_truth_path
    return _parse_crosswalk_csv(path)


def reload_ground_truth() -> None:
    """Clear cached maps (e.g. after path change)."""
    _load_maps.cache_clear()


def eval_source_ids(direction: str) -> list[str]:
    """Source occupation ids in the format used by Chroma embeddings."""
    if direction == "onet_to_esco":
        return sorted(onet_to_esco_map().keys())
    if direction == "esco_to_onet":
        return sorted(esco_to_onet_map().keys())
    return []


def onet_to_esco_map() -> dict[str, dict[str, str]]:
    return _load_maps()[0]


def esco_to_onet_map() -> dict[str, dict[str, str]]:
    return _load_maps()[1]


def direction_map(direction: str) -> dict[str, dict[str, str]]:
    if direction == "onet_to_esco":
        return onet_to_esco_map()
    if direction == "esco_to_onet":
        return esco_to_onet_map()
    return {}


def _normalize_source_id(direction: str, source_id: str) -> str:
    if direction == "esco_to_onet":
        return _esco_canonical_id(source_id)
    return source_id.strip()


def lookup_match_type(
    direction: str,
    source_id: str,
    target_id: str,
) -> str | None:
    """Return direction-correct match type for source → target."""
    source_id = _normalize_source_id(direction, source_id)
    if direction == "onet_to_esco":
        targets = onet_to_esco_map().get(source_id, {})
        for key in _esco_lookup_keys(target_id):
            if key in targets:
                return targets[key]
        return targets.get(_esco_canonical_id(target_id))
    if direction == "esco_to_onet":
        return esco_to_onet_map().get(source_id, {}).get(target_id.strip())
    return None


def lookup_official_match_type(
    source_id: str,
    target_id: str,
    direction: str,
) -> str | None:
    """Return match type in official O*NET-perspective (always from onet_to_esco map)."""
    if direction == "onet_to_esco":
        return lookup_match_type(direction, source_id, target_id)
    # ESCO → O*NET: read original label from forward map
    onet_id = target_id.strip()
    esco_key = _normalize_source_id("esco_to_onet", source_id)
    targets = onet_to_esco_map().get(onet_id, {})
    for key in _esco_lookup_keys(esco_key):
        if key in targets:
            return targets[key]
    return targets.get(esco_key)


def relevant_targets(direction: str, source_id: str) -> dict[str, str]:
    """All ground-truth targets for a source: target_id -> direction-correct match_type."""
    source_id = _normalize_source_id(direction, source_id)
    if direction == "onet_to_esco":
        return dict(onet_to_esco_map().get(source_id, {}))
    if direction == "esco_to_onet":
        return dict(esco_to_onet_map().get(source_id, {}))
    return {}


def _count_by_type(targets: dict[str, str]) -> dict[str, int]:
    counts = {t: 0 for t in MATCH_TYPES}
    for match_type in targets.values():
        if match_type in counts:
            counts[match_type] += 1
    return counts


def summarize_source_ground_truth(direction: str, source_id: str) -> dict[str, Any]:
    """Per-source ground-truth summary for API / UI."""
    relevant = relevant_targets(direction, source_id)
    by_type = _count_by_type(relevant)
    return {
        "total": len(relevant),
        "by_type": by_type,
        "has_exact_match": by_type["exactMatch"] > 0,
        "exact_match_count": by_type["exactMatch"],
    }


def source_has_exact_match(direction: str, source_id: str) -> bool:
    """True when the source has ≥1 exactMatch target (exact@1 eval set)."""
    return summarize_source_ground_truth(direction, source_id)["has_exact_match"]


def cardinality_context(direction: str, source_ids: list[str] | None = None) -> dict[str, Any]:
    """
    Mapping cardinality stats for a direction (eval report / API).
    If source_ids is provided, stats are computed over that subset only.
    """
    m = direction_map(direction)
    ids = source_ids if source_ids is not None else sorted(m.keys())
    counts = [len(m[sid]) for sid in ids if sid in m]
    if not counts:
        return {
            "direction": direction,
            "source_count": 0,
            "mean_targets_per_source": 0.0,
            "median_targets_per_source": 0,
            "max_targets_per_source": 0,
            "percent_single_target": 0.0,
            "sources_with_exact_match": 0,
            "sources_with_any_match": 0,
            "inverted_from_official": direction == "esco_to_onet",
        }

    single = sum(1 for c in counts if c == 1)
    with_exact = sum(
        1 for sid in ids if sid in m and any(t == "exactMatch" for t in m[sid].values())
    )
    return {
        "direction": direction,
        "source_count": len(counts),
        "mean_targets_per_source": round(statistics.mean(counts), 2),
        "median_targets_per_source": int(statistics.median(counts)),
        "max_targets_per_source": max(counts),
        "percent_single_target": round(100.0 * single / len(counts), 1),
        "sources_with_exact_match": with_exact,
        "sources_with_any_match": len(counts),
        "inverted_from_official": direction == "esco_to_onet",
    }


def annotate_matches(
    direction: str,
    source_id: str,
    matches: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """
    Add ground_truth_match_type (direction-correct) and
    ground_truth_match_type_official (O*NET-perspective) to each match.
    Returns annotated matches and summary dict.
    """
    relevant = relevant_targets(direction, source_id)
    found = 0
    found_by_type = {t: 0 for t in MATCH_TYPES}
    annotated: list[dict] = []

    for m in matches:
        target_id = str(m.get("occupation_id", ""))
        match_type = lookup_match_type(direction, source_id, target_id)
        official_type = lookup_official_match_type(source_id, target_id, direction)
        row = dict(m)
        row["ground_truth_match_type"] = match_type
        row["ground_truth_match_type_official"] = official_type
        if match_type:
            found += 1
            if match_type in found_by_type:
                found_by_type[match_type] += 1
        annotated.append(row)

    source_summary = summarize_source_ground_truth(direction, source_id)
    summary: dict[str, Any] = {
        "total": len(relevant),
        "found_in_top_k": found,
        "found_by_type": found_by_type,
        "expected_by_type": source_summary["by_type"],
        "has_exact_match": source_summary["has_exact_match"],
        "exact_match_count": source_summary["exact_match_count"],
    }
    return annotated, summary


def gain_for_match_type(match_type: str | None, strict: bool = False) -> float:
    if not match_type:
        return 0.0
    if strict and match_type not in STRICT_MATCH_TYPES:
        return 0.0
    return MATCH_GAIN.get(match_type, 0.0)


def eval_methodology_notes() -> dict[str, str]:
    return {
        "ground_truth_construction": (
            "Official crosswalk built by validating top-10 BERT model suggestions "
            "plus a manual orphans batch (652 ESCO occupations). Measures agreement "
            "with the published mapping, not absolute semantic correctness."
        ),
        "report_top10_note": (
            "Report Table 3/4 Top-10 accuracy of 1.0 is near-tautological because "
            "ground truth was selected from top-10 model suggestions. Top-1 is the "
            "fairer comparison point."
        ),
        "version_note": (
            "Ground truth: ESCO v1.1.0 + O*NET-SOC 2019. "
            "Embeddings default to ESCO v1.2.1; use chroma_db_1.1.0 / "
            "ESCO_LABELS_JSON=esco_labels_1.1.0.json for aligned v1.1.0 runs."
        ),
    }
