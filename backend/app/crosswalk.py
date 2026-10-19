"""Direction-agnostic embedding match: exact full-pool cosine + weighted ranking."""

from __future__ import annotations

import traceback
from enum import Enum
from typing import Any, Literal

import numpy as np

from . import chroma as chroma_store
from .chroma import FACETS
from .ground_truth import annotate_matches, cardinality_context, summarize_source_ground_truth
from .taxonomies import pair_direction_map
from .weights import DEFAULT_WEIGHTS, get_active_weights

Dataset = Literal["onet", "esco"]


class Direction(str, Enum):
    ONET_TO_ESCO = "onet_to_esco"
    ESCO_TO_ONET = "esco_to_onet"


# Built from the taxonomy registry; reverse direction kept for existing UI search.
_REGISTRY_MAP = pair_direction_map()
DIRECTION_MAP: dict[Direction, tuple[Dataset, Dataset]] = {
    Direction.ONET_TO_ESCO: ("onet", "esco"),
    Direction.ESCO_TO_ONET: ("esco", "onet"),
}
for _dir_key, (_src, _tgt) in _REGISTRY_MAP.items():
    try:
        DIRECTION_MAP[Direction(_dir_key)] = (_src, _tgt)  # type: ignore[assignment]
    except ValueError:
        pass


class CrosswalkError(Exception):
    """Structured matching failure for API / UI detail panels."""

    def __init__(
        self,
        message: str,
        *,
        step: str,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.step = step
        self.context = context or {}
        self.cause = cause
        self.cause_type = type(cause).__name__ if cause else None
        self.traceback_text = (
            "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
            if cause
            else None
        )

    def to_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "message": self.message,
            "error_type": self.cause_type or type(self).__name__,
            "step": self.step,
            "context": self.context,
        }
        if self.cause is not None:
            detail["cause"] = str(self.cause)
        if self.traceback_text:
            detail["traceback"] = self.traceback_text
        return detail


def resolve_direction(direction: Direction | str) -> tuple[Dataset, Dataset]:
    d = Direction(direction) if not isinstance(direction, Direction) else direction
    return DIRECTION_MAP[d]


def _as_float_list(vec: Any) -> list[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in list(vec)]


def _normalize_vector(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return arr
    return arr / norm


def _active_weights(
    source_embeddings: dict[str, list[float]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Drop empty/missing source facets and renormalize remaining weights."""
    base = dict(weights or get_active_weights())
    active = {
        facet: base[facet]
        for facet in FACETS
        if facet in base and source_embeddings.get(facet)
    }
    total = sum(active.values())
    if total <= 0:
        return {}
    return {f: w / total for f, w in active.items()}


def apply_facet_weights(
    facet_scores: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Weighted sum over facets using fixed weights (no renormalization).
    Every candidate in the full-pool path has all active facet scores.
    """
    used = {f: weights[f] for f in weights if f in facet_scores}
    if not used:
        return 0.0, {}
    score = sum(used[f] * facet_scores[f] for f in used)
    return float(score), used


def _score_full_pool(
    target_dataset: Dataset,
    source_embeddings: dict[str, list[float]],
    active_weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Exact cosine similarity against every target occupation for each active facet."""
    pool_ids: list[str] = []
    pool_seen: set[str] = set()
    pool_titles: dict[str, str] = {}
    facet_score_maps: dict[str, dict[str, float]] = {}

    for facet in active_weights:
        try:
            rows = chroma_store.score_facet_all(
                target_dataset,
                facet,
                source_embeddings[facet],
            )
        except Exception as exc:  # noqa: BLE001
            raise CrosswalkError(
                f"Failed to score facet for {target_dataset}_{facet}",
                step="load_facet_matrix",
                context={"target_dataset": target_dataset, "facet": facet},
                cause=exc,
            ) from exc

        if not rows:
            raise CrosswalkError(
                f"No embeddings in {target_dataset}_{facet}",
                step="load_facet_matrix",
                context={"target_dataset": target_dataset, "facet": facet},
            )

        facet_scores: dict[str, float] = {}
        for row in rows:
            occ_id = str(row["occupation_id"])
            if occ_id not in pool_seen:
                pool_seen.add(occ_id)
                pool_ids.append(occ_id)
            pool_titles[occ_id] = str(row["preferred_title"])
            facet_scores[occ_id] = float(row["score"])
        facet_score_maps[facet] = facet_scores

    if not pool_ids:
        return []

    candidates: list[dict[str, Any]] = []
    for occ_id in pool_ids:
        facet_scores = {
            facet: float(facet_score_maps[facet].get(occ_id, 0.0))
            for facet in active_weights
        }
        for facet in FACETS:
            if facet not in facet_scores:
                facet_scores[facet] = 0.0

        weighted, used_weights = apply_facet_weights(facet_scores, active_weights)
        display = {f: round(facet_scores[f], 6) for f in FACETS}

        candidates.append(
            {
                "occupation_id": occ_id,
                "preferred_title": pool_titles.get(occ_id, occ_id),
                "weighted_score": round(weighted, 6),
                "title_similarity": display["title"],
                "description_similarity": display["description"],
                "skills_similarity": display["skills"],
                "knowledge_similarity": display["knowledge"],
                "facet_scores": display,
                "facets_used": list(used_weights.keys()),
                "weights_applied": {k: round(v, 6) for k, v in used_weights.items()},
            }
        )

    candidates.sort(key=lambda r: r["weighted_score"], reverse=True)
    return candidates


def embedding_match(
    occupation_id: str,
    direction: Direction | str,
    top_k: int = 20,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Exact full-pool cosine + weighted sum, ranked by weighted_score."""
    direction_enum = Direction(direction) if not isinstance(direction, Direction) else direction
    source_dataset, target_dataset = resolve_direction(direction_enum)

    try:
        source = chroma_store.load_occupation_embeddings(source_dataset, occupation_id)
    except Exception as exc:  # noqa: BLE001
        raise CrosswalkError(
            f"Failed to load source embeddings for {source_dataset}/{occupation_id}",
            step="load_source",
            context={
                "source_dataset": source_dataset,
                "occupation_id": occupation_id,
                "direction": direction_enum.value,
            },
            cause=exc,
        ) from exc

    if not source:
        raise LookupError(
            f"Source occupation embeddings not found: {source_dataset}/{occupation_id}"
        )

    source_embeddings: dict[str, list[float]] = {
        facet: _as_float_list(vec) for facet, vec in source["embeddings"].items()
    }
    active_weights = _active_weights(source_embeddings, weights)
    if not active_weights:
        raise ValueError("Source occupation has no usable facet embeddings")

    full_pool = _score_full_pool(target_dataset, source_embeddings, active_weights)
    if not full_pool:
        raise CrosswalkError(
            f"No candidates scored for {target_dataset}",
            step="score_pool",
            context={
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "occupation_id": occupation_id,
            },
        )

    embedding_matches = full_pool[: max(1, top_k)]
    emb_annotated, emb_gt = annotate_matches(
        direction_enum.value,
        occupation_id,
        embedding_matches,
    )
    source_gt = summarize_source_ground_truth(direction_enum.value, occupation_id)

    return {
        "direction": direction_enum.value,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source": {
            "occupation_id": source["occupation_id"],
            "preferred_title": source["preferred_title"],
            "dataset": source_dataset,
        },
        "weights": active_weights,
        "top_k": top_k,
        "pool_size": len(full_pool),
        "ground_truth": {
            "source": source_gt,
            "cardinality": cardinality_context(direction_enum.value, [occupation_id]),
            "inverted_from_official": direction_enum.value == "esco_to_onet",
        },
        "embedding": {
            "step": "embedding_match",
            "match_count": len(emb_annotated),
            "matches": emb_annotated,
            "ground_truth_total": emb_gt["total"],
            "ground_truth_found_in_top_k": emb_gt["found_in_top_k"],
            "ground_truth_found_by_type": emb_gt["found_by_type"],
            "ground_truth_expected_by_type": emb_gt["expected_by_type"],
        },
    }


def search_occupations(
    dataset: Dataset,
    query: str,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Substring search over preferred titles stored in the title collection."""
    return chroma_store.search_occupations(dataset, query, limit)
