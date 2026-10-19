"""Score a source occupation against the full target pool using a named variant."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import chroma as chroma_store
from . import signals  # noqa: F401 — register signals
from .crosswalk import CrosswalkError, Direction, resolve_direction
from .ground_truth import annotate_matches, cardinality_context, summarize_source_ground_truth
from .signals.base import get_signal
from .variants import Variant, get_variant
from .weights import get_active_weights, normalize_weights


def _as_float_list(vec: Any) -> list[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in list(vec)]


def score_variant_signal_arrays(
    source_id: str,
    direction: Direction | str,
    variant: Variant | str,
    *,
    context: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """
    Compute per-signal score arrays for the full target pool (no candidate dicts).

    Returns (target_ids, {signal_name: float32 array of shape (n_targets,)}).
    Reuses ``context`` caches across calls (skill matrices, facet matrices, row lists).
    """
    direction_enum = Direction(direction) if not isinstance(direction, Direction) else direction
    source_dataset, target_dataset = resolve_direction(direction_enum)
    var = get_variant(variant) if isinstance(variant, str) else variant

    target_ids, _titles, _ = chroma_store.get_facet_matrix(target_dataset, "title")
    if not target_ids:
        raise CrosswalkError(
            f"No target embeddings for {target_dataset}",
            step="score_variant_signal_arrays",
            context={"target_dataset": target_dataset},
        )

    source = chroma_store.load_occupation_embeddings(source_dataset, source_id)
    source_embeddings: dict[str, list[float]] = {}
    if source:
        source_embeddings = {
            facet: _as_float_list(vec) for facet, vec in source["embeddings"].items()
        }

    ctx: dict[str, Any] = context if context is not None else {}
    ctx["source_embeddings"] = source_embeddings

    pool_map = {
        "title_pool": "title",
        "description_pool": "description",
        "skills_pool": "skills",
        "knowledge_pool": "knowledge",
    }
    facet_matrices = ctx.setdefault("facet_matrices", {})
    for sig_name in var.signals:
        if sig_name in pool_map:
            facet = pool_map[sig_name]
            key = (target_dataset, facet)
            if key not in facet_matrices:
                facet_matrices[key] = chroma_store.get_facet_matrix(target_dataset, facet)

    signal_scores: dict[str, np.ndarray] = {}
    tid_list = list(target_ids)
    for sig_name in var.signals:
        sig = get_signal(sig_name)
        try:
            signal_scores[sig_name] = sig.score(
                source_id,
                source_dataset,
                tid_list,
                target_dataset,
                ctx,
            ).astype(np.float32, copy=False)
        except Exception as exc:  # noqa: BLE001
            raise CrosswalkError(
                f"Signal {sig_name} failed for {source_id}",
                step="score_signal",
                context={"signal": sig_name, "source_id": source_id},
                cause=exc,
            ) from exc
    return tid_list, signal_scores


def score_variant_pool(
    source_id: str,
    direction: Direction | str,
    variant: Variant | str,
    *,
    weights: dict[str, float] | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Full-pool scoring for one variant. Returns candidates sorted by weighted_score desc."""
    direction_enum = Direction(direction) if not isinstance(direction, Direction) else direction
    source_dataset, target_dataset = resolve_direction(direction_enum)
    var = get_variant(variant) if isinstance(variant, str) else variant

    if weights is not None:
        active_weights = normalize_weights(weights) if var.name == "P" else dict(weights)
        # keep only signals this variant uses
        active_weights = {k: v for k, v in active_weights.items() if k in var.signals}
        if not active_weights:
            active_weights = dict(var.default_weights)
    elif var.name == "P":
        active_weights = get_active_weights()
    else:
        active_weights = dict(var.default_weights)

    target_ids, target_titles_list, _ = chroma_store.get_facet_matrix(target_dataset, "title")
    if not target_ids:
        raise CrosswalkError(
            f"No target embeddings for {target_dataset}",
            step="score_variant_pool",
            context={"target_dataset": target_dataset},
        )
    title_by_id = {tid: title for tid, title in zip(target_ids, target_titles_list)}

    ctx: dict[str, Any] = dict(context or {})
    tid_list, signal_scores = score_variant_signal_arrays(
        source_id, direction_enum, var, context=ctx
    )

    usable = {
        k: v
        for k, v in active_weights.items()
        if k in signal_scores and float(np.abs(signal_scores[k]).sum()) > 1e-12
    }
    if not usable:
        usable = {k: active_weights[k] for k in active_weights if k in signal_scores}
    total_w = sum(usable.values()) or 1.0
    norm_w = {k: v / total_w for k, v in usable.items()}

    combined = np.zeros(len(tid_list), dtype=np.float64)
    for k, w in norm_w.items():
        combined += w * signal_scores[k].astype(np.float64)

    legacy_map = {
        "title_pool": "title",
        "description_pool": "description",
        "skills_pool": "skills",
        "knowledge_pool": "knowledge",
        "label_max": "title",
        "label_median": "title",
        "skill_bertscore_f1": "skills",
    }

    candidates: list[dict[str, Any]] = []
    for i, tid in enumerate(tid_list):
        facet_scores = {f: 0.0 for f in ("title", "description", "skills", "knowledge")}
        signal_display: dict[str, float] = {}
        for sig_name, arr in signal_scores.items():
            val = float(arr[i])
            signal_display[sig_name] = round(val, 6)
            legacy = legacy_map.get(sig_name)
            if legacy:
                facet_scores[legacy] = max(facet_scores[legacy], val)

        candidates.append(
            {
                "occupation_id": tid,
                "preferred_title": title_by_id.get(tid, tid),
                "weighted_score": round(float(combined[i]), 6),
                "title_similarity": round(facet_scores["title"], 6),
                "description_similarity": round(facet_scores["description"], 6),
                "skills_similarity": round(facet_scores["skills"], 6),
                "knowledge_similarity": round(facet_scores["knowledge"], 6),
                "facet_scores": {k: round(v, 6) for k, v in facet_scores.items()},
                "signal_scores": signal_display,
                "facets_used": list(norm_w.keys()),
                "weights_applied": {k: round(v, 6) for k, v in norm_w.items()},
            }
        )

    candidates.sort(key=lambda r: r["weighted_score"], reverse=True)
    return candidates


def variant_match(
    occupation_id: str,
    direction: Direction | str,
    variant: str = "P",
    top_k: int = 20,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Crosswalk with a named variant: weighted signal fusion, ranked by score."""
    direction_enum = Direction(direction) if not isinstance(direction, Direction) else direction
    source_dataset, target_dataset = resolve_direction(direction_enum)
    var = get_variant(variant)

    full_pool = score_variant_pool(
        occupation_id,
        direction_enum,
        var,
        weights=weights,
    )
    embedding_matches = full_pool[: max(1, top_k)]
    emb_annotated, emb_gt = annotate_matches(
        direction_enum.value, occupation_id, embedding_matches
    )
    source_gt = summarize_source_ground_truth(direction_enum.value, occupation_id)

    preferred = ""
    src = chroma_store.load_occupation_embeddings(source_dataset, occupation_id)
    preferred = (src or {}).get("preferred_title") or occupation_id

    return {
        "direction": direction_enum.value,
        "variant": var.name,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source": {
            "occupation_id": occupation_id,
            "preferred_title": preferred,
            "dataset": source_dataset,
        },
        "weights": dict(weights or (get_active_weights() if var.name == "P" else var.default_weights)),
        "top_k": top_k,
        "pool_size": len(full_pool),
        "ground_truth": {
            "source": source_gt,
            "cardinality": cardinality_context(direction_enum.value, [occupation_id]),
            "inverted_from_official": direction_enum.value == "esco_to_onet",
        },
        "embedding": {
            "step": "variant_match",
            "match_count": len(emb_annotated),
            "matches": emb_annotated,
            "ground_truth_total": emb_gt["total"],
            "ground_truth_found_in_top_k": emb_gt["found_in_top_k"],
            "ground_truth_found_by_type": emb_gt["found_by_type"],
            "ground_truth_expected_by_type": emb_gt["expected_by_type"],
        },
    }
