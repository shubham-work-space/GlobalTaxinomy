"""Live O*NET↔ESCO crosswalk: Top-20 embeddings → Top-10 (embedding or Jev).

Flow:
1. Score full pool, take Top-20.
2. If Top-1 similarity ≥ threshold → return embedding Top-10.
3. If Top-1 < threshold → reuse a prior Jev ranking when available, else call Jev,
   then return ordered Top-10.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import chroma as chroma_store
from . import runs as run_store
from .config import ROOT
from .crosswalk import Direction, resolve_direction
from .ground_truth import annotate_matches, cardinality_context, summarize_source_ground_truth
from .jev_resolve import RANKED_TOP_N, resolve_batch
from .pipeline import _occupation_facets
from .variant_score import score_variant_pool
from .variants import get_variant
from .weights import get_active_weights

DEFAULT_THRESHOLD = 0.60
CANDIDATE_POOL = 20
LIVE_CACHE_ROOT = ROOT / "output" / "live_jev_cache"


def _safe_id(occupation_id: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", (occupation_id or "").strip())
    return cleaned[:160] or "occupation"


def _live_cache_path(direction: str, source_id: str) -> Path:
    return LIVE_CACHE_ROOT / direction / f"{_safe_id(source_id)}.json"


def load_live_jev_cache(direction: str, source_id: str) -> list[str] | None:
    path = _live_cache_path(direction, source_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ranked = data.get("jev_ranked_target_ids") if isinstance(data, dict) else None
    if not isinstance(ranked, list) or not ranked:
        return None
    return [str(x) for x in ranked if x][:RANKED_TOP_N]


def save_live_jev_cache(
    direction: str,
    source_id: str,
    ranked_ids: list[str],
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    path = _live_cache_path(direction, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": source_id,
        "direction": direction,
        "jev_ranked_target_ids": list(ranked_ids)[:RANKED_TOP_N],
        **(meta or {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_cached_jev_ranking(source_id: str, direction: str) -> tuple[list[str], str] | None:
    """Return (ranked_target_ids, origin) from live cache or completed eval runs."""
    live = load_live_jev_cache(direction, source_id)
    if live:
        return live, "jev_cached_live"

    for run in run_store.list_runs():
        if run.get("status") != "completed":
            continue
        if run.get("direction") != direction:
            continue
        run_id = str(run.get("run_id") or "")
        path = run_store.run_dir(run_id) / "results.jsonl"
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if source_id not in text:
            continue
        for line in text.splitlines():
            if source_id not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source_id") != source_id:
                continue
            ranked = row.get("jev_ranked_target_ids") or []
            if ranked and row.get("sent_to_jev"):
                ids = [str(x) for x in ranked if x][:RANKED_TOP_N]
                if ids:
                    return ids, f"jev_cached_run:{run.get('label') or run_id}"
    return None


def _to_candidate_rows(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, m in enumerate(matches, start=1):
        tid = m.get("occupation_id") or m.get("target_id")
        rows.append(
            {
                "rank": rank,
                "target_id": tid,
                "occupation_id": tid,
                "preferred_title": m.get("preferred_title") or tid,
                "similarity": float(m.get("weighted_score") or m.get("similarity") or 0.0),
                "weighted_score": float(m.get("weighted_score") or m.get("similarity") or 0.0),
                "title_similarity": m.get("title_similarity"),
                "description_similarity": m.get("description_similarity"),
                "skills_similarity": m.get("skills_similarity"),
                "knowledge_similarity": m.get("knowledge_similarity"),
                "facet_scores": m.get("facet_scores"),
                "signal_scores": m.get("signal_scores"),
                "facets_used": m.get("facets_used"),
                "weights_applied": m.get("weights_applied"),
            }
        )
    return rows


def _enrich_for_jev(
    target_dataset: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for cand in candidates:
        tid = str(cand.get("target_id") or cand.get("occupation_id") or "")
        facets = _occupation_facets(target_dataset, tid)
        enriched.append(
            {
                **cand,
                "preferred_title": facets.get("preferred_title")
                or cand.get("preferred_title"),
                "alternate_title": facets.get("alternate_title") or "",
                "description": facets.get("description") or "",
                "skills": facets.get("skills") or "",
                "knowledge": facets.get("knowledge") or "",
            }
        )
    return enriched


def _reorder_by_ids(
    pool: list[dict[str, Any]],
    ranked_ids: list[str],
    *,
    target_dataset: str,
) -> list[dict[str, Any]]:
    """Build ordered matches from ranked ids; fill titles from pool or chroma."""
    by_id = {
        str(m.get("occupation_id") or m.get("target_id")): m for m in pool
    }
    out: list[dict[str, Any]] = []
    for tid in ranked_ids:
        if len(out) >= RANKED_TOP_N:
            break
        base = by_id.get(tid)
        if base:
            out.append(dict(base))
            continue
        facets = _occupation_facets(target_dataset, tid)
        out.append(
            {
                "occupation_id": tid,
                "preferred_title": facets.get("preferred_title") or tid,
                "weighted_score": 0.0,
                "title_similarity": None,
                "description_similarity": None,
                "skills_similarity": None,
                "knowledge_similarity": None,
            }
        )
    # If ranking shorter than 10, append remaining embedding pool order
    if len(out) < RANKED_TOP_N:
        seen = {str(m.get("occupation_id")) for m in out}
        for m in pool:
            tid = str(m.get("occupation_id") or "")
            if tid and tid not in seen:
                out.append(dict(m))
                seen.add(tid)
            if len(out) >= RANKED_TOP_N:
                break
    return out[:RANKED_TOP_N]


def _build_response(
    *,
    direction: str,
    source_dataset: str,
    target_dataset: str,
    occupation_id: str,
    preferred: str,
    weights: dict[str, float],
    variant_name: str,
    pool_size: int,
    threshold: float,
    top1_similarity: float,
    ranking_source: str,
    sent_to_jev: bool,
    top20: list[dict[str, Any]],
    top10: list[dict[str, Any]],
    jev_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    top10_ann, top10_gt = annotate_matches(direction, occupation_id, top10)
    top20_ann, top20_gt = annotate_matches(direction, occupation_id, top20)
    source_gt = summarize_source_ground_truth(direction, occupation_id)
    return {
        "direction": direction,
        "variant": variant_name,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source": {
            "occupation_id": occupation_id,
            "preferred_title": preferred,
            "dataset": source_dataset,
        },
        "weights": weights,
        "top_k": RANKED_TOP_N,
        "candidate_pool": CANDIDATE_POOL,
        "pool_size": pool_size,
        "threshold": threshold,
        "top1_similarity": round(float(top1_similarity), 6),
        "ranking_source": ranking_source,
        "sent_to_jev": sent_to_jev,
        "matches": top10_ann,
        "ground_truth": {
            "source": source_gt,
            "cardinality": cardinality_context(direction, [occupation_id]),
            "inverted_from_official": direction == "esco_to_onet",
        },
        "embedding": {
            "step": "live_crosswalk",
            "match_count": len(top20_ann),
            "matches": top20_ann,
            "ground_truth_total": top20_gt["total"],
            "ground_truth_found_in_top_k": top20_gt["found_in_top_k"],
            "ground_truth_found_by_type": top20_gt["found_by_type"],
            "ground_truth_expected_by_type": top20_gt["expected_by_type"],
        },
        "ranked": {
            "step": ranking_source,
            "match_count": len(top10_ann),
            "matches": top10_ann,
            "ground_truth_total": top10_gt["total"],
            "ground_truth_found_in_top_k": top10_gt["found_in_top_k"],
            "ground_truth_found_by_type": top10_gt["found_by_type"],
            "ground_truth_expected_by_type": top10_gt["expected_by_type"],
        },
        "jev": jev_meta,
    }


async def live_crosswalk(
    occupation_id: str,
    direction: Direction | str,
    *,
    variant: str = "P",
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Live hybrid match returning Top-10 (embedding or Jev-ordered)."""
    direction_enum = Direction(direction) if not isinstance(direction, Direction) else direction
    source_dataset, target_dataset = resolve_direction(direction_enum)
    var = get_variant(variant)
    active_weights = weights or (
        get_active_weights() if var.name == "P" else dict(var.default_weights)
    )

    full_pool = score_variant_pool(
        occupation_id,
        direction_enum,
        var,
        weights=active_weights,
    )
    top20 = full_pool[:CANDIDATE_POOL]
    if not top20:
        raise LookupError(f"No candidates for {occupation_id}")

    top1_similarity = float(top20[0].get("weighted_score") or 0.0)
    preferred = ""
    src = chroma_store.load_occupation_embeddings(source_dataset, occupation_id)
    preferred = (src or {}).get("preferred_title") or occupation_id

    common = dict(
        direction=direction_enum.value,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        occupation_id=occupation_id,
        preferred=preferred,
        weights=dict(active_weights),
        variant_name=var.name,
        pool_size=len(full_pool),
        threshold=float(threshold),
        top1_similarity=top1_similarity,
        top20=top20,
    )

    # High confidence: embedding Top-10
    if top1_similarity >= float(threshold):
        return _build_response(
            **common,
            ranking_source="embedding",
            sent_to_jev=False,
            top10=top20[:RANKED_TOP_N],
            jev_meta=None,
        )

    # Low confidence: reuse prior Jev ranking when available
    cached = find_cached_jev_ranking(occupation_id, direction_enum.value)
    if cached:
        ranked_ids, origin = cached
        top10 = _reorder_by_ids(top20, ranked_ids, target_dataset=target_dataset)
        return _build_response(
            **common,
            ranking_source="jev_cached",
            sent_to_jev=True,
            top10=top10,
            jev_meta={
                "cache_origin": origin,
                "jev_ranked_target_ids": ranked_ids,
            },
        )

    # Call Jev for a fresh ranking
    src_facets = _occupation_facets(source_dataset, occupation_id)
    candidates = _enrich_for_jev(target_dataset, _to_candidate_rows(top20))
    record = {
        "source_id": occupation_id,
        "source_title": preferred,
        "input": {
            "title": src_facets.get("preferred_title") or src_facets.get("title") or preferred,
            "alternate_title": src_facets.get("alternate_title") or "",
            "description": src_facets.get("description") or "",
            "skills": src_facets.get("skills") or "",
            "knowledge": src_facets.get("knowledge") or "",
        },
        "top1_target_id": candidates[0].get("target_id"),
        "top1_preferred_title": candidates[0].get("preferred_title"),
        "top1_similarity": top1_similarity,
        "candidates": candidates,
        "sent_to_jev": True,
    }
    resolved_list = await resolve_batch([record], run_seq=0, concurrency=1)
    resolved = resolved_list[0] if resolved_list else {}
    ranked_ids = list(resolved.get("jev_ranked_target_ids") or [])[:RANKED_TOP_N]
    if not ranked_ids:
        ranked_ids = [
            str(c.get("target_id") or c.get("occupation_id"))
            for c in candidates[:RANKED_TOP_N]
            if c.get("target_id") or c.get("occupation_id")
        ]

    if ranked_ids and not resolved.get("jev_error"):
        save_live_jev_cache(
            direction_enum.value,
            occupation_id,
            ranked_ids,
            meta={"top1_similarity": top1_similarity, "threshold": float(threshold)},
        )

    top10 = _reorder_by_ids(top20, ranked_ids, target_dataset=target_dataset)
    return _build_response(
        **common,
        ranking_source="jev",
        sent_to_jev=True,
        top10=top10,
        jev_meta={
            "cache_origin": None,
            "jev_ranked_target_ids": ranked_ids,
            "jev_error": resolved.get("jev_error"),
            "jev_match_label": resolved.get("jev_match_label"),
            "jev_confidence": resolved.get("jev_confidence"),
        },
    )
