"""Ground-truth accuracy metrics before and after Jev resolution (Accuracy@K)."""

from __future__ import annotations

from typing import Any, Iterable

from .ground_truth import direction_map

ACCURACY_KS = (1, 2, 3, 5, 10)
RANKED_TOP_N = 10


def _normalize_id(value: str | None) -> str:
    if not value:
        return ""
    return str(value).rstrip("/")


def _gold_has_exact(gold: dict[str, str]) -> bool:
    """Exact@1 denominator: only sources with ≥1 exactMatch target (≈490 O*NET)."""
    return any(mt == "exactMatch" for mt in gold.values())


def _resolve_gold(
    gold_map: dict[str, dict[str, str]],
    source_id: str | None,
) -> dict[str, str]:
    if not source_id:
        return {}
    gold = gold_map.get(source_id) or gold_map.get(_normalize_id(source_id)) or {}
    if gold:
        return gold
    for key in gold_map:
        if _normalize_id(key) == _normalize_id(source_id):
            return gold_map[key]
    return {}


def _target_hits_gold(
    target_id: str | None,
    gold: dict[str, str],
    *,
    exact_only: bool,
) -> bool:
    if not target_id or not gold:
        return False
    tid = _normalize_id(target_id)
    uuid = tid.rsplit("/", 1)[-1]
    for gid, match_type in gold.items():
        gnorm = _normalize_id(gid)
        guuid = gnorm.rsplit("/", 1)[-1]
        if tid == gnorm or tid == guuid or uuid == guuid or uuid == gnorm:
            if exact_only:
                return match_type == "exactMatch"
            return True
    return False


def _sent_to_resolver(row: dict[str, Any]) -> bool:
    return bool(row.get("sent_to_jev") or row.get("sent_to_llm"))


def _pick_final_target(row: dict[str, Any]) -> str | None:
    if _sent_to_resolver(row):
        return (
            row.get("jev_target_id")
            or row.get("llm_target_id")
            or row.get("final_target_id")
            or row.get("top1_target_id")
        )
    return row.get("top1_target_id") or row.get("final_target_id")


def _embedding_ranked_ids(row: dict[str, Any], *, limit: int = RANKED_TOP_N) -> list[str]:
    out: list[str] = []
    for cand in row.get("candidates") or []:
        tid = cand.get("target_id") or cand.get("occupation_id")
        if tid and str(tid) not in out:
            out.append(str(tid))
        if len(out) >= limit:
            break
    return out


def final_ranked_target_ids(row: dict[str, Any], *, limit: int = RANKED_TOP_N) -> list[str]:
    """Ordered prediction list used for Accuracy@K.

    JEV-sent rows prefer ``jev_ranked_target_ids``; otherwise embedding candidate order.
    Legacy rows without a ranked list fall back to a single final/top1 id.
    """
    if _sent_to_resolver(row):
        ranked = row.get("jev_ranked_target_ids")
        if isinstance(ranked, list) and ranked:
            out: list[str] = []
            for tid in ranked:
                if tid and str(tid) not in out:
                    out.append(str(tid))
                if len(out) >= limit:
                    break
            return out
    emb = _embedding_ranked_ids(row, limit=limit)
    if emb:
        return emb
    # Legacy / incomplete rows: Acc@1 only from final pick
    final = _pick_final_target(row)
    return [str(final)] if final else []


def _hit_at_k(
    ranked_ids: list[str],
    gold: dict[str, str],
    k: int,
) -> bool:
    for tid in ranked_ids[:k]:
        if _target_hits_gold(tid, gold, exact_only=True):
            return True
    return False


def compute_accuracy_metrics(
    results: list[dict[str, Any]],
    *,
    direction: str = "onet_to_esco",
) -> dict[str, Any]:
    """
    Compute embedding Top1 and hybrid Accuracy@K (post-Jev ranked list).

    Accuracy@K uses the same denominator as 3.0 optimizer exact@1:
    only sources that have ≥1 exactMatch ground-truth target (~490 for O*NET→ESCO).

    Each result row should include:
      - source_id
      - top1_target_id
      - candidates (embedding order)
      - sent_to_jev / sent_to_llm
      - jev_ranked_target_ids (optional; required for Acc@K>1 on JEV rows)
      - jev_target_id / llm_target_id (optional)
      - final_target_id (optional)
    """
    gold_map = direction_map(direction)
    total = 0
    emb_exact_hits = 0
    high_conf_exact = 0
    high_conf_n = 0
    hits_at_k = {k: 0 for k in ACCURACY_KS}

    jev_rows: list[dict[str, Any]] = []
    for row in results:
        gold = _resolve_gold(gold_map, row.get("source_id"))
        # Skip close/broad/narrow-only sources — they cannot score exact@1 hits.
        if not gold or not _gold_has_exact(gold):
            continue
        total += 1
        top1 = row.get("top1_target_id")
        top1_hit = _target_hits_gold(top1, gold, exact_only=True)
        if top1_hit:
            emb_exact_hits += 1

        ranked = final_ranked_target_ids(row)
        for k in ACCURACY_KS:
            if _hit_at_k(ranked, gold, k):
                hits_at_k[k] += 1

        if _sent_to_resolver(row):
            jev_rows.append({**row, "_gold": gold})
        else:
            high_conf_n += 1
            if top1_hit:
                high_conf_exact += 1

    def _ratio(hits: int, denom: int) -> float | None:
        if denom <= 0:
            return None
        return round(hits / denom, 6)

    jev_before_exact = 0
    jev_after_exact = 0
    for row in jev_rows:
        gold = row["_gold"]
        if _target_hits_gold(row.get("top1_target_id"), gold, exact_only=True):
            jev_before_exact += 1
        final = _pick_final_target(row)
        if _target_hits_gold(final, gold, exact_only=True):
            jev_after_exact += 1

    n_jev = len(jev_rows)
    emb_exact = _ratio(emb_exact_hits, total)
    accuracy_at = {k: _ratio(hits_at_k[k], total) for k in ACCURACY_KS}
    final_exact = accuracy_at[1]
    jev_before = _ratio(jev_before_exact, n_jev)
    jev_after = _ratio(jev_after_exact, n_jev)
    gain = None
    if jev_before is not None and jev_after is not None:
        gain = round(jev_after - jev_before, 6)

    return {
        "total_records": total,
        "records_sent_to_jev": n_jev,
        "records_sent_to_llm": n_jev,  # compat
        "accuracy_at_1": accuracy_at[1],
        "accuracy_at_2": accuracy_at[2],
        "accuracy_at_3": accuracy_at[3],
        "accuracy_at_5": accuracy_at[5],
        "accuracy_at_10": accuracy_at[10],
        "exact_match_at_1": final_exact,  # compat alias
        "embedding_top1_accuracy": emb_exact,
        "final_accuracy_after_llm": final_exact,
        "jev_subset_accuracy_before": jev_before,
        "jev_subset_accuracy_after": jev_after,
        "llm_subset_accuracy_before": jev_before,
        "llm_subset_accuracy_after": jev_after,
        "accuracy_gain": gain,
        "exact_hits_before": emb_exact_hits,
        "exact_hits_after": hits_at_k[1],
        "resolver": "jev",
        "high_confidence_records": high_conf_n,
        "high_confidence_correct": high_conf_exact,
        "high_confidence_accuracy": _ratio(high_conf_exact, high_conf_n),
        "resolver_subset_records": n_jev,
        "resolver_subset_correct": jev_after_exact,
        "resolver_subset_accuracy": jev_after,
    }


def annotate_results_with_ground_truth(
    results: Iterable[dict[str, Any]],
    *,
    direction: str = "onet_to_esco",
) -> list[dict[str, Any]]:
    """Attach ground-truth match flags to each result row."""
    gold_map = direction_map(direction)
    out: list[dict[str, Any]] = []
    for row in results:
        gold = _resolve_gold(gold_map, row.get("source_id"))
        top1 = row.get("top1_target_id")
        final = _pick_final_target(row)
        ranked = final_ranked_target_ids(row)
        annotated = dict(row)
        annotated["ground_truth_ids"] = list(gold.keys())
        annotated["ground_truth_types"] = dict(gold)
        annotated["has_exact_match"] = _gold_has_exact(gold)
        annotated["top1_exact_hit"] = _target_hits_gold(top1, gold, exact_only=True)
        annotated["final_exact_hit"] = _target_hits_gold(final, gold, exact_only=True)
        annotated["final_ranked_target_ids"] = ranked
        annotated["final_hit_at_k"] = {
            str(k): _hit_at_k(ranked, gold, k) for k in ACCURACY_KS
        }
        out.append(annotated)
    return out
