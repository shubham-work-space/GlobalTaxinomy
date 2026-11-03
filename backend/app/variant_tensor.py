"""Build / cache / load per-variant signal tensors for weight optimization."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .accel import backend_info
from .config import ROOT
from .ground_truth import direction_map, eval_source_ids
from .variant_score import score_variant_signal_arrays
from .variants import get_variant

OUTPUT_DIR = ROOT / "output" / "cache"
DEFAULT_WORKERS = 20

ProgressCallback = Callable[[str, dict[str, Any]], None]


def tensor_path(variant_name: str) -> Path:
    """Cache path; SIGNAL_TENSOR_SUFFIX (e.g. _1.1.0) isolates versioned tensors."""
    import os

    suffix = (os.environ.get("SIGNAL_TENSOR_SUFFIX") or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return OUTPUT_DIR / f"{variant_name}_tensor{suffix}.npz"


def save_signal_tensor(variant_name: str, data: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = tensor_path(variant_name)
    np.savez_compressed(
        path,
        scores=data["scores"],
        mask=data["mask"],
        exact_gold=data["exact_gold"],
        any_gold=data["any_gold"],
        has_exact=data["has_exact"],
        has_any=data["has_any"],
        source_ids=np.array(data["source_ids"], dtype=object),
        target_ids=np.array(data["target_ids"], dtype=object),
        signals=np.array(data["signals"], dtype=object),
    )
    return path


def load_signal_tensor(variant_name: str) -> dict | None:
    path = tensor_path(variant_name)
    if not path.is_file():
        return None
    packed = np.load(path, allow_pickle=True)
    return {
        "signals": [str(x) for x in packed["signals"].tolist()],
        "source_ids": [str(x) for x in packed["source_ids"].tolist()],
        "target_ids": [str(x) for x in packed["target_ids"].tolist()],
        "scores": packed["scores"],
        "mask": packed["mask"],
        "exact_gold": packed["exact_gold"],
        "any_gold": packed["any_gold"],
        "has_exact": packed["has_exact"],
        "has_any": packed["has_any"],
    }


def _resolve_target_index(target_ids: list[str], key: str) -> int | None:
    key = key.rstrip("/")
    uuid = key.rsplit("/", 1)[-1] if "/" in key else key
    canonical = f"http://data.europa.eu/esco/occupation/{uuid}"
    id_to_idx = {tid: i for i, tid in enumerate(target_ids)}
    for candidate in (key, canonical, uuid):
        if candidate in id_to_idx:
            return id_to_idx[candidate]
    return None


def build_signal_tensor(
    variant_name: str,
    *,
    direction: str = "onet_to_esco",
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
    progress: ProgressCallback | None = None,
) -> dict:
    """Score sources in parallel and pack signal tensors for grid search."""
    var = get_variant(variant_name)
    sources = eval_source_ids(direction)
    if limit:
        sources = sources[:limit]
    dm = direction_map(direction)
    accel = backend_info()

    if progress:
        progress(
            "building_matrices",
            {
                "message": (
                    f"{var.name}: warming caches "
                    f"(accel={accel['backend']}"
                    f"{', ' + accel['gpu_name'] if accel.get('gpu_name') else ''})"
                ),
                "done": 0,
                "total": len(sources),
                "percent": 0.0,
            },
        )

    # Warm shared context (skill matrices, facet matrices, target row lists)
    shared_ctx: dict[str, Any] = {}
    probe_ids, probe_scores = score_variant_signal_arrays(
        sources[0], direction, var, context=shared_ctx
    )
    target_ids = list(probe_ids)
    n_t = len(target_ids)
    n_s = len(sources)
    n_sig = len(var.signals)
    sig_index = {s: i for i, s in enumerate(var.signals)}

    scores = np.zeros((n_s, n_sig, n_t), dtype=np.float32)
    mask = np.zeros((n_s, n_sig), dtype=bool)
    exact_gold = np.zeros((n_s, n_t), dtype=bool)
    any_gold = np.zeros((n_s, n_t), dtype=bool)
    has_exact = np.zeros(n_s, dtype=bool)
    has_any = np.zeros(n_s, dtype=bool)

    # Fill first source from probe
    for sig, arr in probe_scores.items():
        if sig in sig_index:
            scores[0, sig_index[sig], :] = arr.astype(np.float32)
            mask[0, sig_index[sig]] = True
    _fill_gold(0, sources[0], dm, target_ids, exact_gold, any_gold, has_exact, has_any)

    done = 1
    done_lock = threading.Lock()
    if progress:
        progress(
            "building_matrices",
            {
                "message": f"{var.name}: scoring occupation 1 of {n_s} (workers={workers})",
                "done": 1,
                "total": n_s,
                "percent": round(70.0 * 1 / max(n_s, 1), 2),
            },
        )

    remaining = list(enumerate(sources))[1:]

    def _work(si: int, sid: str) -> tuple[int, str, dict[str, np.ndarray]]:
        # Per-thread shallow copy of shared caches (read-only matrices)
        ctx = {
            "skill_item_cache": shared_ctx.get("skill_item_cache", {}),
            "facet_matrices": shared_ctx.get("facet_matrices", {}),
            "_target_row_lists": shared_ctx.get("_target_row_lists"),
            "_target_row_lists_key": shared_ctx.get("_target_row_lists_key"),
        }
        _ids, sig_scores = score_variant_signal_arrays(sid, direction, var, context=ctx)
        return si, sid, sig_scores

    n_workers = max(1, min(workers, max(len(remaining), 1)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_work, si, sid) for si, sid in remaining]
        for fut in as_completed(futures):
            si, sid, sig_scores = fut.result()
            for sig, arr in sig_scores.items():
                if sig in sig_index:
                    scores[si, sig_index[sig], :] = arr.astype(np.float32)
                    mask[si, sig_index[sig]] = True
            _fill_gold(si, sid, dm, target_ids, exact_gold, any_gold, has_exact, has_any)
            with done_lock:
                done += 1
                if progress and (done % 5 == 0 or done == n_s):
                    progress(
                        "building_matrices",
                        {
                            "message": (
                                f"{var.name}: scored {done}/{n_s} "
                                f"(parallel={n_workers}, accel={accel['backend']})"
                            ),
                            "done": done,
                            "total": n_s,
                            "percent": round(70.0 * done / max(n_s, 1), 2),
                        },
                    )

    return {
        "signals": list(var.signals),
        "source_ids": sources,
        "target_ids": target_ids,
        "scores": scores,
        "mask": mask,
        "exact_gold": exact_gold,
        "any_gold": any_gold,
        "has_exact": has_exact,
        "has_any": has_any,
        "accel": accel,
        "workers": n_workers,
    }


def _fill_gold(
    si: int,
    sid: str,
    dm: dict,
    target_ids: list[str],
    exact_gold: np.ndarray,
    any_gold: np.ndarray,
    has_exact: np.ndarray,
    has_any: np.ndarray,
) -> None:
    targets = dm.get(sid, {})
    if not targets:
        return
    has_any[si] = True
    for esco_key, mt in targets.items():
        col = _resolve_target_index(target_ids, esco_key)
        if col is None:
            continue
        any_gold[si, col] = True
        if mt == "exactMatch":
            exact_gold[si, col] = True
            has_exact[si] = True


def load_or_build_signal_tensor(
    variant_name: str,
    *,
    direction: str = "onet_to_esco",
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
    progress: ProgressCallback | None = None,
    force_rebuild: bool = False,
    cache: bool = True,
) -> dict:
    """
    Load cached tensor when possible; otherwise build (parallel) and optionally cache.

    Cache hit rules:
    - If a cached tensor exists and ``force_rebuild`` is false, reuse it when it has
      at least ``limit`` sources (or any size when ``limit`` is None).
    - Fresh builds are written to disk when ``cache`` is true.
    """
    if not force_rebuild:
        cached = load_signal_tensor(variant_name)
        if cached is not None:
            n_cached = len(cached["source_ids"])
            full_n = len(eval_source_ids(direction))
            usable = False
            if limit is not None and n_cached >= limit:
                if n_cached > limit:
                    cached = _slice_tensor(cached, limit)
                usable = True
            elif limit is None and n_cached >= full_n:
                usable = True
            if usable:
                if progress:
                    progress(
                        "building_matrices",
                        {
                            "message": (
                                f"{variant_name}: loaded cached tensor "
                                f"{tensor_path(variant_name)} ({len(cached['source_ids'])} sources)"
                            ),
                            "done": len(cached["source_ids"]),
                            "total": len(cached["source_ids"]),
                            "percent": 70.0,
                        },
                    )
                return cached

    data = build_signal_tensor(
        variant_name,
        direction=direction,
        limit=limit,
        workers=workers,
        progress=progress,
    )
    if cache:
        path = save_signal_tensor(variant_name, data)
        if progress:
            progress(
                "building_matrices",
                {
                    "message": f"{variant_name}: wrote cache {path}",
                    "done": len(data["source_ids"]),
                    "total": len(data["source_ids"]),
                    "percent": 70.0,
                },
            )
    return data


def _slice_tensor(data: dict, limit: int) -> dict:
    out = dict(data)
    out["source_ids"] = data["source_ids"][:limit]
    out["scores"] = data["scores"][:limit]
    out["mask"] = data["mask"][:limit]
    out["exact_gold"] = data["exact_gold"][:limit]
    out["any_gold"] = data["any_gold"][:limit]
    out["has_exact"] = data["has_exact"][:limit]
    out["has_any"] = data["has_any"][:limit]
    return out
