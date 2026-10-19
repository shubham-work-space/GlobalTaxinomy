"""Seven-phase crosswalk run pipeline with progress reporting."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Callable, Awaitable

import numpy as np

from . import chroma as chroma_store
from .accuracy import annotate_results_with_ground_truth, compute_accuracy_metrics
from .ground_truth import source_has_exact_match
from .jev_resolve import resolve_batch
from .runs import PHASES, create_run, save_run_artifacts, update_run, validate_config
from .variant_tensor import load_or_build_signal_tensor

ProgressAsync = Callable[[dict[str, Any]], Awaitable[None]]


def _combined_from_tensor(data: dict, weights: dict[str, float]) -> np.ndarray:
    signals = data["signals"]
    w = np.array([float(weights.get(s, 0.0)) for s in signals], dtype=np.float32)
    scores = data["scores"]
    mask = data["mask"].astype(np.float32)
    ww = w.reshape(1, -1)
    eff = ww * mask
    denom_w = np.maximum(eff.sum(axis=1, keepdims=True), 1e-12)
    eff = eff / denom_w
    return (eff[:, :, np.newaxis] * scores).sum(axis=1)


def _occupation_facets(dataset: str, occupation_id: str) -> dict[str, str]:
    rec = chroma_store.load_occupation_embeddings(dataset, occupation_id)  # type: ignore[arg-type]
    if not rec:
        return {
            "title": "",
            "description": "",
            "skills": "",
            "knowledge": "",
            "preferred_title": "",
            "alternate_title": "",
        }
    texts = rec.get("facet_texts") or {}
    alt = ""
    if dataset == "esco":
        try:
            from .item_index import esco_labels

            rows = esco_labels().get(occupation_id) or esco_labels().get(
                occupation_id.rstrip("/")
            )
            if rows:
                alts = [r.get("text") or "" for r in rows if not r.get("is_preferred")]
                alt = "; ".join(a for a in alts if a)
        except Exception:
            alt = ""
    return {
        "title": texts.get("title") or rec.get("preferred_title") or "",
        "description": texts.get("description") or "",
        "skills": texts.get("skills") or "",
        "knowledge": texts.get("knowledge") or "",
        "preferred_title": rec.get("preferred_title") or texts.get("title") or "",
        "alternate_title": alt,
    }


def _top_k_indices(row: np.ndarray, k: int) -> list[int]:
    n = row.shape[0]
    if n == 0:
        return []
    kk = min(k, n)
    if kk == n:
        order = np.argsort(-row)
        return [int(i) for i in order]
    part = np.argpartition(-row, kk - 1)[:kk]
    part_sorted = part[np.argsort(-row[part])]
    return [int(i) for i in part_sorted]


class RunPipelineJob:
    """In-process job state for a single crosswalk run."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "run_id": None,
            "label": None,
            "phase": None,
            "phases": list(PHASES),
            "phase_index": 0,
            "percent": 0.0,
            "message": "",
            "done": 0,
            "total": 0,
            "error": None,
            "version": 0,
        }
        self._task: asyncio.Task | None = None

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def is_running(self) -> bool:
        return self._state.get("status") == "running"

    async def _bump(self, **patch: Any) -> None:
        async with self._lock:
            self._state.update(patch)
            self._state["version"] = int(self._state.get("version") or 0) + 1

    async def start(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.is_running():
            return {"ok": False, "error": "A run is already in progress", "job": self.snapshot()}
        cfg = validate_config(config)
        entry = create_run(cfg)
        await self._bump(
            status="running",
            run_id=entry["run_id"],
            label=entry["label"],
            phase=PHASES[0],
            phase_index=0,
            percent=0.0,
            message="Starting run…",
            done=0,
            total=0,
            error=None,
        )
        self._task = asyncio.create_task(self._execute(entry, cfg))
        return {"ok": True, "reused": False, "run": entry, "job": self.snapshot()}

    async def _set_phase(self, phase: str, message: str, *, percent: float | None = None) -> None:
        idx = PHASES.index(phase) if phase in PHASES else 0
        base = round(100.0 * idx / len(PHASES), 2)
        await self._bump(
            phase=phase,
            phase_index=idx,
            message=message,
            percent=percent if percent is not None else base,
        )
        run_id = self._state.get("run_id")
        if run_id:
            update_run(run_id, phase=phase, message=message, percent=self._state["percent"])

    async def _execute(self, entry: dict[str, Any], cfg: dict[str, Any]) -> None:
        run_id = entry["run_id"]
        run_seq = int(entry["seq"])
        try:
            # 1. Generate embeddings (skip if inventory already present)
            await self._set_phase("generate_embeddings", "Checking embeddings…", percent=2)
            inv = chroma_store.inventory_summary()
            onet_n = inv["datasets"]["onet"]["occupations"]
            esco_n = inv["datasets"]["esco"]["occupations"]
            labels_n = chroma_store.collection_count("esco_label")
            if onet_n == 0 or esco_n == 0 or labels_n == 0:
                await self._set_phase(
                    "generate_embeddings",
                    "Embeddings missing — copy Chroma data into data/chroma_db first",
                    percent=5,
                )
                raise RuntimeError(
                    "Embeddings not ready. Ensure data/chroma_db has onet, esco, and esco_label collections."
                )
            await self._set_phase(
                "generate_embeddings",
                f"Embeddings ready (onet={onet_n}, esco={esco_n}, labels={labels_n})",
                percent=12,
            )

            # 2. Calculate similarities (parallel tensor)
            await self._set_phase("calculate_similarities", "Building/loading signal tensor…", percent=15)
            loop = asyncio.get_running_loop()

            def tensor_progress(phase: str, info: dict[str, Any]) -> None:
                msg = info.get("message") or "Calculating similarities…"
                pct = 15 + 0.35 * float(info.get("percent") or 0)
                asyncio.run_coroutine_threadsafe(
                    self._bump(phase="calculate_similarities", message=msg, percent=round(pct, 2)),
                    loop,
                )

            data = await asyncio.to_thread(
                load_or_build_signal_tensor,
                cfg["model"],
                direction=cfg["direction"],
                progress=tensor_progress,
            )
            await self._set_phase(
                "calculate_similarities",
                f"Tensor ready ({len(data['source_ids'])} sources × {len(data['target_ids'])} targets)",
                percent=50,
            )

            # 3. Select top candidates
            await self._set_phase("select_top_candidates", "Selecting top-20 candidates…", percent=55)
            combined = _combined_from_tensor(data, cfg["weights"])
            source_ids = data["source_ids"]
            target_ids = data["target_ids"]
            top_k = int(cfg.get("top_k") or 20)
            # Preload title index for targets
            target_titles = {
                row["occupation_id"]: row["preferred_title"]
                for row in chroma_store._title_index(cfg["output_taxonomy"])  # type: ignore[arg-type]
            }

            results: list[dict[str, Any]] = []
            for si, sid in enumerate(source_ids):
                idxs = _top_k_indices(combined[si], top_k)
                candidates = []
                for rank, ti in enumerate(idxs, start=1):
                    tid = target_ids[ti]
                    candidates.append(
                        {
                            "rank": rank,
                            "target_id": tid,
                            "occupation_id": tid,
                            "preferred_title": target_titles.get(tid, tid),
                            "similarity": float(combined[si, ti]),
                        }
                    )
                top1 = candidates[0] if candidates else {}
                src_facets = _occupation_facets(cfg["input_taxonomy"], sid)
                results.append(
                    {
                        "source_id": sid,
                        "source_title": src_facets.get("preferred_title") or sid,
                        "input": {
                            "title": src_facets.get("preferred_title") or src_facets.get("title"),
                            "alternate_title": src_facets.get("alternate_title") or "",
                            "description": src_facets.get("description"),
                            "skills": src_facets.get("skills"),
                            "knowledge": src_facets.get("knowledge"),
                        },
                        "top1_target_id": top1.get("target_id"),
                        "top1_preferred_title": top1.get("preferred_title"),
                        "top1_similarity": top1.get("similarity"),
                        "candidates": candidates,
                        "sent_to_jev": False,
                        "sent_to_llm": False,
                        "final_target_id": top1.get("target_id"),
                        "final_preferred_title": top1.get("preferred_title"),
                    }
                )
            await self._set_phase(
                "select_top_candidates",
                f"Selected top-{top_k} for {len(results)} occupations",
                percent=62,
            )

            # 4. Apply threshold (exact@1 set only — sources with ≥1 exactMatch GT)
            await self._set_phase("apply_threshold", "Applying threshold…", percent=65)
            threshold = float(cfg["threshold"])
            has_exact = data.get("has_exact")
            needs_llm: list[dict[str, Any]] = []
            for si, row in enumerate(results):
                if has_exact is not None:
                    in_exact_set = bool(has_exact[si])
                else:
                    in_exact_set = source_has_exact_match(cfg["direction"], row["source_id"])
                if not in_exact_set:
                    continue
                sim = float(row.get("top1_similarity") or 0.0)
                if sim < threshold:
                    # Enrich candidates with facet texts for Jev prompt
                    enriched = []
                    for cand in row.get("candidates") or []:
                        facets = _occupation_facets(cfg["output_taxonomy"], cand["target_id"])
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
                    row["candidates"] = enriched
                    row["sent_to_jev"] = True
                    row["sent_to_llm"] = True  # compat
                    needs_llm.append(row)
            await self._set_phase(
                "apply_threshold",
                f"{len(needs_llm)} exact-GT records below threshold {threshold}",
                percent=70,
            )

            # 5. Run Jev evaluation
            await self._set_phase(
                "run_jev_evaluation",
                f"Sending {len(needs_llm)} records to Jev…" if needs_llm else "No Jev records",
                percent=72,
            )

            async def jev_progress(info: dict[str, Any]) -> None:
                done = int(info.get("done") or 0)
                total = max(int(info.get("total") or 1), 1)
                pct = 72 + 18 * (done / total)
                await self._bump(
                    phase="run_jev_evaluation",
                    message=info.get("message") or "Jev…",
                    done=done,
                    total=total,
                    percent=round(pct, 2),
                )

            if needs_llm:
                resolved = await resolve_batch(
                    needs_llm,
                    run_seq=run_seq,
                    progress=jev_progress,
                )
                by_source = {r["source_id"]: r for r in resolved}
                for i, row in enumerate(results):
                    if row["source_id"] in by_source:
                        results[i] = by_source[row["source_id"]]

            await self._set_phase("run_jev_evaluation", "Jev evaluation complete", percent=90)

            # 6. Calculate Accuracy@K
            await self._set_phase("calculate_accuracy", "Calculating Accuracy@K…", percent=92)
            results = annotate_results_with_ground_truth(results, direction=cfg["direction"])
            metrics = compute_accuracy_metrics(results, direction=cfg["direction"])
            metrics["model"] = cfg["model"]
            metrics["input_taxonomy"] = cfg["input_taxonomy"]
            metrics["output_taxonomy"] = cfg["output_taxonomy"]
            metrics["weights"] = cfg["weights"]
            metrics["threshold"] = cfg["threshold"]
            await self._set_phase("calculate_accuracy", "Accuracy calculated", percent=96)

            # 7. Save run
            await self._set_phase("save_run", "Saving run artifacts…", percent=98)
            save_run_artifacts(run_id, results=results, metrics=metrics)
            await self._bump(
                status="completed",
                phase="save_run",
                percent=100.0,
                message="Run complete",
            )
        except Exception as exc:  # noqa: BLE001
            err = str(exc).strip() or f"{type(exc).__name__}: {exc!r}"
            update_run(
                run_id,
                status="failed",
                message=err,
                error=err,
                finished_at=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            )
            await self._bump(status="failed", error=err, message=err)


run_pipeline_job = RunPipelineJob()
