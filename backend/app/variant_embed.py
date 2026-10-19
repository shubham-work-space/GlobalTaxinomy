"""Embed additive label collections required by model P (e.g. esco_label)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from openai import AsyncAzureOpenAI, RateLimitError

from . import chroma as chroma_store
from . import item_index
from .config import Settings
from .embed_manifest import compute_fingerprint, write_manifest
from .taxonomies import label_collections

ProgressCb = Callable[[dict[str, Any]], Awaitable[None]]

STEPS = label_collections() or ("esco_label",)


async def _embed_batch(
    client: AsyncAzureOpenAI,
    deployment: str,
    texts: list[str],
    *,
    batch_size: int = 64,
    on_batch: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[list[float]]:
    results: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        chunk = texts[start : start + batch_size]
        delay = 1.0
        for _ in range(8):
            try:
                resp = await client.embeddings.create(
                    model=deployment,
                    input=[t or " " for t in chunk],
                )
                ordered = sorted(resp.data, key=lambda x: x.index)
                results.extend(list(item.embedding) for item in ordered)
                break
            except RateLimitError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
            except Exception:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20)
        else:
            raise RuntimeError(f"Failed to embed batch at {start}")
        if on_batch:
            await on_batch(min(start + batch_size, total), total)
    return results


async def embed_variant_collections(
    client: AsyncAzureOpenAI,
    settings: Settings,
    *,
    skip_existing: bool = False,
    progress: ProgressCb | None = None,
) -> dict[str, int]:
    """Embed taxonomy label collections (currently esco_label only)."""
    deployment = settings.embedding_deployment
    model = settings.embedding_model_name
    created_at = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {}
    steps = list(STEPS)

    async def report(**patch: Any) -> None:
        if progress:
            await progress(patch)

    async def upsert(
        name: str,
        ids: list[str],
        texts: list[str],
        metas: list[dict],
        *,
        step_i: int,
    ) -> int:
        if not ids:
            counts[name] = 0
            await report(
                current=name,
                completed=step_i,
                percent=round(100.0 * step_i / len(steps), 1),
                message=f"{name}: empty",
                counts=dict(counts),
            )
            return 0

        existing = chroma_store.collection_count(name)
        if skip_existing and existing >= len(ids):
            counts[name] = existing
            await report(
                current=name,
                completed=step_i,
                percent=round(100.0 * step_i / len(steps), 1),
                message=f"{name}: skip ({existing})",
                counts=dict(counts),
            )
            return existing

        await report(
            current=name,
            completed=step_i - 1,
            items_done=0,
            items_total=len(ids),
            message=f"{name}: embedding {len(ids)}…",
        )

        async def on_batch(done: int, total: int) -> None:
            await report(
                current=name,
                items_done=done,
                items_total=total,
                message=f"{name}: {done}/{total}",
                percent=round(
                    100.0 * ((step_i - 1) + done / max(total, 1)) / len(steps),
                    1,
                ),
            )

        for meta in metas:
            meta.setdefault("model", model)
            meta.setdefault("created_at", created_at)

        vectors = await _embed_batch(client, deployment, texts, on_batch=on_batch)
        chroma_store.upsert_items(name, ids, vectors, texts, metas)
        counts[name] = len(ids)
        await report(
            current=name,
            completed=step_i,
            percent=round(100.0 * step_i / len(steps), 1),
            items_done=len(ids),
            items_total=len(ids),
            message=f"{name}: done",
            counts=dict(counts),
        )
        return len(ids)

    await report(
        status="running",
        total=len(steps),
        completed=0,
        percent=0.0,
        current=steps[0] if steps else None,
        message="Starting label embeddings…",
        counts={},
    )

    labels = item_index.esco_labels()
    ids, texts, metas = [], [], []
    for occ_id, rows in labels.items():
        for row in rows:
            ids.append(row["label_id"])
            texts.append(row["text"])
            metas.append(
                {
                    "dataset": "esco",
                    "collection_kind": "label",
                    "occupation_id": occ_id,
                    "parent_occupation_id": occ_id,
                    "text": row["text"],
                    "is_preferred": bool(row.get("is_preferred")),
                }
            )
    await upsert("esco_label", ids, texts, metas, step_i=1)

    write_manifest(
        model=model,
        collections=counts,
        fingerprint=compute_fingerprint(model),
    )
    chroma_store.invalidate_item_matrix_cache()
    await report(
        status="done",
        completed=len(steps),
        percent=100.0,
        current=None,
        message="Label embeddings complete",
        counts=dict(counts),
    )
    return counts
