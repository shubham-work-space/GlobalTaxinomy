"""Max / median aggregation over item-level Chroma collections."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .. import chroma as chroma_store
from .base import register_signal

AggOp = Literal["max", "median"]


class AggregateItemSignal:
    """
    Score each target occupation by aggregating (max or median) cosine
    similarities between a source query vector and all item vectors belonging
    to that occupation (e.g. ESCO preferred + alternate labels).
    """

    def __init__(
        self,
        name: str,
        *,
        collection: str,
        query_facet: str,
        op: AggOp,
        occupancy_key: str,
    ) -> None:
        self.name = name
        self.collection = collection
        self.query_facet = query_facet
        self.op = op
        self.occupancy_key = occupancy_key
        self.required_collections = (collection,)

    def score(
        self,
        source_id: str,
        source_dataset: str,
        target_ids: list[str],
        target_dataset: str,
        context: dict[str, Any],
    ) -> np.ndarray:
        query = self._resolve_query(source_id, source_dataset, context)
        if query is None:
            return np.zeros(len(target_ids), dtype=np.float32)

        cache = context.get("aggregate_cache") or {}
        packed = cache.get(self.collection)
        if packed is None:
            packed = chroma_store.get_item_collection_matrix(self.collection)
            cache[self.collection] = packed
            context["aggregate_cache"] = cache

        _item_ids, item_matrix, occ_to_rows = packed
        if item_matrix.size == 0:
            return np.zeros(len(target_ids), dtype=np.float32)

        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 1e-12:
            q = q / qn
        sims = np.clip(item_matrix @ q, 0.0, 1.0)

        out = np.zeros(len(target_ids), dtype=np.float32)
        for ti, tid in enumerate(target_ids):
            rows = occ_to_rows.get(tid) or occ_to_rows.get(tid.rstrip("/"))
            if not rows:
                uuid = tid.rsplit("/", 1)[-1]
                rows = occ_to_rows.get(uuid)
            if not rows:
                continue
            vals = sims[np.asarray(rows, dtype=np.int64)]
            if self.op == "max":
                out[ti] = float(vals.max())
            else:
                out[ti] = float(np.median(vals))
        return out

    def _resolve_query(
        self,
        source_id: str,
        source_dataset: str,
        context: dict[str, Any],
    ) -> list[float] | None:
        source_emb = context.get("source_embeddings") or {}
        if self.query_facet in source_emb and source_emb[self.query_facet]:
            return list(source_emb[self.query_facet])

        loaded = chroma_store.load_occupation_embeddings(source_dataset, source_id)  # type: ignore[arg-type]
        if not loaded:
            return None
        return loaded["embeddings"].get(self.query_facet)


def _register_label_signals_from_registry() -> None:
    """Register label_max / label_median from taxonomy pair label collections."""
    from ..taxonomies import TAXONOMIES, TAXONOMY_PAIRS

    registered: set[str] = set()
    for pair in TAXONOMY_PAIRS.values():
        out = TAXONOMIES.get(pair.output_taxonomy)
        if not out or not out.label_collection:
            continue
        collection = out.label_collection
        if collection in registered:
            continue
        registered.add(collection)
        occupancy_key = f"{out.key}_labels"
        register_signal(
            AggregateItemSignal(
                "label_max",
                collection=collection,
                query_facet="title",
                op="max",
                occupancy_key=occupancy_key,
            )
        )
        register_signal(
            AggregateItemSignal(
                "label_median",
                collection=collection,
                query_facet="title",
                op="median",
                occupancy_key=occupancy_key,
            )
        )


_register_label_signals_from_registry()
