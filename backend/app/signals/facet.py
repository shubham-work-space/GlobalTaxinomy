"""Pooled facet cosine signals (wrap existing Chroma facet collections)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import chroma as chroma_store
from .base import register_signal


class FacetCosineSignal:
    """Cosine similarity of one source facet against every target occupation."""

    def __init__(self, name: str, source_facet: str, target_facet: str | None = None) -> None:
        self.name = name
        self.source_facet = source_facet
        self.target_facet = target_facet or source_facet
        self.required_collections = (f"*_{self.target_facet}",)

    def score(
        self,
        source_id: str,
        source_dataset: str,
        target_ids: list[str],
        target_dataset: str,
        context: dict[str, Any],
    ) -> np.ndarray:
        source_emb = context.get("source_embeddings") or {}
        query = source_emb.get(self.source_facet)
        if not query:
            loaded = chroma_store.load_occupation_embeddings(source_dataset, source_id)  # type: ignore[arg-type]
            if not loaded:
                return np.zeros(len(target_ids), dtype=np.float32)
            query = loaded["embeddings"].get(self.source_facet)
        if not query:
            return np.zeros(len(target_ids), dtype=np.float32)

        matrices = context.get("facet_matrices") or {}
        key = (target_dataset, self.target_facet)
        if key in matrices:
            ids, _titles, matrix = matrices[key]
            id_to_row = {oid: i for i, oid in enumerate(ids)}
            q = np.asarray(query, dtype=np.float32)
            qn = float(np.linalg.norm(q))
            if qn > 1e-12:
                q = q / qn
            scores = np.clip(matrix @ q, 0.0, 1.0)
            out = np.zeros(len(target_ids), dtype=np.float32)
            for ti, tid in enumerate(target_ids):
                row = id_to_row.get(tid)
                if row is not None:
                    out[ti] = float(scores[row])
            return out

        rows = chroma_store.score_facet_all(target_dataset, self.target_facet, list(query))  # type: ignore[arg-type]
        by_id = {str(r["occupation_id"]): float(r["score"]) for r in rows}
        return np.asarray([by_id.get(tid, 0.0) for tid in target_ids], dtype=np.float32)


register_signal(FacetCosineSignal("title_pool", "title"))
register_signal(FacetCosineSignal("description_pool", "description"))
register_signal(FacetCosineSignal("skills_pool", "skills"))
register_signal(FacetCosineSignal("knowledge_pool", "knowledge"))
