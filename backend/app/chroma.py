"""ChromaDB store: 8 facet collections (onet_* + esco_*), cosine space."""

from __future__ import annotations

from functools import lru_cache
import pickle
import subprocess
import sys
from typing import Any, Literal

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.errors import InternalError as ChromaInternalError

from .config import get_settings
from .taxonomies import datasets as registry_datasets
from .taxonomies import facets as registry_facets
from .taxonomies import label_collections

FACETS = registry_facets()
DATASETS = registry_datasets()
Dataset = Literal["onet", "esco"]

# Additive collections for multi-label title pillars (e.g. ESCO preferred + alternate).
VARIANT_COLLECTIONS = label_collections()

# Full-collection get() can hit Chroma HNSW "Error finding id" after concurrent
# PersistentClient opens (uvicorn reload, eval + weights in parallel). Paginated
# get + client reopen + subprocess fallback handle it for facets and item cols.
_FACET_GET_BATCH = 256


@lru_cache
def get_chroma() -> chromadb.ClientAPI:
    settings = get_settings()
    path = settings.chroma_dir
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def reset_chroma_client() -> None:
    """Drop cached client + matrices so the next call opens a fresh PersistentClient."""
    invalidate_facet_matrix_cache()
    invalidate_item_matrix_cache()
    get_chroma.cache_clear()


def collection_name(dataset: Dataset, facet: str) -> str:
    return f"{dataset}_{facet}"


def get_collection(dataset: Dataset, facet: str) -> Collection:
    return get_chroma().get_or_create_collection(
        name=collection_name(dataset, facet),
        metadata={"hnsw:space": "cosine"},
    )


_facet_matrix_cache: dict[tuple[str, str], tuple[list[str], list[str], np.ndarray]] = {}
_item_matrix_cache: dict[str, Any] = {}


def invalidate_facet_matrix_cache() -> None:
    """Clear cached facet matrices (call after embedding refresh)."""
    _facet_matrix_cache.clear()


def invalidate_item_matrix_cache() -> None:
    """Clear cached item/vocab matrices."""
    _item_matrix_cache.clear()


def get_named_collection(name: str) -> Collection:
    """Get or create a named collection (facet or variant)."""
    return get_chroma().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_all_collections(*, include_variants: bool = False) -> None:
    """Drop and recreate facet collections (refresh). Optionally wipe variant collections."""
    invalidate_facet_matrix_cache()
    invalidate_item_matrix_cache()
    client = get_chroma()
    existing = {c.name for c in client.list_collections()}
    for dataset in DATASETS:
        for facet in FACETS:
            name = collection_name(dataset, facet)  # type: ignore[arg-type]
            if name in existing:
                client.delete_collection(name)
            client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
    if include_variants:
        for name in VARIANT_COLLECTIONS:
            if name in existing:
                client.delete_collection(name)
            client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )


def embeddings_exist() -> bool:
    client = get_chroma()
    try:
        names = {c.name for c in client.list_collections()}
    except Exception:  # noqa: BLE001
        return False
    for dataset in DATASETS:
        name = collection_name(dataset, "title")  # type: ignore[arg-type]
        if name not in names:
            continue
        if get_collection(dataset, "title").count() > 0:  # type: ignore[arg-type]
            return True
    return False


def occupation_exists(dataset: Dataset, occupation_id: str) -> bool:
    col = get_collection(dataset, "title")
    got = col.get(ids=[occupation_id], include=[])
    return bool(got and got.get("ids"))


def count_dataset(dataset: Dataset) -> int:
    try:
        return get_collection(dataset, "title").count()
    except Exception:  # noqa: BLE001
        return 0


def _sample_meta(dataset: Dataset) -> dict[str, str]:
    """Peek one stored vector for model / created_at labels."""
    try:
        col = get_collection(dataset, "title")
        if col.count() == 0:
            return {}
        got = col.peek(limit=1)
        metas = got.get("metadatas") or []
        if not metas:
            return {}
        meta = metas[0] or {}
        return {
            "model": str(meta.get("model") or ""),
            "created_at": str(meta.get("created_at") or ""),
        }
    except Exception:  # noqa: BLE001
        return {}


def inventory_summary() -> dict[str, Any]:
    """Counts and labels for embeddings already stored in ChromaDB."""
    collections: dict[str, int] = {}
    datasets: dict[str, Any] = {}
    total_vectors = 0
    total_occupations = 0
    model = ""
    created_at = ""

    for dataset in DATASETS:
        facet_counts: dict[str, int] = {}
        for facet in FACETS:
            name = collection_name(dataset, facet)  # type: ignore[arg-type]
            try:
                n = get_collection(dataset, facet).count()  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                n = 0
            facet_counts[facet] = n
            collections[name] = n
            total_vectors += n

        occupations = facet_counts.get("title", 0)
        total_occupations += occupations
        sample = _sample_meta(dataset)  # type: ignore[arg-type]
        if sample.get("model") and not model:
            model = sample["model"]
        if sample.get("created_at") and not created_at:
            created_at = sample["created_at"]

        datasets[dataset] = {
            "occupations": occupations,
            "vectors": sum(facet_counts.values()),
            "facets": facet_counts,
        }

    return {
        "exists": total_vectors > 0,
        "total_occupations": total_occupations,
        "total_vectors": total_vectors,
        "model": model or "text-embedding-3-large",
        "created_at": created_at or None,
        "store": "chromadb",
        "path": str(get_settings().chroma_dir),
        "datasets": datasets,
        "collections": collections,
    }


def upsert_occupation(
    dataset: Dataset,
    occupation_id: str,
    preferred_title: str,
    facet_texts: dict[str, str],
    embeddings: dict[str, list[float]],
    model: str,
    created_at: str,
) -> None:
    for facet in FACETS:
        get_collection(dataset, facet).upsert(
            ids=[occupation_id],
            embeddings=[embeddings[facet]],
            documents=[facet_texts.get(facet, "")],
            metadatas=[
                {
                    "occupation_id": occupation_id,
                    "dataset": dataset,
                    "preferred_title": preferred_title,
                    "facet": facet,
                    "model": model,
                    "created_at": created_at,
                }
            ],
        )


def load_occupation_embeddings(dataset: Dataset, occupation_id: str) -> dict[str, Any] | None:
    """Load all 4 facet vectors + texts for one occupation from ChromaDB only."""
    facet_texts: dict[str, str] = {}
    embeddings: dict[str, list[float]] = {}
    preferred_title = ""
    model = "text-embedding-3-large"
    created_at = ""

    for facet in FACETS:
        col = get_collection(dataset, facet)
        got = col.get(ids=[occupation_id], include=["embeddings", "documents", "metadatas"])
        if not got or not got.get("ids"):
            return None
        emb = got["embeddings"]
        if emb is None or len(emb) == 0:
            return None
        embeddings[facet] = list(emb[0])
        docs = got.get("documents") or [""]
        facet_texts[facet] = docs[0] or ""
        metas = got.get("metadatas") or [{}]
        meta = metas[0] or {}
        preferred_title = str(meta.get("preferred_title") or preferred_title)
        model = str(meta.get("model") or model)
        created_at = str(meta.get("created_at") or created_at)

    return {
        "occupation_id": occupation_id,
        "preferred_title": preferred_title,
        "dataset": dataset,
        "facet_texts": facet_texts,
        "embeddings": embeddings,
        "model": model,
        "created_at": created_at,
    }


def _title_index(dataset: Dataset) -> list[dict[str, str]]:
    """All occupation ids + preferred titles from the title collection."""
    col = get_collection(dataset, "title")
    total = col.count()
    if total == 0:
        return []
    got = col.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    rows: list[dict[str, str]] = []
    for i, occ_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        title = str(meta.get("preferred_title") or occ_id)
        rows.append({"occupation_id": str(occ_id), "preferred_title": title})
    return rows


def _fetch_facet_rows(
    col: Collection,
) -> tuple[list[str], list[str], list[list[float]]]:
    """
    Load ids / titles / embedding vectors for one collection.

    Prefer paginated get(limit/offset) over get(ids=...), which is the Chroma
    path most often raising InternalError("Error finding id") on this store.
    """
    total = col.count()
    if total == 0:
        return [], [], []

    all_ids: list[str] = []
    titles: list[str] = []
    vectors: list[list[float]] = []

    for start in range(0, total, _FACET_GET_BATCH):
        got = col.get(
            limit=_FACET_GET_BATCH,
            offset=start,
            include=["embeddings", "metadatas"],
        )
        got_ids = [str(x) for x in (got.get("ids") or [])]
        metas = got.get("metadatas") or []
        raw_embs = got.get("embeddings")
        embs = list(raw_embs) if raw_embs is not None else []
        all_ids.extend(got_ids)
        for i, occ_id in enumerate(got_ids):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            emb = embs[i] if i < len(embs) else None
            titles.append(str(meta.get("preferred_title") or occ_id))
            vectors.append([float(x) for x in emb] if emb is not None else [])

    return all_ids, titles, vectors


def _load_facet_matrix_subprocess(
    dataset: Dataset,
    facet: str,
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Last-resort fallback for Chroma get() instability in the API process.

    A fresh helper Python process can often read the persisted collections even
    when the current process keeps hitting InternalError("Error finding id").
    """
    script = """
import pickle
import sys
import chromadb
import numpy as np

path, collection_name, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
client = chromadb.PersistentClient(path=path)
col = client.get_collection(collection_name)
total = col.count()
ids = []
titles = []
vectors = []
for start in range(0, total, batch_size):
    got = col.get(limit=batch_size, offset=start, include=["embeddings", "metadatas"])
    got_ids = [str(x) for x in (got.get("ids") or [])]
    metas = got.get("metadatas") or []
    raw_embs = got.get("embeddings")
    embs = list(raw_embs) if raw_embs is not None else []
    ids.extend(got_ids)
    for i, occ_id in enumerate(got_ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        emb = embs[i] if i < len(embs) else None
        titles.append(str(meta.get("preferred_title") or occ_id))
        vectors.append([float(x) for x in emb] if emb is not None else [])

if not ids:
    matrix = np.zeros((0, 0), dtype=np.float32)
elif not vectors or not vectors[0]:
    matrix = np.zeros((len(ids), 0), dtype=np.float32)
else:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

pickle.dump((ids, titles, matrix), sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(get_settings().chroma_dir),
            collection_name(dataset, facet),
            str(_FACET_GET_BATCH),
        ],
        capture_output=True,
        check=True,
    )
    ids, titles, matrix = pickle.loads(proc.stdout)
    return ids, titles, matrix


def get_facet_matrix(
    dataset: Dataset,
    facet: str,
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Load all embeddings for one facet as an L2-normalized matrix.
    Returns (occupation_ids, preferred_titles, matrix) where matrix shape is (n, dim).
    """
    key = (dataset, facet)
    cached = _facet_matrix_cache.get(key)
    if cached is not None:
        return cached

    def _load() -> tuple[list[str], list[str], np.ndarray]:
        col = get_collection(dataset, facet)
        ids, titles, vectors = _fetch_facet_rows(col)
        if not ids:
            return [], [], np.zeros((0, 0), dtype=np.float32)
        if not vectors or not vectors[0]:
            return ids, titles, np.zeros((len(ids), 0), dtype=np.float32)

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return ids, titles, matrix / norms

    try:
        result = _load()
    except ChromaInternalError:
        # Stale HNSW in the long-lived client — reopen and retry once.
        reset_chroma_client()
        try:
            result = _load()
        except ChromaInternalError:
            result = _load_facet_matrix_subprocess(dataset, facet)

    _facet_matrix_cache[key] = result
    return result


def score_facet_all(
    dataset: Dataset,
    facet: str,
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    """
    Score every occupation in a facet collection for one query vector.

    This avoids the Chroma bulk get() path used by matrix loading. It relies on
    query() for the bulk of the pool, then fills any missing ids with single-id
    reads, which are much more reliable on this store.
    """
    col = get_collection(dataset, facet)
    total = col.count()
    if total == 0:
        return []

    got = col.query(
        query_embeddings=[query_embedding],
        n_results=total,
        include=["distances", "metadatas"],
    )
    ids = [str(x) for x in ((got.get("ids") or [[]])[0] or [])]
    metas = ((got.get("metadatas") or [[]])[0] or [])
    distances = ((got.get("distances") or [[]])[0] or [])

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, occ_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        distance = float(distances[i]) if i < len(distances) else 1.0
        rows.append(
            {
                "occupation_id": occ_id,
                "preferred_title": str(meta.get("preferred_title") or occ_id),
                "score": float(np.clip(1.0 - distance, 0.0, 1.0)),
            }
        )
        seen.add(occ_id)

    return rows


def search_occupations(
    dataset: Dataset,
    query: str,
    limit: int = 20,
) -> list[dict[str, str]]:
    """Case-insensitive substring match on preferred_title (and id fallback)."""
    q = (query or "").strip().lower()
    rows = _title_index(dataset)
    if not q:
        return rows[: max(1, limit)]

    scored: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        title = row["preferred_title"].lower()
        oid = row["occupation_id"].lower()
        if q in title or q in oid:
            # Prefer prefix / earlier match
            pos = title.find(q)
            if pos < 0:
                pos = 1000 + oid.find(q)
            scored.append((pos, row))

    scored.sort(key=lambda t: (t[0], t[1]["preferred_title"].lower()))
    return [row for _, row in scored[: max(1, limit)]]


# ---------------------------------------------------------------------------
# Variant / item-level collections
# ---------------------------------------------------------------------------


def upsert_items(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Batch upsert into a named collection. Dedupes IDs within the request."""
    if not ids:
        return
    # Chroma rejects batches that contain duplicate IDs.
    seen: set[str] = set()
    keep: list[int] = []
    for i, iid in enumerate(ids):
        if iid in seen:
            continue
        seen.add(iid)
        keep.append(i)
    if len(keep) != len(ids):
        ids = [ids[i] for i in keep]
        embeddings = [embeddings[i] for i in keep]
        documents = [documents[i] for i in keep]
        metadatas = [metadatas[i] for i in keep]
    col = get_named_collection(collection_name)
    batch = 128
    for start in range(0, len(ids), batch):
        end = start + batch
        col.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    invalidate_item_matrix_cache()


def load_item_vector(collection_name: str, item_id: str) -> list[float] | None:
    col = get_named_collection(collection_name)
    got = col.get(ids=[item_id], include=["embeddings"])
    if not got or not got.get("ids"):
        return None
    emb = got.get("embeddings")
    if emb is None or len(emb) == 0:
        return None
    return [float(x) for x in emb[0]]


def collection_count(name: str) -> int:
    try:
        return get_named_collection(name).count()
    except Exception:  # noqa: BLE001
        return 0


def _fetch_named_rows(
    col: Collection,
) -> tuple[list[str], list[dict[str, Any]], list[list[float]]]:
    total = col.count()
    if total == 0:
        return [], [], []
    all_ids: list[str] = []
    metas: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    for start in range(0, total, _FACET_GET_BATCH):
        got = col.get(
            limit=_FACET_GET_BATCH,
            offset=start,
            include=["embeddings", "metadatas"],
        )
        got_ids = [str(x) for x in (got.get("ids") or [])]
        got_metas = got.get("metadatas") or []
        raw_embs = got.get("embeddings")
        embs = list(raw_embs) if raw_embs is not None else []
        all_ids.extend(got_ids)
        for i, _occ_id in enumerate(got_ids):
            meta = got_metas[i] if i < len(got_metas) and got_metas[i] else {}
            emb = embs[i] if i < len(embs) else None
            metas.append(dict(meta) if meta else {})
            vectors.append([float(x) for x in emb] if emb is not None else [])
    return all_ids, metas, vectors


def _fetch_named_rows_subprocess(
    name: str,
) -> tuple[list[str], list[dict[str, Any]], list[list[float]]]:
    """
    Last-resort fallback for named-collection get() instability (e.g. esco_label).

    Same pattern as _load_facet_matrix_subprocess: a fresh process often succeeds
    when the long-lived API PersistentClient keeps hitting Error finding id.
    """
    script = """
import pickle
import sys
import chromadb

path, collection_name, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
client = chromadb.PersistentClient(path=path)
col = client.get_collection(collection_name)
total = col.count()
ids = []
metas = []
vectors = []
for start in range(0, total, batch_size):
    got = col.get(limit=batch_size, offset=start, include=["embeddings", "metadatas"])
    got_ids = [str(x) for x in (got.get("ids") or [])]
    got_metas = got.get("metadatas") or []
    raw_embs = got.get("embeddings")
    embs = list(raw_embs) if raw_embs is not None else []
    ids.extend(got_ids)
    for i, _ in enumerate(got_ids):
        meta = got_metas[i] if i < len(got_metas) and got_metas[i] else {}
        emb = embs[i] if i < len(embs) else None
        metas.append(dict(meta) if meta else {})
        vectors.append([float(x) for x in emb] if emb is not None else [])
pickle.dump((ids, metas, vectors), sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(get_settings().chroma_dir),
            name,
            str(_FACET_GET_BATCH),
        ],
        capture_output=True,
        check=True,
    )
    return pickle.loads(proc.stdout)


def _load_named_rows_resilient(
    name: str,
) -> tuple[list[str], list[dict[str, Any]], list[list[float]]]:
    """Paginated get with client reopen + subprocess fallback on HNSW errors."""

    def _load() -> tuple[list[str], list[dict[str, Any]], list[list[float]]]:
        return _fetch_named_rows(get_named_collection(name))

    try:
        return _load()
    except ChromaInternalError:
        reset_chroma_client()
        try:
            return _load()
        except ChromaInternalError:
            return _fetch_named_rows_subprocess(name)


def get_vocab_matrix(
    collection_name: str,
) -> tuple[list[str], list[str], np.ndarray]:
    """
    Load a vocabulary collection as L2-normalized matrix.
    Returns (item_ids, texts, matrix).
    """
    cached = _item_matrix_cache.get(("vocab", collection_name))
    if cached is not None:
        return cached

    ids, metas, vectors = _load_named_rows_resilient(collection_name)
    if not ids:
        result: tuple[list[str], list[str], np.ndarray] = (
            [],
            [],
            np.zeros((0, 0), dtype=np.float32),
        )
        _item_matrix_cache[("vocab", collection_name)] = result
        return result

    texts = [str(m.get("text") or "") for m in metas]
    if not vectors or not vectors[0]:
        result = (ids, texts, np.zeros((len(ids), 0), dtype=np.float32))
    else:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        result = (ids, texts, matrix / norms)
    _item_matrix_cache[("vocab", collection_name)] = result
    return result


def get_item_collection_matrix(
    collection_name: str,
) -> tuple[list[str], np.ndarray, dict[str, list[int]]]:
    """
    Load an item collection (e.g. esco_label) with occupation grouping.

    Returns:
        item_ids, L2-normalized matrix, occupation_id -> list of row indices
    """
    cached = _item_matrix_cache.get(("item", collection_name))
    if cached is not None:
        return cached

    ids, metas, vectors = _load_named_rows_resilient(collection_name)
    if not ids:
        empty = ([], np.zeros((0, 0), dtype=np.float32), {})
        _item_matrix_cache[("item", collection_name)] = empty
        return empty

    if not vectors or not vectors[0]:
        matrix = np.zeros((len(ids), 0), dtype=np.float32)
    else:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

    occ_to_rows: dict[str, list[int]] = {}
    for i, meta in enumerate(metas):
        occ = str(
            meta.get("parent_occupation_id")
            or meta.get("occupation_id")
            or ""
        ).strip()
        if not occ:
            continue
        occ_to_rows.setdefault(occ, []).append(i)
        # Also index by uuid for ESCO URIs
        if "/" in occ:
            uuid = occ.rsplit("/", 1)[-1]
            if uuid != occ:
                occ_to_rows.setdefault(uuid, []).append(i)

    result = (ids, matrix, occ_to_rows)
    _item_matrix_cache[("item", collection_name)] = result
    return result


def variant_inventory() -> dict[str, int]:
    return {name: collection_count(name) for name in VARIANT_COLLECTIONS}
