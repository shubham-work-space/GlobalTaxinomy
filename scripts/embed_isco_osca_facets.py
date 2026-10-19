"""Embed ISCO/OSCA title and description facets into ChromaDB.

Creates four collections (exact CSV cell text, no labels/prefixes):
  - isco_title         ← preferred_title
  - isco_description   ← Definition
  - osca_title         ← preferred_title
  - osca_description   ← lead_statement

Requires AZURE_OAI_ENDPOINT, AZURE_OAI_KEY, and a large embedding deployment
(AZURE_OAI_EMBEDDING_DEPLOYMENT_LARGE or a deployment name containing 'large').

Embedding API batches run in a multiprocessing pool for throughput.

Usage:
  python scripts/embed_isco_osca_facets.py
  python scripts/embed_isco_osca_facets.py --workers 8
  python scripts/embed_isco_osca_facets.py --skip-existing
  python scripts/embed_isco_osca_facets.py --only isco_title,osca_title
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import chroma as chroma_store  # noqa: E402
from backend.app.config import Settings, get_settings  # noqa: E402

ISCO_CSV = ROOT / "ISCO_OSCA" / "ISCO_OSCA" / "processed" / "isco08" / "isco08_master.csv"
OSCA_CSV = ROOT / "ISCO_OSCA" / "ISCO_OSCA" / "processed" / "osca" / "osca_master.csv"

ALL_COLLECTIONS = (
    "isco_title",
    "isco_description",
    "osca_title",
    "osca_description",
)

# Populated in each worker process via initializer (avoids pickling secrets repeatedly).
_WORKER_CFG: dict[str, str] = {}


def _init_worker(endpoint: str, api_key: str, api_version: str, deployment: str) -> None:
    global _WORKER_CFG
    _WORKER_CFG = {
        "endpoint": endpoint,
        "api_key": api_key,
        "api_version": api_version,
        "deployment": deployment,
    }


def _embed_chunk(job: tuple[int, list[str]]) -> tuple[int, list[list[float]]]:
    """Embed one text chunk in a worker process. Returns (start_index, vectors)."""
    from openai import AzureOpenAI, RateLimitError

    start, texts = job
    client = AzureOpenAI(
        azure_endpoint=_WORKER_CFG["endpoint"],
        api_key=_WORKER_CFG["api_key"],
        api_version=_WORKER_CFG["api_version"],
    )
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(8):
        try:
            resp = client.embeddings.create(
                model=_WORKER_CFG["deployment"],
                input=[t or " " for t in texts],
            )
            ordered = sorted(resp.data, key=lambda x: x.index)
            return start, [list(item.embedding) for item in ordered]
        except RateLimitError as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise RuntimeError(f"Failed to embed batch at {start}: {last_err!r}")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _build_isco(rows: list[dict[str, str]]) -> dict[str, tuple[list[str], list[str], list[dict]]]:
    title_ids: list[str] = []
    title_texts: list[str] = []
    title_metas: list[dict] = []
    desc_ids: list[str] = []
    desc_texts: list[str] = []
    desc_metas: list[dict] = []

    for row in rows:
        occ_id = (row.get("normalized_code") or row.get("source_code") or "").strip()
        preferred = (row.get("preferred_title") or "").strip()
        definition = row.get("Definition") or ""
        if not occ_id:
            continue
        title_ids.append(occ_id)
        title_texts.append(preferred)
        title_metas.append(
            {
                "occupation_id": occ_id,
                "dataset": "isco",
                "preferred_title": preferred,
                "facet": "title",
            }
        )
        desc_ids.append(occ_id)
        desc_texts.append(definition)
        desc_metas.append(
            {
                "occupation_id": occ_id,
                "dataset": "isco",
                "preferred_title": preferred,
                "facet": "description",
            }
        )

    return {
        "isco_title": (title_ids, title_texts, title_metas),
        "isco_description": (desc_ids, desc_texts, desc_metas),
    }


def _build_osca(rows: list[dict[str, str]]) -> dict[str, tuple[list[str], list[str], list[dict]]]:
    title_ids: list[str] = []
    title_texts: list[str] = []
    title_metas: list[dict] = []
    desc_ids: list[str] = []
    desc_texts: list[str] = []
    desc_metas: list[dict] = []

    for row in rows:
        occ_id = (row.get("occupation_code") or "").strip()
        preferred = (row.get("preferred_title") or "").strip()
        lead = row.get("lead_statement") or ""
        if not occ_id:
            continue
        title_ids.append(occ_id)
        title_texts.append(preferred)
        title_metas.append(
            {
                "occupation_id": occ_id,
                "dataset": "osca",
                "preferred_title": preferred,
                "facet": "title",
            }
        )
        desc_ids.append(occ_id)
        desc_texts.append(lead)
        desc_metas.append(
            {
                "occupation_id": occ_id,
                "dataset": "osca",
                "preferred_title": preferred,
                "facet": "description",
            }
        )

    return {
        "osca_title": (title_ids, title_texts, title_metas),
        "osca_description": (desc_ids, desc_texts, desc_metas),
    }


def _require_large_model(settings: Settings) -> tuple[str, str]:
    deployment = settings.embedding_deployment
    model = settings.embedding_model_name
    if "large" not in deployment.lower() and model != "text-embedding-3-large":
        raise SystemExit(
            "Large embedding model required. Set AZURE_OAI_EMBEDDING_DEPLOYMENT_LARGE "
            "(e.g. text-embedding-3-large) in .env."
        )
    if not settings.azure_oai_endpoint.strip() or not settings.azure_oai_key.strip():
        raise SystemExit(
            "Missing Azure OpenAI credentials. Set AZURE_OAI_ENDPOINT and AZURE_OAI_KEY in .env."
        )
    return deployment, model


def _parse_only(raw: str | None) -> list[str]:
    if not raw:
        return list(ALL_COLLECTIONS)
    wanted: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name == "isco":
            wanted.extend(["isco_title", "isco_description"])
        elif name == "osca":
            wanted.extend(["osca_title", "osca_description"])
        elif name in ALL_COLLECTIONS:
            wanted.append(name)
        else:
            raise SystemExit(
                f"Unknown --only value: {part!r}. "
                f"Use isco, osca, or one of: {', '.join(ALL_COLLECTIONS)}"
            )
    seen: set[str] = set()
    out: list[str] = []
    for n in wanted:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _chunk_jobs(texts: list[str], batch_size: int) -> list[tuple[int, list[str]]]:
    return [
        (start, texts[start : start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]


def _embed_texts_mp(
    texts: list[str],
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    workers: int,
    batch_size: int,
    label: str,
) -> list[list[float]]:
    jobs = _chunk_jobs(texts, batch_size)
    if not jobs:
        return []

    results: dict[int, list[list[float]]] = {}
    done_items = 0
    total = len(texts)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(endpoint, api_key, api_version, deployment),
    ) as pool:
        futures = {pool.submit(_embed_chunk, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            start, vectors = fut.result()
            results[start] = vectors
            done_items += len(vectors)
            print(f"  {label}: {done_items}/{total}", flush=True)

    ordered: list[list[float]] = []
    for start, _ in jobs:
        ordered.extend(results[start])
    if len(ordered) != len(texts):
        raise RuntimeError(
            f"{label}: expected {len(texts)} vectors, got {len(ordered)}"
        )
    return ordered


def _embed_collection(
    *,
    name: str,
    ids: list[str],
    texts: list[str],
    metas: list[dict],
    endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
    model: str,
    created_at: str,
    skip_existing: bool,
    workers: int,
    batch_size: int,
) -> int:
    if not ids:
        print(f"{name}: empty — skipped")
        return 0

    existing = chroma_store.collection_count(name)
    if skip_existing and existing >= len(ids):
        print(f"{name}: skip ({existing} already present, need {len(ids)})")
        return existing

    print(f"{name}: embedding {len(ids)} with {workers} workers…")
    for meta in metas:
        meta.setdefault("model", model)
        meta.setdefault("created_at", created_at)

    vectors = _embed_texts_mp(
        texts,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment=deployment,
        workers=workers,
        batch_size=batch_size,
        label=name,
    )
    chroma_store.upsert_items(name, ids, vectors, texts, metas)
    print(f"{name}: done ({len(ids)})")
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed ISCO/OSCA title & description facets")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a collection if Chroma already has >= source row count",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset: isco, osca, isco_title, isco_description, "
        "osca_title, osca_description",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Process-pool size for parallel embedding batches (default: min(8, CPU count))",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts per Azure embeddings API call (default: 64)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    get_settings.cache_clear()
    settings = get_settings()
    deployment, model = _require_large_model(settings)
    created_at = datetime.now(timezone.utc).isoformat()
    selected = _parse_only(args.only)

    if not ISCO_CSV.is_file():
        raise SystemExit(f"ISCO master CSV not found: {ISCO_CSV}")
    if not OSCA_CSV.is_file():
        raise SystemExit(f"OSCA master CSV not found: {OSCA_CSV}")

    payloads: dict[str, tuple[list[str], list[str], list[dict]]] = {}
    if any(n.startswith("isco_") for n in selected):
        payloads.update(_build_isco(_load_csv(ISCO_CSV)))
    if any(n.startswith("osca_") for n in selected):
        payloads.update(_build_osca(_load_csv(OSCA_CSV)))

    endpoint = settings.azure_oai_endpoint.strip()
    api_key = settings.azure_oai_key.strip()
    api_version = settings.azure_openai_api_version

    print(f"Model: {model}  deployment: {deployment}")
    print(f"Chroma: {settings.chroma_dir}")
    print(f"Workers: {args.workers}  batch_size: {args.batch_size}")
    print(f"Collections: {', '.join(selected)}")

    counts: dict[str, int] = {}
    for name in selected:
        ids, texts, metas = payloads[name]
        counts[name] = _embed_collection(
            name=name,
            ids=ids,
            texts=texts,
            metas=metas,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment=deployment,
            model=model,
            created_at=created_at,
            skip_existing=args.skip_existing,
            workers=args.workers,
            batch_size=args.batch_size,
        )

    print("\n--- verification ---")
    for name in selected:
        col = chroma_store.get_named_collection(name)
        n = col.count()
        got = col.get(limit=1, include=["documents", "metadatas"])
        doc = (got.get("documents") or [None])[0]
        meta = (got.get("metadatas") or [{}])[0] or {}
        sample_id = got["ids"][0] if got.get("ids") else None
        print(f"{name}: count={n}  sample_id={sample_id}")
        print(f"  document[:80]={doc[:80]!r}" if doc is not None else "  document=None")
        print(f"  facet={meta.get('facet')}  model={meta.get('model')}")
        ids, texts, _ = payloads[name]
        if sample_id and texts and sample_id in ids:
            expected = texts[ids.index(sample_id)]
            if doc != expected:
                print(f"  WARNING: document mismatch for {sample_id}")
            else:
                print("  document matches source cell")

    print("\nFinal counts:", counts)


if __name__ == "__main__":
    main()
