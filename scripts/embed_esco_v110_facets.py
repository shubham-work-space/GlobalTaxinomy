"""Build data/chroma_db_1.1.0: copy O*NET vectors, embed ESCO v1.1.0 facets + labels.

Prerequisites:
  python scripts/prepare_esco_v110_facets.py
  .env with Azure large embedding deployment

Usage:
  python scripts/embed_esco_v110_facets.py
  python scripts/embed_esco_v110_facets.py --skip-existing
  python scripts/embed_esco_v110_facets.py --workers 8 --only esco_label
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import Settings, get_settings  # noqa: E402

SRC_CHROMA = ROOT / "data" / "chroma_db"
DST_CHROMA = ROOT / "data" / "chroma_db_1.1.0"
FACET_ROOT = ROOT / "data" / "esco_1.1.0"
LABELS_JSON = ROOT / "data" / "item_index" / "esco_labels_1.1.0.json"
MANIFEST_PATH = ROOT / "data" / "embed_manifest_1.1.0.json"
URI_PREFIX = "http://data.europa.eu/esco/occupation/"

ONET_COLLECTIONS = ("onet_title", "onet_description", "onet_skills", "onet_knowledge")
ESCO_FACETS = ("title", "description", "skills", "knowledge")
ESCO_COLLECTIONS = tuple(f"esco_{f}" for f in ESCO_FACETS)

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


def _require_large_model(settings: Settings) -> tuple[str, str]:
    deployment = settings.embedding_deployment
    model = settings.embedding_model_name
    if "large" not in deployment.lower() and model != "text-embedding-3-large":
        raise SystemExit(
            "Large embedding model required. Set AZURE_OAI_EMBEDDING_DEPLOYMENT_LARGE "
            "in .env."
        )
    if not settings.azure_oai_endpoint.strip() or not settings.azure_oai_key.strip():
        raise SystemExit(
            "Missing Azure OpenAI credentials. Set AZURE_OAI_ENDPOINT and AZURE_OAI_KEY."
        )
    return deployment, model


def _client(path: Path) -> chromadb.PersistentClient:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _copy_collection(
    src: chromadb.PersistentClient,
    dst: chromadb.PersistentClient,
    name: str,
    *,
    batch: int = 256,
) -> int:
    src_col = src.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    total = src_col.count()
    if total == 0:
        print(f"  {name}: source empty — skipped")
        return 0
    existing = {c.name for c in dst.list_collections()}
    if name in existing:
        dst.delete_collection(name)
    dst_col = dst.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    copied = 0
    for start in range(0, total, batch):
        got = src_col.get(
            limit=batch,
            offset=start,
            include=["embeddings", "documents", "metadatas"],
        )
        ids = list(got.get("ids") or [])
        if not ids:
            continue
        embs = got.get("embeddings")
        docs = got.get("documents") or [""] * len(ids)
        metas = got.get("metadatas") or [{}] * len(ids)
        dst_col.upsert(
            ids=ids,
            embeddings=list(embs) if embs is not None else None,
            documents=list(docs),
            metadatas=[dict(m) if m else {} for m in metas],
        )
        copied += len(ids)
        print(f"  {name}: copied {copied}/{total}", flush=True)
    return copied


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
    jobs = [
        (start, texts[start : start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]
    if not jobs:
        return []
    results: dict[int, list[list[float]]] = {}
    done_items = 0
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
            print(f"  {label}: {done_items}/{len(texts)}", flush=True)
    ordered: list[list[float]] = []
    for start, _ in jobs:
        ordered.extend(results[start])
    if len(ordered) != len(texts):
        raise RuntimeError(f"{label}: expected {len(texts)} vectors, got {len(ordered)}")
    return ordered


def _upsert_named(
    client: chromadb.PersistentClient,
    name: str,
    ids: list[str],
    vectors: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    *,
    batch: int = 128,
) -> None:
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        # Keep collection; upsert overwrites ids
        col = client.get_collection(name)
    else:
        col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(ids), batch):
        end = start + batch
        col.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def _load_esco_facet_payloads() -> dict[str, tuple[list[str], list[str], list[dict]]]:
    if not FACET_ROOT.is_dir():
        raise SystemExit(f"Missing facets dir: {FACET_ROOT} (run prepare_esco_v110_facets.py)")
    out: dict[str, tuple[list[str], list[str], list[dict]]] = {
        f"esco_{f}": ([], [], []) for f in ESCO_FACETS
    }
    dirs = sorted(d for d in FACET_ROOT.iterdir() if d.is_dir())
    for occ_dir in dirs:
        uuid = occ_dir.name
        uri = f"{URI_PREFIX}{uuid}"
        preferred = (occ_dir / "title.txt").read_text(encoding="utf-8").strip()
        for facet in ESCO_FACETS:
            text = (occ_dir / f"{facet}.txt").read_text(encoding="utf-8")
            ids, texts, metas = out[f"esco_{facet}"]
            ids.append(uri)
            texts.append(text)
            metas.append(
                {
                    "occupation_id": uri,
                    "dataset": "esco",
                    "preferred_title": preferred,
                    "facet": facet,
                }
            )
    return out


def _load_label_payload() -> tuple[list[str], list[str], list[dict]]:
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    seen: set[str] = set()
    for occ_id, rows in labels.items():
        for row in rows:
            lid = row["label_id"]
            if lid in seen:
                continue
            seen.add(lid)
            ids.append(lid)
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
    return ids, texts, metas


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed ESCO v1.1.0 into chroma_db_1.1.0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma list: onet_copy, esco_title, esco_description, esco_skills, "
        "esco_knowledge, esco_label (or 'esco' / 'labels')",
    )
    parser.add_argument("--skip-onet-copy", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    deployment, model = _require_large_model(settings)
    created_at = datetime.now(timezone.utc).isoformat()

    wanted: set[str] | None = None
    if args.only:
        wanted = set()
        for part in args.only.split(","):
            name = part.strip().lower()
            if name == "onet_copy":
                wanted.add("onet_copy")
            elif name == "esco":
                wanted.update(ESCO_COLLECTIONS)
            elif name in ("labels", "esco_label"):
                wanted.add("esco_label")
            elif name in ESCO_COLLECTIONS or name == "esco_label":
                wanted.add(name)
            else:
                raise SystemExit(f"Unknown --only value: {part!r}")

    def do(step: str) -> bool:
        return wanted is None or step in wanted

    if not SRC_CHROMA.is_dir():
        raise SystemExit(f"Missing source Chroma: {SRC_CHROMA}")

    dst = _client(DST_CHROMA)
    counts: dict[str, int] = {}

    # 1) Copy O*NET collections
    if do("onet_copy") and not args.skip_onet_copy:
        print(f"Copying O*NET collections from {SRC_CHROMA} -> {DST_CHROMA}")
        src = _client(SRC_CHROMA)
        for name in ONET_COLLECTIONS:
            if args.skip_existing:
                try:
                    n = dst.get_collection(name).count()
                except Exception:
                    n = 0
                if n > 0:
                    print(f"  {name}: skip ({n} already present)")
                    counts[name] = n
                    continue
            counts[name] = _copy_collection(src, dst, name)
    else:
        print("Skipping O*NET copy")

    # Point chroma module at destination for upsert helpers if needed
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(DST_CHROMA)

    # 2) Embed ESCO facets
    if any(do(c) for c in ESCO_COLLECTIONS):
        payloads = _load_esco_facet_payloads()
        for name in ESCO_COLLECTIONS:
            if not do(name):
                continue
            ids, texts, metas = payloads[name]
            if args.skip_existing:
                try:
                    n = dst.get_collection(name).count()
                except Exception:
                    n = 0
                if n >= len(ids) and len(ids) > 0:
                    print(f"{name}: skip ({n} already present)")
                    counts[name] = n
                    continue
            print(f"{name}: embedding {len(ids)} with {args.workers} workers…")
            for meta in metas:
                meta.setdefault("model", model)
                meta.setdefault("created_at", created_at)
            vectors = _embed_texts_mp(
                texts,
                endpoint=settings.azure_oai_endpoint,
                api_key=settings.azure_oai_key,
                api_version=settings.azure_openai_api_version,
                deployment=deployment,
                workers=args.workers,
                batch_size=args.batch_size,
                label=name,
            )
            # Recreate collection for clean v1.1.0 set
            existing = {c.name for c in dst.list_collections()}
            if name in existing:
                dst.delete_collection(name)
            _upsert_named(dst, name, ids, vectors, texts, metas)
            counts[name] = len(ids)
            print(f"{name}: done ({len(ids)})")

    # 3) Embed esco_label
    if do("esco_label"):
        ids, texts, metas = _load_label_payload()
        skip_labels = False
        if args.skip_existing:
            try:
                n = dst.get_collection("esco_label").count()
            except Exception:
                n = 0
            if n >= len(ids) and len(ids) > 0:
                print(f"esco_label: skip ({n} already present)")
                counts["esco_label"] = n
                skip_labels = True
        if not skip_labels:
            print(f"esco_label: embedding {len(ids)} with {args.workers} workers…")
            for meta in metas:
                meta.setdefault("model", model)
                meta.setdefault("created_at", created_at)
            vectors = _embed_texts_mp(
                texts,
                endpoint=settings.azure_oai_endpoint,
                api_key=settings.azure_oai_key,
                api_version=settings.azure_openai_api_version,
                deployment=deployment,
                workers=args.workers,
                batch_size=args.batch_size,
                label="esco_label",
            )
            existing = {c.name for c in dst.list_collections()}
            if "esco_label" in existing:
                dst.delete_collection("esco_label")
            _upsert_named(dst, "esco_label", ids, vectors, texts, metas)
            counts["esco_label"] = len(ids)
            print(f"esco_label: done ({len(ids)})")

    manifest = {
        "schema_version": "variant_collections_v2",
        "source_version": "esco_v1.1.0_onet_31.0",
        "model": model,
        "created_at": created_at,
        "chroma_path": str(DST_CHROMA),
        "collections": counts,
        "labels_json": str(LABELS_JSON),
        "facet_root": str(FACET_ROOT),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
