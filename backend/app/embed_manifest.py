"""Embedding manifest: fingerprint + counts for label/skill collections."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT

MANIFEST_PATH = ROOT / "data" / "embed_manifest.json"
SCHEMA_VERSION = "variant_collections_v2"
SOURCE_VERSION = "esco_v1.2.1_onet_31.0"


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_fingerprint(model: str = "text-embedding-3-large") -> str:
    """Fingerprint over item-index sources + model + schema."""
    index_dir = ROOT / "data" / "item_index"
    parts = [SCHEMA_VERSION, SOURCE_VERSION, model]
    for name in ("esco_labels.json", "esco_skills.json"):
        parts.append(f"{name}:{_file_sha256(index_dir / name)}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(
    *,
    model: str,
    collections: dict[str, int],
    fingerprint: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_version": SOURCE_VERSION,
        "model": model,
        "fingerprint": fingerprint or compute_fingerprint(model),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collections": collections,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def manifest_is_current(model: str = "text-embedding-3-large") -> bool:
    m = load_manifest()
    if not m:
        return False
    return (
        m.get("fingerprint") == compute_fingerprint(model)
        and m.get("schema_version") == SCHEMA_VERSION
        and m.get("model") == model
    )
