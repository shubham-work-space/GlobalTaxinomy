"""Chroma inventory fingerprint for run reuse (no optimizer dependency)."""

from __future__ import annotations

import hashlib
import json

from . import chroma as chroma_store

SIGNAL_KEYS = (
    "label_max",
    "label_median",
    "description_pool",
    "skills_pool",
    "knowledge_pool",
)


def inventory_fingerprint() -> str:
    """SHA256 prefix over Chroma inventory summary used in run fingerprints."""
    try:
        inv = chroma_store.inventory_summary()
        raw = json.dumps(
            {
                "model": inv.get("model"),
                "created_at": inv.get("created_at"),
                "onet": inv["datasets"]["onet"]["occupations"],
                "esco": inv["datasets"]["esco"]["occupations"],
                "signals": list(SIGNAL_KEYS),
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"
