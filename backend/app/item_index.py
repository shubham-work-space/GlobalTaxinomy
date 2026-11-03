"""Load sidecar item indexes from data/item_index/."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import ROOT

ITEM_INDEX_DIR = ROOT / "data" / "item_index"


def _load_json(name: str) -> Any:
    path = ITEM_INDEX_DIR / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _esco_labels_path() -> Path:
    """Allow ESCO_LABELS_JSON env override (absolute or repo-relative)."""
    raw = (os.environ.get("ESCO_LABELS_JSON") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p
    return ITEM_INDEX_DIR / "esco_labels.json"


@lru_cache
def esco_labels() -> dict[str, list[dict[str, Any]]]:
    path = _esco_labels_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def esco_skills() -> dict[str, list[dict[str, Any]]]:
    return _load_json("esco_skills.json")


def clear_item_index_cache() -> None:
    esco_labels.cache_clear()
    esco_skills.cache_clear()
