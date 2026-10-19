"""Active production weights — defaults, file-backed overrides, apply from optimizer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT

from .taxonomies import DEFAULT_PAIR_WEIGHTS

# Canonical signal keys for production model P (maps 1:1 to pair signals)
DEFAULT_WEIGHTS: dict[str, float] = dict(DEFAULT_PAIR_WEIGHTS)

# Friendly aliases accepted from older UI / optimizer payloads
_ALIAS_TO_SIGNAL = {
    "title_max": "label_max",
    "title_median": "label_median",
    "description": "description_pool",
    "skills": "skills_pool",
    "knowledge": "knowledge_pool",
    "title": "label_max",  # legacy 4-facet — fold into title_max only if migrating
}

ACTIVE_WEIGHTS_PATH = ROOT / "config" / "active_weights.json"


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Map aliases → signal keys; keep only known keys."""
    out: dict[str, float] = {}
    for k, v in weights.items():
        key = _ALIAS_TO_SIGNAL.get(k, k)
        if key in DEFAULT_WEIGHTS:
            out[key] = float(v)
    return out


def get_active_weights() -> dict[str, float]:
    """Return current production weights (file override or defaults)."""
    if not ACTIVE_WEIGHTS_PATH.is_file():
        return dict(DEFAULT_WEIGHTS)
    try:
        data = json.loads(ACTIVE_WEIGHTS_PATH.read_text(encoding="utf-8"))
        weights = data.get("weights")
        if isinstance(weights, dict):
            norm = normalize_weights({k: float(v) for k, v in weights.items()})
            if len(norm) == len(DEFAULT_WEIGHTS):
                return {k: norm[k] for k in DEFAULT_WEIGHTS}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return dict(DEFAULT_WEIGHTS)


def get_weights_metadata() -> dict[str, Any]:
    """Metadata about active weights (source run, applied_at, etc.)."""
    if not ACTIVE_WEIGHTS_PATH.is_file():
        return {"source": "default", "weights": dict(DEFAULT_WEIGHTS)}
    try:
        data = json.loads(ACTIVE_WEIGHTS_PATH.read_text(encoding="utf-8"))
        if "weights" in data and isinstance(data["weights"], dict):
            data["weights"] = get_active_weights()
        return data
    except (json.JSONDecodeError, TypeError):
        return {"source": "default", "weights": get_active_weights()}


def apply_weights(
    weights: dict[str, float],
    *,
    run_id: str | None = None,
    metric: str | None = None,
    accuracy: float | None = None,
) -> dict[str, Any]:
    """Persist weights to config/active_weights.json."""
    norm = normalize_weights(weights)
    if len(norm) != len(DEFAULT_WEIGHTS):
        raise ValueError(f"weights must include all of {list(DEFAULT_WEIGHTS)}")
    ACTIVE_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "weights": {k: round(float(norm[k]), 6) for k in DEFAULT_WEIGHTS},
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "source": "optimizer",
    }
    if run_id:
        payload["run_id"] = run_id
    if metric:
        payload["metric"] = metric
    if accuracy is not None:
        payload["accuracy_at_1"] = round(float(accuracy), 6)
    ACTIVE_WEIGHTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
