"""Run registry: config fingerprinting, result reuse, sequential Run1..RunN IDs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT
from .taxonomies import (
    DEFAULT_DIRECTION,
    DEFAULT_INPUT_TAXONOMY,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_TAXONOMY,
    DEFAULT_PAIR_WEIGHTS,
    get_pair,
)
from .weights import normalize_weights

RUNS_DIR = ROOT / "output" / "runs"
INDEX_PATH = RUNS_DIR / "index.json"
WEIGHT_TOLERANCE = 0.001
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99

PHASES = (
    "generate_embeddings",
    "calculate_similarities",
    "select_top_candidates",
    "apply_threshold",
    "run_jev_evaluation",
    "calculate_accuracy",
    "save_run",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_index() -> dict[str, Any]:
    if not INDEX_PATH.is_file():
        return {"runs": [], "next_seq": 1}
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            data.setdefault("next_seq", 1)
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {"runs": [], "next_seq": 1}


def _save_index(index: dict[str, Any]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate a run configuration. Raises ValueError on bad input."""
    model = str(config.get("model") or DEFAULT_MODEL).strip().upper()
    if model != DEFAULT_MODEL:
        raise ValueError(f"Only model {DEFAULT_MODEL} is supported, got {model!r}")

    input_taxonomy = str(
        config.get("input_taxonomy") or DEFAULT_INPUT_TAXONOMY
    ).strip().lower()
    output_taxonomy = str(
        config.get("output_taxonomy") or DEFAULT_OUTPUT_TAXONOMY
    ).strip().lower()
    pair = get_pair(input_taxonomy, output_taxonomy)

    raw_weights = config.get("weights") or dict(DEFAULT_PAIR_WEIGHTS)
    if not isinstance(raw_weights, dict):
        raise ValueError("weights must be an object")
    weights = normalize_weights({k: float(v) for k, v in raw_weights.items()})
    missing = [k for k in pair.signals if k not in weights]
    if missing:
        raise ValueError(f"weights must include {', '.join(pair.signals)}")
    total = sum(weights[k] for k in pair.signals)
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"weights must sum to 1.0 (100%), got {total:.4f}")
    weights = {k: round(float(weights[k]), 6) for k in pair.signals}

    try:
        threshold = float(config.get("threshold", 0.60))
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a number") from exc
    if threshold < THRESHOLD_MIN or threshold > THRESHOLD_MAX:
        raise ValueError(
            f"threshold must be between {THRESHOLD_MIN} and {THRESHOLD_MAX}, got {threshold}"
        )

    return {
        "model": model,
        "input_taxonomy": pair.input_taxonomy,
        "output_taxonomy": pair.output_taxonomy,
        "direction": pair.direction,
        "weights": weights,
        "threshold": round(threshold, 4),
        "top_k": int(config.get("top_k") or 20),
    }


def _inventory_fingerprint() -> str:
    from .inventory import inventory_fingerprint

    return inventory_fingerprint()


def _ground_truth_version(direction: str) -> str:
    try:
        pair = get_pair(direction=direction)
        path = Path(pair.ground_truth_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            st = path.stat()
            return f"{path.name}:{int(st.st_mtime)}:{st.st_size}"
    except Exception:
        pass
    return "unknown"


def run_fingerprint(config: dict[str, Any]) -> str:
    """SHA256 over model | taxonomies | weights | threshold | top_k | rank mode | GT | inventory."""
    cfg = validate_config(config)
    parts = [
        cfg["model"],
        cfg["input_taxonomy"],
        cfg["output_taxonomy"],
        "|".join(f"{k}:{cfg['weights'][k]:.6f}" for k in sorted(cfg["weights"])),
        f"{cfg['threshold']:.4f}",
        f"top_k:{int(cfg['top_k'])}",
        "jev_rank_v1",
        _ground_truth_version(cfg["direction"]),
        _inventory_fingerprint(),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def find_reusable(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return a completed run with matching fingerprint, if any."""
    fp = run_fingerprint(config)
    index = _load_index()
    for entry in reversed(index.get("runs", [])):
        if entry.get("fingerprint") == fp and entry.get("status") == "completed":
            return deepcopy(entry)
    return None


def list_runs() -> list[dict[str, Any]]:
    index = _load_index()
    return list(reversed(index.get("runs", [])))


def get_run(run_id: str) -> dict[str, Any] | None:
    index = _load_index()
    for entry in index.get("runs", []):
        if entry.get("run_id") == run_id or entry.get("label") == run_id:
            detail = deepcopy(entry)
            metrics_path = run_dir(entry["run_id"]) / "metrics.json"
            config_path = run_dir(entry["run_id"]) / "config.json"
            if metrics_path.is_file():
                detail["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            if config_path.is_file():
                detail["config"] = json.loads(config_path.read_text(encoding="utf-8"))
            return detail
    return None


def load_results(run_id: str) -> list[dict[str, Any]]:
    path = run_dir(run_id) / "results.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def create_run(config: dict[str, Any], *, reused_from: str | None = None) -> dict[str, Any]:
    """Allocate RunN, persist config stub, and append to the index."""
    cfg = validate_config(config)
    fp = run_fingerprint(cfg)
    index = _load_index()
    seq = int(index.get("next_seq") or 1)
    label = f"Run{seq}"
    run_id = f"run_{seq:04d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    index["next_seq"] = seq + 1

    entry: dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "seq": seq,
        "fingerprint": fp,
        "model": cfg["model"],
        "input_taxonomy": cfg["input_taxonomy"],
        "output_taxonomy": cfg["output_taxonomy"],
        "direction": cfg["direction"],
        "weights": cfg["weights"],
        "threshold": cfg["threshold"],
        "status": "pending" if reused_from else "running",
        "reused_from": reused_from,
        "created_at": _now(),
        "finished_at": None,
        "phase": None,
        "percent": 0.0,
        "message": "",
        "metrics_summary": None,
    }
    index.setdefault("runs", []).append(entry)
    _save_index(index)

    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(
        json.dumps({**cfg, "fingerprint": fp, "label": label, "run_id": run_id}, indent=2),
        encoding="utf-8",
    )
    return deepcopy(entry)


def update_run(run_id: str, **patch: Any) -> dict[str, Any] | None:
    index = _load_index()
    for entry in index.get("runs", []):
        if entry.get("run_id") != run_id:
            continue
        entry.update(patch)
        _save_index(index)
        return deepcopy(entry)
    return None


def metrics_summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build the Evals-table metrics_summary (incl. resolver subset / emb≥thr)."""
    return {
        "total_records": metrics.get("total_records"),
        "records_sent_to_jev": metrics.get("records_sent_to_jev"),
        "exact_match_at_1": metrics.get("exact_match_at_1"),
        "accuracy_at_1": metrics.get("accuracy_at_1", metrics.get("exact_match_at_1")),
        "accuracy_at_2": metrics.get("accuracy_at_2"),
        "accuracy_at_3": metrics.get("accuracy_at_3"),
        "accuracy_at_5": metrics.get("accuracy_at_5"),
        "accuracy_at_10": metrics.get("accuracy_at_10"),
        "embedding_top1_accuracy": metrics.get("embedding_top1_accuracy"),
        "final_accuracy_after_llm": metrics.get("exact_match_at_1")
        or metrics.get("final_accuracy_after_llm"),
        "records_sent_to_llm": metrics.get("records_sent_to_llm")
        or metrics.get("records_sent_to_jev"),
        "llm_subset_accuracy_before": metrics.get("llm_subset_accuracy_before"),
        "llm_subset_accuracy_after": metrics.get("llm_subset_accuracy_after"),
        "jev_subset_accuracy_after": metrics.get("jev_subset_accuracy_after"),
        "accuracy_gain": metrics.get("accuracy_gain"),
        "resolver": metrics.get("resolver") or "jev",
        "high_confidence_records": metrics.get("high_confidence_records"),
        "high_confidence_correct": metrics.get("high_confidence_correct"),
        "high_confidence_accuracy": metrics.get("high_confidence_accuracy"),
        "resolver_subset_records": metrics.get("resolver_subset_records"),
        "resolver_subset_correct": metrics.get("resolver_subset_correct"),
        "resolver_subset_accuracy": metrics.get("resolver_subset_accuracy"),
    }


def save_run_artifacts(
    run_id: str,
    *,
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    results_path = rd / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_run(
        run_id,
        status="completed",
        finished_at=_now(),
        phase="save_run",
        percent=100.0,
        message="Run complete",
        metrics_summary=metrics_summary_from_metrics(metrics),
    )


def delete_run(run_id: str) -> bool:
    import shutil

    index = _load_index()
    matched = [
        r
        for r in index.get("runs", [])
        if r.get("run_id") == run_id or r.get("label") == run_id
    ]
    if not matched:
        return False
    canonical_ids = {r["run_id"] for r in matched}
    index["runs"] = [r for r in index.get("runs", []) if r.get("run_id") not in canonical_ids]
    _save_index(index)
    for cid in canonical_ids:
        rd = run_dir(cid)
        if rd.is_dir():
            shutil.rmtree(rd, ignore_errors=True)
    return True


def clone_reused_run(source_run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Create a new RunN that points at an earlier completed run's artifacts."""
    source = get_run(source_run_id)
    if not source:
        raise LookupError(f"Source run not found: {source_run_id}")
    entry = create_run(config, reused_from=source_run_id)
    # Copy artifacts
    src_dir = run_dir(source["run_id"])
    dst_dir = run_dir(entry["run_id"])
    for name in ("results.jsonl", "metrics.json"):
        src = src_dir / name
        if src.is_file():
            (dst_dir / name).write_bytes(src.read_bytes())
    metrics = {}
    metrics_path = dst_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # Prefer a full summary from metrics (resolver subset / emb≥thr), then fall
    # back to the source index entry when metrics lack those fields.
    summary = metrics_summary_from_metrics(metrics) if metrics else {}
    source_summary = source.get("metrics_summary") or {}
    for key, value in source_summary.items():
        if summary.get(key) is None and value is not None:
            summary[key] = value
    update_run(
        entry["run_id"],
        status="completed",
        finished_at=_now(),
        phase="save_run",
        percent=100.0,
        message=f"Reused results from {source.get('label') or source_run_id}",
        metrics_summary=summary,
        reused=True,
    )
    return get_run(entry["run_id"]) or entry


def default_config() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL,
        "input_taxonomy": DEFAULT_INPUT_TAXONOMY,
        "output_taxonomy": DEFAULT_OUTPUT_TAXONOMY,
        "direction": DEFAULT_DIRECTION,
        "weights": dict(DEFAULT_PAIR_WEIGHTS),
        "threshold": 0.60,
        "top_k": 20,
    }
