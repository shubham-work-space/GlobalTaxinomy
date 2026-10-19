"""Recompute resolver-subset / emb≥thr metrics into metrics.json + index summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.accuracy import compute_accuracy_metrics
from backend.app.runs import metrics_summary_from_metrics, run_dir, update_run


def backfill(run_id: str) -> dict:
    rd = run_dir(run_id)
    rows = [
        json.loads(line)
        for line in (rd / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cfg = {}
    cfg_path = rd / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    direction = cfg.get("direction") or "onet_to_esco"
    metrics = compute_accuracy_metrics(rows, direction=direction)
    old = {}
    metrics_path = rd / "metrics.json"
    if metrics_path.is_file():
        old = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("model", "input_taxonomy", "output_taxonomy", "weights", "threshold", "resolver"):
        if key in old and key not in metrics:
            metrics[key] = old[key]
        elif key in cfg and key not in metrics:
            metrics[key] = cfg[key]
    if old.get("resolver"):
        metrics["resolver"] = old["resolver"]
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = metrics_summary_from_metrics(metrics)
    update_run(run_id, metrics_summary=summary)
    return {
        "run_id": run_id,
        "high_confidence": (
            f"{metrics['high_confidence_correct']}/"
            f"{metrics['high_confidence_records']} "
            f"({metrics['high_confidence_accuracy']})"
        ),
        "resolver_subset": (
            f"{metrics['resolver_subset_correct']}/"
            f"{metrics['resolver_subset_records']} "
            f"({metrics['resolver_subset_accuracy']})"
        ),
        "acc1": metrics["accuracy_at_1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="Run ids, e.g. run_0012_...")
    args = parser.parse_args()
    for run_id in args.run_ids:
        print(json.dumps(backfill(run_id), indent=2))


if __name__ == "__main__":
    main()
