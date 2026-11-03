#!/usr/bin/env python3
"""Compare gated enriched JEV Choice vs Noul runs (Acc@K + token usage).

Requires JEV_API_KEY and raw O*NET/ESCO paths (ONET_SQL_DIR, ESCO_CSV_DIR).

Example:
  python scripts/run_jev_choice_noul_ab.py
  python scripts/run_jev_choice_noul_ab.py --base-url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _post_run(client: httpx.Client, base: str, mode: str, force: bool) -> dict:
    resp = client.post(
        f"{base}/api/runs",
        json={
            "model": "P",
            "input_taxonomy": "onet",
            "output_taxonomy": "esco",
            "threshold": 0.60,
            "top_k": 20,
            "jev_rank_mode": mode,
            "force": force,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _wait_run(client: httpx.Client, base: str, run_id: str, timeout_s: float = 7200) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"{base}/api/runs/{run_id}", timeout=30.0)
        r.raise_for_status()
        run = r.json().get("run") or r.json()
        status = run.get("status")
        if status in {"completed", "failed"}:
            return run
        time.sleep(5)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_s}s")


def _metrics(run: dict) -> dict:
    metrics = run.get("metrics") or {}
    if not metrics:
        run_id = run.get("run_id")
        if run_id:
            path = ROOT / "output" / "runs" / run_id / "metrics.json"
            if path.is_file():
                metrics = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Choice vs Noul JEV A/B eval")
    parser.add_argument("--base-url", default=os.getenv("CROSSWALK_API", "http://127.0.0.1:8001"))
    parser.add_argument("--force", action="store_true", help="Do not reuse prior runs")
    args = parser.parse_args()

    if not (os.getenv("JEV_API_KEY") or "").strip():
        print("Set JEV_API_KEY in .env before running live A/B eval.", file=sys.stderr)
        return 1

    base = args.base_url.rstrip("/")
    with httpx.Client() as client:
        print("Starting Choice run…")
        choice = _post_run(client, base, "choice", args.force)
        choice_run = _wait_run(client, base, choice["run"]["run_id"])
        print("Starting Noul run…")
        noul = _post_run(client, base, "noul", True)
        noul_run = _wait_run(client, base, noul["run"]["run_id"])

    cm = _metrics(choice_run)
    nm = _metrics(noul_run)
    print("\n=== Acc@K ===")
    for k in (1, 2, 3, 5, 10):
        print(f"  Acc@{k}: choice={cm.get(f'accuracy_at_{k}')}  noul={nm.get(f'accuracy_at_{k}')}")
    print("\n=== JEV tokens (gated subset) ===")
    for key in (
        "jev_api_calls_total",
        "jev_input_tokens_total",
        "jev_output_tokens_total",
        "jev_total_tokens",
        "jev_avg_total_tokens_per_gated_source",
    ):
        print(f"  {key}: choice={cm.get(key)}  noul={nm.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
