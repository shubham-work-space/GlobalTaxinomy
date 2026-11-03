"""Force-run Model P O*NET→ESCO eval against ESCO v1.1.0 Chroma + labels.

Sets version-isolated env vars, then executes the same RunPipelineJob path as the API.

Prerequisites:
  python scripts/prepare_esco_v110_facets.py
  python scripts/embed_esco_v110_facets.py

Usage:
  python scripts/run_esco_v110_eval.py
  python scripts/run_esco_v110_eval.py --threshold 0.60 --top-k 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _configure_env() -> None:
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(ROOT / "data" / "chroma_db_1.1.0")
    os.environ["ESCO_LABELS_JSON"] = str(
        ROOT / "data" / "item_index" / "esco_labels_1.1.0.json"
    )
    os.environ["ESCO_FACET_ROOT"] = str(ROOT / "data" / "esco_1.1.0")
    os.environ["SIGNAL_TENSOR_SUFFIX"] = "_1.1.0"


async def _run(threshold: float, top_k: int) -> dict:
    _configure_env()

    # Reset cached settings / chroma / item index after env patch
    from backend.app.config import get_settings
    from backend.app import chroma as chroma_store
    from backend.app.item_index import clear_item_index_cache
    from backend.app.pipeline import RunPipelineJob
    from backend.app.runs import default_config, validate_config
    from backend.app.variant_tensor import tensor_path

    get_settings.cache_clear()
    chroma_store.reset_chroma_client()
    clear_item_index_cache()

    settings = get_settings()
    print(f"Chroma: {settings.chroma_dir}")
    print(f"Labels: {os.environ['ESCO_LABELS_JSON']}")
    print(f"Tensor: {tensor_path('P')}")

    inv = chroma_store.inventory_summary()
    onet_n = inv["datasets"]["onet"]["occupations"]
    esco_n = inv["datasets"]["esco"]["occupations"]
    labels_n = chroma_store.collection_count("esco_label")
    print(f"Inventory: onet={onet_n} esco={esco_n} labels={labels_n}")
    if onet_n == 0 or esco_n == 0 or labels_n == 0:
        raise SystemExit(
            "chroma_db_1.1.0 incomplete — run scripts/embed_esco_v110_facets.py first"
        )

    cfg = validate_config(
        {
            **default_config(),
            "threshold": threshold,
            "top_k": top_k,
        }
    )
    job = RunPipelineJob()

    async def watch() -> None:
        last = -1
        while True:
            snap = job.snapshot()
            ver = int(snap.get("version") or 0)
            if ver != last:
                last = ver
                print(
                    f"[{snap.get('status')}] {snap.get('phase')} "
                    f"{snap.get('percent')}% — {snap.get('message')}",
                    flush=True,
                )
            if snap.get("status") in {"completed", "failed"}:
                return
            await asyncio.sleep(2)

    start = await job.start(cfg)
    if not start.get("ok"):
        raise SystemExit(start.get("error") or "failed to start run")
    print(f"Started {start['run'].get('label')} id={start['run'].get('run_id')}")
    await watch()
    # Wait for background task
    if job._task is not None:
        await job._task
    snap = job.snapshot()
    if snap.get("status") != "completed":
        raise SystemExit(f"Run failed: {snap.get('error') or snap.get('message')}")
    return snap


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESCO v1.1.0 gated eval")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    snap = asyncio.run(_run(args.threshold, args.top_k))
    print("Done:", snap.get("run_id"), snap.get("message"))


if __name__ == "__main__":
    main()
