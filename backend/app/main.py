"""FastAPI app: Taxonomy Crosswalk 4.0 — Model P + Jev Top-1 hybrid runs."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import chroma as chroma_store
from . import runs as run_store
from .config import get_settings
from .crosswalk import Direction, search_occupations
from .pipeline import run_pipeline_job
from .taxonomies import api_payload as taxonomies_payload
from .live_crosswalk import DEFAULT_THRESHOLD, live_crosswalk
from .weights import DEFAULT_WEIGHTS, get_active_weights

app = FastAPI(title="Taxonomy Crosswalk 4.0", version="4.0.0")

settings = get_settings()
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrosswalkBody(BaseModel):
    direction: Literal["onet_to_esco", "esco_to_onet"]
    occupation_id: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=20)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.01, le=0.99)
    weights: dict[str, float] | None = None
    model: str = "P"


class RunStartBody(BaseModel):
    model: str = "P"
    input_taxonomy: str = "onet"
    output_taxonomy: str = "esco"
    weights: dict[str, float] | None = None
    threshold: float = Field(0.60, ge=0.01, le=0.99)
    top_k: int = Field(20, ge=1, le=20)
    force: bool = False


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "store": "chromadb",
        "chroma_path": str(settings.chroma_dir),
        "onet_count": chroma_store.count_dataset("onet"),
        "esco_count": chroma_store.count_dataset("esco"),
        "version": "4.0.0",
        "resolver": "jev",
    }


@app.get("/api/taxonomies")
def get_taxonomies():
    return taxonomies_payload()


@app.get("/api/runs")
def list_crosswalk_runs():
    return {"runs": run_store.list_runs(), "defaults": run_store.default_config()}


@app.post("/api/runs")
async def start_crosswalk_run(body: RunStartBody):
    config: dict[str, Any] = {
        "model": body.model,
        "input_taxonomy": body.input_taxonomy,
        "output_taxonomy": body.output_taxonomy,
        "weights": body.weights or dict(DEFAULT_WEIGHTS),
        "threshold": body.threshold,
        "top_k": body.top_k,
    }
    try:
        run_store.validate_config(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    if not body.force:
        existing = run_store.find_reusable(config)
        if existing:
            cloned = run_store.clone_reused_run(existing["run_id"], config)
            return {"ok": True, "reused": True, "run": cloned}

    if run_pipeline_job.is_running():
        raise HTTPException(status_code=409, detail={"message": "A run is already in progress"})

    result = await run_pipeline_job.start(config)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail={"message": result.get("error")})
    return result


@app.get("/api/runs/progress")
async def runs_progress_stream():
    async def gen():
        last_ver = -1
        idle_ticks = 0
        while True:
            state = run_pipeline_job.snapshot()
            if state["version"] != last_ver:
                last_ver = state["version"]
                yield f"data: {json.dumps(state)}\n\n"
                idle_ticks = 0
            else:
                idle_ticks += 1
            if state["status"] in ("completed", "failed"):
                yield f"data: {json.dumps(state)}\n\n"
                break
            if state["status"] == "idle" and idle_ticks > 3:
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/runs/{run_id}")
def get_crosswalk_run(
    run_id: str,
    include_results: bool = Query(False),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    payload: dict[str, Any] = {"run": run}
    if include_results:
        rows = run_store.load_results(run["run_id"])
        payload["results"] = rows[offset : offset + limit]
        payload["results_total"] = len(rows)
        payload["offset"] = offset
        payload["limit"] = limit
    return payload


@app.get("/api/runs/{run_id}/progress")
async def run_progress_stream(run_id: str):
    async def gen():
        last_ver = -1
        idle_ticks = 0
        while True:
            state = run_pipeline_job.snapshot()
            persisted = run_store.get_run(run_id)
            if persisted and state.get("run_id") not in (run_id, persisted.get("run_id")):
                payload = {
                    "status": persisted.get("status"),
                    "run_id": persisted.get("run_id"),
                    "label": persisted.get("label"),
                    "phase": persisted.get("phase"),
                    "percent": persisted.get("percent") or 100.0,
                    "message": persisted.get("message") or "",
                    "version": 0,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                break
            if state["version"] != last_ver:
                last_ver = state["version"]
                yield f"data: {json.dumps(state)}\n\n"
                idle_ticks = 0
            else:
                idle_ticks += 1
            if state["status"] in ("completed", "failed"):
                yield f"data: {json.dumps(state)}\n\n"
                break
            if state["status"] == "idle" and idle_ticks > 3:
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.delete("/api/runs/{run_id}")
def delete_crosswalk_run(run_id: str):
    ok = run_store.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"ok": True}


@app.get("/api/occupations/{dataset}")
def list_or_search_occupations(
    dataset: Literal["onet", "esco"],
    q: str = Query("", description="Occupation title search"),
    limit: int = Query(20, ge=1, le=100),
):
    if chroma_store.count_dataset(dataset) == 0:
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"No {dataset} embeddings in ChromaDB — copy data/chroma_db first",
                "error_type": "EmbeddingsMissing",
                "step": "search",
                "context": {"dataset": dataset},
            },
        )
    try:
        return {
            "dataset": dataset,
            "query": q,
            "results": search_occupations(dataset, q, limit),
        }
    except Exception as exc:  # noqa: BLE001
        import traceback as tb

        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Occupation search failed: {exc}",
                "error_type": type(exc).__name__,
                "step": "search",
                "context": {"dataset": dataset, "query": q},
                "cause": str(exc),
                "traceback": "".join(tb.format_exception(type(exc), exc, exc.__traceback__)),
            },
        ) from exc


@app.post("/api/crosswalk")
async def crosswalk_match(body: CrosswalkBody):
    """Live hybrid match: Top-20 → Top-10 (embedding if ≥ threshold, else Jev)."""
    occupation_id = unquote(body.occupation_id).strip()
    weights = body.weights or get_active_weights()
    try:
        direction = Direction(body.direction)
        return await live_crosswalk(
            occupation_id=occupation_id,
            direction=direction,
            variant=(body.model or "P").strip(),
            weights=weights,
            threshold=float(body.threshold),
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "message": str(exc),
                "error_type": "LookupError",
                "step": "load_source",
                "context": {
                    "direction": body.direction,
                    "occupation_id": occupation_id,
                },
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        import traceback as tb

        raise HTTPException(
            status_code=400,
            detail={
                "message": str(exc),
                "error_type": type(exc).__name__,
                "traceback": "".join(tb.format_exception(type(exc), exc, exc.__traceback__)),
            },
        ) from exc


@app.get("/api/models")
def list_models():
    return {"models": ["P"], "default": "P"}
