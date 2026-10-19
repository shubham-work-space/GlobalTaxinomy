"""Jev-assisted resolution for low-confidence crosswalk candidates (Top-20 Choice ranking).

Uses the documented TypeSafe System One API:
  POST https://api.typesafe.ai/v1/systemone
  https://docs.typesafe.ai/api

Choice returns a single pick plus per-option probabilities; we sort probabilities
descending to produce an ordered ranking (top 10 used for Accuracy@K).
"""

from __future__ import annotations

import asyncio
import json
import re
import ssl
import time
from pathlib import Path
from typing import Any

import httpx

from .config import ROOT, get_settings

# A–T covers up to 20 candidates (TypeSafe Choice allows ≤255 options).
MAX_CANDIDATES = 20
RANKED_TOP_N = 10
LABELS = tuple(chr(ord("A") + i) for i in range(MAX_CANDIDATES))

JEV_IO_ROOT = ROOT / "output" / "jev_io_pairs"
DEFAULT_TYPESAFE_BASE = "https://api.typesafe.ai"
SYSTEMONE_PATH = "/v1/systemone"
# Only these hosts may receive the Bearer API key (SSRF / key-exfil guard).
ALLOWED_TYPESAFE_HOSTS = frozenset({"api.typesafe.ai"})


def ssl_verify() -> ssl.SSLContext | bool:
    """Use the OS trust store when available (fixes corporate/proxy SSL on Windows).

    Keeps certificate verification ON (never verify=False / CERT_NONE).
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return True


def candidate_labels(n: int) -> tuple[str, ...]:
    """Return the first n Choice labels (A, B, …)."""
    n = max(0, min(int(n), MAX_CANDIDATES))
    return LABELS[:n]


def _safe_folder_name(title: str, row_idx: int) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", (title or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "occupation"
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip(" .")
    return f"{row_idx:03d}_{cleaned}"


def _candidate_summary(cand: dict[str, Any]) -> str:
    title = cand.get("preferred_title") or cand.get("title") or ""
    alt = cand.get("alternate_title") or ""
    desc = cand.get("description") or ""
    skills = cand.get("skills") or ""
    knowledge = cand.get("knowledge") or ""
    parts = [f"Job Title: {title}"]
    if alt:
        parts.append(f"Alternate Title: {alt}")
    if desc:
        parts.append(f"Job Description: {desc}")
    if skills:
        parts.append(f"Skills: {skills}")
    if knowledge:
        parts.append(f"Knowledge: {knowledge}")
    return " | ".join(parts)


def build_jev_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build TypeSafe Jev decide payload: Choice over Top-N candidates (≤20)."""
    inp = record.get("input") or {}
    input_state = {
        "job_title": inp.get("title") or inp.get("preferred_title") or record.get("source_title") or "",
        "alternate_title": inp.get("alternate_title") or "",
        "job_description": inp.get("description") or "",
        "skills": inp.get("skills") or "",
        "knowledge": inp.get("knowledge") or "",
    }
    candidates = list(record.get("candidates") or [])[:MAX_CANDIDATES]
    labels = candidate_labels(len(candidates))
    cand_state: dict[str, Any] = {}
    criteria: dict[str, str] = {}
    for i, label in enumerate(labels):
        cand = candidates[i] if i < len(candidates) else {}
        cand_state[label] = {
            "job_title": cand.get("preferred_title") or cand.get("title") or "",
            "alternate_title": cand.get("alternate_title") or "",
            "job_description": cand.get("description") or "",
            "skills": cand.get("skills") or "",
            "knowledge": cand.get("knowledge") or "",
            "target_id": cand.get("target_id") or cand.get("occupation_id") or "",
        }
        criteria[label] = _candidate_summary(cand) or f"Candidate {label}"

    label_list = ", ".join(labels) if labels else "(none)"
    return {
        "state": {
            "input": input_state,
            "candidates": cand_state,
        },
        "model": "jev-latest",
        "questions": {
            "match": {
                "type": "choice",
                "instructions": (
                    "Using only the supplied Job Title, Alternate Title, Job Description, "
                    "Skills, and Knowledge for the input occupation and each candidate, "
                    "which candidate is the closest overall match? "
                    "Do not use web search or any external sources. "
                    f"Pick exactly one of: {label_list}."
                ),
                "criteria": criteria,
            }
        },
    }


def parse_jev_choice(body: dict[str, Any], labels: tuple[str, ...] | None = None) -> str:
    """Extract a Choice label from a Jev decide response."""
    allowed = labels or LABELS
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        return ""
    match = answers.get("match")
    if not isinstance(match, dict):
        return ""
    choice = match.get("choice")
    if isinstance(choice, str):
        up = choice.strip().upper()
        if up in allowed:
            return up
        # Tolerate "Job A" / "A." style
        for label in allowed:
            if re.search(rf"\b{label}\b", up):
                return label
    return ""


def parse_jev_ranking(
    body: dict[str, Any],
    labels: tuple[str, ...],
) -> list[str]:
    """Return labels ordered by descending Choice probability.

    Falls back to [choice] then remaining labels in original order when
    probabilities are missing or incomplete.
    """
    if not labels:
        return []
    answers = body.get("answers") if isinstance(body, dict) else None
    match = answers.get("match") if isinstance(answers, dict) else None
    probs_raw = match.get("probabilities") if isinstance(match, dict) else None

    ranked: list[str] = []
    if isinstance(probs_raw, dict) and probs_raw:
        scored: list[tuple[str, float]] = []
        for label in labels:
            val = probs_raw.get(label)
            if val is None:
                # Tolerate lowercase keys
                val = probs_raw.get(label.lower())
            try:
                score = float(val) if val is not None else -1.0
            except (TypeError, ValueError):
                score = -1.0
            scored.append((label, score))
        scored.sort(key=lambda x: (-x[1], labels.index(x[0])))
        ranked = [lab for lab, _ in scored]
    else:
        choice = parse_jev_choice(body, labels)
        if choice and choice in labels:
            ranked = [choice] + [lab for lab in labels if lab != choice]
        else:
            ranked = list(labels)
    return ranked


def _embedding_ranked_ids(record: dict[str, Any], *, limit: int = RANKED_TOP_N) -> list[str]:
    """Ordered target ids from embedding candidates (fallback / non-JEV path)."""
    out: list[str] = []
    for cand in record.get("candidates") or []:
        tid = cand.get("target_id") or cand.get("occupation_id")
        if tid and tid not in out:
            out.append(str(tid))
        if len(out) >= limit:
            break
    return out


def save_io_pair(
    io_dir: Path,
    row_idx: int,
    title: str,
    request_payload: dict[str, Any],
    response_body: dict[str, Any] | str,
) -> Path:
    folder = io_dir / _safe_folder_name(title, row_idx)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Input.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if isinstance(response_body, str):
        (folder / "Output.json").write_text(response_body, encoding="utf-8")
    else:
        (folder / "Output.json").write_text(
            json.dumps(response_body, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return folder


def io_dir_for_run(run_seq: int) -> Path:
    path = JEV_IO_ROOT / f"run_{run_seq}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _exc_message(exc: BaseException) -> str:
    """Prefer a useful message; many httpx/asyncio errors stringify to ''."""
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def systemone_endpoint(base_url: str | None = None) -> str:
    """Resolve System One URL; pin HTTPS + allowlisted host (docs.typesafe.ai).

    Rejects http:// and unknown hosts so the Bearer key cannot be sent
    in cleartext or to an attacker-controlled URL via misconfigured env.
    """
    from urllib.parse import urlparse

    raw = (base_url or DEFAULT_TYPESAFE_BASE).strip() or DEFAULT_TYPESAFE_BASE
    # Legacy host / path → documented endpoint
    if "jevtypesafeai.com" in raw or raw.rstrip("/").endswith("/api/v1/decide"):
        raw = DEFAULT_TYPESAFE_BASE

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")

    if scheme != "https":
        raise RuntimeError(
            f"JEV_BASE_URL must use https (got {scheme!r}). "
            f"Refusing to send API key over cleartext."
        )
    if host not in ALLOWED_TYPESAFE_HOSTS:
        raise RuntimeError(
            f"JEV_BASE_URL host {host!r} is not allowlisted. "
            f"Allowed: {sorted(ALLOWED_TYPESAFE_HOSTS)}. "
            f"See https://docs.typesafe.ai/api"
        )
    if path in ("", "/"):
        path = SYSTEMONE_PATH
    elif not path.endswith(SYSTEMONE_PATH):
        path = SYSTEMONE_PATH
    return f"https://{host}{path}"


class JevClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = (settings.jev_base_url or DEFAULT_TYPESAFE_BASE).rstrip("/")
        self.api_key = (settings.jev_api_key or "").strip()
        if not self.api_key:
            raise RuntimeError("Jev credentials missing. Set JEV_API_KEY.")
        self.endpoint = systemone_endpoint(self.base_url)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def decide(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        *,
        timeout_seconds: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        try:
            response = await client.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            latency = time.perf_counter() - t0
            return {
                "_error": f"timeout: {_exc_message(exc)}",
                "status_code": 0,
                "_latency_s": round(latency, 4),
            }, latency
        except httpx.HTTPError as exc:
            latency = time.perf_counter() - t0
            return {
                "_error": f"transport: {_exc_message(exc)}",
                "status_code": 0,
                "_latency_s": round(latency, 4),
            }, latency

        latency = time.perf_counter() - t0
        try:
            body = response.json() if response.content else {}
        except Exception:
            body = {"raw_text": response.text, "status_code": response.status_code}
        if not isinstance(body, dict):
            body = {"raw": body, "status_code": response.status_code}
        body.setdefault("status_code", response.status_code)
        if response.status_code >= 400:
            body["_error"] = f"HTTP {response.status_code}"
        body["_latency_s"] = round(latency, 4)
        return body, latency


def _fallback_resolved(record: dict[str, Any], error: str) -> dict[str, Any]:
    """Keep embedding order when a single Jev call fails hard; do not abort the batch."""
    ranked_ids = _embedding_ranked_ids(record)
    top = ranked_ids[0] if ranked_ids else record.get("top1_target_id")
    title_by_id = {
        (c.get("target_id") or c.get("occupation_id")): (
            c.get("preferred_title") or ""
        )
        for c in (record.get("candidates") or [])
    }
    return {
        **record,
        "sent_to_jev": True,
        "sent_to_llm": True,
        "jev_match_label": "",
        "llm_match_label": "",
        "llm_primary_label": "",
        "jev_ranked_labels": [],
        "jev_ranked_target_ids": ranked_ids,
        "jev_target_id": None,
        "llm_target_id": None,
        "jev_preferred_title": "",
        "llm_preferred_title": "",
        "final_target_id": top,
        "final_preferred_title": title_by_id.get(top) or record.get("top1_preferred_title"),
        "jev_error": error,
        "llm_error": error,
        "jev_confidence": None,
        "jev_raw": {"_error": error},
    }


async def resolve_one(
    record: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    jev: JevClient,
    io_dir: Path,
    row_idx: int,
    max_retries: int = 3,
) -> dict[str, Any]:
    payload = build_jev_payload(record)
    candidates = list(record.get("candidates") or [])[:MAX_CANDIDATES]
    labels = candidate_labels(len(candidates))
    title = (
        (record.get("input") or {}).get("title")
        or record.get("source_title")
        or record.get("source_id")
        or f"row_{row_idx}"
    )
    last_body: dict[str, Any] = {}
    choice = ""
    ranked_labels: list[str] = []
    error: str | None = None
    for _attempt in range(1, max_retries + 1):
        body, _latency = await jev.decide(client, payload)
        last_body = body
        if body.get("_error"):
            error = str(body["_error"])
            await asyncio.sleep(0.5)
            continue
        choice = parse_jev_choice(body, labels)
        ranked_labels = parse_jev_ranking(body, labels)
        if choice in labels and ranked_labels:
            error = None
            break
        error = "unparsable_choice"
        await asyncio.sleep(0.5)

    save_io_pair(io_dir, row_idx, str(title), payload, last_body)

    if error or not ranked_labels:
        return _fallback_resolved(record, error or "unparsable_choice")

    label_to_cand = {labels[i]: candidates[i] for i in range(len(labels))}
    ranked_ids: list[str] = []
    for lab in ranked_labels:
        cand = label_to_cand.get(lab) or {}
        tid = cand.get("target_id") or cand.get("occupation_id")
        if tid and tid not in ranked_ids:
            ranked_ids.append(str(tid))
        if len(ranked_ids) >= RANKED_TOP_N:
            break

    if not ranked_ids:
        return _fallback_resolved(record, "empty_ranking")

    top_cand = label_to_cand.get(ranked_labels[0]) or {}
    # Prefer probability-rank #1; fall back to explicit choice if needed
    if choice in label_to_cand:
        chosen = label_to_cand[choice]
        # Ensure choice is treated as rank-1 when probabilities disagree oddly
        top_id = chosen.get("target_id") or chosen.get("occupation_id")
        if top_id and ranked_ids and ranked_ids[0] != top_id:
            ranked_ids = [str(top_id)] + [t for t in ranked_ids if t != top_id]
            ranked_ids = ranked_ids[:RANKED_TOP_N]
            top_cand = chosen

    jev_target = top_cand.get("target_id") or top_cand.get("occupation_id") or ranked_ids[0]
    jev_title = top_cand.get("preferred_title") or ""

    return {
        **record,
        "sent_to_jev": True,
        "sent_to_llm": True,  # compat alias for shared accuracy helpers
        "jev_match_label": choice or (ranked_labels[0] if ranked_labels else ""),
        "llm_match_label": choice or (ranked_labels[0] if ranked_labels else ""),
        "llm_primary_label": choice or (ranked_labels[0] if ranked_labels else ""),
        "jev_ranked_labels": ranked_labels[:RANKED_TOP_N],
        "jev_ranked_target_ids": ranked_ids,
        "jev_target_id": jev_target,
        "llm_target_id": jev_target,
        "jev_preferred_title": jev_title,
        "llm_preferred_title": jev_title,
        "final_target_id": jev_target or record.get("top1_target_id"),
        "final_preferred_title": jev_title or record.get("top1_preferred_title"),
        "jev_error": error,
        "llm_error": error,
        "jev_confidence": (
            (last_body.get("answers") or {}).get("match", {}).get("confidence")
            if isinstance(last_body.get("answers"), dict)
            else None
        ),
        "jev_raw": {
            k: last_body.get(k)
            for k in ("answers", "usage", "status_code", "_error")
            if k in last_body
        },
    }


async def resolve_batch(
    records: list[dict[str, Any]],
    *,
    run_seq: int,
    concurrency: int | None = None,
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    """Resolve low-confidence records via Jev in parallel; write I/O under jev_io_pairs/run_N."""
    if not records:
        return []
    settings = get_settings()
    conc = max(1, int(concurrency if concurrency is not None else (settings.jev_concurrency or 16)))
    jev = JevClient()
    io_dir = io_dir_for_run(run_seq)
    sem = asyncio.Semaphore(conc)
    progress_lock = asyncio.Lock()
    results: list[dict[str, Any] | None] = [None] * len(records)
    done = 0

    # Allow true parallel in-flight POSTs up to concurrency.
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    timeout = httpx.Timeout(120.0, connect=30.0)

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        verify=ssl_verify(),
    ) as client:

        async def _one(i: int, rec: dict[str, Any]) -> None:
            nonlocal done
            async with sem:
                try:
                    resolved = await resolve_one(
                        rec,
                        client=client,
                        jev=jev,
                        io_dir=io_dir,
                        row_idx=i + 1,
                    )
                except Exception as exc:  # noqa: BLE001 — keep sibling tasks running
                    resolved = _fallback_resolved(rec, _exc_message(exc))
                    try:
                        save_io_pair(
                            io_dir,
                            i + 1,
                            str(
                                (rec.get("input") or {}).get("title")
                                or rec.get("source_title")
                                or rec.get("source_id")
                                or f"row_{i + 1}"
                            ),
                            build_jev_payload(rec),
                            {"_error": resolved["jev_error"]},
                        )
                    except Exception:
                        pass
                results[i] = resolved
                async with progress_lock:
                    done += 1
                    current = done
                if progress:
                    await progress(
                        {
                            "phase": "run_jev_evaluation",
                            "message": f"Jev {current}/{len(records)} (parallel×{conc})",
                            "done": current,
                            "total": len(records),
                            "percent": round(100.0 * current / max(len(records), 1), 2),
                        }
                    )

        # All tasks scheduled together; semaphore caps in-flight Jev calls.
        await asyncio.gather(*[_one(i, r) for i, r in enumerate(records)])

    return [r for r in results if r is not None]
