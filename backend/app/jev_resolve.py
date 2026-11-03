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
    """Text summary for Choice criteria (enriched ESCO when present)."""
    title = cand.get("preferred_title") or cand.get("title") or ""
    alt = cand.get("alternate_title") or cand.get("alternate_titles") or ""
    desc = cand.get("description") or ""
    parts = [f"Preferred Title: {title}"]
    if alt:
        parts.append(f"Alternate Titles: {alt}")
    if desc:
        parts.append(f"Description: {desc}")
    for key, label in (
        ("essential_skills", "Essential Skills"),
        ("optional_skills", "Optional Skills"),
        ("essential_knowledge", "Essential Knowledge"),
        ("optional_knowledge", "Optional Knowledge"),
        ("skills", "Skills"),
        ("knowledge", "Knowledge"),
    ):
        val = cand.get(key)
        if val:
            parts.append(f"{label}: {val}")
    return " | ".join(parts)


def _onet_input_state(record: dict[str, Any]) -> dict[str, Any]:
    inp = record.get("input") or {}
    return {
        "occupation_id": inp.get("occupation_id") or record.get("source_id") or "",
        "title": inp.get("title") or inp.get("preferred_title") or record.get("source_title") or "",
        "description": inp.get("description") or inp.get("job_description") or "",
        "titles": inp.get("titles") or inp.get("alternate_title") or "",
        "tasks": inp.get("tasks") or "",
        "abilities": inp.get("abilities") or "",
        "work_activities": inp.get("work_activities") or "",
        "software_skills": inp.get("software_skills") or "",
        "essential_skills": inp.get("essential_skills") or inp.get("skills") or "",
        "transferable_skills": inp.get("transferable_skills") or "",
        "knowledge": inp.get("knowledge") or "",
    }


def _esco_candidate_state(cand: dict[str, Any]) -> dict[str, Any]:
    return {
        "occupation_id": cand.get("occupation_id") or cand.get("target_id") or "",
        "preferred_title": cand.get("preferred_title") or cand.get("title") or "",
        "alternate_titles": cand.get("alternate_titles") or cand.get("alternate_title") or "",
        "description": cand.get("description") or "",
        "essential_skills": cand.get("essential_skills") or "",
        "optional_skills": cand.get("optional_skills") or "",
        "essential_knowledge": cand.get("essential_knowledge") or "",
        "optional_knowledge": cand.get("optional_knowledge") or "",
    }


def usage_from_body(body: dict[str, Any]) -> tuple[int, int]:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return 0, 0
    try:
        inp = int(usage.get("input_tokens") or 0)
    except (TypeError, ValueError):
        inp = 0
    try:
        out = int(usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        out = 0
    return inp, out


def aggregate_jev_token_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum token usage across JEV-sent rows for run metrics."""
    sent = [r for r in results if r.get("sent_to_jev")]
    if not sent:
        return {
            "jev_input_tokens_total": 0,
            "jev_output_tokens_total": 0,
            "jev_total_tokens": 0,
            "jev_api_calls_total": 0,
            "jev_avg_input_tokens_per_gated_source": 0.0,
            "jev_avg_total_tokens_per_gated_source": 0.0,
        }
    inp = sum(int(r.get("jev_input_tokens") or 0) for r in sent)
    out = sum(int(r.get("jev_output_tokens") or 0) for r in sent)
    calls = sum(int(r.get("jev_api_calls") or 0) for r in sent)
    n = len(sent)
    total = inp + out
    return {
        "jev_input_tokens_total": inp,
        "jev_output_tokens_total": out,
        "jev_total_tokens": total,
        "jev_api_calls_total": calls,
        "jev_avg_input_tokens_per_gated_source": round(inp / n, 2),
        "jev_avg_total_tokens_per_gated_source": round(total / n, 2),
    }


def _apply_token_fields(
    row: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
    api_calls: int,
) -> dict[str, Any]:
    row["jev_input_tokens"] = input_tokens
    row["jev_output_tokens"] = output_tokens
    row["jev_total_tokens"] = input_tokens + output_tokens
    row["jev_api_calls"] = api_calls
    return row


def build_jev_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build TypeSafe Jev decide payload: Choice over Top-N candidates (≤20)."""
    input_state = _onet_input_state(record)
    candidates = list(record.get("candidates") or [])[:MAX_CANDIDATES]
    labels = candidate_labels(len(candidates))
    cand_state: dict[str, Any] = {}
    criteria: dict[str, str] = {}
    for i, label in enumerate(labels):
        cand = candidates[i] if i < len(candidates) else {}
        state = _esco_candidate_state(cand)
        state["target_id"] = cand.get("target_id") or cand.get("occupation_id") or ""
        cand_state[label] = state
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
                    "Using only the supplied occupation fields for the input and each "
                    "candidate (titles, descriptions, skills, knowledge, tasks, abilities, "
                    "work activities, and related lists), which candidate is the closest "
                    "overall match? Do not use web search or any external sources. "
                    f"Pick exactly one of: {label_list}."
                ),
                "criteria": criteria,
            }
        },
    }


def build_jev_noul_payload(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """One Noul question for a single (input, candidate) pair."""
    return {
        "state": {
            "input": _onet_input_state(record),
            "candidate": _esco_candidate_state(candidate),
        },
        "model": "jev-latest",
        "questions": {
            "match": {
                "type": "noul",
                "instructions": (
                    "Using only the supplied occupation fields, is this candidate an overall "
                    "equivalent match for the input occupation for crosswalk purposes? "
                    "Do not use web search or external sources."
                ),
                "criteria": {
                    "true": (
                        "Same or equivalent occupation based on titles, description, skills, "
                        "knowledge, tasks, abilities, and work activities when provided."
                    ),
                    "false": (
                        "Different role, only tangential overlap, or insufficient evidence "
                        "of equivalence."
                    ),
                },
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


def parse_jev_noul(body: dict[str, Any]) -> float | None:
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        return None
    match = answers.get("match")
    if not isinstance(match, dict):
        return None
    val = match.get("noul")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


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


def save_noul_candidate_io(
    folder: Path,
    label: str,
    request_payload: dict[str, Any],
    response_body: dict[str, Any] | str,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"cand_{label}_Input.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if isinstance(response_body, str):
        (folder / f"cand_{label}_Output.json").write_text(response_body, encoding="utf-8")
    else:
        (folder / f"cand_{label}_Output.json").write_text(
            json.dumps(response_body, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def save_noul_summary(folder: Path, summary: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
        "jev_input_tokens": 0,
        "jev_output_tokens": 0,
        "jev_total_tokens": 0,
        "jev_api_calls": 0,
    }


def _record_title(record: dict[str, Any], row_idx: int) -> str:
    return str(
        (record.get("input") or {}).get("title")
        or record.get("source_title")
        or record.get("source_id")
        or f"row_{row_idx}"
    )


def _build_success_result(
    record: dict[str, Any],
    *,
    ranked_labels: list[str],
    ranked_ids: list[str],
    choice: str,
    jev_target: Any,
    jev_title: str,
    error: str | None,
    jev_confidence: Any,
    jev_raw: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    api_calls: int,
) -> dict[str, Any]:
    out = {
        **record,
        "sent_to_jev": True,
        "sent_to_llm": True,
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
        "jev_confidence": jev_confidence,
        "jev_raw": jev_raw,
    }
    return _apply_token_fields(
        out,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=api_calls,
    )


async def resolve_one_choice(
    record: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    jev: JevClient,
    io_dir: Path,
    row_idx: int,
    request_sem: asyncio.Semaphore | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    payload = build_jev_payload(record)
    candidates = list(record.get("candidates") or [])[:MAX_CANDIDATES]
    labels = candidate_labels(len(candidates))
    title = _record_title(record, row_idx)
    last_body: dict[str, Any] = {}
    choice = ""
    ranked_labels: list[str] = []
    error: str | None = None
    total_in, total_out = 0, 0
    for _attempt in range(1, max_retries + 1):
        if request_sem:
            async with request_sem:
                body, _latency = await jev.decide(client, payload)
        else:
            body, _latency = await jev.decide(client, payload)
        last_body = body
        inp, out = usage_from_body(body)
        total_in += inp
        total_out += out
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

    save_io_pair(io_dir, row_idx, title, payload, last_body)

    if error or not ranked_labels:
        fb = _fallback_resolved(record, error or "unparsable_choice")
        return _apply_token_fields(fb, input_tokens=total_in, output_tokens=total_out, api_calls=1)

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
        fb = _fallback_resolved(record, "empty_ranking")
        return _apply_token_fields(fb, input_tokens=total_in, output_tokens=total_out, api_calls=1)

    top_cand = label_to_cand.get(ranked_labels[0]) or {}
    if choice in label_to_cand:
        chosen = label_to_cand[choice]
        top_id = chosen.get("target_id") or chosen.get("occupation_id")
        if top_id and ranked_ids and ranked_ids[0] != top_id:
            ranked_ids = [str(top_id)] + [t for t in ranked_ids if t != top_id]
            ranked_ids = ranked_ids[:RANKED_TOP_N]
            top_cand = chosen

    jev_target = top_cand.get("target_id") or top_cand.get("occupation_id") or ranked_ids[0]
    jev_title = top_cand.get("preferred_title") or ""

    return _build_success_result(
        record,
        ranked_labels=ranked_labels,
        ranked_ids=ranked_ids,
        choice=choice,
        jev_target=jev_target,
        jev_title=jev_title,
        error=error,
        jev_confidence=(
            (last_body.get("answers") or {}).get("match", {}).get("confidence")
            if isinstance(last_body.get("answers"), dict)
            else None
        ),
        jev_raw={
            k: last_body.get(k)
            for k in ("answers", "usage", "status_code", "_error")
            if k in last_body
        },
        input_tokens=total_in,
        output_tokens=total_out,
        api_calls=1,
    )


async def resolve_one_noul(
    record: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    jev: JevClient,
    io_dir: Path,
    row_idx: int,
    request_sem: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    candidates = list(record.get("candidates") or [])[:MAX_CANDIDATES]
    labels = candidate_labels(len(candidates))
    title = _record_title(record, row_idx)
    folder = io_dir / _safe_folder_name(title, row_idx)
    folder.mkdir(parents=True, exist_ok=True)

    scored: list[tuple[str, float, dict[str, Any]]] = []
    total_in, total_out, api_calls = 0, 0, 0
    per_candidate: dict[str, Any] = {}

    for i, label in enumerate(labels):
        cand = candidates[i]
        payload = build_jev_noul_payload(record, cand)
        body: dict[str, Any] = {}
        for _attempt in range(3):
            if request_sem:
                async with request_sem:
                    body, _ = await jev.decide(client, payload)
            else:
                body, _ = await jev.decide(client, payload)
            if not body.get("_error"):
                break
            await asyncio.sleep(0.5)
        save_noul_candidate_io(folder, label, payload, body)
        inp, out = usage_from_body(body)
        total_in += inp
        total_out += out
        api_calls += 1
        noul = parse_jev_noul(body)
        score = noul if noul is not None else -1.0
        scored.append((label, score, body))
        per_candidate[label] = {
            "noul": noul,
            "target_id": cand.get("target_id") or cand.get("occupation_id"),
            "usage": body.get("usage"),
            "error": body.get("_error"),
        }

    scored.sort(key=lambda x: (-x[1], labels.index(x[0])))
    ranked_labels = [lab for lab, _, _ in scored if lab in labels]
    if not ranked_labels:
        ranked_labels = list(labels)

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
        fb = _fallback_resolved(record, "empty_noul_ranking")
        save_noul_summary(
            folder,
            {
                "ranked_target_ids": fb.get("jev_ranked_target_ids"),
                "per_candidate": per_candidate,
                "jev_input_tokens": total_in,
                "jev_output_tokens": total_out,
                "jev_api_calls": api_calls,
            },
        )
        return _apply_token_fields(fb, input_tokens=total_in, output_tokens=total_out, api_calls=api_calls)

    top_cand = label_to_cand.get(ranked_labels[0]) or {}
    top_noul = scored[0][1] if scored else None
    jev_target = top_cand.get("target_id") or top_cand.get("occupation_id") or ranked_ids[0]
    jev_title = top_cand.get("preferred_title") or ""

    save_noul_summary(
        folder,
        {
            "ranked_labels": ranked_labels[:RANKED_TOP_N],
            "ranked_target_ids": ranked_ids,
            "per_candidate": per_candidate,
            "jev_input_tokens": total_in,
            "jev_output_tokens": total_out,
            "jev_api_calls": api_calls,
        },
    )

    return _build_success_result(
        record,
        ranked_labels=ranked_labels,
        ranked_ids=ranked_ids,
        choice=ranked_labels[0],
        jev_target=jev_target,
        jev_title=jev_title,
        error=None,
        jev_confidence=top_noul if top_noul is not None and top_noul >= 0 else None,
        jev_raw={"mode": "noul", "per_candidate": per_candidate},
        input_tokens=total_in,
        output_tokens=total_out,
        api_calls=api_calls,
    )


async def resolve_one(
    record: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    jev: JevClient,
    io_dir: Path,
    row_idx: int,
    jev_rank_mode: str = "choice",
    request_sem: asyncio.Semaphore | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    mode = (jev_rank_mode or "choice").strip().lower()
    if mode == "noul":
        return await resolve_one_noul(
            record,
            client=client,
            jev=jev,
            io_dir=io_dir,
            row_idx=row_idx,
            request_sem=request_sem,
        )
    return await resolve_one_choice(
        record,
        client=client,
        jev=jev,
        io_dir=io_dir,
        row_idx=row_idx,
        request_sem=request_sem,
        max_retries=max_retries,
    )


async def resolve_batch(
    records: list[dict[str, Any]],
    *,
    run_seq: int,
    concurrency: int | None = None,
    jev_rank_mode: str = "choice",
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    """Resolve low-confidence records via Jev in parallel; write I/O under jev_io_pairs/run_N."""
    if not records:
        return []
    settings = get_settings()
    conc = max(1, int(concurrency if concurrency is not None else (settings.jev_concurrency or 16)))
    jev = JevClient()
    io_dir = io_dir_for_run(run_seq)
    record_sem = asyncio.Semaphore(conc)
    request_sem = asyncio.Semaphore(conc)
    progress_lock = asyncio.Lock()
    results: list[dict[str, Any] | None] = [None] * len(records)
    done = 0
    mode = (jev_rank_mode or "choice").strip().lower()

    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    timeout = httpx.Timeout(120.0, connect=30.0)

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        verify=ssl_verify(),
    ) as client:

        async def _one(i: int, rec: dict[str, Any]) -> None:
            nonlocal done
            async with record_sem:
                try:
                    resolved = await resolve_one(
                        rec,
                        client=client,
                        jev=jev,
                        io_dir=io_dir,
                        row_idx=i + 1,
                        jev_rank_mode=mode,
                        request_sem=request_sem,
                    )
                except Exception as exc:  # noqa: BLE001 — keep sibling tasks running
                    resolved = _fallback_resolved(rec, _exc_message(exc))
                    try:
                        title = _record_title(rec, i + 1)
                        if mode == "noul":
                            folder = io_dir / _safe_folder_name(title, i + 1)
                            save_noul_summary(
                                folder,
                                {"_error": resolved["jev_error"], "mode": "noul"},
                            )
                        else:
                            save_io_pair(
                                io_dir,
                                i + 1,
                                title,
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
