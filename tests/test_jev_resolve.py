"""Jev payload / parse / resolve_batch flow tests (mocked HTTP)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from backend.app.jev_resolve import (
    LABELS,
    MAX_CANDIDATES,
    build_jev_payload,
    candidate_labels,
    parse_jev_choice,
    parse_jev_ranking,
    resolve_batch,
    resolve_one,
    systemone_endpoint,
    JevClient,
)


def _make_candidates(n: int = 4):
    return [
        {
            "rank": i + 1,
            "target_id": f"cand-{chr(ord('a') + i)}",
            "preferred_title": f"Job {chr(ord('A') + i)}",
            "alternate_title": f"Alt {chr(ord('A') + i)}",
            "description": f"{chr(ord('A') + i)} desc",
            "skills": f"{chr(ord('A') + i)} skills",
            "knowledge": f"{chr(ord('A') + i)} knowledge",
        }
        for i in range(n)
    ]


def _sample_record(**overrides):
    cands = overrides.pop("candidates", None)
    if cands is None:
        cands = _make_candidates(4)
    base = {
        "source_id": "11-1011.00",
        "source_title": "Chief Executives",
        "input": {
            "title": "Chief Executives",
            "alternate_title": "CEO; Managing Director",
            "description": "Determine and formulate policies",
            "skills": "Leadership",
            "knowledge": "Administration",
        },
        "top1_target_id": cands[0]["target_id"],
        "top1_preferred_title": cands[0]["preferred_title"],
        "top1_similarity": 0.45,
        "candidates": cands,
        "sent_to_jev": True,
        "final_target_id": cands[0]["target_id"],
        "final_preferred_title": cands[0]["preferred_title"],
    }
    base.update(overrides)
    return base


def test_candidate_labels_caps_at_20():
    assert candidate_labels(4) == ("A", "B", "C", "D")
    assert len(candidate_labels(20)) == 20
    assert candidate_labels(20)[-1] == "T"
    assert len(LABELS) == MAX_CANDIDATES == 20


def test_build_jev_payload_matches_candidate_count():
    payload = build_jev_payload(_sample_record())
    assert set(payload["state"]["candidates"].keys()) == {"A", "B", "C", "D"}
    assert set(payload["questions"]["match"]["criteria"].keys()) == {"A", "B", "C", "D"}
    assert payload["questions"]["match"]["type"] == "choice"
    inp = payload["state"]["input"]
    assert inp["job_title"] == "Chief Executives"
    assert inp["alternate_title"] == "CEO; Managing Director"
    assert "job_description" in inp
    instructions = payload["questions"]["match"]["instructions"]
    assert "web search" in instructions.lower() or "external" in instructions.lower()
    assert "O*NET" not in instructions
    assert "ESCO" not in instructions


def test_build_jev_payload_top20():
    payload = build_jev_payload(_sample_record(candidates=_make_candidates(20)))
    assert len(payload["state"]["candidates"]) == 20
    assert set(payload["state"]["candidates"].keys()) == set(LABELS)


def test_parse_jev_choice():
    labels = ("A", "B", "C", "D")
    assert parse_jev_choice({"answers": {"match": {"type": "choice", "choice": "C"}}}, labels) == "C"
    assert parse_jev_choice({"answers": {"match": {"choice": "b"}}}, labels) == "B"
    assert parse_jev_choice({"answers": {}}, labels) == ""
    assert parse_jev_choice({}, labels) == ""


def test_parse_jev_ranking_by_probabilities():
    labels = ("A", "B", "C", "D")
    body = {
        "answers": {
            "match": {
                "type": "choice",
                "choice": "C",
                "probabilities": {"A": 0.05, "B": 0.15, "C": 0.6, "D": 0.2},
            }
        }
    }
    assert parse_jev_ranking(body, labels) == ["C", "D", "B", "A"]


def test_parse_jev_ranking_fallback_without_probs():
    labels = ("A", "B", "C", "D")
    body = {"answers": {"match": {"type": "choice", "choice": "B"}}}
    assert parse_jev_ranking(body, labels) == ["B", "A", "C", "D"]


def test_systemone_endpoint_pins_https_allowlist():
    assert systemone_endpoint("https://api.typesafe.ai") == "https://api.typesafe.ai/v1/systemone"
    assert systemone_endpoint("https://api.typesafe.ai/v1/systemone") == (
        "https://api.typesafe.ai/v1/systemone"
    )
    # Legacy host is rewritten, not used as-is
    assert systemone_endpoint("https://jevtypesafeai.com") == (
        "https://api.typesafe.ai/v1/systemone"
    )
    with pytest.raises(RuntimeError, match="https"):
        systemone_endpoint("http://api.typesafe.ai")
    with pytest.raises(RuntimeError, match="allowlisted"):
        systemone_endpoint("https://evil.example")


@pytest.mark.asyncio
@respx.mock
async def test_resolve_one_promotes_choice_c(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://api.typesafe.ai")
    from backend.app import config as cfg

    cfg.get_settings.cache_clear()

    route = respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "answers": {
                    "match": {
                        "type": "choice",
                        "choice": "C",
                        "confidence": 0.8,
                        "probabilities": {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1},
                    }
                }
            },
        )
    )

    # Write I/O under tmp
    monkeypatch.setattr(
        "backend.app.jev_resolve.JEV_IO_ROOT",
        tmp_path / "jev_io_pairs",
    )

    client = httpx.AsyncClient()
    jev = JevClient()
    io_dir = tmp_path / "run_1"
    io_dir.mkdir()
    out = await resolve_one(
        _sample_record(),
        client=client,
        jev=jev,
        io_dir=io_dir,
        row_idx=1,
    )
    await client.aclose()

    assert route.called
    assert out["jev_match_label"] == "C"
    assert out["final_target_id"] == "cand-c"
    assert out["jev_target_id"] == "cand-c"
    assert out["jev_ranked_target_ids"] == ["cand-c", "cand-a", "cand-b", "cand-d"]
    assert out["jev_error"] is None
    assert list(io_dir.iterdir())  # wrote audit folder


@pytest.mark.asyncio
@respx.mock
async def test_resolve_one_ranks_by_probability_order(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://api.typesafe.ai")
    from backend.app import config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setattr(
        "backend.app.jev_resolve.JEV_IO_ROOT",
        tmp_path / "jev_io_pairs",
    )

    respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "answers": {
                    "match": {
                        "type": "choice",
                        "choice": "D",
                        "probabilities": {"A": 0.05, "B": 0.25, "C": 0.1, "D": 0.6},
                    }
                }
            },
        )
    )

    client = httpx.AsyncClient()
    out = await resolve_one(
        _sample_record(),
        client=client,
        jev=JevClient(),
        io_dir=tmp_path / "run_rank",
        row_idx=1,
    )
    await client.aclose()

    assert out["jev_ranked_target_ids"] == ["cand-d", "cand-b", "cand-c", "cand-a"]
    assert out["final_target_id"] == "cand-d"


@pytest.mark.asyncio
@respx.mock
async def test_resolve_one_falls_back_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://api.typesafe.ai")
    from backend.app import config as cfg

    cfg.get_settings.cache_clear()

    respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    monkeypatch.setattr(
        "backend.app.jev_resolve.JEV_IO_ROOT",
        tmp_path / "jev_io_pairs",
    )

    client = httpx.AsyncClient()
    jev = JevClient()
    io_dir = tmp_path / "run_1"
    io_dir.mkdir()
    out = await resolve_one(
        _sample_record(),
        client=client,
        jev=jev,
        io_dir=io_dir,
        row_idx=1,
        max_retries=2,
    )
    await client.aclose()

    assert out["final_target_id"] == "cand-a"  # embedding Top1 fallback
    assert out["jev_ranked_target_ids"][0] == "cand-a"
    assert out["jev_error"]


@pytest.mark.asyncio
@respx.mock
async def test_resolve_batch_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://api.typesafe.ai")
    from backend.app import config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setattr(
        "backend.app.jev_resolve.JEV_IO_ROOT",
        tmp_path / "jev_io_pairs",
    )

    respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "answers": {
                    "match": {
                        "type": "choice",
                        "choice": "B",
                        "probabilities": {"A": 0.2, "B": 0.5, "C": 0.2, "D": 0.1},
                    }
                }
            },
        )
    )

    events: list[dict] = []

    async def progress(info):
        events.append(info)

    out = await resolve_batch(
        [_sample_record(), _sample_record(source_id="other")],
        run_seq=99,
        concurrency=2,
        progress=progress,
    )
    assert len(out) == 2
    assert all(r["final_target_id"] == "cand-b" for r in out)
    assert events[-1]["done"] == 2
    assert events[-1]["phase"] == "run_jev_evaluation"
    assert "parallel" in (events[-1].get("message") or "")


@pytest.mark.asyncio
@respx.mock
async def test_resolve_batch_survives_transport_error(tmp_path, monkeypatch):
    """Transport failures must not abort sibling parallel Jev calls."""
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEV_BASE_URL", "https://api.typesafe.ai")
    from backend.app import config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setattr(
        "backend.app.jev_resolve.JEV_IO_ROOT",
        tmp_path / "jev_io_pairs",
    )

    def _side_effect(request):
        payload = json.loads(request.content.decode("utf-8"))
        title = ((payload.get("state") or {}).get("input") or {}).get("job_title") or ""
        if title == "FAIL_TITLE":
            raise httpx.ConnectError("")
        return httpx.Response(
            200,
            json={
                "answers": {
                    "match": {
                        "type": "choice",
                        "choice": "B",
                        "probabilities": {"A": 0.2, "B": 0.5, "C": 0.2, "D": 0.1},
                    }
                }
            },
        )

    respx.post("https://api.typesafe.ai/v1/systemone").mock(side_effect=_side_effect)

    out = await resolve_batch(
        [
            _sample_record(
                source_id="fail",
                source_title="FAIL_TITLE",
                input={
                    "title": "FAIL_TITLE",
                    "description": "x",
                    "skills": "y",
                    "knowledge": "z",
                },
            ),
            _sample_record(source_id="ok"),
        ],
        run_seq=100,
        concurrency=2,
    )
    assert len(out) == 2
    by_id = {r["source_id"]: r for r in out}
    assert by_id["ok"]["final_target_id"] == "cand-b"
    assert by_id["ok"]["jev_error"] is None
    # Failed record falls back to Top-1 after soft transport retries
    assert by_id["fail"]["final_target_id"] == "cand-a"
    assert by_id["fail"]["jev_error"]
    assert "transport" in by_id["fail"]["jev_error"]
