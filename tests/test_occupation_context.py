"""Tests for gated JEV occupation context loader."""

from __future__ import annotations

import pytest

from backend.app import occupation_context as oc


def test_enrich_gated_record_merges_fields(monkeypatch):
    monkeypatch.setattr(
        oc,
        "get_onet_jev_context",
        lambda sid: {
            "occupation_id": sid,
            "title": "Chief Executives",
            "tasks": "Direct organization",
        },
    )
    monkeypatch.setattr(
        oc,
        "get_esco_jev_context",
        lambda tid: {
            "occupation_id": tid,
            "preferred_title": "CEO",
            "essential_skills": "lead teams",
        },
    )
    row = {
        "source_id": "11-1011.00",
        "input": {"title": "Old Title"},
        "candidates": [{"target_id": "http://esco/uuid", "preferred_title": "x"}],
    }
    out = oc.enrich_gated_record(row, input_taxonomy="onet", output_taxonomy="esco")
    assert out["input"]["tasks"] == "Direct organization"
    assert out["candidates"][0]["essential_skills"] == "lead teams"
    assert out["jev_context_version"] == oc.JEV_CONTEXT_VERSION


def test_validate_jev_rank_mode_in_runs():
    from backend.app.runs import validate_config

    cfg = validate_config({"jev_rank_mode": "noul"})
    assert cfg["jev_rank_mode"] == "noul"
    with pytest.raises(ValueError):
        validate_config({"jev_rank_mode": "score"})


@pytest.mark.skipif(
    not oc.context_sources_available(),
    reason="O*NET/ESCO raw data dirs not present on this machine",
)
def test_live_onet_context_sample():
    ctx = oc.get_onet_jev_context("11-1011.00")
    assert ctx.get("title") == "Chief Executives"
    assert ctx.get("description")
    assert ctx.get("essential_skills") or ctx.get("knowledge")
