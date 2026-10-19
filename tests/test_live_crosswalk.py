"""Live crosswalk threshold / cache / reorder helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.live_crosswalk import (
    _reorder_by_ids,
    find_cached_jev_ranking,
    load_live_jev_cache,
    save_live_jev_cache,
)


def test_reorder_by_ids_prefers_ranked_order():
    pool = [
        {"occupation_id": "a", "preferred_title": "A", "weighted_score": 0.9},
        {"occupation_id": "b", "preferred_title": "B", "weighted_score": 0.8},
        {"occupation_id": "c", "preferred_title": "C", "weighted_score": 0.7},
    ]
    out = _reorder_by_ids(pool, ["c", "a"], target_dataset="esco")
    assert [m["occupation_id"] for m in out[:2]] == ["c", "a"]
    assert out[2]["occupation_id"] == "b"


def test_live_jev_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "backend.app.live_crosswalk.LIVE_CACHE_ROOT",
        tmp_path / "live_jev_cache",
    )
    save_live_jev_cache("onet_to_esco", "11-1011.00", ["t1", "t2", "t3"])
    assert load_live_jev_cache("onet_to_esco", "11-1011.00") == ["t1", "t2", "t3"]
    assert load_live_jev_cache("onet_to_esco", "missing") is None


def test_find_cached_jev_ranking_prefers_live_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "backend.app.live_crosswalk.LIVE_CACHE_ROOT",
        tmp_path / "live_jev_cache",
    )
    save_live_jev_cache("onet_to_esco", "11-1011.00", ["cached-a", "cached-b"])
    hit = find_cached_jev_ranking("11-1011.00", "onet_to_esco")
    assert hit is not None
    ids, origin = hit
    assert ids[0] == "cached-a"
    assert origin == "jev_cached_live"


@pytest.mark.asyncio
async def test_live_crosswalk_high_confidence_skips_jev():
    from backend.app.live_crosswalk import live_crosswalk

    pool = [
        {
            "occupation_id": f"t{i}",
            "preferred_title": f"Title {i}",
            "weighted_score": 0.75 - i * 0.01,
            "title_similarity": 0.7,
            "description_similarity": 0.6,
            "skills_similarity": 0.5,
            "knowledge_similarity": 0.4,
        }
        for i in range(20)
    ]
    with (
        patch("backend.app.live_crosswalk.score_variant_pool", return_value=pool),
        patch(
            "backend.app.live_crosswalk.chroma_store.load_occupation_embeddings",
            return_value={"preferred_title": "Chief Executives"},
        ),
        patch("backend.app.live_crosswalk.resolve_batch", new_callable=AsyncMock) as mock_jev,
        patch(
            "backend.app.live_crosswalk.annotate_matches",
            side_effect=lambda _d, _s, matches: (
                matches,
                {
                    "total": 0,
                    "found_in_top_k": 0,
                    "found_by_type": {},
                    "expected_by_type": {},
                },
            ),
        ),
        patch(
            "backend.app.live_crosswalk.summarize_source_ground_truth",
            return_value={
                "total": 0,
                "by_type": {},
                "has_exact_match": False,
                "exact_match_count": 0,
            },
        ),
        patch(
            "backend.app.live_crosswalk.cardinality_context",
            return_value={"direction": "onet_to_esco"},
        ),
    ):
        out = await live_crosswalk("11-1011.00", "onet_to_esco", threshold=0.60)

    assert out["ranking_source"] == "embedding"
    assert out["sent_to_jev"] is False
    assert len(out["matches"]) == 10
    assert out["matches"][0]["occupation_id"] == "t0"
    mock_jev.assert_not_called()


@pytest.mark.asyncio
async def test_live_crosswalk_low_confidence_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from backend.app.live_crosswalk import live_crosswalk, save_live_jev_cache

    monkeypatch.setattr(
        "backend.app.live_crosswalk.LIVE_CACHE_ROOT",
        tmp_path / "live_jev_cache",
    )
    save_live_jev_cache("onet_to_esco", "11-1011.00", ["t5", "t1", "t0"])

    pool = [
        {
            "occupation_id": f"t{i}",
            "preferred_title": f"Title {i}",
            "weighted_score": 0.50 - i * 0.001,
            "title_similarity": 0.5,
            "description_similarity": 0.4,
            "skills_similarity": 0.3,
            "knowledge_similarity": 0.2,
        }
        for i in range(20)
    ]
    with (
        patch("backend.app.live_crosswalk.score_variant_pool", return_value=pool),
        patch(
            "backend.app.live_crosswalk.chroma_store.load_occupation_embeddings",
            return_value={"preferred_title": "Chief Executives"},
        ),
        patch("backend.app.live_crosswalk.resolve_batch", new_callable=AsyncMock) as mock_jev,
        patch(
            "backend.app.live_crosswalk.annotate_matches",
            side_effect=lambda _d, _s, matches: (
                matches,
                {
                    "total": 0,
                    "found_in_top_k": 0,
                    "found_by_type": {},
                    "expected_by_type": {},
                },
            ),
        ),
        patch(
            "backend.app.live_crosswalk.summarize_source_ground_truth",
            return_value={
                "total": 0,
                "by_type": {},
                "has_exact_match": False,
                "exact_match_count": 0,
            },
        ),
        patch(
            "backend.app.live_crosswalk.cardinality_context",
            return_value={"direction": "onet_to_esco"},
        ),
    ):
        out = await live_crosswalk("11-1011.00", "onet_to_esco", threshold=0.60)

    assert out["ranking_source"] == "jev_cached"
    assert out["sent_to_jev"] is True
    assert [m["occupation_id"] for m in out["matches"][:3]] == ["t5", "t1", "t0"]
    mock_jev.assert_not_called()
