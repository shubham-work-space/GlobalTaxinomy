"""Accuracy@K correctness tests."""

from __future__ import annotations

from unittest.mock import patch

from backend.app.accuracy import (
    _pick_final_target,
    annotate_results_with_ground_truth,
    compute_accuracy_metrics,
    final_ranked_target_ids,
)

GOLD = {
    "src-1": {"tgt-exact": "exactMatch", "tgt-close": "closeMatch"},
    "src-2": {"tgt-b": "exactMatch"},
    "src-3": {"tgt-c": "exactMatch"},
    # closeMatch-only — excluded from exact@1 denominator (like the 450 non-exact O*NET)
    "src-close-only": {"tgt-x": "closeMatch"},
}


def test_pick_final_target_uses_jev_when_sent():
    row = {
        "sent_to_jev": True,
        "top1_target_id": "tgt-a",
        "jev_target_id": "tgt-c",
        "final_target_id": "tgt-a",
    }
    assert _pick_final_target(row) == "tgt-c"


def test_pick_final_target_keeps_embedding_when_not_sent():
    row = {
        "sent_to_jev": False,
        "top1_target_id": "tgt-a",
        "jev_target_id": "tgt-c",
    }
    assert _pick_final_target(row) == "tgt-a"


@patch("backend.app.accuracy.direction_map", return_value=GOLD)
def test_exact_match_at_1_hybrid_promotion(_mock_dm):
    """Wrong embedding Top1 + correct Jev promotion counts as Exact Match @1 hit."""
    results = [
        {
            "source_id": "src-1",
            "top1_target_id": "tgt-close",  # wrong for exact
            "sent_to_jev": True,
            "jev_target_id": "tgt-exact",
            "jev_ranked_target_ids": ["tgt-exact", "tgt-close"],
            "final_target_id": "tgt-exact",
        },
        {
            "source_id": "src-2",
            "top1_target_id": "tgt-b",
            "sent_to_jev": False,
            "final_target_id": "tgt-b",
            "candidates": [{"target_id": "tgt-b"}],
        },
    ]
    metrics = compute_accuracy_metrics(results)
    assert metrics["total_records"] == 2
    assert metrics["records_sent_to_jev"] == 1
    assert metrics["embedding_top1_accuracy"] == 0.5  # only src-2
    assert metrics["exact_match_at_1"] == 1.0  # both final exact
    assert metrics["accuracy_at_1"] == 1.0
    assert metrics["high_confidence_records"] == 1
    assert metrics["high_confidence_correct"] == 1
    assert metrics["resolver_subset_records"] == 1
    assert metrics["resolver_subset_correct"] == 1
    assert metrics["resolver_subset_accuracy"] == 1.0


@patch("backend.app.accuracy.direction_map", return_value=GOLD)
def test_threshold_gate_keeps_wrong_top1_when_not_sent(_mock_dm):
    """Score ≥ threshold path: Acc@1 keeps embedding Top1 even if gold is later."""
    results = [
        {
            "source_id": "src-3",
            "top1_target_id": "wrong",
            "sent_to_jev": False,
            "final_target_id": "wrong",
            # candidate list includes gold but gate did not send to Jev
            "candidates": [
                {"target_id": "wrong"},
                {"target_id": "tgt-c"},
            ],
        }
    ]
    metrics = compute_accuracy_metrics(results)
    assert metrics["exact_match_at_1"] == 0.0
    assert metrics["accuracy_at_1"] == 0.0
    assert metrics["accuracy_at_2"] == 1.0  # gold at rank 2 via embedding order
    assert metrics["records_sent_to_jev"] == 0


@patch("backend.app.accuracy.direction_map", return_value=GOLD)
def test_accuracy_at_k_from_jev_ranking(_mock_dm):
    """Gold at JEV rank 3 → miss Acc@1/@2, hit Acc@3/@5/@10."""
    results = [
        {
            "source_id": "src-1",
            "top1_target_id": "tgt-close",
            "sent_to_jev": True,
            "jev_target_id": "other-1",
            "jev_ranked_target_ids": ["other-1", "other-2", "tgt-exact", "x", "y"],
            "final_target_id": "other-1",
        }
    ]
    metrics = compute_accuracy_metrics(results)
    assert metrics["accuracy_at_1"] == 0.0
    assert metrics["accuracy_at_2"] == 0.0
    assert metrics["accuracy_at_3"] == 1.0
    assert metrics["accuracy_at_5"] == 1.0
    assert metrics["accuracy_at_10"] == 1.0
    assert metrics["exact_match_at_1"] == 0.0


@patch("backend.app.accuracy.direction_map", return_value=GOLD)
def test_legacy_row_without_ranked_list_only_scores_at_1(_mock_dm):
    """Older runs without jev_ranked_target_ids still compute Acc@1 via final pick."""
    results = [
        {
            "source_id": "src-2",
            "top1_target_id": "wrong",
            "sent_to_jev": True,
            "jev_target_id": "tgt-b",
            "final_target_id": "tgt-b",
        }
    ]
    ranked = final_ranked_target_ids(results[0])
    assert ranked == ["tgt-b"]
    metrics = compute_accuracy_metrics(results)
    assert metrics["accuracy_at_1"] == 1.0
    assert metrics["accuracy_at_2"] == 1.0  # same single-id list


@patch("backend.app.accuracy.direction_map", return_value=GOLD)
def test_annotate_final_exact_hit(_mock_dm):
    rows = annotate_results_with_ground_truth(
        [
            {
                "source_id": "src-1",
                "top1_target_id": "tgt-close",
                "sent_to_jev": True,
                "jev_target_id": "tgt-exact",
                "jev_ranked_target_ids": ["tgt-exact", "tgt-close"],
            }
        ]
    )
    assert rows[0]["top1_exact_hit"] is False
    assert rows[0]["final_exact_hit"] is True
    assert rows[0]["has_exact_match"] is True
    assert rows[0]["final_hit_at_k"]["1"] is True
    assert rows[0]["final_ranked_target_ids"][0] == "tgt-exact"


@patch("backend.app.accuracy.direction_map", return_value=GOLD)
def test_exact_at_1_excludes_sources_without_exact_gt(_mock_dm):
    """closeMatch-only sources must not inflate total_records or Jev counts."""
    results = [
        {
            "source_id": "src-2",
            "top1_target_id": "tgt-b",
            "sent_to_jev": False,
            "final_target_id": "tgt-b",
            "candidates": [{"target_id": "tgt-b"}],
        },
        {
            "source_id": "src-close-only",
            "top1_target_id": "tgt-x",
            "sent_to_jev": True,  # would have been sent under old 940-wide gate
            "jev_target_id": "tgt-x",
            "final_target_id": "tgt-x",
        },
    ]
    metrics = compute_accuracy_metrics(results)
    assert metrics["total_records"] == 1
    assert metrics["records_sent_to_jev"] == 0
    assert metrics["exact_match_at_1"] == 1.0
