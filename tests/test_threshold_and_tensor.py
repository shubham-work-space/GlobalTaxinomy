"""Threshold gating helpers and combined-score math."""

from __future__ import annotations

import numpy as np

from backend.app.pipeline import _combined_from_tensor, _top_k_indices


def test_combined_from_tensor_weight_normalization():
    data = {
        "signals": ["a", "b"],
        "scores": np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],  # source 0
            ],
            dtype=np.float32,
        ),
        "mask": np.array([[True, True]], dtype=bool),
    }
    # Equal weights → average of signals
    combined = _combined_from_tensor(data, {"a": 0.5, "b": 0.5})
    assert combined.shape == (1, 2)
    np.testing.assert_allclose(combined[0], [0.5, 0.5], atol=1e-5)

    # All weight on signal a
    combined_a = _combined_from_tensor(data, {"a": 1.0, "b": 0.0})
    np.testing.assert_allclose(combined_a[0], [1.0, 0.0], atol=1e-5)


def test_top_k_indices_order():
    row = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)
    idxs = _top_k_indices(row, 4)
    assert idxs == [1, 3, 2, 0]
    assert _top_k_indices(row, 1) == [1]


def test_threshold_gate_selection_logic():
    """Mirror pipeline threshold rule without full run."""
    threshold = 0.60
    rows = [
        {"top1_similarity": 0.72, "source_id": "high"},
        {"top1_similarity": 0.59, "source_id": "low"},
        {"top1_similarity": 0.60, "source_id": "edge"},
    ]
    needs = [r for r in rows if float(r["top1_similarity"]) < threshold]
    assert [r["source_id"] for r in needs] == ["low"]
