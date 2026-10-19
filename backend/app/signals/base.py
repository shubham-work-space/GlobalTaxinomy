"""Signal protocol and registry for variant scoring."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

SIGNAL_REGISTRY: dict[str, "Signal"] = {}


@runtime_checkable
class Signal(Protocol):
    name: str
    required_collections: tuple[str, ...]

    def score(
        self,
        source_id: str,
        source_dataset: str,
        target_ids: list[str],
        target_dataset: str,
        context: dict[str, Any],
    ) -> np.ndarray:
        """Return score vector shape (len(target_ids),) aligned with target_ids."""
        ...


def register_signal(signal: Signal) -> Signal:
    SIGNAL_REGISTRY[signal.name] = signal
    return signal


def get_signal(name: str) -> Signal:
    if name not in SIGNAL_REGISTRY:
        raise KeyError(f"Unknown signal: {name}")
    return SIGNAL_REGISTRY[name]


def list_signals() -> list[str]:
    return sorted(SIGNAL_REGISTRY.keys())
