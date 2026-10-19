"""Register all signal implementations."""

from . import aggregate as _aggregate  # noqa: F401
from . import facet as _facet  # noqa: F401
from .base import SIGNAL_REGISTRY, Signal, get_signal, list_signals, register_signal

__all__ = [
    "SIGNAL_REGISTRY",
    "Signal",
    "get_signal",
    "list_signals",
    "register_signal",
]
