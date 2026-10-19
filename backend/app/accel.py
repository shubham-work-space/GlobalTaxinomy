"""Optional torch CUDA / MPS acceleration with NumPy fallback."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import numpy as np

Backend = Literal["cuda", "mps", "cpu", "numpy"]


@lru_cache(maxsize=1)
def detect_backend() -> Backend:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def backend_info() -> dict[str, Any]:
    b = detect_backend()
    info: dict[str, Any] = {"backend": b, "torch": False, "device": None}
    if b == "numpy":
        return info
    import torch

    info["torch"] = True
    info["torch_version"] = torch.__version__
    if b == "cuda":
        info["device"] = "cuda:0"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    elif b == "mps":
        info["device"] = "mps"
        info["gpu_name"] = "Apple MPS"
    else:
        info["device"] = "cpu"
    return info


def matmul_f1_all_targets(
    t_matrix: np.ndarray,
    a_matrix: np.ndarray,
    target_row_lists: list[list[int] | None],
) -> np.ndarray:
    """
    Soft BERTScore F1 for every target occupation vs one source skill set.

    t_matrix / a_matrix must already be L2-normalized (float32 preferred).
    target_row_lists[i] = row indices into t_matrix for target i, or None/empty.
    """
    n_t = len(target_row_lists)
    out = np.zeros(n_t, dtype=np.float32)
    if t_matrix.size == 0 or a_matrix.size == 0:
        return out

    backend = detect_backend()
    if backend in ("cuda", "mps", "cpu"):
        try:
            return _torch_f1_all_targets(t_matrix, a_matrix, target_row_lists, backend)
        except Exception:  # noqa: BLE001 — fall back to NumPy
            pass
    return _numpy_f1_all_targets(t_matrix, a_matrix, target_row_lists)


def _numpy_f1_all_targets(
    t_matrix: np.ndarray,
    a_matrix: np.ndarray,
    target_row_lists: list[list[int] | None],
) -> np.ndarray:
    # Matrices are L2-normalized → cosine = matmul
    sim = np.clip(t_matrix @ a_matrix.T, 0.0, 1.0)  # (N_skills, m)
    out = np.zeros(len(target_row_lists), dtype=np.float32)
    m = sim.shape[1]
    for i, rows in enumerate(target_row_lists):
        if not rows:
            continue
        M = sim[rows]  # (n, m)
        n = M.shape[0]
        if n == 0 or m == 0:
            continue
        p = float(M.max(axis=1).mean())
        r = float(M.max(axis=0).mean())
        out[i] = (2.0 * p * r / (p + r)) if (p + r) > 1e-12 else 0.0
    return out


def _torch_f1_all_targets(
    t_matrix: np.ndarray,
    a_matrix: np.ndarray,
    target_row_lists: list[list[int] | None],
    backend: Backend,
) -> np.ndarray:
    import torch

    device = torch.device("cuda:0" if backend == "cuda" else backend)
    t = torch.as_tensor(t_matrix, dtype=torch.float32, device=device)
    a = torch.as_tensor(a_matrix, dtype=torch.float32, device=device)
    sim = (t @ a.T).clamp_(0.0, 1.0)
    sim_cpu = sim.detach().cpu().numpy()
    return _numpy_f1_from_sim(sim_cpu, target_row_lists)


def _numpy_f1_from_sim(
    sim: np.ndarray,
    target_row_lists: list[list[int] | None],
) -> np.ndarray:
    out = np.zeros(len(target_row_lists), dtype=np.float32)
    m = sim.shape[1]
    for i, rows in enumerate(target_row_lists):
        if not rows:
            continue
        M = sim[rows]
        n = M.shape[0]
        if n == 0 or m == 0:
            continue
        p = float(M.max(axis=1).mean())
        r = float(M.max(axis=0).mean())
        out[i] = (2.0 * p * r / (p + r)) if (p + r) > 1e-12 else 0.0
    return out
