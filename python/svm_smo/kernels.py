"""NumPy implementations of the kernel functions used by the solver (for prediction)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def kernel_matrix(
    x: ArrayLike,
    z: ArrayLike,
    kernel: str = "rbf",
    *,
    gamma: float = 1.0,
    coef0: float = 0.0,
    degree: int = 3,
) -> NDArray[np.float64]:
    """Kernel matrix ``K[i, j] = k(x_i, z_j)`` with the same conventions as the solver.

    ``rbf``: ``exp(-gamma |x - z|^2)``, ``linear``: ``<x, z>``,
    ``poly``: ``(gamma <x, z> + coef0)^degree``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    if kernel in ("rbf", "gaussian"):
        d2 = (x * x).sum(axis=1)[:, None] + (z * z).sum(axis=1)[None, :] - 2.0 * x @ z.T
        return np.exp(-gamma * np.maximum(d2, 0.0))
    if kernel == "linear":
        return x @ z.T
    if kernel in ("poly", "polynomial"):
        return (gamma * (x @ z.T) + coef0) ** degree
    raise ValueError(f"unknown kernel {kernel!r}")


def smooth_max(t: ArrayLike, delta: float) -> NDArray[np.float64]:
    """Quadratically smoothed maximum ``m_delta(t)`` (``max(0, t)`` for ``delta = 0``)."""
    t = np.asarray(t, dtype=np.float64)
    if delta == 0.0:
        return np.maximum(t, 0.0)
    return np.where(t >= delta, t, np.where(t <= -delta, 0.0, 0.25 / delta * (t + delta) ** 2))
