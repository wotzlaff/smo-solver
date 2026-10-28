"""Fast SMO solver for smoothed and weighted support vector machines.

The low-level functions solve the dual training problem and return a :class:`SmoResult`;
scikit-learn compatible estimators (:class:`SmoSVC`, :class:`SmoSVR`, :class:`SmoLSSVR`)
are available if scikit-learn is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._core import (
    SmoResult,
    __version__,
    solve_classification,
    solve_dual,
    solve_least_squares,
    solve_regression,
)
from .kernels import kernel_matrix, smooth_max

if TYPE_CHECKING:
    from .estimators import SmoLSSVR, SmoSVC, SmoSVR

__all__ = [
    "SmoLSSVR",
    "SmoResult",
    "SmoSVC",
    "SmoSVR",
    "__version__",
    "kernel_matrix",
    "smooth_max",
    "solve_classification",
    "solve_dual",
    "solve_least_squares",
    "solve_regression",
]

_ESTIMATORS = {"SmoSVC", "SmoSVR", "SmoLSSVR"}


def __getattr__(name: str) -> Any:
    # estimators need scikit-learn, which is an optional dependency
    if name in _ESTIMATORS:
        from . import estimators

        return getattr(estimators, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
