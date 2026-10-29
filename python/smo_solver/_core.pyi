from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__version__: str

Kernel = Literal["rbf", "gaussian", "linear", "poly", "polynomial", "precomputed"]
Stopping = Literal["violation", "bound_aware"]
WorkingSet = Literal[
    "mvp", "mud", "md", "max_violating_pair", "max_unbounded_decrease", "max_decrease"
]
Callback = Callable[[dict[str, Any]], bool | None]
Array1 = NDArray[np.float64]
Array2 = NDArray[np.float64]

class SmoResult:
    alpha: Array1
    beta: Array1
    b: float
    c: float
    ka: Array1
    status: Literal["converged", "max_steps", "time_limit", "interrupted"]
    steps: int
    elapsed: float
    violation: float
    dual_objective: float
    primal_objective: float
    gap: float
    n_active: int
    shrinks: int
    unshrinks: int
    min_active: int
    bound_aware: float
    cache_hits: int
    cache_misses: int
    lam: float
    @property
    def converged(self) -> bool: ...

def solve_classification(
    x: Array2,
    y: Array1,
    lam: float,
    *,
    kernel: Kernel = "rbf",
    gamma: float = 1.0,
    coef0: float = 0.0,
    degree: int = 3,
    weights: Array1 | None = None,
    shift: float = 1.0,
    smoothing: float = 0.0,
    max_asum: float | None = None,
    tol: float = 1e-3,
    max_steps: int = 100_000_000,
    time_limit: float | None = None,
    working_set: WorkingSet = "md",
    two_sided: bool = True,
    stopping: Stopping = "violation",
    rho: float | None = None,
    shrinking: bool = True,
    shrinking_period: int = 0,
    shrinking_exponent: float = 1.0,
    shrinking_factor: float = 1.0,
    shrinking_min_fraction: float = 0.1,
    cache_size_mb: float = 256.0,
    alpha0: Array1 | None = None,
    callback: Callback | None = None,
    callback_period: int = 1000,
) -> SmoResult: ...
def solve_regression(
    x: Array2,
    y: Array1,
    lam: float,
    *,
    epsilon: float = 0.1,
    kernel: Kernel = "rbf",
    gamma: float = 1.0,
    coef0: float = 0.0,
    degree: int = 3,
    weights: Array1 | None = None,
    smoothing: float = 0.0,
    max_asum: float | None = None,
    tol: float = 1e-3,
    max_steps: int = 100_000_000,
    time_limit: float | None = None,
    working_set: WorkingSet = "md",
    two_sided: bool = True,
    stopping: Stopping = "violation",
    rho: float | None = None,
    shrinking: bool = True,
    shrinking_period: int = 0,
    shrinking_exponent: float = 1.0,
    shrinking_factor: float = 1.0,
    shrinking_min_fraction: float = 0.1,
    cache_size_mb: float = 256.0,
    alpha0: Array1 | None = None,
    callback: Callback | None = None,
    callback_period: int = 1000,
) -> SmoResult: ...
def solve_least_squares(
    x: Array2,
    y: Array1,
    lam: float,
    *,
    kernel: Kernel = "rbf",
    gamma: float = 1.0,
    coef0: float = 0.0,
    degree: int = 3,
    weights: Array1 | None = None,
    tol: float = 1e-3,
    max_steps: int = 100_000_000,
    time_limit: float | None = None,
    working_set: WorkingSet = "md",
    two_sided: bool = True,
    stopping: Stopping = "violation",
    rho: float | None = None,
    cache_size_mb: float = 256.0,
    alpha0: Array1 | None = None,
    callback: Callback | None = None,
    callback_period: int = 1000,
) -> SmoResult: ...
def solve_dual(
    x: Array2,
    lam: float,
    q: Array1,
    p: Array1,
    lo: Array1,
    hi: Array1,
    *,
    kernel_index: Sequence[int] | None = None,
    sign: Array1 | None = None,
    max_asum: float | None = None,
    kernel: Kernel = "rbf",
    gamma: float = 1.0,
    coef0: float = 0.0,
    degree: int = 3,
    tol: float = 1e-3,
    max_steps: int = 100_000_000,
    time_limit: float | None = None,
    working_set: WorkingSet = "md",
    two_sided: bool = True,
    stopping: Stopping = "violation",
    rho: float | None = None,
    shrinking: bool = True,
    shrinking_period: int = 0,
    shrinking_exponent: float = 1.0,
    shrinking_factor: float = 1.0,
    shrinking_min_fraction: float = 0.1,
    cache_size_mb: float = 256.0,
    alpha0: Array1 | None = None,
    callback: Callback | None = None,
    callback_period: int = 1000,
) -> SmoResult: ...
