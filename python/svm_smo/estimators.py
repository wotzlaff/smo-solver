"""scikit-learn compatible estimators backed by the SMO solver."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data

from . import _core
from .kernels import kernel_matrix


class _BaseSmo(BaseEstimator):
    """Shared logic: kernel parameters, solver options and the kernel expansion."""

    _solver_options = ("tol", "max_steps", "time_limit", "working_set", "cache_size_mb")

    def _gamma(self, x: NDArray[np.float64]) -> float:
        if self.gamma == "scale":
            var = x.var()
            return 1.0 / (x.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / x.shape[1]
        if not isinstance(self.gamma, Real) or self.gamma <= 0:
            raise ValueError(f"gamma must be 'scale', 'auto' or positive, got {self.gamma!r}")
        return float(self.gamma)

    def _kernel_args(self) -> dict[str, Any]:
        return dict(kernel=self.kernel, gamma=self._gamma_, coef0=self.coef0, degree=self.degree)

    def _options(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._solver_options}

    def _check_lam(self) -> None:
        if not isinstance(self.lam, Real) or self.lam <= 0:
            raise ValueError(f"lam must be positive, got {self.lam!r}")

    @staticmethod
    def _weights(sample_weight: ArrayLike | None, x: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = _check_sample_weight(sample_weight, x, dtype=np.float64, ensure_non_negative=True)
        if not np.any(weights > 0):
            raise ValueError("All sample weights are zero.")
        return np.ascontiguousarray(weights)

    def _store(self, x: NDArray[np.float64], results: list[_core.SmoResult]) -> None:
        """Stores the kernel expansion of one or several (one-vs-rest) solutions."""
        coef = np.stack([np.asarray(r.beta) / self.lam for r in results])
        support = np.flatnonzero(np.any(coef != 0.0, axis=0))
        self.support_ = support
        self.dual_coef_ = coef[:, support]
        self.intercept_ = np.array([r.b for r in results])
        self.n_support_ = support.size
        self.support_vectors_ = x[support] if self.kernel != "precomputed" else np.empty((0, 0))
        self.n_iter_ = np.array([r.steps for r in results])
        self.fit_status_ = [r.status for r in results]
        self.duality_gap_ = np.array([r.gap for r in results])

    def _decision(self, X: ArrayLike) -> NDArray[np.float64]:
        """Decision values ``f(x) = sum_k coef_k K(x, sv_k) + b``, shape ``(n, n_outputs)``."""
        check_is_fitted(self)
        x = validate_data(self, X, reset=False)
        if self.kernel == "precomputed":
            k = x[:, self.support_]
        else:
            k = kernel_matrix(
                x,
                self.support_vectors_,
                self.kernel,
                gamma=self._gamma_,
                coef0=self.coef0,
                degree=self.degree,
            )
        return k @ self.dual_coef_.T + self.intercept_

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags


class SmoSVC(ClassifierMixin, _BaseSmo):
    """Support vector classifier with optionally smoothed hinge loss.

    More than two classes are handled by one-vs-rest.

    Solves ``min_{w,b} lam/2 |w|^2 + sum_i c_i m_delta(shift - y_i (<w, phi(x_i)> + b))``
    where ``m_delta`` is the quadratically smoothed maximum function (``smoothing = delta``).
    ``lam`` corresponds to ``1 / C`` in LIBSVM / :class:`sklearn.svm.SVC`.

    Parameters
    ----------
    lam : float
        Regularization parameter ``lambda > 0``.
    kernel : {"rbf", "linear", "poly", "precomputed"}
    gamma : float or {"scale", "auto"}
        Kernel parameter, same conventions as scikit-learn.
    coef0, degree : float, int
        Parameters of the polynomial kernel.
    shift : float
        Margin of the hinge loss (``1`` for the classical SVM).
    smoothing : float
        Smoothing parameter ``delta >= 0`` of the loss.
    max_asum : float, optional
        Bound on ``sum |alpha_i|`` (generalised nu-SVM).
    class_weight : dict or "balanced", optional
        Weights per class, multiplied with ``sample_weight``.
    tol, max_steps, time_limit, working_set, cache_size_mb
        Solver options, see :func:`svm_smo.solve_classification`.
    """

    def __init__(
        self,
        lam: float = 1.0,
        *,
        kernel: str = "rbf",
        gamma: float | str = "scale",
        coef0: float = 0.0,
        degree: int = 3,
        shift: float = 1.0,
        smoothing: float = 0.0,
        max_asum: float | None = None,
        class_weight: dict[Any, float] | str | None = None,
        tol: float = 1e-3,
        max_steps: int = 100_000_000,
        time_limit: float | None = None,
        working_set: str = "md",
        cache_size_mb: float = 256.0,
    ) -> None:
        self.lam = lam
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.shift = shift
        self.smoothing = smoothing
        self.max_asum = max_asum
        self.class_weight = class_weight
        self.tol = tol
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.working_set = working_set
        self.cache_size_mb = cache_size_mb

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> SmoSVC:
        self._check_lam()
        x, y = validate_data(self, X, y, dtype=np.float64, order="C")
        check_classification_targets(y)
        self.classes_, yi = np.unique(y, return_inverse=True)
        n_classes = self.classes_.size
        if n_classes < 2:
            raise ValueError(f"The number of classes has to be at least 2; got {n_classes} class")
        weights = self._weights(sample_weight, x)
        if self.class_weight == "balanced":
            counts = np.bincount(yi, minlength=n_classes)
            weights = weights * (yi.size / (n_classes * counts[yi]))
        elif isinstance(self.class_weight, dict):
            weights = weights * np.array([self.class_weight.get(c, 1.0) for c in self.classes_])[yi]
        self._gamma_ = self._gamma(x) if self.kernel != "precomputed" else 1.0
        positives = [1] if n_classes == 2 else range(n_classes)
        results = [
            _core.solve_classification(
                x,
                np.where(yi == k, 1.0, -1.0),
                self.lam,
                weights=np.ascontiguousarray(weights),
                shift=self.shift,
                smoothing=self.smoothing,
                max_asum=self.max_asum,
                **self._kernel_args(),
                **self._options(),
            )
            for k in positives
        ]
        self._store(x, results)
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Signed decision values: shape ``(n,)`` for two classes (positive means
        ``classes_[1]``), otherwise ``(n, n_classes)`` (one-vs-rest)."""
        dec = self._decision(X)
        return dec[:, 0] if self.classes_.size == 2 else dec

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        dec = self.decision_function(X)
        idx = (dec > 0).astype(int) if dec.ndim == 1 else dec.argmax(axis=1)
        return self.classes_[idx]


class SmoSVR(RegressorMixin, _BaseSmo):
    """epsilon-insensitive support vector regression with optionally smoothed loss.

    Solves ``min lam/2 |w|^2 + sum_i c_i (m_delta(f_i - y_i - eps) + m_delta(y_i - f_i - eps))``.
    See :class:`SmoSVC` for the shared parameters.
    """

    def __init__(
        self,
        lam: float = 1.0,
        *,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        gamma: float | str = "scale",
        coef0: float = 0.0,
        degree: int = 3,
        smoothing: float = 0.0,
        max_asum: float | None = None,
        tol: float = 1e-3,
        max_steps: int = 100_000_000,
        time_limit: float | None = None,
        working_set: str = "md",
        cache_size_mb: float = 256.0,
    ) -> None:
        self.lam = lam
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.smoothing = smoothing
        self.max_asum = max_asum
        self.tol = tol
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.working_set = working_set
        self.cache_size_mb = cache_size_mb

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> SmoSVR:
        self._check_lam()
        x, y = validate_data(self, X, y, dtype=np.float64, order="C", y_numeric=True)
        weights = self._weights(sample_weight, x)
        self._gamma_ = self._gamma(x) if self.kernel != "precomputed" else 1.0
        result = _core.solve_regression(
            x,
            np.ascontiguousarray(y, dtype=np.float64),
            self.lam,
            epsilon=self.epsilon,
            weights=weights,
            smoothing=self.smoothing,
            max_asum=self.max_asum,
            **self._kernel_args(),
            **self._options(),
        )
        self._store(x, [result])
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        return self._decision(X)[:, 0]


class SmoLSSVR(RegressorMixin, _BaseSmo):
    """Least-squares SVM regression ``min lam/2 |w|^2 + sum_i c_i/2 (f(x_i) - y_i)^2``.

    Equivalent to kernel ridge regression with an unregularized bias.
    """

    def __init__(
        self,
        lam: float = 1.0,
        *,
        kernel: str = "rbf",
        gamma: float | str = "scale",
        coef0: float = 0.0,
        degree: int = 3,
        tol: float = 1e-3,
        max_steps: int = 100_000_000,
        time_limit: float | None = None,
        working_set: str = "md",
        cache_size_mb: float = 256.0,
    ) -> None:
        self.lam = lam
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.tol = tol
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.working_set = working_set
        self.cache_size_mb = cache_size_mb

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> SmoLSSVR:
        self._check_lam()
        x, y = validate_data(self, X, y, dtype=np.float64, order="C", y_numeric=True)
        weights = self._weights(sample_weight, x)
        self._gamma_ = self._gamma(x) if self.kernel != "precomputed" else 1.0
        result = _core.solve_least_squares(
            x,
            np.ascontiguousarray(y, dtype=np.float64),
            self.lam,
            weights=weights,
            **self._kernel_args(),
            **self._options(),
        )
        self._store(x, [result])
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        return self._decision(X)[:, 0]
