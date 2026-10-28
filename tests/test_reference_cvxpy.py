"""Independent reference: solve the *primal* training problems with cvxpy.

The smoothed maximum is written as an infimal convolution (DCP compliant),
``m_delta(t) = min_s max(0, s) + (t + delta - s)^2 / (4 delta)``, so the reference does not
rely on the dual derivation used by the solver.
"""

import numpy as np
import pytest

import svm_smo
from svm_smo import kernel_matrix

cp = pytest.importorskip("cvxpy")


def smooth_max_expr(t, delta):
    if delta == 0.0:
        return cp.pos(t)
    s = cp.Variable(t.shape)
    return cp.pos(s) + cp.square(t + delta - s) / (4 * delta)


def kernel_factor(k):
    vals, vecs = np.linalg.eigh(k)
    return vecs * np.sqrt(np.maximum(vals, 0.0))


def solve_primal(k, losses, lam, max_asum=None):
    """min lam/2 v'Kv + sum(losses(K v + b + s c)) (+ A c, c >= 0)."""
    n = k.shape[0]
    v, b = cp.Variable(n), cp.Variable()
    f = k @ v + b
    objective = lam / 2 * cp.sum_squares(kernel_factor(k).T @ v)
    c = None
    if max_asum is not None:
        c = cp.Variable(nonneg=True)
        objective += max_asum * c
    objective += losses(f, c)
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve()
    assert prob.status == cp.OPTIMAL
    return prob.value


def make_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = np.where(x[:, 0] * x[:, 1] + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    t = np.sin(2 * x[:, 0]) + 0.3 * x[:, 1]
    w = rng.uniform(0.5, 2.0, size=n)
    return x, y, t, w


@pytest.mark.parametrize("delta", [0.0, 0.4])
@pytest.mark.parametrize("max_asum", [None, 3.0])
@pytest.mark.parametrize("lam", [1.0, 0.05])
def test_classification(delta, max_asum, lam):
    x, y, _, w = make_data()
    k = kernel_matrix(x, x, gamma=0.8)

    def losses(f, c):
        arg = 1.0 - cp.multiply(y, f if c is None else f + y * c)
        m = smooth_max_expr(arg, delta)
        return w @ m

    ref = solve_primal(k, losses, lam, max_asum)
    res = svm_smo.solve_classification(
        x, y, lam, gamma=0.8, weights=w, smoothing=delta, max_asum=max_asum, tol=1e-9
    )
    assert res.converged
    np.testing.assert_allclose(res.primal_objective, ref, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(-res.dual_objective, ref, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("delta", [0.0, 0.2])
@pytest.mark.parametrize("max_asum", [None, 2.0])
def test_regression(delta, max_asum):
    x, _, t, w = make_data(seed=1)
    lam, eps = 0.1, 0.1
    k = kernel_matrix(x, x, gamma=0.5)

    def losses(f, c):
        # copies with sign +1 (penalise y - f > eps) and -1 (penalise f - y > eps)
        up = f if c is None else f + c
        dn = f if c is None else f - c
        m1 = smooth_max_expr(t - up - eps, delta)
        m2 = smooth_max_expr(dn - t - eps, delta)
        return w @ (m1 + m2)

    ref = solve_primal(k, losses, lam, max_asum)
    res = svm_smo.solve_regression(
        x,
        t,
        lam,
        epsilon=eps,
        gamma=0.5,
        weights=w,
        smoothing=delta,
        max_asum=max_asum,
        tol=1e-9,
    )
    assert res.converged
    np.testing.assert_allclose(res.primal_objective, ref, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(-res.dual_objective, ref, rtol=1e-5, atol=1e-7)


def test_least_squares():
    x, _, t, w = make_data(seed=2)
    lam = 0.3
    k = kernel_matrix(x, x, "poly", gamma=0.5, coef0=1.0, degree=2)
    ref = solve_primal(k, lambda f, c: 0.5 * w @ cp.square(f - t), lam)
    res = svm_smo.solve_least_squares(
        x, t, lam, kernel="poly", gamma=0.5, coef0=1.0, degree=2, weights=w, tol=1e-10
    )
    np.testing.assert_allclose(res.primal_objective, ref, rtol=1e-6)
