"""Tests of the Python bindings against scikit-learn/LIBSVM, closed forms and SciPy."""

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.svm import SVC, SVR

import smo_solver
from smo_solver import kernel_matrix, smooth_max


def make_data(n=120, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = np.where(np.sin(2 * x[:, 0]) + x[:, 1] ** 2 - 0.5 + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    t = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] + 0.1 * rng.normal(size=n)
    return x, y, t


def decision(x_train, res, x, gamma):
    return kernel_matrix(x, x_train, gamma=gamma) @ (res.beta / res.lam) + res.b


@pytest.mark.parametrize("lam", [1.0, 0.1, 0.01])
def test_classification_matches_libsvm(lam):
    x, y, _ = make_data()
    gamma = 0.5
    res = smo_solver.solve_classification(x, y, lam, gamma=gamma, tol=1e-8)
    assert res.converged
    ref = SVC(C=1 / lam, gamma=gamma, tol=1e-8).fit(x, y)
    xt = make_data(50, seed=1)[0]
    # LIBSVM stops with the same (scaled) tolerance, so compare relative to C = 1 / lam
    np.testing.assert_allclose(
        decision(x, res, xt, gamma), ref.decision_function(xt), atol=1e-6 / lam
    )
    np.testing.assert_allclose(res.b, ref.intercept_[0], atol=1e-6 / lam)


@pytest.mark.parametrize("eps", [0.0, 0.1])
def test_regression_matches_libsvm(eps):
    x, _, t = make_data()
    gamma, lam = 0.3, 0.05
    res = smo_solver.solve_regression(x, t, lam, epsilon=eps, gamma=gamma, tol=1e-8)
    assert res.converged and res.alpha.shape == (2 * len(t),)
    ref = SVR(C=1 / lam, epsilon=eps, gamma=gamma, tol=1e-8).fit(x, t)
    xt = make_data(50, seed=2)[0]
    np.testing.assert_allclose(decision(x, res, xt, gamma), ref.predict(xt), atol=1e-5)


def test_least_squares_closed_form():
    x, _, t = make_data(80)
    lam, gamma = 0.1, 0.7
    w = np.linspace(0.5, 2.0, len(t))
    res = smo_solver.solve_least_squares(x, t, lam, gamma=gamma, weights=w, tol=1e-10)
    k = kernel_matrix(x, x, gamma=gamma)
    n = len(t)
    a = np.block([[k / lam + np.diag(1 / w), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    sol = np.linalg.solve(a, np.append(t, 0.0))
    np.testing.assert_allclose(res.alpha, sol[:n], atol=1e-7)
    np.testing.assert_allclose(res.b, sol[n], atol=1e-7)


def test_smoothed_weighted_constrained_problem_matches_scipy():
    x, y, _ = make_data(30, 2, seed=3)
    lam, gamma, delta, shift, max_asum = 0.2, 1.0, 0.3, 1.0, 6.0
    w = np.linspace(0.5, 1.5, len(y))
    res = smo_solver.solve_classification(
        x,
        y,
        lam,
        gamma=gamma,
        weights=w,
        smoothing=delta,
        shift=shift,
        max_asum=max_asum,
        tol=1e-10,
    )
    assert res.converged
    k = kernel_matrix(x, x, gamma=gamma)
    q, p = 2 * delta / w, -(shift + delta) * y

    def obj(a):
        return 0.5 / lam * a @ k @ a + (0.5 * q * a**2 + p * a).sum()

    def grad(a):
        return k @ a / lam + q * a + p

    bounds = [(0, wi) if yi > 0 else (-wi, 0) for yi, wi in zip(y, w, strict=True)]
    cons = [
        {"type": "eq", "fun": lambda a: a.sum(), "jac": lambda a: np.ones_like(a)},
        {"type": "ineq", "fun": lambda a: max_asum - (y * a).sum(), "jac": lambda a: -y},
    ]
    ref = minimize(
        obj,
        np.zeros(len(y)),
        jac=grad,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert ref.success
    assert res.dual_objective <= ref.fun + 1e-8
    np.testing.assert_allclose(res.dual_objective, ref.fun, rtol=1e-6)
    assert abs(res.gap) < 1e-7
    # the primal objective evaluated in NumPy agrees with the solver's value
    f = res.ka + res.b + y * res.c
    loss = (w * smooth_max(shift - y * f, delta)).sum()
    primal = 0.5 * res.alpha @ res.ka + loss + max_asum * res.c
    np.testing.assert_allclose(primal, res.primal_objective, rtol=1e-10)


def test_solve_dual_reproduces_classification():
    x, y, _ = make_data(60)
    lam = 0.1
    ref = smo_solver.solve_classification(x, y, lam, tol=1e-9)
    lo, hi = np.minimum(0, y), np.maximum(0, y)
    res = smo_solver.solve_dual(x, lam, np.zeros_like(y), -y, lo, hi, tol=1e-9)
    np.testing.assert_allclose(res.alpha, ref.alpha, atol=1e-8)


def test_precomputed_kernel_and_other_kernels():
    x, y, _ = make_data(60)
    k = kernel_matrix(x, x, "poly", gamma=0.5, coef0=1.0, degree=2)
    poly = dict(kernel="poly", gamma=0.5, coef0=1.0, degree=2)
    a = smo_solver.solve_classification(x, y, 0.1, **poly, tol=1e-9)
    b = smo_solver.solve_classification(k, y, 0.1, kernel="precomputed", tol=1e-9)
    np.testing.assert_allclose(a.alpha, b.alpha, atol=1e-8)
    lin = smo_solver.solve_classification(x, y, 0.1, kernel="linear")
    assert lin.converged


@pytest.mark.parametrize("ws", ["mvp", "mud", "md"])
def test_working_sets_and_shrinking(ws):
    x, y, _ = make_data(150)
    objs = [
        smo_solver.solve_classification(
            x, y, 0.01, working_set=ws, shrinking=s, shrinking_period=10, tol=1e-9
        ).dual_objective
        for s in (False, True)
    ]
    np.testing.assert_allclose(objs[0], objs[1], rtol=1e-9)


def test_callback_and_warm_start():
    x, y, _ = make_data(200)
    seen = []

    def cb(info):
        seen.append(info["step"])
        return info["step"] >= 20

    res = smo_solver.solve_classification(x, y, 0.001, callback=cb, callback_period=10, tol=1e-9)
    assert res.status == "interrupted" and seen == [10, 20] and res.steps == 20
    partial = smo_solver.solve_classification(x, y, 0.001, max_steps=600, tol=1e-6)
    assert partial.status == "max_steps"
    warm = smo_solver.solve_classification(x, y, 0.001, alpha0=partial.alpha, tol=1e-6)
    cold = smo_solver.solve_classification(x, y, 0.001, tol=1e-6)
    assert warm.steps < cold.steps - 400
    np.testing.assert_allclose(warm.dual_objective, cold.dual_objective, rtol=1e-6)


def test_callback_exception_propagates():
    x, y, _ = make_data(200)

    def cb(info):
        raise RuntimeError("stop!")

    with pytest.raises(RuntimeError, match="stop!"):
        smo_solver.solve_classification(x, y, 0.001, callback=cb, callback_period=1)


def test_input_validation():
    x, y, _ = make_data(20)
    with pytest.raises(ValueError, match="rows"):
        smo_solver.solve_classification(x, y[:-1], 1.0)
    with pytest.raises(ValueError, match="y"):
        smo_solver.solve_classification(x, y * 2, 1.0)
    with pytest.raises(ValueError, match="lambda"):
        smo_solver.solve_classification(x, y, -1.0)
    with pytest.raises(ValueError, match="kernel"):
        smo_solver.solve_classification(x, y, 1.0, kernel="sigmoid")
    with pytest.raises(ValueError, match="working set"):
        smo_solver.solve_classification(x, y, 1.0, working_set="foo")
    with pytest.raises(ValueError, match="contiguous"):
        smo_solver.solve_classification(np.asfortranarray(x), y, 1.0)


def bound_aware_measure(alpha, g, lo, hi, rho):
    """Brute-force ``r(alpha, rho) / rho`` of Strasdat, Def. 7.21 (O(n^2))."""
    up = np.minimum(hi - alpha, rho)  # possible increase of variable i
    down = np.minimum(alpha - lo, rho)  # possible decrease of variable j
    step = np.minimum(up[:, None], down[None, :])
    return float(np.max(step * (g[None, :] - g[:, None]), initial=0.0)) / rho


@pytest.mark.parametrize("lam", [0.1, 0.01])
def test_bound_aware_stopping_criterion(lam):
    x, y, _ = make_data(300, seed=4)
    kw = dict(gamma=0.5, tol=1e-3, shrinking_period=50)
    plain = smo_solver.solve_classification(x, y, lam, **kw)
    aware = smo_solver.solve_classification(x, y, lam, stopping="bound_aware", **kw)
    assert aware.converged
    assert aware.steps <= plain.steps
    # independent check of the criterion; hinge loss: g = ka - y, bounds [min(0, y), max(0, y)]
    g = aware.ka - y
    measure = bound_aware_measure(aware.alpha, g, np.minimum(0.0, y), np.maximum(0.0, y), lam)
    assert measure <= 1e-3 * (1 + 1e-6)
    # the bound-aware measure never exceeds the violation
    assert measure <= aware.violation + 1e-12
    # stopping early costs little accuracy
    tight = smo_solver.solve_classification(x, y, lam, gamma=0.5, tol=1e-8)
    assert abs(aware.dual_objective - tight.dual_objective) < 1e-4 * abs(tight.dual_objective)


def test_bound_aware_stopping_for_all_problem_types():
    x, y, t = make_data(200, seed=5)
    calls = {
        "regression": lambda **k: smo_solver.solve_regression(
            x, t, 0.05, epsilon=0.1, shrinking_period=50, **k
        ),
        "smoothed": lambda **k: smo_solver.solve_classification(
            x, y, 0.05, smoothing=0.2, shrinking_period=50, **k
        ),
        "least_squares": lambda **k: smo_solver.solve_least_squares(x, t, 0.05, **k),
    }
    for name, call in calls.items():
        plain = call(gamma=0.5)
        aware = call(gamma=0.5, stopping="bound_aware")
        assert aware.converged, name
        assert aware.steps <= plain.steps, name


def test_bound_aware_rho_and_validation():
    x, y, _ = make_data(100)
    res = smo_solver.solve_classification(x, y, 0.1, stopping="bound_aware", rho=0.5)
    assert res.converged
    with pytest.raises(ValueError, match="stopping"):
        smo_solver.solve_classification(x, y, 0.1, stopping="nope")
    with pytest.raises(ValueError, match="rho"):
        smo_solver.solve_classification(x, y, 0.1, stopping="bound_aware", rho=-1.0)
    with pytest.raises(ValueError, match="max_asum"):
        smo_solver.solve_classification(x, y, 0.1, stopping="bound_aware", max_asum=5.0)
    # the default criterion still supports the constraint
    assert smo_solver.solve_classification(x, y, 0.1, max_asum=5.0).converged
