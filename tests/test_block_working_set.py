"""Larger working sets against independent primal/dual convex programs and invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import smo_solver
from smo_solver import kernel_matrix

cp = pytest.importorskip("cvxpy")


def data(n=24, seed=19):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    targets = np.sin(x[:, 0]) + 0.3 * x[:, 1]
    labels = np.where(targets > 0, 1.0, -1.0)
    weights = rng.uniform(0.5, 2.0, n)
    weights[::7] = 0.0
    return x, targets, labels, weights


def smooth_hinge(t, delta):
    if delta == 0:
        return cp.pos(t)
    v = cp.Variable(t.shape)
    return cp.pos(v) + cp.square(t + delta - v) / (4 * delta)


def primal_oracle(k, target, weight, lam, kind, delta, bound):
    n = len(target)
    eig, vec = np.linalg.eigh(k)
    factor = vec * np.sqrt(np.maximum(eig, 0.0))
    u, bias = cp.Variable(n), cp.Variable()
    f = factor @ u + bias
    objective = lam / 2 * cp.sum_squares(u)
    c = cp.Variable(nonneg=True) if bound is not None else 0.0
    if bound is not None:
        objective += bound * c
    if kind == "classification":
        losses = smooth_hinge(1 - cp.multiply(target, f) - c, delta)
    elif kind == "regression":
        losses = smooth_hinge(target - f - 0.1 - c, delta) + smooth_hinge(
            f - target - 0.1 - c, delta
        )
    else:
        losses = cp.square(f - target) / 2
    problem = cp.Problem(cp.Minimize(objective + weight @ losses))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    assert problem.status == cp.OPTIMAL
    return problem.value


@pytest.mark.parametrize("size", [4, 8, 16])
@pytest.mark.parametrize(
    "kind,delta,bound",
    [
        ("classification", 0.0, None),
        ("classification", 0.2, 2.0),
        ("regression", 0.15, 1.5),
        ("least_squares", 0.0, None),
    ],
)
def test_block_matches_independent_primal(size, kind, delta, bound):
    x, targets, labels, weights = data()
    y = labels if kind == "classification" else targets
    lam, gamma = 0.15, 0.7
    kwargs = dict(
        weights=weights,
        gamma=gamma,
        tol=1e-8,
        max_steps=50_000,
        time_limit=3.0,
        working_set_size=size,
        cache_size_mb=0.0,
        shrinking=True,
        shrinking_period=7,
    )
    if kind != "least_squares":
        kwargs.update(smoothing=delta, max_asum=bound)
    else:
        kwargs.pop("shrinking")
        kwargs.pop("shrinking_period")
    result = getattr(smo_solver, f"solve_{kind}")(x, y, lam, **kwargs)
    ref = primal_oracle(kernel_matrix(x, x, gamma=gamma), y, weights, lam, kind, delta, bound)
    assert result.converged
    assert result.violation <= 1e-8 * (1 + 1e-4)
    assert_allclose(-result.dual_objective, ref, rtol=2e-6, atol=2e-7)
    assert_allclose(result.primal_objective, ref, rtol=2e-6, atol=2e-7)
    assert abs(result.alpha.sum()) <= 1e-9
    if kind == "regression":
        assert_allclose(result.alpha[np.r_[weights == 0, weights == 0]], 0)
    else:
        assert_allclose(result.alpha[weights == 0], 0)
    if bound is not None:
        assert abs(result.alpha).sum() <= bound + 1e-9


@pytest.mark.parametrize("size", [4, 8, 16])
@pytest.mark.parametrize("bound", [None, 0.0, 1.2])
def test_custom_singular_dual_and_constraint_activation(size, bound):
    # Rank-one kernel, duplicate kernel indices, zero curvature and fixed variables.
    x = np.array([[-1.0], [0.2], [1.0]])
    indices = np.array([0, 1, 2, 0, 1, 2])
    sign = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    lo = np.array([0.0, 0.0, 0.0, -1.0, 0.0, -1.0])
    hi = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q = np.zeros(6)
    p = np.array([-0.7, 0.0, -1.2, 0.4, 0.0, 0.9])
    k = kernel_matrix(x, x, kernel="linear")[indices][:, indices]
    a = cp.Variable(6)
    constraints = [cp.sum(a) == 0, a >= lo, a <= hi]
    if bound is not None:
        constraints.append(sign @ a <= bound)
    problem = cp.Problem(cp.Minimize(cp.quad_form(a, cp.psd_wrap(k)) / 0.6 + p @ a), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10)
    result = smo_solver.solve_dual(
        x,
        0.3,
        q,
        p,
        lo,
        hi,
        kernel="linear",
        kernel_index=indices,
        sign=sign,
        max_asum=bound,
        working_set_size=size,
        tol=1e-9,
        max_steps=50_000,
        time_limit=3.0,
        cache_size_mb=0,
    )
    assert result.converged
    assert_allclose(result.dual_objective, problem.value, atol=2e-7)
    assert abs(result.alpha.sum()) <= 1e-10
    assert np.all(result.alpha >= lo) and np.all(result.alpha <= hi)
    if bound is not None:
        assert sign @ result.alpha <= bound + 1e-10


@pytest.mark.parametrize("size", [4, 8, 16])
def test_callback_monotonicity_warm_start_and_permutations(size):
    x, targets, _, _ = data(100)
    history = []

    def callback(info):
        history.append(info)
        return info["step"] >= 12

    kwargs = dict(
        working_set_size=size,
        smoothing=0.2,
        max_asum=3.0,
        gamma=0.4,
        tol=1e-8,
        max_steps=50_000,
        time_limit=3.0,
        shrinking_period=3,
        cache_size_mb=0,
    )
    partial = smo_solver.solve_regression(
        x, targets, 0.02, callback=callback, callback_period=3, **kwargs
    )
    assert partial.status == "interrupted"
    assert [p["step"] for p in history] == [3, 6, 9, 12]
    values = np.array([p["dual_objective"] for p in history])
    assert np.max(np.diff(values)) <= 1e-9
    for info in history:
        a = info["alpha"]
        assert abs(a.sum()) <= 1e-9
        assert abs(a).sum() <= 3 + 1e-9
        assert np.all((a[: len(x)] >= 0.0) & (a[: len(x)] <= 1.0))
        assert np.all((a[len(x) :] >= -1.0) & (a[len(x) :] <= 0.0))
    warm = smo_solver.solve_regression(x, targets, 0.02, alpha0=partial.alpha, **kwargs)
    cold = smo_solver.solve_regression(x, targets, 0.02, **kwargs)
    assert warm.converged and cold.converged
    assert_allclose(warm.dual_objective, cold.dual_objective, atol=1e-7)
    assert_allclose(warm.ka + warm.b, cold.ka + cold.b, atol=2e-6)


def test_explicit_two_is_unchanged():
    x, targets, _, _ = data(80)
    default = smo_solver.solve_regression(x, targets, 0.05, smoothing=0.1)
    two = smo_solver.solve_regression(x, targets, 0.05, smoothing=0.1, working_set_size=2)
    assert_allclose(default.alpha, two.alpha, rtol=0, atol=0)
    assert default.steps == two.steps and default.cache_misses == two.cache_misses


@pytest.mark.parametrize("size", [4, 8])
def test_infinite_bound_matches_no_bound(size):
    x, targets, _, _ = data(40)
    kwargs = dict(
        working_set_size=size, smoothing=0.2, gamma=0.4, tol=1e-8, max_steps=50_000, time_limit=3.0
    )
    unconstrained = smo_solver.solve_regression(x, targets, 0.1, **kwargs)
    infinite = smo_solver.solve_regression(x, targets, 0.1, max_asum=np.inf, **kwargs)
    assert unconstrained.converged and infinite.converged
    assert np.isfinite(infinite.primal_objective) and np.isfinite(infinite.gap)
    assert_allclose(infinite.dual_objective, unconstrained.dual_objective, atol=1e-8)
    assert_allclose(infinite.ka + infinite.b, unconstrained.ka + unconstrained.b, atol=2e-7)


@pytest.mark.parametrize("size", [4, 5, 8, 16])
def test_active_inequality_is_released(size):
    # Warm start lies on s'a=A; the exact minimizer lies strictly inside that inequality.
    # A block solve that incorrectly treats the active inequality as a permanent equality fails.
    optimum = np.array([0.1, 0.2, -0.15, -0.15])
    start = np.array([0.5, 0.5, -0.5, -0.5])
    sign = np.array([1.0, 1.0, -1.0, -1.0])
    result = smo_solver.solve_dual(
        np.zeros((4, 1)),
        1.0,
        np.ones(4),
        -optimum,
        np.array([0.0, 0.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
        kernel="linear",
        sign=sign,
        max_asum=2.0,
        alpha0=start,
        working_set_size=size,
        tol=1e-10,
        cache_size_mb=0,
        max_steps=1000,
        time_limit=3.0,
    )
    assert result.converged
    assert_allclose(result.alpha, optimum, atol=1e-9)
    assert sign @ result.alpha < 2.0 - 1.0


@pytest.mark.parametrize("size", [0, 1, -1, 65])
def test_invalid_working_set_sizes(size):
    x, _, labels, _ = data(10)
    with pytest.raises((ValueError, OverflowError), match="working_set_size|negative"):
        smo_solver.solve_classification(x, labels, 1.0, working_set_size=size)


def test_odd_size_and_bound_aware_stop():
    x, _, labels, _ = data(80)
    result = smo_solver.solve_classification(
        x,
        labels,
        0.1,
        working_set_size=7,
        stopping="bound_aware",
        tol=1e-6,
        shrinking_period=5,
        max_steps=50_000,
        time_limit=3.0,
    )
    assert result.converged
    assert result.bound_aware <= 1e-6 * (1 + 1e-4)


@pytest.mark.parametrize("size", [4, 8, 16])
def test_saved_small_lambda_nu_regression(size):
    # Exact LCG output of saved proptest seed1519. Pair SMO was excluded from the randomized
    # suite at these parameters because it needs around a million zig-zagging updates.
    x = np.array(
        [
            [-0.33201509919734706, 0.22237521967956664],
            [-0.35341405483857236, 0.48234369492998896],
            [0.07660345206527153, 1.3903624714490896],
            [-0.3488088560358985, -0.47056472997740373],
            [0.5596777628764761, 1.1742080323217992],
        ]
    )
    target = np.array(
        [
            -0.17288565267883627,
            0.15904852618645093,
            -0.6129115132559828,
            -1.8577536477187102,
            0.5731978940142429,
        ]
    )
    weights = np.array(
        [
            2.1143428432757743,
            1.7801047290834415,
            0.6400150712573394,
            0.7156240302190875,
            2.099654271249062,
        ]
    )
    bound = 1.539719423248258
    result = smo_solver.solve_regression(
        x,
        target,
        10**-3.9700411425769886,
        weights=weights,
        epsilon=0.05,
        gamma=4.863234423924022,
        smoothing=0.09944635523522093,
        max_asum=bound,
        shrinking=False,
        working_set_size=size,
        tol=1e-7,
        max_steps=5000,
        time_limit=3.0,
    )
    assert result.converged
    assert abs(result.alpha.sum()) <= 1e-10
    assert abs(result.alpha).sum() <= bound + 1e-10
    assert abs(result.gap) <= 1e-6


@pytest.mark.parametrize("kind", ["SmoSVC", "SmoSVR", "SmoLSSVR"])
def test_estimator_exposes_block_size(kind):
    from sklearn.base import clone

    x, targets, labels, _ = data(30)
    y = labels if kind == "SmoSVC" else targets
    est = getattr(smo_solver, kind)(working_set_size=8, lam=0.1, tol=1e-7)
    assert clone(est).get_params()["working_set_size"] == 8
    est.fit(x, y)
    assert np.all(np.isfinite(est.predict(x)))


def test_full_sixty_four_variable_block_matches_analytic_solution():
    # Exercise the actual upper block size, rather than clipping 64 to a tiny problem.
    optimum = np.linspace(-1.0, 1.0, 64)
    result = smo_solver.solve_dual(
        np.zeros((64, 1)),
        1.0,
        np.ones(64),
        -optimum,
        np.full(64, -2.0),
        np.full(64, 2.0),
        kernel="linear",
        working_set_size=64,
        tol=1e-10,
        time_limit=3,
        max_steps=100,
        cache_size_mb=0,
    )
    assert result.converged
    assert_allclose(result.alpha, optimum, atol=1e-9)
    assert abs(result.alpha.sum()) < 1e-10
