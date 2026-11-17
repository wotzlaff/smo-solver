"""Convergence of SMO on the checkerboard problem (Strasdat, Sec. 10.1, 10.2.4-10.2.7, 10.2.9)

Data (Sec. 10.1): ``n`` points uniform on ``[0, 1]^2``, ``y = +1`` iff
``sin(4 pi x1) sin(4 pi x2) >= 0``, Gaussian kernel ``gamma = 30``, hinge loss (``smoothing = 0``)
or the quadratically smoothed hinge ``m_delta`` of Obs. 4.16 / Ex. 4.12 (``delta = smoothing``,
thesis: 1/4), ``c_i = 1``, working set rule MDP, no shrinking in the trace studies. For every
step ``k`` the callback (``callback_period=1``) maintains the dual gradient incrementally
(``O(n)``) and yields

* ``r``    violation ``r(alpha^k)``,
* ``tau``  ``|alpha^{k+1} - alpha^k| / sqrt(2)``,
* ``dpsi`` decrease ``psi(alpha^k) - psi(alpha^{k+1})`` (exact from the two changed coordinates),
* ``rt``   ``sqrt(2) r_k / |d^k|``, ``r_k = -grad psi(alpha^k)^T d^k``, ``d^k`` the step direction.

At thinned steps (logarithmic plus uniform) the ``O(n^2)`` quantities are added: the duality gap
``Delta*(alpha^k)`` (minimised over the bias), ``D = |alpha^k - alpha*|`` and
``V = psi(alpha^k) - psi(alpha*)`` (evaluated as ``(g* + b*)^T d + d^T H d / 2`` with
``d = alpha^k - alpha*`` to avoid cancellation); ``alpha*``: reference solution, ``r < 1e-12``.

Studies (CSV on stdout; ``summary`` and ``plot`` read the files)::

    python benchmarks/convergence_study.py relations | gzip > convergence_relations.csv.gz  # 10.2.4
    python benchmarks/convergence_study.py factor    | gzip > convergence_factor.csv.gz     # 10.2.5
    python benchmarks/convergence_study.py smooth    | gzip > convergence_smooth.csv.gz     # 10.2.6
    python benchmarks/convergence_study.py lambda    > convergence_lambda.csv               # 10.2.7
    python benchmarks/convergence_study.py size      > convergence_size.csv                 # 10.2.9
    python benchmarks/convergence_study.py size --smoke   # tiny version
    python benchmarks/convergence_study.py size --reps 2  # repetitions (the minimum time is used)
    python benchmarks/convergence_study.py summary benchmarks/results
    uv run --with matplotlib python benchmarks/convergence_study.py plot benchmarks/results OUTDIR
"""

from __future__ import annotations

import csv
import gzip
import os
import sys
import time
from pathlib import Path

# small matrix-vector products: BLAS threads only add contention (set before NumPy is loaded)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

import smo_solver  # noqa: E402
from smo_solver import smooth_max  # noqa: E402

GAMMA = 30.0
LAM = 1e-3
MAX_STEPS = 500_000
R_STOP = 1e-10
R_REF = 1e-12
FIT_RANGE = (1e-9, 1e-3)  # range of r(alpha) used for the factor fits (thesis 10.2.5)
SQRT2 = np.sqrt(2.0)

TRACE_FIELDS = ["study", "n", "lam", "gamma", "delta", "step", "r", "tau", "dpsi", "rt",
                "gap", "D", "V"]  # fmt: skip
LAMBDA_FIELDS = ["delta", "lam", "n", "steps", "status", "r", "n_sv", "n_free", "n_upper",
                 "seconds"]  # fmt: skip
SIZE_FIELDS = ["n", "lam", "tol", "steps", "seconds", "n_free", "n_sv", "status", "violation",
               "shrinks", "unshrinks", "min_active", "rep"]  # fmt: skip

# (measure, name in the CSV, exponent of r in the thesis relation v ~ c(lam) r^p, thesis fit)
MEASURES = [
    ("gap", "gap", 1, (5.770, -0.04)),
    ("tau", "tau", 1, (1.648, 0.958)),
    ("dpsi", "dpsi", 2, (0.427, 0.989)),
    ("rt", "rt", 1, (0.707, 0.014)),
    ("D", "D", 1, None),
    ("V", "V", 2, None),
]


# ---------------------------------------------------------------------------------------------
# data and problem


def checkerboard(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 2))
    y = np.where(np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) >= 0, 1.0, -1.0)
    return x, y


class Problem:
    """Dual ``psi(a) = a^T K a / (2 lam) + sum_i (q_i a_i^2 / 2 + p_i a_i)`` with ``sum a = 0``."""

    def __init__(self, n: int, lam: float, gamma: float, delta: float, seed: int = 0):
        self.n, self.lam, self.gamma, self.delta = n, lam, gamma, delta
        self.x, self.y = checkerboard(n, seed)
        # direct differences: the usual |x|^2 + |z|^2 - 2 x.z expansion loses ~1e-12 in K
        d2 = ((self.x[:, None, :] - self.x[None, :, :]) ** 2).sum(-1)
        self.K = np.exp(-gamma * d2)
        self.q = np.full(n, 2.0 * delta)
        self.p = -(1.0 + delta) * self.y
        self.lo = np.where(self.y > 0, 0.0, -1.0)
        self.hi = np.where(self.y > 0, 1.0, 0.0)

    def solve(self, **kw):
        kw.setdefault("working_set", "md")
        kw.setdefault("shrinking", False)
        return smo_solver.solve_classification(
            self.x, self.y, self.lam, gamma=self.gamma, smoothing=self.delta, **kw
        )

    def ka(self, alpha):
        return self.K @ alpha / self.lam

    def grad(self, alpha):
        return self.ka(alpha) + self.q * alpha + self.p

    def psi(self, alpha):
        return float(np.sum(0.5 * alpha * self.ka(alpha) + (0.5 * self.q * alpha + self.p) * alpha))

    def violation(self, alpha, g):
        return max(float(g[alpha > self.lo].max() - g[alpha < self.hi].min()), 0.0)

    def gap_direct(self, alpha):
        """Primal value (best bias) plus dual value; accurate only down to about 1e-12."""
        ka = self.ka(alpha)
        y, d = self.y, self.delta
        quad = 0.5 * float(alpha @ ka)
        dual = quad + float(np.sum((0.5 * self.q * alpha + self.p) * alpha))
        loss = np.maximum(1.0 - y[None, :] * (ka[None, :] + (y - ka)[:, None]), 0.0).sum(1)
        if d > 0.0:
            bs = np.linspace(-np.abs(ka).max() - 2, np.abs(ka).max() + 2, 20001)
            loss = np.array([smooth_max(1.0 - y * (ka + b), d).sum() for b in bs])
        return quad + float(loss.min()) + dual

    def gap(self, alpha):
        """Duality gap ``Delta*(alpha)`` minimised over the bias.

        The gap is evaluated as the sum of the Fenchel-Young terms
        ``(g_i + b)(a_i - a^_i) - q_i (a_i - a^_i)^2 / 2`` with ``a^_i`` the maximiser of the
        concave quadratic over ``[lo_i, hi_i]``. This avoids the cancellation between primal and
        dual objective (which limits their sum to about 1e-12). The sum is convex in ``b`` with
        derivative ``sum_i (a_i - a^_i)``; the minimum is found by bisection.
        """
        g, q = self.grad(alpha), self.q

        def parts(b):
            if self.delta > 0.0:
                ahat = np.clip(alpha - (g + b) / q, self.lo, self.hi)
            else:
                ahat = np.where(g + b < 0.0, self.hi, self.lo)
            return alpha - ahat

        lo_b, hi_b = -np.abs(g).max() - 2.0, np.abs(g).max() + 2.0
        for _ in range(100):
            b = 0.5 * (lo_b + hi_b)
            if parts(b).sum() > 0.0:
                hi_b = b
            else:
                lo_b = b
        b = 0.5 * (lo_b + hi_b)
        da = parts(b)
        return float(np.sum((g + b) * da - 0.5 * q * da * da))


class Reference:
    """Tight reference solution used for ``D_k`` and ``V_k``."""

    def __init__(self, prob: Problem, tol: float = R_REF):
        self.prob = prob
        res = prob.solve(tol=tol, max_steps=20_000_000)
        self.res = res
        self.alpha = np.asarray(res.alpha)
        self.g = prob.grad(self.alpha)
        self.b = res.b
        self.r = res.violation
        self.steps = res.steps
        self.dual = res.dual_objective

    def distances(self, alpha):
        p = self.prob
        d = alpha - self.alpha
        hd = p.K @ d / p.lam + p.q * d
        v = float((self.g + self.b) @ d + 0.5 * d @ hd)
        return float(np.linalg.norm(d)), max(v, 0.0)


# ---------------------------------------------------------------------------------------------
# trace of one run


def thinned_steps(total: int, n_log: int = 300, n_lin: int = 700) -> set[int]:
    pts = set(np.unique(np.geomspace(1, max(total, 2), n_log).astype(int)).tolist())
    pts |= set(np.unique(np.linspace(1, max(total, 2), n_lin).astype(int)).tolist())
    return pts


def trace(prob: Problem, ref: Reference, study: str, *, max_steps=MAX_STEPS, r_stop=R_STOP,
          cheap_stride: int = 25, expensive: bool = True, resync: int = 5000):  # fmt: skip
    """Run SMO with a per-step callback; returns the list of CSV rows (dicts)."""
    n = prob.n
    alpha_prev = np.zeros(n)
    state = {"g": prob.grad(alpha_prev), "alpha": alpha_prev, "drift": 0.0}
    rows: list[dict] = []
    # steps with O(n^2) quantities: unknown total length, so use a generous log grid up to max_steps
    exp_steps = thinned_steps(max_steps) if expensive else set()
    q, K, lam = prob.q, prob.K, prob.lam

    def emit(k, alpha, g, step_vec):
        r = prob.violation(alpha, g)
        idx = np.flatnonzero(step_vec)
        if len(idx) == 0:
            return
        # remove the rounding error of the invariant sum(alpha) = 0 (about 1e-17 per coordinate):
        # multiplied with g = O(1) it would swamp r_k * |d| and the decrease at small r
        dv = step_vec[idx]
        dv = dv - dv.mean()
        nrm = float(np.linalg.norm(dv))
        if nrm == 0.0:
            return
        lin = float(g[idx] @ dv)
        hess = K[np.ix_(idx, idx)] / lam + np.diag(q[idx])
        dec = -(lin + 0.5 * float(dv @ hess @ dv))
        row = {"study": study, "n": n, "lam": prob.lam, "gamma": prob.gamma, "delta": prob.delta,
               "step": k, "r": r, "tau": nrm / SQRT2, "dpsi": dec,
               "rt": SQRT2 * (-lin) / nrm}  # fmt: skip
        if k in exp_steps:
            row["gap"] = prob.gap(alpha)
            row["D"], row["V"] = ref.distances(alpha)
        if k in exp_steps or k % cheap_stride == 0:
            rows.append(row)

    def cb(info):
        k = info["step"]  # alpha is alpha^k; finalise quantities of step k-1
        alpha = info["alpha"]
        prev, g = state["alpha"], state["g"]
        step_vec = alpha - prev
        emit(k - 1, prev, g, step_vec)
        idx = np.flatnonzero(step_vec)
        g = g + K[:, idx] @ step_vec[idx] / lam
        g[idx] += q[idx] * step_vec[idx]
        if k % resync == 0:  # remove accumulated rounding errors
            g_exact = prob.grad(alpha)
            state["drift"] = max(state["drift"], float(np.abs(g - g_exact).max()))
            g = g_exact
        state["g"], state["alpha"] = g, alpha
        return prob.violation(alpha, g) < r_stop

    res = prob.solve(tol=r_stop * 1e-3, max_steps=max_steps, callback=cb, callback_period=1)
    # final iterate: the solver's own result; the last step's quantities are already in `rows`
    alpha = np.asarray(res.alpha)
    g = prob.grad(alpha)
    final = {"study": study, "n": n, "lam": prob.lam, "gamma": prob.gamma, "delta": prob.delta,
             "step": res.steps, "r": prob.violation(alpha, g), "gap": prob.gap(alpha)}  # fmt: skip
    final["D"], final["V"] = ref.distances(alpha)
    rows.append(final)
    return rows, state["drift"], res


def write_rows(w, rows):
    for row in rows:
        w.writerow(row)


def trace_study(study: str, settings, max_steps=MAX_STEPS):
    """Traces for a list of ``(n, lam, gamma, delta)`` settings, CSV on stdout."""
    w = csv.DictWriter(sys.stdout, TRACE_FIELDS)
    w.writeheader()
    for n, lam, gamma, delta in settings:
        t0 = time.perf_counter()
        prob = Problem(n, lam, gamma, delta)
        ref = Reference(prob)
        t1 = time.perf_counter()
        rows, drift, res = trace(prob, ref, study, max_steps=max_steps)
        write_rows(w, rows)
        sys.stdout.flush()
        print(
            f"{study} n={n} lam={lam:g} gamma={gamma:g} delta={delta:g}: ref {ref.steps} steps "
            f"(r={ref.r:.1e}, {t1 - t0:.1f}s), trace {res.steps} steps r={res.violation:.1e} "
            f"({time.perf_counter() - t1:.1f}s), rows={len(rows)}, gradient drift {drift:.1e}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------------------------
# studies


def study_relations():
    settings = [(1000, 1e-3, 30.0, 0.0), (1000, 1e-2, 30.0, 0.0), (1000, 1e-1, 30.0, 0.0),
                (1000, 1e-3, 10.0, 0.0), (1000, 1e-3, 100.0, 0.0)]  # fmt: skip
    trace_study("relations", settings)


def study_factor():
    lams = [10.0**e for e in np.arange(-5, -0.9, 0.5)]
    trace_study("factor", [(1000, float(lam), GAMMA, 0.0) for lam in lams])


def study_smooth():
    trace_study("smooth", [(1000, lam, GAMMA, 0.25) for lam in (1e-3, 1e-2, 1e-1)])


def study_lambda():
    """Support vectors, free variables and steps over lambda (hinge and smoothed hinge)."""
    w = csv.DictWriter(sys.stdout, LAMBDA_FIELDS)
    w.writeheader()
    for delta in (0.25, 0.0):
        for lam in [10.0**e for e in np.arange(-5, 0.1, 0.5)]:
            prob = Problem(1000, float(lam), GAMMA, delta)
            t0 = time.perf_counter()
            res = prob.solve(tol=R_STOP, max_steps=MAX_STEPS)
            sec = time.perf_counter() - t0
            a = np.asarray(res.alpha)
            eps = 1e-12
            free = (a > prob.lo + eps) & (a < prob.hi - eps)
            sv = np.abs(a) > eps
            w.writerow({"delta": delta, "lam": lam, "n": 1000, "steps": res.steps,
                        "status": res.status, "r": res.violation, "n_sv": int(sv.sum()),
                        "n_free": int(free.sum()), "n_upper": int((sv & ~free).sum()),
                        "seconds": sec})  # fmt: skip
            sys.stdout.flush()


def study_size(smoke: bool, ns=None, lams=None, reps: int = 1, cache_mb: float = 1024.0):
    """Steps, runtime and free variables over n (shrinking on, ``tol = 1e-8``, MDP)."""
    if ns is None:
        ns = (500, 1000, 2000) if smoke else (1000, 2000, 5000, 10000, 20000, 50000)
    if lams is None:
        lams = (1e-2, 1e-3) if smoke else (1e-1, 1e-2, 1e-3, 1e-4)
    w = csv.DictWriter(sys.stdout, SIZE_FIELDS)
    w.writeheader()
    for n in ns:
        x, y = checkerboard(n)
        for lam in lams:
            for rep in range(reps):
                t0 = time.perf_counter()
                res = smo_solver.solve_classification(
                    x, y, lam, gamma=GAMMA, tol=1e-8, working_set="md", shrinking=True,
                    cache_size_mb=cache_mb)  # fmt: skip
                sec = time.perf_counter() - t0
                a = np.asarray(res.alpha)
                lo = np.where(y > 0, 0.0, -1.0)
                hi = np.where(y > 0, 1.0, 0.0)
                free = (a > lo + 1e-12) & (a < hi - 1e-12)
                w.writerow({"n": n, "lam": lam, "tol": 1e-8, "steps": res.steps, "seconds": sec,
                            "n_free": int(free.sum()), "n_sv": int((np.abs(a) > 1e-12).sum()),
                            "status": res.status, "violation": res.violation,
                            "shrinks": res.shrinks, "unshrinks": res.unshrinks,
                            "min_active": res.min_active, "rep": rep})  # fmt: skip
                sys.stdout.flush()
                print(
                    f"n={n} lam={lam:g}: {res.steps} steps, {sec:.1f}s, free={int(free.sum())}",
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------------------------
# analysis of the CSV files


def result_file(d: Path, name: str) -> Path:
    """``d / name``, or its gzipped version ``name + ".gz"`` if only that exists."""
    f = d / name
    return f if f.exists() or not f.with_name(name + ".gz").exists() else f.with_name(name + ".gz")


def read_csv(path) -> list[dict[str, float | str]]:
    rows = []
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: _num(v) for k, v in row.items()})
    return rows


def _num(v: str):
    if v == "":
        return np.nan
    try:
        return float(v)
    except ValueError:
        return v


def column(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def groups(rows, *keys):
    out: dict[tuple, list] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def factor_fit(rows, name: str, power: int):
    """``c`` in ``v ~ c r^power`` over ``FIT_RANGE`` (``n_points``); two estimators.

    ``c_log``: geometric mean of ``v / r^power``; ``c_lin``: least squares through the origin
    (dominated by the largest ``r``); ``slope``: local log-log exponent of ``v`` over ``r``.
    """
    r = column(rows, "r")
    v = column(rows, name)
    m = (r >= FIT_RANGE[0]) & (r <= FIT_RANGE[1]) & np.isfinite(v) & (v > 0)
    if m.sum() < 5:
        return None
    r, v = r[m], v[m]
    x = r**power
    c_log = float(np.exp(np.mean(np.log(v / x))))
    c_lin = float((x @ v) / (x @ x))
    slope = float(np.polyfit(np.log(r), np.log(v), 1)[0])
    scatter = float(np.std(np.log(v / x)))  # spread of log(v / r^p)
    return {"c_log": c_log, "c_lin": c_lin, "slope": slope, "scatter": scatter, "n": int(m.sum())}


def loglog_fit(lams, cs):
    s, b = np.polyfit(np.log(lams), np.log(cs), 1)
    res = np.log(cs) - (s * np.log(lams) + b)
    r2 = 1.0 - float(res @ res) / float(np.var(np.log(cs)) * len(cs))
    return float(np.exp(b)), float(s), r2


def factor_table(rows, key="c_log"):
    """Per measure: lambda values, factors and the fit ``c ~ a lam^s``."""
    out = {}
    byl = groups(rows, "lam")
    for label, name, power, thesis in MEASURES:
        lams, cs = [], []
        for (lam,), g in sorted(byl.items()):
            f = factor_fit(g, name, power)
            if f is not None:
                lams.append(lam)
                cs.append(f[key])
        if len(lams) >= 3:
            a, s, r2 = loglog_fit(np.array(lams), np.array(cs))
            out[label] = {"lams": np.array(lams), "cs": np.array(cs), "a": a, "s": s, "r2": r2,
                          "thesis": thesis}  # fmt: skip
    return out


def summary(resdir: str):
    d = Path(resdir)
    f = result_file(d, "convergence_relations.csv")
    if f.exists():
        print("== 10.2.4 relations: local exponents of v over r (fit range), spread of log(v/r^p)")
        rows = read_csv(f)
        for (_n, lam, gamma, _delta), g in sorted(
            groups(rows, "n", "lam", "gamma", "delta").items()
        ):
            parts = []
            for label, name, power, _ in MEASURES:
                ft = factor_fit(g, name, power)
                if ft:
                    parts.append(f"{label}: c={ft['c_log']:.3g} slope={ft['slope']:.2f}")
            print(f"  lam={lam:g} gamma={gamma:g}: " + "; ".join(parts))
            gp, rr = column(g, "gap"), column(g, "r")
            ok = np.isfinite(gp) & (rr > 0) & (gp > 0) & (rr <= FIT_RANGE[1])
            ratio = gp[ok] / rr[ok]
            print(f"    gap/r for r <= 1e-3: min {ratio.min():.3g} max {ratio.max():.3g}")
    f = result_file(d, "convergence_factor.csv")
    if f.exists():
        rows = read_csv(f)
        for key, title in (("c_log", "geometric-mean factor"), ("c_lin", "least squares thru 0")):
            print(f"== 10.2.5 factor fits c ~ a lam^s ({title}); thesis in brackets")
            for label, t in factor_table(rows, key).items():
                th = t["thesis"]
                ths = f"   [thesis {th[0]:g} lam^{th[1]:g}]" if th else ""
                print(f"  {label:5s} c = {t['a']:.4g} lam^{t['s']:.3f}  R2={t['r2']:.3f}{ths}")
        print("== 10.2.5 per-lambda local exponents v ~ r^e (should be 1, 1, 2, 1, 1, 2)")
        for (lam,), g in sorted(groups(rows, "lam").items()):
            parts = []
            for label, name, power, _ in MEASURES:
                ft = factor_fit(g, name, power)
                parts.append(f"{label}={ft['slope']:.2f}" if ft else f"{label}=-")
            print(f"  lam={lam:.2g}: " + " ".join(parts))
        parts = []
        for label, name, power, _ in MEASURES:
            sc = [factor_fit(g, name, power) for g in groups(rows, "lam").values()]
            sc = [x["scatter"] / np.log(10) for x in sc if x]
            parts.append(f"{label}={np.mean(sc):.2f}")
        print("  mean spread (std of log10(v / r^p)) per measure: " + " ".join(parts))
        print("  (r_end, steps) per lambda: ", end="")
        for (lam,), g in sorted(groups(rows, "lam").items()):
            last = max(g, key=lambda r: r["step"])
            print(f"{lam:.1e}:{last['step']:.0f}/{last['r']:.1e}", end=" ")
        print()
        tab = factor_table(rows, "c_log")
        if all(k in tab for k in ("gap", "rt")):
            ratio = tab["gap"]["cs"] / 1.0
            print(f"  gap/r factor range {ratio.min():.3g}..{ratio.max():.3g}; "
                  f"r/gap ~ {np.mean(1 / ratio):.3f} (thesis 0.173)")  # fmt: skip
    f = result_file(d, "convergence_smooth.csv")
    if f.exists():
        print("== 10.2.6 smoothed hinge (delta = 1/4): local exponents over r")
        rows = read_csv(f)
        for (lam,), g in sorted(groups(rows, "lam").items()):
            parts = []
            for label, name, power, _ in MEASURES:
                ft = factor_fit(g, name, power)
                if ft:
                    parts.append(f"{label}: slope={ft['slope']:.2f} c={ft['c_log']:.3g}")
            print(f"  lam={lam:g}: " + "; ".join(parts))
    f = result_file(d, "convergence_lambda.csv")
    if f.exists():
        print("== 10.2.7 support vectors, free variables, steps")
        for (delta,), g in sorted(groups(read_csv(f), "delta").items()):
            print(f"  delta={delta:g}")
            for r in g:
                print(
                    f"    lam={r['lam']:.1e}: sv={r['n_sv']:.0f} (upper bound {r['n_upper']:.0f}) "
                    f"free={r['n_free']:.0f} steps={r['steps']:.0f} {r['status']} r={r['r']:.1e}"
                )
    f = result_file(d, "convergence_size.csv")
    if f.exists():
        print("== 10.2.9 size: log-log exponents (thesis: steps n^0.6, time n^1.6)")
        rows = read_csv(f)
        for (lam,), g in sorted(groups(rows, "lam").items()):
            ns = sorted({r["n"] for r in g})
            if len(ns) < 2:
                continue
            st = [np.mean([r["steps"] for r in g if r["n"] == n]) for n in ns]
            tm = [np.min([r["seconds"] for r in g if r["n"] == n]) for n in ns]
            fr = [np.mean([r["n_free"] for r in g if r["n"] == n]) for n in ns]
            e = [float(np.polyfit(np.log(ns), np.log(v), 1)[0]) for v in (st, tm, fr)]
            print(f"  lam={lam:g}: n={[int(v) for v in ns]} steps~n^{e[0]:.2f} time~n^{e[1]:.2f} "
                  f"free~n^{e[2]:.2f}")  # fmt: skip


# ---------------------------------------------------------------------------------------------
# plots


def plot(resdir: str, outdir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, out = Path(resdir), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip
    cmap = plt.get_cmap("viridis")

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(out / name)
        plt.close(fig)
        print(out / name)

    f = result_file(d, "convergence_relations.csv")
    if f.exists():  # Fig. 10.2.6
        rows = read_csv(f)
        gs = sorted(groups(rows, "n", "lam", "gamma", "delta").items())
        fig, ax = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True)
        specs = [("r", 1, "r(alpha)"), ("tau", 1, "tau~"), ("dpsi", 0.5, "sqrt(dpsi)"),
                 ("D", 1, "D_k"), ("V", 0.5, "sqrt(V_k)")]  # fmt: skip
        for i, ((_n, lam, gamma, _delta), g) in enumerate(gs):
            gap = column(g, "gap")
            for a, (name, pw, lab) in zip(ax, specs, strict=True):
                v = column(g, name)
                m = np.isfinite(gap) & np.isfinite(v) & (v > 0) & (gap > 0)
                a.loglog(gap[m], v[m] ** pw, ".", ms=3, color=cmap(i / max(len(gs) - 1, 1)),
                         label=f"lam={lam:g}, gamma={gamma:g}")  # fmt: skip
                a.set_ylabel(lab)
                a.set_xlabel("gap")
        for a in ax:
            lim = np.array([1e-9, 1e2])
            a.plot(lim, lim, "k--", lw=0.8)
        ax[0].legend(fontsize=7)
        fig.suptitle("Fig. 10.2.6: measures versus primal-dual gap (hinge, MDP)")
        finish(fig, "fig_10_2_6_relations.png")

    f = result_file(d, "convergence_factor.csv")
    if f.exists():  # Fig. 10.2.7
        rows = read_csv(f)
        tab = factor_table(rows)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for a, labels in zip(ax, (("gap", "tau", "dpsi", "rt"), ("D", "V")), strict=True):
            for i, label in enumerate(labels):
                t = tab.get(label)
                if t is None:
                    continue
                col = f"C{i}"
                a.loglog(t["lams"], t["cs"], "o", color=col)
                a.loglog(t["lams"], t["a"] * t["lams"] ** t["s"], "-", color=col,
                         label=f"{label}: {t['a']:.3g} lam^{t['s']:.3f}")  # fmt: skip
                if t["thesis"]:
                    th = t["thesis"]
                    a.loglog(t["lams"], th[0] * t["lams"] ** th[1], ":", color=col)
            a.set_xlabel("lambda")
            a.set_ylabel("c(lambda)")
            a.legend(fontsize=8)
        ax[0].set_title("tight relations (dotted: thesis fit)")
        ax[1].set_title("loose relations (V_k against r^2)")
        fig.suptitle("Fig. 10.2.7: factor c(lambda) in v ~ c r^p, gamma = 30")
        finish(fig, "fig_10_2_7_factor.png")

    f = result_file(d, "convergence_smooth.csv")
    if f.exists():  # Fig. 10.2.8
        rows = read_csv(f)
        g = groups(rows, "lam")[(1e-3,)] if (1e-3,) in groups(rows, "lam") else rows
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        step = column(g, "step")
        for name, lab in (("r", "r"), ("gap", "gap"), ("D", "D_k"), ("V", "V_k")):
            v = column(g, name)
            m = np.isfinite(v) & (v > 0)
            ax[0].loglog(step[m], v[m], ".", ms=3, label=lab)
        ax[0].set_xlabel("step")
        ax[0].legend()
        ax[0].set_title("(a) development (lam = 1e-3, delta = 1/4)")
        r = column(g, "r")
        for a, (name, pw, ref) in zip(ax[1:], (("gap", 2, "r^2"), ("D", 1, "r")), strict=True):
            v = column(g, name)
            m = np.isfinite(v) & (v > 0) & (r > 0)
            a.loglog(r[m], v[m], ".", ms=3, label=name)
            xs = np.array([r[m].min(), r[m].max()])
            c = np.exp(np.mean(np.log(v[m] / r[m] ** pw)))
            a.loglog(xs, c * xs**pw, "k--", lw=0.8, label=f"{c:.3g} {ref}")
            a.set_xlabel("r(alpha)")
            a.set_ylabel(name)
            a.legend()
        ax[1].set_title("(b) gap against r")
        ax[2].set_title("(b) D_k against r")
        fig.suptitle("Fig. 10.2.8: smoothed hinge, MDP")
        finish(fig, "fig_10_2_8_smooth.png")

    f = result_file(d, "convergence_lambda.csv")
    if f.exists():  # Fig. 10.2.9
        rows = read_csv(f)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, ((delta,), g) in enumerate(sorted(groups(rows, "delta").items())):
            lam = column(g, "lam")
            ax[0].semilogx(
                lam, column(g, "n_sv"), "o-", color=f"C{i}", label=f"SV, delta={delta:g}"
            )
            ax[0].semilogx(lam, column(g, "n_free"), "s--", color=f"C{i}",
                           label=f"free, delta={delta:g}")  # fmt: skip
            ax[1].loglog(lam, column(g, "steps"), "o-", color=f"C{i}", label=f"delta={delta:g}")
        ax[0].set_xlabel("lambda")
        ax[0].set_ylabel("number of variables")
        ax[0].legend(fontsize=8)
        ax[1].set_xlabel("lambda")
        ax[1].set_ylabel("steps")
        ax[1].legend()
        ax[0].set_title("(a) support vectors and free variables")
        ax[1].set_title("(b) steps to r < 1e-10 (cap 500000)")
        fig.suptitle("Fig. 10.2.9: dependence on lambda (n = 1000)")
        finish(fig, "fig_10_2_9_lambda.png")

    f = result_file(d, "convergence_size.csv")
    if f.exists():  # Fig. 10.2.12 / 10.2.13
        rows = read_csv(f)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for i, ((lam,), g) in enumerate(sorted(groups(rows, "lam").items())):
            ns = sorted({r["n"] for r in g})
            for a, key, fn in zip(
                ax, ("steps", "seconds", "n_free"), (np.mean, np.min, np.mean), strict=True
            ):
                a.loglog(ns, [fn([r[key] for r in g if r["n"] == n]) for n in ns], "o-",
                         color=f"C{i}", label=f"lam={lam:g}")  # fmt: skip
        titles = ("steps (Fig. 10.2.12a)", "runtime in s (10.2.12b)", "free variables (10.2.13)")
        for a, t in zip(ax, titles, strict=True):
            a.set_xlabel("n")
            a.set_title(t)
        ax[0].legend(fontsize=8)
        fig.suptitle("Figs. 10.2.12/13: increasing problem size (shrinking, tol 1e-8)")
        finish(fig, "fig_10_2_12_size.png")


def main(argv):
    cmd = argv[0] if argv else ""
    if cmd == "relations":
        study_relations()
    elif cmd == "factor":
        study_factor()
    elif cmd == "smooth":
        study_smooth()
    elif cmd == "lambda":
        study_lambda()
    elif cmd == "size":
        reps = int(argv[argv.index("--reps") + 1]) if "--reps" in argv else 1
        study_size("--smoke" in argv, reps=reps)
    elif cmd == "summary" and len(argv) > 1:
        summary(argv[1])
    elif cmd == "plot" and len(argv) > 2:
        plot(argv[1], argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
