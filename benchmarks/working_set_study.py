"""Working set selection on the checkerboard problem (Strasdat, Sec. 10.1 and 10.2.1-10.2.3)

The study of the dissertation (prototype in Python) is repeated with the Rust solver. Data:
``n = 1000`` points uniform on ``[0, 1]^2``, ``y = +1`` iff ``sin(4 pi x1) sin(4 pi x2) >= 0``,
Gaussian kernel, hinge loss, ``c_i = 1``. The base setting is ``lambda = 1e-3``, ``gamma = 30``;
``lam1e-2``, ``lam1e-4`` and ``gamma10`` vary one parameter each.

Five selection rules are compared: the maximal violating pair ``mvp`` (Def. 7.6), the
bound-aware pair ``bamvp`` realizing ``r_diamond(alpha, lambda)`` (Def. 7.21), ``mud`` (Def.
7.27), ``md`` (Def. 7.28) and the total maximal decrease pair ``tmd`` (Def. 7.29, all ``O(n^2)``
pairs). ``mvp``, ``mud`` and ``md`` are the rules of the Rust solver (``shrinking=False``,
observed through the callback); ``bamvp`` and ``tmd`` do not exist there and run in a small
NumPy reference SMO (precomputed kernel matrix, the same clipped two-variable step as the
solver), which is validated against the Rust iterates.

Per step (``tracked`` through all steps, stored thinned) the file ``working_set_traj.csv.gz`` holds

* ``r``, ``rba`` (= ``r_diamond(alpha, lambda)``), ``gap`` (``Delta*``, bias at the midpoint,
  Obs. 7.4), ``D`` (= ``||alpha - alpha*||``), ``V`` (= ``psi(alpha) - psi(alpha*)``) and their
  running minima ``*_min``,
* the step quantities of Sec. 10.2: ``eps`` (maximal step ``eps^{ij}``), ``viol`` (``b^-_i -
  b^+_j`` of the chosen pair), ``rk`` (``-grad psi^T d`` with ``d = min(eps, 1) (e^i - e^j)``,
  i.e. ``rho_0 = 1``), ``rtilde`` (``sqrt(2) rk / ||d||``), ``tau`` (``||alpha^{k+1} - alpha^k||
  / sqrt(2)``) and ``decrease`` (``psi(alpha^k) - psi(alpha^{k+1})``).

``working_set_hits.csv`` has the exact first step at which each measure drops below 1e-1 ...
1e-10, ``working_set_timing.csv`` wall times (Rust solver: time to tolerance, minimum of 3 runs;
reference SMO: time per step) and ``working_set_validation.csv`` the comparison of the reference
SMO with the solver.

Usage::

    python benchmarks/working_set_study.py run [--out DIR] [--settings base,lam1e-2] [--quick]
    python benchmarks/working_set_study.py summary [DIR]
    uv run --with matplotlib python benchmarks/working_set_study.py plot OUTDIR [--data DIR]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import time
from pathlib import Path

import numpy as np

import smo_solver
from smo_solver import kernel_matrix

N = 1000
SEED = 0
TAU = 1e-12  # lower bound of the curvature, as in the solver
RULES = ["mvp", "bamvp", "mud", "md", "tmd"]
RUST_RULES = ["mvp", "mud", "md"]
SETTINGS = {  # name -> (lambda, gamma)
    "base": (1e-3, 30.0),
    "lam1e-2": (1e-2, 30.0),
    "lam1e-4": (1e-4, 30.0),
    "gamma10": (1e-3, 10.0),
}
# stop when r <= target, with a step cap (the O(n^2) rules are expensive per step)
TARGET = {"mvp": 1e-8, "mud": 1e-8, "md": 1e-8, "bamvp": 1e-8, "tmd": 1e-6}
CAP = {"mvp": 2_000_000, "mud": 2_000_000, "md": 2_000_000, "bamvp": 600_000, "tmd": 20_000}
THRESHOLDS = [10.0**-k for k in range(1, 11)]
MEASURES = ["r", "rba_lam", "gap", "D", "V"]
THIN_ALL, THIN_FACTOR = 1000, 1.01  # store all steps up to 1000, then log-spaced

TRAJ_FIELDS = [
    "setting",
    "rule",
    "step",
    "r",
    "rba",
    "rba_lam",
    "gap",
    "D",
    "V",
    "r_min",
    "rba_min",
    "gap_min",
    "D_min",
    "V_min",
    "eps",
    "viol",
    "rk",
    "rtilde",
    "tau",
    "decrease",
]
HIT_FIELDS = ["setting", "rule", "measure", "threshold", "step"]
TIMING_FIELDS = ["setting", "rule", "impl", "kind", "threshold", "steps", "seconds", "us_per_step"]
VALID_FIELDS = [
    "setting",
    "rule",
    "steps",
    "first_pair_mismatch",
    "max_alpha_diff",
    "max_obj_diff",
    "end_alpha_diff",
    "r_diff",
    "rba_lam_diff",
    "gap_diff",
    "dual_diff",
]


# ----------------------------------------------------------------------------------------------
# problem and measures
# ----------------------------------------------------------------------------------------------


def make_data(n: int = N, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 2))
    y = np.where(np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) >= 0, 1.0, -1.0)
    return x, y


class Problem:
    """Dual hinge-loss SVM ``psi(a) = a^T K a / (2 lam) - y^T a`` with ``a_i in [0, 1]`` (``y_i = +1``)
    or ``[-1, 0]`` (``y_i = -1``), gradient ``g = K a / lam - y`` and ``sum a = 0``."""  # noqa: E501

    def __init__(self, x, y, lam, gamma):
        self.x, self.y, self.lam, self.gamma = x, y, lam, gamma
        self.n = len(y)
        self.K = np.ascontiguousarray(kernel_matrix(x, x, gamma=gamma))
        self.lo = np.where(y > 0, 0.0, -1.0)
        self.hi = np.where(y > 0, 1.0, 0.0)
        # curvature of the pair (i, j): (K_ii + K_jj - 2 K_ij) / lam (q = 0 for the hinge loss)
        self.eta = np.maximum(
            (np.diag(self.K)[:, None] + np.diag(self.K)[None, :] - 2.0 * self.K) / lam, TAU
        )

    def gradient(self, a):
        return self.K @ a / self.lam - self.y

    def psi(self, a, g=None):
        g = self.gradient(a) if g is None else g
        return 0.5 * float(a @ g) - 0.5 * float(self.y @ a)

    def extremes(self, a, g):
        """``max g`` over variables that can decrease and ``min g`` over those that can increase."""
        dmax = g[a > self.lo].max(initial=-np.inf)
        umin = g[a < self.hi].min(initial=np.inf)
        return dmax, umin

    def violation(self, a, g):
        dmax, umin = self.extremes(a, g)
        return max(dmax - umin, 0.0)

    def gap(self, a, g, bias="mid"):
        """Primal-dual gap of the primal point induced by ``a`` (Obs. 7.4).

        ``bias='mid'``: ``b = -(max + min)/2`` of the extremal gradients (the thesis); ``'free'``: mean over
        the free variables (the solver, see ``SmoResult.gap``)."""  # noqa: E501
        dmax, umin = self.extremes(a, g)
        free = (a > self.lo) & (a < self.hi)
        if bias == "free" and free.any():
            b = -float(g[free].mean())
        elif np.isfinite(dmax) and np.isfinite(umin):
            b = -0.5 * (dmax + umin)
        else:
            b = -(dmax if np.isfinite(dmax) else umin)
        ka = g + self.y
        hinge = np.maximum(0.0, 1.0 - self.y * (ka + b)).sum()
        return float(a @ ka + hinge - self.y @ a)

    def bound_aware(self, a, g, rho, pair=False):
        """``r_diamond(a, rho)`` in O(n log n) (Remark 7.22), optionally with a realizing pair ``(i, j)``
        (``i`` decreases, ``j`` increases)."""  # noqa: E501
        room_d, room_u = a - self.lo, self.hi - a
        dec = np.flatnonzero(room_d > 0)
        inc = np.flatnonzero(room_u > 0)
        best, arg = 0.0, None
        for fixed, other, sgn_f, sgn_o in ((dec, inc, 1.0, -1.0), (inc, dec, -1.0, 1.0)):
            if len(fixed) == 0 or len(other) == 0:
                continue
            room_f = np.minimum(room_d if sgn_f > 0 else room_u, rho)[fixed]
            room_o = np.minimum(room_d if sgn_o > 0 else room_u, rho)[other]
            val_f, val_o = sgn_f * g[fixed], sgn_o * g[other]
            order = np.argsort(-room_o, kind="stable")
            neg_sorted = -room_o[order]
            vals = val_o[order]
            pm = np.maximum.accumulate(vals)
            cnt = np.searchsorted(neg_sorted, -room_f, side="right")
            ok = cnt > 0
            if not ok.any():
                continue
            cand = np.where(ok, room_f * (pm[np.maximum(cnt - 1, 0)] + val_f), -np.inf)
            k = int(np.argmax(cand))
            if cand[k] > best:
                best = float(cand[k])
                if pair:
                    pos = np.arange(len(vals))
                    last = np.maximum.accumulate(np.where(vals == pm, pos, 0))
                    f_idx, o_idx = int(fixed[k]), int(other[order[last[cnt[k] - 1]]])
                    arg = (f_idx, o_idx) if sgn_f > 0 else (o_idx, f_idx)
        return (best, arg) if pair else best


# ----------------------------------------------------------------------------------------------
# NumPy reference SMO
# ----------------------------------------------------------------------------------------------


class RefSmo:
    """Plain SMO with the selection rules ``mvp``, ``bamvp``, ``mud``, ``md`` and ``tmd``."""

    def __init__(self, prob: Problem, rule: str):
        self.p, self.rule = prob, rule
        self.a = np.zeros(prob.n)
        self.g = -prob.y.copy()
        self.k = 0

    def mvp(self):
        p, a, g = self.p, self.a, self.g
        i0 = int(np.argmax(np.where(a > p.lo, g, -np.inf)))
        j1 = int(np.argmin(np.where(a < p.hi, g, np.inf)))
        return i0, j1

    def second_order(self):
        """Def. 7.27 / 7.28 with the tie-breaking of the solver (first candidate wins)."""
        p, a, g = self.p, self.a, self.g
        i0, j1 = self.mvp()
        unbounded = self.rule == "mud"

        def decrease(d, eta, tmax):
            if unbounded:
                return 0.5 * d * d / eta
            t = np.minimum(d / eta, tmax)
            return t * (d - 0.5 * eta * t)

        d1 = g[i0] - g
        ok1 = (a < p.hi) & (d1 > 0)
        dec1 = np.where(
            ok1, decrease(d1, p.eta[i0], np.minimum(a[i0] - p.lo[i0], p.hi - a)), -np.inf
        )
        d2 = g - g[j1]
        ok2 = (a > p.lo) & (d2 > 0)
        dec2 = np.where(
            ok2, decrease(d2, p.eta[j1], np.minimum(a - p.lo, p.hi[j1] - a[j1])), -np.inf
        )
        flat = np.empty(2 * p.n)
        flat[0::2], flat[1::2] = dec1, dec2
        k = int(np.argmax(flat))
        if not np.isfinite(flat[k]):
            return i0, j1
        return (i0, k // 2) if k % 2 == 0 else (k // 2, j1)

    def total_maximal_decrease(self):
        p, a, g = self.p, self.a, self.g
        d = g[:, None] - g[None, :]
        tmax = np.minimum((a - p.lo)[:, None], (p.hi - a)[None, :])
        t = np.minimum(d / p.eta, tmax)
        dec = t * (d - 0.5 * p.eta * t)
        dec[(d <= 0) | ((a <= p.lo)[:, None]) | ((a >= p.hi)[None, :])] = -np.inf
        k = int(np.argmax(dec))
        return divmod(k, p.n)

    def select(self):
        if self.rule == "mvp":
            return self.mvp()
        if self.rule == "bamvp":
            return self.p.bound_aware(self.a, self.g, self.p.lam, pair=True)[1]
        if self.rule in ("mud", "md"):
            return self.second_order()
        return self.total_maximal_decrease()

    def step(self) -> bool:
        """One SMO step; ``False`` if no violating pair exists."""
        p, a, g = self.p, self.a, self.g
        pair = self.select()
        if pair is None:
            return False
        i, j = pair
        d = g[i] - g[j]
        box_i, box_j = a[i] - p.lo[i], p.hi[j] - a[j]
        t = min(d / p.eta[i, j], box_i, box_j)
        if not t > 0:
            return False
        a[i] = p.lo[i] if t == box_i else a[i] - t
        a[j] = p.hi[j] if t == box_j else a[j] + t
        g += (t / p.lam) * (p.K[:, j] - p.K[:, i])
        self.k += 1
        return True


# ----------------------------------------------------------------------------------------------
# tracking of the measures along an iterate sequence
# ----------------------------------------------------------------------------------------------


class Tracker:
    """Receives the iterates ``alpha^0, alpha^1, ...`` (from the solver callback or the reference SMO),
    computes the measures of Sec. 10.2 at every step and stores a thinned version of them."""  # noqa: E501

    def __init__(self, prob: Problem, ref_alpha, setting: str, rule: str):
        self.p, self.setting, self.rule = prob, setting, rule
        self.alpha_ref = ref_alpha
        self.psi_ref = prob.psi(ref_alpha)
        self.prev = None  # (alpha, g, row) of the previous iterate
        self.k = -1
        self.rows: list[dict] = []
        self.next_emit = THIN_ALL + 1
        self.run_min = dict.fromkeys(["r", "rba", "gap", "D", "V"], np.inf)
        self.hits = {m: [None] * len(THRESHOLDS) for m in MEASURES}

    def _emit(self, row):
        k = row["step"]
        if k <= THIN_ALL or k >= self.next_emit:
            self.rows.append(row)
            if k >= self.next_emit:
                self.next_emit = max(math.ceil(k * THIN_FACTOR), k + 1)

    def push(self, alpha) -> float:
        """Processes the next iterate and returns its violation ``r``."""
        p = self.p
        alpha = np.array(alpha, dtype=float)
        self.k += 1
        if self.prev is None:
            g = p.gradient(alpha)
        else:
            a0, g0, row0 = self.prev
            nz = np.flatnonzero(alpha != a0)
            if len(nz) == 2 and self.k % 2000:
                g = g0 + p.K[:, nz] @ (alpha - a0)[nz] / p.lam
            else:
                g = p.gradient(alpha)
            self._finish_step(a0, g0, alpha, nz, row0)
        r = p.violation(alpha, g)
        rba = p.bound_aware(alpha, g, p.lam)
        gap = p.gap(alpha, g)
        dist = float(np.linalg.norm(alpha - self.alpha_ref))
        v = p.psi(alpha, g) - self.psi_ref
        cur = {"r": r, "rba": rba, "gap": gap, "D": dist, "V": v}
        for key, val in cur.items():
            self.run_min[key] = min(self.run_min[key], val)
        row = {
            "setting": self.setting,
            "rule": self.rule,
            "step": self.k,
            **cur,
            "rba_lam": rba / p.lam,
            **{f"{key}_min": self.run_min[key] for key in cur},
        }
        for m in MEASURES:
            val = row[m] if m != "V" else v
            for t, thr in enumerate(THRESHOLDS):
                if self.hits[m][t] is None and val <= thr:
                    self.hits[m][t] = self.k
        self.prev = (alpha, g, row)
        return r

    def _finish_step(self, a0, g0, a1, nz, row):
        """Step quantities of the transition ``alpha^k -> alpha^{k+1}`` (all ``nan`` for a no-op)."""  # noqa: E501
        p = self.p
        for key in ("eps", "viol", "rk", "rtilde", "tau", "decrease"):
            row[key] = math.nan
        if len(nz) == 2:
            i, j = (nz[0], nz[1]) if a1[nz[0]] < a0[nz[0]] else (nz[1], nz[0])
            t = float(a0[i] - a1[i])
            eps = min(a0[i] - p.lo[i], p.hi[j] - a0[j])
            viol = float(g0[i] - g0[j])
            eta = p.eta[i, j]
            row["eps"] = float(eps)
            row["viol"] = viol
            row["rk"] = min(eps, 1.0) * viol  # rho_0 = 1
            row["rtilde"] = math.sqrt(2.0) * row["rk"] / (math.sqrt(2.0) * min(eps, 1.0))
            row["tau"] = t
            row["decrease"] = t * (viol - 0.5 * eta * t)
        self._emit(row)

    def finish(self):
        if self.prev is not None:
            row = self.prev[2]
            for key in ("eps", "viol", "rk", "rtilde", "tau", "decrease"):
                row[key] = math.nan
            self.rows.append(row)  # the last iterate is always stored
            self.prev = None


def reference_solution(prob: Problem):
    r = smo_solver.solve_classification(
        prob.x,
        prob.y,
        prob.lam,
        gamma=prob.gamma,
        working_set="md",
        shrinking=False,
        tol=1e-12,
        max_steps=20_000_000,
    )
    return np.asarray(r.alpha), r


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------


def rust_trajectory(prob: Problem, tracker: Tracker, rule: str, target: float, cap: int):
    """Runs the solver with a per-step callback feeding the tracker; returns the steps made."""
    tracker.push(np.zeros(prob.n))
    done = [False]

    def cb(info):
        done[0] = tracker.push(info["alpha"]) <= target
        return done[0]

    res = smo_solver.solve_classification(
        prob.x,
        prob.y,
        prob.lam,
        gamma=prob.gamma,
        working_set=rule,
        shrinking=False,
        tol=0.0,
        max_steps=cap,
        callback=cb,
        callback_period=1,
    )
    if not done[0]:  # the callback sees steps 1 .. steps-1 only
        tracker.push(res.alpha)
    tracker.finish()
    return tracker.k


def ref_trajectory(prob: Problem, tracker: Tracker, rule: str, target: float, cap: int):
    smo = RefSmo(prob, rule)
    r = tracker.push(smo.a)
    while r > target and smo.k < cap and smo.step():
        r = tracker.push(smo.a)
    tracker.finish()
    return tracker.k


def validate(prob: Problem, setting: str, steps: int = 2000):
    """Reference SMO against the solver iterates (first ``steps`` steps of mvp, mud, md)."""
    out = []
    for rule in RUST_RULES:
        traj = []
        res = smo_solver.solve_classification(
            prob.x,
            prob.y,
            prob.lam,
            gamma=prob.gamma,
            working_set=rule,
            shrinking=False,
            tol=0.0,
            max_steps=steps,
            callback=lambda d: traj.append(d["alpha"]) and None,  # noqa: B023
            callback_period=1,
        )
        traj = [np.zeros(prob.n), *traj, np.asarray(res.alpha)]
        smo = RefSmo(prob, rule)
        a_diff = o_diff = 0.0  # largest deviation before the first differing pair
        mismatch = None
        prev = smo.a.copy()
        for k in range(1, len(traj)):
            smo.step()
            if mismatch is None and (
                set(np.flatnonzero(smo.a != prev)) != set(np.flatnonzero(traj[k] != traj[k - 1]))
            ):
                mismatch = k
            if mismatch is None:
                a_diff = max(a_diff, float(np.abs(smo.a - traj[k]).max()))
                o_diff = max(o_diff, abs(prob.psi(smo.a) - prob.psi(traj[k])))
            prev = smo.a.copy()
        end_diff = float(np.abs(smo.a - traj[-1]).max())
        a = np.asarray(res.alpha)
        g = prob.gradient(a)
        out.append(
            {
                "setting": setting,
                "rule": rule,
                "steps": len(traj) - 1,
                "first_pair_mismatch": "" if mismatch is None else mismatch,
                "max_alpha_diff": a_diff,
                "max_obj_diff": o_diff,
                "end_alpha_diff": end_diff,
                "r_diff": abs(prob.violation(a, g) - res.violation),
                "rba_lam_diff": abs(prob.bound_aware(a, g, prob.lam) / prob.lam - res.bound_aware),
                "gap_diff": abs(prob.gap(a, g, bias="free") - res.gap),
                "dual_diff": abs(prob.psi(a, g) - res.dual_objective),
            }
        )
    return out


def timing(prob: Problem, setting: str, quick: bool):
    rows = []
    reps = 1 if quick else 3
    for rule in RUST_RULES:
        for thr in (1e-3, 1e-5, 1e-8):
            best = None
            for _ in range(reps):
                t0 = time.perf_counter()
                res = smo_solver.solve_classification(
                    prob.x,
                    prob.y,
                    prob.lam,
                    gamma=prob.gamma,
                    working_set=rule,
                    shrinking=False,
                    tol=thr,
                    max_steps=20_000_000,
                )
                dt = time.perf_counter() - t0
                if best is None or dt < best[0]:
                    best = (dt, res.steps)
            rows.append(
                {
                    "setting": setting,
                    "rule": rule,
                    "impl": "rust",
                    "kind": "to_tol",
                    "threshold": thr,
                    "steps": best[1],
                    "seconds": best[0],
                    "us_per_step": 1e6 * best[0] / max(best[1], 1),
                }
            )
    # reference SMO: cost of the selection plus the update (no measures)
    for rule in RULES:
        steps = {"tmd": 150, "bamvp": 1500}.get(rule, 3000)
        best = None
        for _ in range(reps):
            smo = RefSmo(prob, rule)
            t0 = time.perf_counter()
            while smo.k < steps and smo.step():
                pass
            dt = time.perf_counter() - t0
            if best is None or dt < best:
                best = dt
        rows.append(
            {
                "setting": setting,
                "rule": rule,
                "impl": "numpy",
                "kind": "per_step",
                "threshold": "",
                "steps": steps,
                "seconds": best,
                "us_per_step": 1e6 * best / steps,
            }
        )
    return rows


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.settings.split(",") if args.settings else list(SETTINGS)
    x, y = make_data()
    files = {
        "traj": gzip.open(out / "working_set_traj.csv.gz", "wt", newline=""),
        "hits": open(out / "working_set_hits.csv", "w", newline=""),
        "timing": open(out / "working_set_timing.csv", "w", newline=""),
        "valid": open(out / "working_set_validation.csv", "w", newline=""),
    }
    writers = {
        "traj": csv.DictWriter(files["traj"], TRAJ_FIELDS),
        "hits": csv.DictWriter(files["hits"], HIT_FIELDS),
        "timing": csv.DictWriter(files["timing"], TIMING_FIELDS),
        "valid": csv.DictWriter(files["valid"], VALID_FIELDS),
    }
    for w in writers.values():
        w.writeheader()
    t_start = time.perf_counter()
    for name in names:
        lam, gamma = SETTINGS[name]
        prob = Problem(x, y, lam, gamma)
        alpha_ref, ref = reference_solution(prob)
        g_ref = prob.gradient(alpha_ref)
        print(
            f"[{name}] lambda={lam} gamma={gamma}: reference {ref.steps} steps, r={ref.violation:.1e},"  # noqa: E501
            f" gap(mid)={prob.gap(alpha_ref, g_ref):.2e}, gap(solver)={ref.gap:.2e},"
            f" SVs={int((np.abs(alpha_ref) > 0).sum())}",
            file=sys.stderr,
        )
        writers["valid"].writerows(validate(prob, name, 300 if args.quick else 2000))
        writers["timing"].writerows(timing(prob, name, args.quick))
        for key in ("valid", "timing"):
            files[key].flush()
        for rule in RULES:
            cap = min(CAP[rule], (1_000 if rule == "tmd" else 20_000) if args.quick else CAP[rule])
            tr = Tracker(prob, alpha_ref, name, rule)
            t0 = time.perf_counter()
            if rule in RUST_RULES:
                steps = rust_trajectory(prob, tr, rule, TARGET[rule], cap)
            else:
                steps = ref_trajectory(prob, tr, rule, TARGET[rule], cap)
            writers["traj"].writerows(tr.rows)
            for m in MEASURES:
                for thr, step in zip(THRESHOLDS, tr.hits[m]):  # noqa: B905
                    writers["hits"].writerow(
                        {
                            "setting": name,
                            "rule": rule,
                            "measure": m,
                            "threshold": thr,
                            "step": "" if step is None else step,
                        }
                    )
            for f in files.values():
                f.flush()
            print(
                f"[{name}] {rule}: {steps} steps, r={tr.rows[-1]['r']:.1e}, {len(tr.rows)} rows,"
                f" {time.perf_counter() - t0:.0f} s",
                file=sys.stderr,
            )
    for f in files.values():
        f.close()
    print(f"total {time.perf_counter() - t_start:.0f} s", file=sys.stderr)


# ----------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------


def read(path: Path):
    """Reads ``path`` or, if only that exists, its gzipped version ``path.gz``."""
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    return float(s) if s not in ("", None) else math.nan


def summary(args) -> None:
    d = Path(args.dir)
    valid = read(d / "working_set_validation.csv")
    hits = read(d / "working_set_hits.csv")
    timing_rows = read(d / "working_set_timing.csv")
    traj = read(d / "working_set_traj.csv")
    settings = list(dict.fromkeys(r["setting"] for r in traj))

    print("Validation of the NumPy reference SMO against the solver iterates\n")
    print(
        f"{'setting':<9}{'rule':<5}{'steps':>6}{'1st pair mismatch':>19}{'max|da|':>10}{'max|dpsi|':>11}{'end|da|':>9}"  # noqa: E501
        f"{'|dr|':>9}{'|drba/l|':>10}{'|dgap|':>9}{'|ddual|':>9}"
    )
    for r in valid:
        print(
            f"{r['setting']:<9}{r['rule']:<5}{r['steps']:>6}{r['first_pair_mismatch'] or '-':>19}"
            f"{num(r['max_alpha_diff']):>10.1e}{num(r['max_obj_diff']):>11.1e}{num(r['end_alpha_diff']):>9.1e}{num(r['r_diff']):>9.1e}"
            f"{num(r['rba_lam_diff']):>10.1e}{num(r['gap_diff']):>9.1e}{num(r['dual_diff']):>9.1e}"
        )

    per_step = {
        (r["setting"], r["rule"], r["impl"]): num(r["us_per_step"])
        for r in timing_rows
        if r["kind"] == "per_step" or (r["kind"] == "to_tol" and num(r["threshold"]) == 1e-8)
    }
    for s in settings:
        print(f"\n=== setting {s} ===")
        h = {
            (r["rule"], r["measure"], num(r["threshold"])): r["step"]
            for r in hits
            if r["setting"] == s
        }
        tt = {
            (r["rule"], num(r["threshold"])): r
            for r in timing_rows
            if r["setting"] == s and r["kind"] == "to_tol"
        }
        print(
            "steps (and Rust time in ms, or estimated time = steps x numpy per-step time) to reach a threshold\n"  # noqa: E501
        )
        head = f"{'rule':<6}{'us/step':>9}"
        cols = [
            ("r", 1e-3),
            ("r", 1e-5),
            ("r", 1e-8),
            ("rba_lam", 1e-3),
            ("rba_lam", 1e-5),
            ("gap", 1e-3),
            ("gap", 1e-5),
            ("gap", 1e-7),
        ]
        for m, t in cols:
            head += f"{m + '<=' + format(t, '.0e'):>20}"
        print(head)
        for rule in RULES:
            impl = "rust" if rule in RUST_RULES else "numpy"
            us = per_step.get((s, rule, impl), math.nan)
            line = f"{rule:<6}{us:>9.1f}"
            for m, t in cols:
                step = h.get((rule, m, t), "")
                if step == "":
                    line += f"{'-':>20}"
                    continue
                if m == "r" and rule in RUST_RULES and (rule, t) in tt:
                    tm = num(tt[(rule, t)]["seconds"]) * 1e3
                    line += f"{step + ' / ' + format(tm, '.0f') + 'ms':>20}"
                else:
                    tm = int(step) * us * 1e-3
                    line += f"{step + ' / ~' + format(tm, '.0f') + 'ms':>20}"
            print(line)

        rows = [r for r in traj if r["setting"] == s]
        print(
            "\nfirst- vs second-order: steps to reach (running minimum of the measure) relative to mvp"  # noqa: E501
        )
        for m, t in (("r", 1e-3), ("r", 1e-5), ("rba_lam", 1e-5), ("gap", 1e-5)):
            ref = h.get(("mvp", m, t), "")
            ratios = []
            for rule in RULES:
                st = h.get((rule, m, t), "")
                ratios.append(
                    f"{rule} {st or '-':>7}"
                    + (f" ({int(ref) / int(st):.1f}x)" if st and ref and rule != "mvp" else "")
                )
            print(f"  {m}<={t:.0e}: " + "  ".join(ratios))

        print(
            "\nmedian [10 %, 90 %] of ratios over stored iterates (late: r <= 1e-2; all: every stored step)"  # noqa: E501
        )
        specs = [
            ("r / (rba/lam)", "r", "rba_lam"),
            ("gap / (rba/lam)", "gap", "rba_lam"),
            ("rk / rba", "rk", "rba"),
            ("rk / (rba/lam)", "rk", "rba_lam"),
            ("rtilde / r", "rtilde", "r"),
        ]
        print(f"{'rule':<6}" + "".join(f"{n:>46}" for n, _, _ in specs))
        for rule in RULES:
            rr = [r for r in rows if r["rule"] == rule]
            line = f"{rule:<6}"
            for _, a, b in specs:
                cells = []
                for late in (True, False):
                    q = []
                    for r in rr:
                        if late and not num(r["r"]) <= 1e-2:
                            continue
                        va, vb = num(r[a]), num(r[b])
                        if va > 0 and vb > 0 and math.isfinite(va) and math.isfinite(vb):
                            q.append(va / vb)
                    if q:
                        lo, med, hi = np.percentile(q, [10, 50, 90])
                        cells.append(f"{med:.2f}[{lo:.2f},{hi:.2f}]")
                    else:
                        cells.append("-")
                line += f"{'late ' + cells[0] + ' all ' + cells[1]:>46}"
            print(line)

        print(
            "\nfraction of stored iterates with rk < rba (rk below the bound-aware measure, Sec. 10.2.3)"  # noqa: E501
        )
        for rule in RULES:
            q = [
                num(r["rk"]) < num(r["rba"])
                for r in rows
                if r["rule"] == rule and math.isfinite(num(r["rk"]))
            ]
            late = [
                num(r["rk"]) < num(r["rba"])
                for r in rows
                if r["rule"] == rule and math.isfinite(num(r["rk"])) and num(r["r"]) <= 1e-2
            ]
            print(
                f"  {rule:<6} all {np.mean(q):.3f}   late {np.mean(late) if late else math.nan:.3f}"
            )


# ----------------------------------------------------------------------------------------------
# plots (Figures 10.2.1 - 10.2.5)
# ----------------------------------------------------------------------------------------------

COLORS = {"mvp": "#1f77b4", "bamvp": "#17becf", "mud": "#2ca02c", "md": "#ff7f0e", "tmd": "#d62728"}
LABELS = {"mvp": "MVP", "bamvp": "baMVP", "mud": "MuDP", "md": "MDP", "tmd": "tMDP"}


def plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    traj = read(Path(args.data) / "working_set_traj.csv")
    timing_rows = read(Path(args.data) / "working_set_timing.csv")
    settings = list(dict.fromkeys(r["setting"] for r in traj))

    def col(rows, key):
        return np.array([num(r[key]) for r in rows])

    for s in settings:
        lam, gamma = SETTINGS[s]
        title = f"checkerboard, n=1000, lambda={lam:g}, gamma={gamma:g}"
        data = {
            rule: [r for r in traj if r["setting"] == s and r["rule"] == rule] for rule in RULES
        }
        data = {k: v for k, v in data.items() if v}
        lam_ = lam

        # Figure 10.2.1: running minima of the measures, all rules
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (key, name) in zip(  # noqa: B905
            axes.flat,
            [
                ("rba_min", "r_diamond(alpha^k, lambda)"),
                ("gap_min", "Delta*(alpha^k)"),
                ("r_min", "r(alpha^k)"),
                ("D_min", "D_k"),
            ],
        ):
            for rule, rows in data.items():
                ax.loglog(
                    np.maximum(col(rows, "step"), 1),
                    col(rows, key),
                    color=COLORS[rule],
                    label=LABELS[rule],
                )
            ax.set_xlabel("step k")
            ax.set_ylabel(f"minimum so far of {name}")
            ax.grid(alpha=0.3)
        axes[0, 0].legend()
        fig.suptitle(f"Fig. 10.2.1  {title}")
        fig.tight_layout()
        fig.savefig(out / f"ws_{s}_fig10.2.1.png", dpi=120)
        plt.close(fig)

        # Figure 10.2.1 vs. time: steps weighted with the measured cost per step
        cost = {}
        for r in timing_rows:
            if r["setting"] == s and (
                r["kind"] == "per_step" or (r["kind"] == "to_tol" and num(r["threshold"]) == 1e-8)
            ):
                if r["kind"] == "per_step" or r["rule"] in RUST_RULES:
                    cost[r["rule"]] = (
                        num(r["us_per_step"])
                        if (r["impl"] == "rust" or r["rule"] not in RUST_RULES)
                        else cost.get(r["rule"], math.nan)
                    )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (key, name) in zip(axes, [("r_min", "r(alpha^k)"), ("gap_min", "Delta*(alpha^k)")]):  # noqa: B905
            for rule, rows in data.items():
                t = np.maximum(col(rows, "step"), 1) * cost.get(rule, math.nan) * 1e-6
                ax.loglog(
                    t,
                    col(rows, key),
                    color=COLORS[rule],
                    label=f"{LABELS[rule]} ({cost.get(rule, math.nan):.0f} us/step)",
                )
            ax.set_xlabel("estimated time [s] (steps x cost per step)")
            ax.set_ylabel(f"minimum so far of {name}")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"Fig. 10.2.1 over time  {title}\n"
            f"(cost per step: Rust for MVP, MuDP, MDP; NumPy for baMVP, tMDP)"
        )
        fig.tight_layout()
        fig.savefig(out / f"ws_{s}_fig10.2.1_time.png", dpi=120)
        plt.close(fig)

        # Figure 10.2.2: second-order rules
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (key, name) in zip(  # noqa: B905
            axes, [("gap_min", "Delta*(alpha^k)"), ("rba_min", "r_diamond(alpha^k, lambda)")]
        ):
            for rule in ("tmd", "md", "mud"):
                if rule in data:
                    ax.loglog(
                        np.maximum(col(data[rule], "step"), 1),
                        col(data[rule], key),
                        color=COLORS[rule],
                        label=LABELS[rule],
                    )
            ax.set_xlabel("step k")
            ax.set_ylabel(f"minimum so far of {name}")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"Fig. 10.2.2  {title}")
        fig.tight_layout()
        fig.savefig(out / f"ws_{s}_fig10.2.2.png", dpi=120)
        plt.close(fig)

        def scatter(fig_no, pairs, rules, xkey, ylabel_x, lines):
            fig, axes = plt.subplots(1, len(rules), figsize=(5.6 * len(rules), 5))
            for ax, rule in zip(np.atleast_1d(axes), rules):  # noqa: B905
                rows = data.get(rule)  # noqa: B023
                if rows is None:
                    continue
                xs = col(rows, xkey)
                for ykey, label, color in pairs:
                    ys = col(rows, ykey)
                    ok = (xs > 0) & (ys > 0)
                    ax.scatter(xs[ok], ys[ok], s=4, alpha=0.5, color=color, label=label)
                lim = ax.get_xlim()
                xx = np.geomspace(max(lim[0], 1e-12), lim[1], 50)
                for slope_name, f in lines:
                    ax.plot(xx, f(xx), "k--", lw=0.8, label=slope_name)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(ylabel_x)
                ax.set_title(LABELS[rule])
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            fig.suptitle(f"Fig. {fig_no}  {title}")  # noqa: B023
            fig.tight_layout()
            fig.savefig(out / f"ws_{s}_fig{fig_no}.png", dpi=120)  # noqa: B023
            plt.close(fig)

        # Figures 10.2.3 - 10.2.5
        scatter(
            "10.2.3",
            [("r", "r(alpha^k)", "#1f77b4"), ("gap", "Delta*(alpha^k)", "#d62728")],
            ("mvp", "bamvp"),
            "rba_lam",
            "r_diamond(alpha^k, lambda) / lambda",
            [("y = x", lambda v: v)],
        )
        scatter(
            "10.2.4",
            [("rk", "r_k", "#2ca02c")],
            ("mvp", "bamvp"),
            "rba",
            "r_diamond(alpha^k, lambda)",
            [("r_k = r_diamond", lambda v: v), ("r_k = r_diamond / lambda", lambda v: v / lam_)],  # noqa: B023
        )
        scatter(
            "10.2.5",
            [("rtilde", "r~_k", "#9467bd")],
            ("md", "bamvp"),
            "r",
            "r(alpha^k)",
            [("y = x", lambda v: v)],
        )
        print(f"plots for {s} written", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run the study and write the CSV files")
    p.add_argument("--out", default=str(Path(__file__).parent / "results"))
    p.add_argument("--settings", help=f"comma-separated subset of {list(SETTINGS)}")
    p.add_argument("--quick", action="store_true", help="short run for testing")
    p.set_defaults(func=run)
    p = sub.add_parser("summary", help="print the key numbers")
    p.add_argument("dir", nargs="?", default=str(Path(__file__).parent / "results"))
    p.set_defaults(func=summary)
    p = sub.add_parser("plot", help="recreate Figures 10.2.1-10.2.5 (needs matplotlib)")
    p.add_argument("outdir")
    p.add_argument("--data", default=str(Path(__file__).parent / "results"))
    p.set_defaults(func=plot)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
