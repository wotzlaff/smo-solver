"""Which optimality measure is a good stopping criterion? (Strasdat, Sec. 7.4-7.6 and 10.2.2)

SMO is run for an increasing number of steps on classification and epsilon-regression problems
(with and without smoothing, several lambda). At every budget the three candidate measures of the
final iterate are recorded together with the *actual* errors against a tight reference solution:

* ``violation``    ``r(alpha)``, the measure the solver stops on (LIBSVM scaling),
* ``bound_aware``  ``r_diamond(alpha, rho) / rho`` with ``rho = lambda`` (Def. 7.21),
* ``gap``          the duality gap (primal + dual objective),

and ``dual_subopt`` (``D(alpha) - D(alpha*)``), ``f_err`` (maximal deviation of the decision
function on test points) and ``alpha_err`` (mean absolute deviation of the dual variables).

Usage::

    python benchmarks/stopping_study.py run > stopping.csv
    python benchmarks/stopping_study.py summary stopping.csv
"""

from __future__ import annotations

import csv
import sys

import numpy as np

import smo_solver
from smo_solver import kernel_matrix

FIELDS = ["task", "smoothing", "lam", "budget", "steps", "violation", "bound_aware", "gap",
          "dual_subopt", "f_err", "alpha_err"]  # fmt: skip
MEASURES = ["violation", "bound_aware", "gap"]
ERRORS = ["dual_subopt", "f_err", "alpha_err"]


def data(n: int, d: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    f = np.sin(x[:, 0]) + x[:, 1] * x[:, 2] - 0.5 * x[:, 3]
    y = np.where(f + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    return x, y, f + 0.1 * rng.normal(size=n)


def run() -> None:
    n, gamma = 1500, 0.1
    x, y, t = data(n)
    xt = data(500, seed=1)[0]
    kt = kernel_matrix(xt, x, gamma=gamma)
    w = csv.DictWriter(sys.stdout, FIELDS)
    w.writeheader()
    for task in ("classification", "regression"):
        for smoothing in (0.0, 0.5):
            for lam in (0.1, 0.01, 0.001):

                def solve(**kw):
                    common = dict(gamma=gamma, smoothing=smoothing, shrinking=False, **kw)
                    if task == "classification":
                        return smo_solver.solve_classification(x, y, lam, **common)
                    return smo_solver.solve_regression(x, t, lam, epsilon=0.1, **common)

                ref = solve(tol=1e-10)
                f_ref = kt @ (ref.beta / lam) + ref.b
                budgets = np.unique(np.geomspace(20, max(ref.steps, 40), 24).astype(int))
                for budget in budgets:
                    r = solve(tol=1e-12, max_steps=int(budget))
                    f = kt @ (r.beta / lam) + r.b
                    w.writerow({
                        "task": task, "smoothing": smoothing, "lam": lam, "budget": budget,
                        "steps": r.steps, "violation": r.violation, "bound_aware": r.bound_aware,
                        "gap": r.gap, "dual_subopt": max(r.dual_objective - ref.dual_objective, 0.0),
                        "f_err": float(np.max(np.abs(f - f_ref))),
                        "alpha_err": float(np.mean(np.abs(np.asarray(r.alpha) - np.asarray(ref.alpha)))),
                    })  # fmt: skip
                    sys.stdout.flush()
                print(
                    f"{task} delta={smoothing} lam={lam}: {len(budgets)} budgets", file=sys.stderr
                )


def summary(path: str) -> None:
    rows = [r for r in csv.DictReader(open(path)) if float(r["violation"]) > 0]
    problems: dict[tuple, list[dict]] = {}
    for r in rows:
        problems.setdefault((r["task"], r["smoothing"], r["lam"]), []).append(r)
    arr = {
        k: {c: np.array([float(r[c]) for r in v]) for c in MEASURES + ERRORS}
        for k, v in problems.items()
    }
    print("ratio error / measure over all iterates of all problems (log10: median [10 %, 90 %])\n")
    print(f"{'measure':<12}" + "".join(f"{e:>34}" for e in ERRORS))
    for m in MEASURES:
        cells = []
        for e in ERRORS:
            ratios = np.concatenate(
                [np.log10(np.maximum(a[e], 1e-300) / a[m]) for a in arr.values()]
            )
            lo, med, hi = np.percentile(ratios, [10, 50, 90])
            cells.append(f"{med:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] (spread {hi - lo:4.2f})")
        print(f"{m:<12}" + "".join(f"{c:>34}" for c in cells))
    print("\nlargest tolerance (per measure) that guarantees f_err <= target on every problem\n")
    for target in (1e-1, 1e-2, 1e-3):
        line = f"f_err <= {target:.0e}: "
        for m in MEASURES:
            # for every problem: the smallest measure value among iterates that still violate the target
            worst = np.inf
            for a in arr.values():
                bad = a["f_err"] > target
                if bad.any():
                    worst = min(worst, a[m][bad].min())
            line += f"{m} < {worst:.1e}   "
        print(line)


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run()
    elif sys.argv[1:2] == ["summary"] and len(sys.argv) > 2:
        summary(sys.argv[2])
    else:
        raise SystemExit(__doc__)
