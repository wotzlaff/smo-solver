"""Study of the shrinking heuristic (Strasdat, Sec. 7.10 and 10.2.7-10.2.9).

Variables at a bound whose shifted gradient exceeds ``factor * r(alpha)**exponent`` (``r``:
violation measure) are removed from the active set; before termination the full problem is restored and
checked again. The script measures, for classification and epsilon-regression,

* ``scaling``:  time with and without shrinking over problem size and tolerance,
* ``params``:   influence of ``shrinking_exponent`` (eta), ``shrinking_factor`` (c1) and the
                shrinking period on time, steps and the number of restores ("unshrinks"),
* ``lambda``:   behaviour for different regularization parameters (number of free variables),
* ``robust``:   default rule against ``exponent = 1`` over kernel widths, lambda and a 1-norm bound,
* ``tune``:     factor and minimal removed fraction for ``exponent = 1``.

Usage::

    python benchmarks/shrinking_study.py scaling > scaling.csv
    python benchmarks/shrinking_study.py params  > params.csv
    python benchmarks/shrinking_study.py lambda  > lambda.csv
    python benchmarks/shrinking_study.py robust  > robust.csv
    python benchmarks/shrinking_study.py tune    > tune.csv
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

import smo_solver

FIELDS = ["part", "task", "n", "lam", "tol", "variant", "exponent", "factor", "period", "min_fraction",
          "seconds", "steps", "shrinks", "unshrinks", "min_active", "free", "support", "gamma", "max_asum"]  # fmt: skip


def data(n: int, d: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    f = np.sin(x[:, 0]) + x[:, 1] * x[:, 2] - 0.5 * x[:, 3]
    y = np.where(f + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    t = f + 0.1 * rng.normal(size=n)
    return x, y, t


def solve(task, x, y, t, lam, tol, gamma=0.1, **opts):
    if task == "classification":
        return smo_solver.solve_classification(x, y, lam, gamma=gamma, tol=tol, **opts)
    return smo_solver.solve_regression(x, t, lam, epsilon=0.1, gamma=gamma, tol=tol, **opts)


def measure(part, task, n, lam, tol, variant, reps, gamma=0.1, **opts):
    x, y, t = data(n)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = solve(task, x, y, t, lam, tol, gamma=gamma, **opts)
        dt = time.perf_counter() - t0
        if best is None or dt < best[0]:
            best = (dt, r)
    dt, r = best
    alpha = np.asarray(r.alpha)
    free = int(np.sum((np.abs(alpha) > 1e-12) & (np.abs(np.abs(alpha) - 1.0) > 1e-12)))
    return {
        "part": part, "task": task, "n": n, "lam": lam, "tol": tol, "variant": variant,
        "exponent": opts.get("shrinking_exponent", 0.5), "factor": opts.get("shrinking_factor", 1.0),
        "period": opts.get("shrinking_period", 0), "min_fraction": opts.get("shrinking_min_fraction", 0.0), "seconds": dt, "steps": r.steps,
        "shrinks": r.shrinks, "unshrinks": r.unshrinks, "min_active": r.min_active,
        "free": free, "support": int(np.sum(np.abs(alpha) > 1e-12)), "gamma": gamma,
        "max_asum": opts.get("max_asum", ""),
    }  # fmt: skip


def main(part: str, reps: int = 2) -> None:
    w = csv.DictWriter(sys.stdout, FIELDS)
    w.writeheader()

    def emit(row):
        w.writerow(row)
        sys.stdout.flush()

    if part == "scaling":
        for task, sizes in (
            ("classification", (1000, 2000, 5000, 10000)),
            ("regression", (1000, 2000, 3000)),
        ):
            for n in sizes:
                for tol in (1e-2, 1e-3, 1e-4, 1e-5):
                    for variant, opts in (("off", {"shrinking": False}), ("default", {})):
                        emit(measure("scaling", task, n, 0.01, tol, variant, reps, **opts))
    elif part == "params":
        for lam in (0.1, 0.01, 0.001):
            for tol in (1e-3, 1e-5):
                emit(
                    measure(
                        "params", "classification", 5000, lam, tol, "off", reps, shrinking=False
                    )
                )
                for eta in (0.25, 0.5, 0.75, 1.0):
                    for c in (0.1, 1.0, 10.0):
                        emit(measure("params", "classification", 5000, lam, tol, "threshold", reps,
                                     shrinking_exponent=eta, shrinking_factor=c))  # fmt: skip
                for period in (50, 200, 1000, 5000):
                    emit(
                        measure(
                            "params",
                            "classification",
                            5000,
                            lam,
                            tol,
                            "period",
                            reps,
                            shrinking_period=period,
                        )
                    )
        for lam in (0.1, 0.01, 0.001):
            emit(measure("params", "regression", 2000, lam, 1e-3, "off", reps, shrinking=False))
            for eta in (0.25, 0.5, 0.75, 1.0):
                for c in (0.1, 1.0, 10.0):
                    emit(measure("params", "regression", 2000, lam, 1e-3, "threshold", reps,
                                 shrinking_exponent=eta, shrinking_factor=c))  # fmt: skip
            for period in (50, 200, 1000, 2000):
                emit(
                    measure(
                        "params",
                        "regression",
                        2000,
                        lam,
                        1e-3,
                        "period",
                        reps,
                        shrinking_period=period,
                    )
                )
    elif part == "lambda":
        for task, n in (("classification", 5000), ("regression", 2000)):
            for lam in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
                for variant, opts in (("off", {"shrinking": False}), ("default", {})):
                    emit(measure("lambda", task, n, lam, 1e-3, variant, reps, **opts))
    elif part == "robust":
        variants = (
            ("off", {"shrinking": False}),
            ("default", {}),
            ("exp0.75", {"shrinking_exponent": 0.75}),
            ("exp1", {"shrinking_exponent": 1.0}),
        )
        for task, n in (("classification", 5000), ("regression", 2000)):
            for gamma in (0.01, 0.1, 1.0):
                for lam in (0.1, 0.01, 0.001):
                    for tol in (1e-3, 1e-5):
                        for variant, opts in variants:
                            emit(
                                measure(
                                    "robust", task, n, lam, tol, variant, reps, gamma=gamma, **opts
                                )
                            )
        for lam in (0.01, 0.001):  # active 1-norm bound (nu-type constraint)
            for variant, opts in variants:
                emit(
                    measure(
                        "robust",
                        "regression",
                        2000,
                        lam,
                        1e-3,
                        variant,
                        reps,
                        max_asum=100.0,
                        **opts,
                    )
                )
    elif part == "tune":
        variants = [("off", {"shrinking": False})]
        variants += [
            (f"c{c}", {"shrinking_exponent": 1.0, "shrinking_factor": c})
            for c in (0.3, 0.5, 1.0, 2.0, 4.0)
        ]
        variants += [
            (f"min{f}", {"shrinking_exponent": 1.0, "shrinking_min_fraction": f})
            for f in (0.05, 0.1, 0.2)
        ]
        variants.append(("default", {}))
        for task, n in (("classification", 5000), ("regression", 2000)):
            for gamma in (0.01, 0.1, 1.0):
                for lam in (0.1, 0.001):
                    for tol in (1e-3, 1e-5):
                        for variant, opts in variants:
                            emit(
                                measure(
                                    "tune", task, n, lam, tol, variant, reps, gamma=gamma, **opts
                                )
                            )
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
