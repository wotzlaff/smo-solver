"""Compare training time and accuracy of smo_solver with LIBSVM (scikit-learn SVC/SVR).

Usage: python benchmarks/bench_vs_libsvm.py [--sizes 1000 2000 5000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.svm import SVC, SVR

import smo_solver


def data(n: int, d: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    f = np.sin(x[:, 0]) + x[:, 1] * x[:, 2] - 0.5 * x[:, 3]
    y = np.where(f + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    t = f + 0.1 * rng.normal(size=n)
    return x, y, t


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000])
    parser.add_argument("--lam", type=float, default=0.01)
    args = parser.parse_args()
    gamma, lam = 0.1, args.lam
    print(f"{'task':<14}{'n':>7}{'smo_solver [s]':>15}{'libsvm [s]':>12}{'steps':>9}{'libsvm steps':>14}{'max |Δf|':>11}")
    for n in args.sizes:
        x, y, t = data(n)
        xs = x[:200]
        cases = [
            (
                "classification",
                lambda x=x, y=y: smo_solver.solve_classification(x, y, lam, gamma=gamma),
                lambda x=x, y=y: SVC(C=1 / lam, gamma=gamma, cache_size=256).fit(x, y),
                "decision_function",
            ),
            (
                "regression",
                lambda x=x, t=t: smo_solver.solve_regression(x, t, lam, epsilon=0.1, gamma=gamma),
                lambda x=x, t=t: SVR(C=1 / lam, epsilon=0.1, gamma=gamma, cache_size=256).fit(x, t),
                "predict",
            ),
        ]
        for task, ours_fn, ref_fn, method in cases:
            res, t_ours = timed(ours_fn)
            ref, t_ref = timed(ref_fn)
            ours = smo_solver.kernel_matrix(xs, x, gamma=gamma) @ (res.beta / lam) + res.b
            diff = np.abs(ours - getattr(ref, method)(xs)).max()
            print(f"{task:<14}{n:>7}{t_ours:>15.3f}{t_ref:>12.3f}{res.steps:>9}{int(np.sum(ref.n_iter_)):>14}{diff:>11.1e}")


if __name__ == "__main__":
    main()
