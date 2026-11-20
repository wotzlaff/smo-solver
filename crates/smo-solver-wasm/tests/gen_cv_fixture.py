"""Generates cv_fixture.json: reference values of the cross-validation objective.

Run from the optimized-svm repository (it provides the Python reference implementation):

    cd <optimized-svm> && uv run python <this file> <output.json>

The Rust tests (`src/tests.rs`) compare `cv_objective` against these numbers. Hyperparameters are
given as (log lambda, log gamma) with gamma = s^2; the Python reference differentiates w.r.t.
log s = log(gamma) / 2, so its gradient entry is halved here.
"""

import json
import sys

import numpy as np

from optimized_svm import cv_error_and_gradient
from optimized_svm.losses import EpsilonInsensitiveLoss, HingeLoss
from optimized_svm.validation import SmoothAbsoluteValidation, SmoothHingeValidation


def moons(n, rng):
    t = rng.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    x = np.r_[a, b] + 0.12 * rng.standard_normal((n, 2))
    y = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    return x, y


def make_cases():
    rng = np.random.default_rng(7)
    x, y = moons(120, rng)
    fold = rng.permutation(120) % 4
    clf = dict(name="moons-hinge", kind=0, x=x, y=y, fold_of=fold, k=4, smoothing=0.3, epsilon=0.0,
               points=[(-1.0, 0.0), (-2.5, 0.7), (0.5, -1.0)])
    clf0 = dict(clf, name="moons-hinge-delta0", smoothing=0.0, points=[(-1.0, 0.0)])
    x1 = np.sort(rng.uniform(-1.5, 1.5, 80))[:, None]
    y1 = np.sin(3 * x1[:, 0]) + 0.1 * rng.standard_normal(80)
    fold1 = rng.permutation(80) % 4
    reg = dict(name="sine-eps", kind=1, x=x1, y=y1, fold_of=fold1, k=4, smoothing=0.2, epsilon=0.1,
               points=[(-3.0, 0.5), (-4.5, 1.0), (-2.0, -0.5)])
    return [clf, clf0, reg]


def main(path):
    out = []
    for c in make_cases():
        n = len(c["y"])
        folds = [(np.flatnonzero(c["fold_of"] != i), np.flatnonzero(c["fold_of"] == i))
                 for i in range(c["k"])]
        if c["kind"] == 0:
            loss, val = HingeLoss(smoothing=c["smoothing"]), SmoothHingeValidation()
        else:
            loss = EpsilonInsensitiveLoss(epsilon=c["epsilon"], smoothing=c["smoothing"])
            val = SmoothAbsoluteValidation()
        pts = []
        for ll, lg in c["points"]:
            theta = np.array([ll, 0.5 * lg])
            res = cv_error_and_gradient(theta, c["x"], c["y"], np.ones(n), folds, loss, val,
                                        smo_options={"tol": 1e-10})
            pts.append(dict(log_lambda=ll, log_gamma=lg, value=float(res.error),
                            grad_log_lambda=float(res.gradient[0]),
                            grad_log_gamma=float(res.gradient[1]) / 2.0))
        out.append(dict(name=c["name"], kind=c["kind"], dim=c["x"].shape[1],
                        x=c["x"].ravel().tolist(), y=c["y"].tolist(),
                        fold_of=c["fold_of"].tolist(), k=c["k"], smoothing=c["smoothing"],
                        epsilon=c["epsilon"], points=pts))
    with open(path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
