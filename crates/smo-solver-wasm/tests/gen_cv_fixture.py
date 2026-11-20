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
from optimized_svm.kernels import rbf
from optimized_svm.losses import EpsilonInsensitiveLoss, HingeLoss, LogisticLoss
from optimized_svm.problem import TrainingProblem
from optimized_svm.training import train
from optimized_svm.validation import (
    LogisticValidation,
    SmoothAbsoluteValidation,
    SmoothHingeValidation,
)


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
    # logistic cases reuse the moons data (same generator state as above, appended last)
    logi = dict(clf, name="moons-logistic", kind=2, smoothing=0.0,
                points=[(-1.0, 0.0), (-3.0, 0.7), (0.5, -1.0)])
    return [clf, clf0, reg, logi]


def make_ard_cases():
    """ARD cases (one RBF width per feature); own generator so the cases above stay unchanged."""
    rng = np.random.default_rng(23)
    xm, y = moons(120, rng)
    x = np.c_[xm, 0.5 * rng.standard_normal((120, 2))]  # 2 informative + 2 noise dims
    fold = rng.permutation(120) % 4
    clf = dict(name="ard-moons-hinge", kind=0, x=x, y=y, fold_of=fold, k=4, smoothing=0.3,
               epsilon=0.0,
               points=[(-1.0, [0.0, 0.3, -1.0, -2.0]), (-2.5, [0.7, -0.5, -3.0, 0.2]),
                       (0.5, [-1.0, 0.5, 0.0, -0.5])])
    xr = rng.uniform(0, 1, (100, 5))  # Friedman #1 type: 3 informative + 2 noise dims
    yr = (10 * np.sin(np.pi * xr[:, 0] * xr[:, 1]) + 20 * (xr[:, 2] - 0.5) ** 2
          + 0.3 * rng.standard_normal(100))
    reg = dict(name="ard-friedman-eps", kind=1, x=xr, y=yr, fold_of=rng.permutation(100) % 4, k=4,
               smoothing=0.2, epsilon=0.5,
               points=[(-3.0, [0.5, 0.0, -0.5, -2.0, -3.0]), (-4.5, [1.0, 1.5, 0.3, -1.0, -2.0]),
                       (-2.0, [-0.5, 0.2, 0.8, -3.0, 0.0])])
    logi = dict(clf, name="ard-moons-logistic", kind=2, smoothing=0.0,
                points=[(-1.0, [0.0, 0.3, -1.0, -2.0]), (-3.0, [0.7, -0.5, -3.0, 0.2]),
                        (0.5, [-1.0, 0.5, 0.0, -0.5])])
    return [clf, reg, logi]


def main(path):
    out = []
    ard_names = {c["name"] for c in make_ard_cases()}
    for c in make_cases() + make_ard_cases():
        ard = c["name"] in ard_names
        n = len(c["y"])
        folds = [(np.flatnonzero(c["fold_of"] != i), np.flatnonzero(c["fold_of"] == i))
                 for i in range(c["k"])]
        newton = None
        if c["kind"] == 2:
            loss, val = LogisticLoss(), LogisticValidation()
            newton = {"tol": 1e-22}
        elif c["kind"] == 0:
            loss, val = HingeLoss(smoothing=c["smoothing"]), SmoothHingeValidation()
        else:
            loss = EpsilonInsensitiveLoss(epsilon=c["epsilon"], smoothing=c["smoothing"])
            val = SmoothAbsoluteValidation()
        pts = []
        for ll, lg in c["points"]:
            if ard:
                # one block per feature, theta = (log lambda, log s_1, ..., log s_d), gamma_d = s_d^2
                lgs = np.asarray(lg, dtype=float)
                theta = np.r_[ll, 0.5 * lgs]
                res = cv_error_and_gradient(theta, c["x"], c["y"], np.ones(n), folds, loss, val,
                                            blocks=[[d] for d in range(len(lgs))],
                                            smo_options={"tol": 1e-10}, newton_options=newton)
                pts.append(dict(log_lambda=ll, log_gammas=lgs.tolist(), value=float(res.error),
                                grad_log_lambda=float(res.gradient[0]),
                                grad_log_gammas=(np.asarray(res.gradient[1:]) / 2.0).tolist()))
                continue
            theta = np.array([ll, 0.5 * lg])
            res = cv_error_and_gradient(theta, c["x"], c["y"], np.ones(n), folds, loss, val,
                                        smo_options={"tol": 1e-10}, newton_options=newton)
            pts.append(dict(log_lambda=ll, log_gamma=lg, value=float(res.error),
                            grad_log_lambda=float(res.gradient[0]),
                            grad_log_gamma=float(res.gradient[1]) / 2.0))
        extra = {}
        if ard:
            extra = dict(ard=True)
        elif c["kind"] == 2:
            # full-data fit with the low-level API: decision values at fixed test points
            rng = np.random.default_rng(11)
            test = rng.uniform(-1.5, 2.5, (6, 2)) * np.array([1.0, 0.8])
            fits = []
            for ll, lg in c["points"]:
                s_ = np.exp(0.5 * lg)
                prob = TrainingProblem(c["x"] * s_, c["y"], loss, float(np.exp(ll)))
                sol = train(prob, newton_options=newton)
                dec = rbf(test * s_, prob.x) @ (sol.a / prob.lam) + sol.b
                fits.append(dict(log_lambda=ll, log_gamma=lg, bias=float(sol.b),
                                 newton_steps=int(sol.newton_steps), decision=dec.tolist()))
            extra = dict(test_points=test.ravel().tolist(), fits=fits)
        out.append(dict(**extra, name=c["name"], kind=c["kind"], dim=c["x"].shape[1],
                        x=c["x"].ravel().tolist(), y=c["y"].tolist(),
                        fold_of=c["fold_of"].tolist(), k=c["k"], smoothing=c["smoothing"],
                        epsilon=c["epsilon"], points=pts))
    with open(path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
