"""Sequential end-to-end HPO comparison with block sizes 2, 4 and 8.

Needs the development sources of optimized-svm and a release smo-solver extension::

    PYTHONPATH=../optimized-svm/src:../smo-solver/python RAYON_NUM_THREADS=1 \
        python benchmarks/block_hpo.py

The SMO warm stage uses its normal rough tolerance/step budget, followed by Newton and the
adjoint hypergradient. The CSV reports optimizer success separately from having a fitted model:
time/iteration limits are retained as censored runs, not successful optimizations. The optimizer
time limit applies to CV evaluations, so a final full-data fit can extend the total wall time.
All calls are sequential; BLAS is limited to one thread for these small systems.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import optimized_svm
from block_working_sets import provenance
from optimized_svm import OptimizedSVC, OptimizedSVR, cv_error_and_gradient
from threadpoolctl import threadpool_limits

FIELDS = [
    "case",
    "n",
    "working_set_size",
    "repeat",
    "seconds",
    "hpo_seconds",
    "cv_error",
    "cv_error_recomputed",
    "cv_error_difference",
    "cv_max_norm_excess",
    "cv_verified",
    "lam",
    "scaling",
    "evaluations",
    "iterations",
    "optimizer_success",
    "optimizer_status",
    "optimizer_message",
    "max_asum",
    "final_c",
    "bound_active",
    "final_residual",
    "final_smo_steps",
    "final_newton_steps",
]


def run(args):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(args.n, 4))
    signal = np.sin(1.8 * x[:, 0]) + 0.4 * x[:, 1] + 0.2 * x[:, 2] ** 2
    targets = signal + 0.08 * rng.normal(size=args.n)
    labels = np.where(signal + 0.2 * rng.normal(size=args.n) > np.median(signal), 1.0, -1.0)
    rows = []
    with threadpool_limits(limits=1):
        for case, cls, y in [("svc", OptimizedSVC, labels), ("nu_svr", OptimizedSVR, targets)]:
            for size in args.sizes:
                for repeat in range(args.repeats):
                    kw = dict(
                        lam_init=0.01,
                        scaling_init=0.6,
                        cv=3,
                        n_jobs=1,
                        random_state=0,
                        lam_bounds=(1e-4, 0.1),
                        scaling_bounds=(0.25, 2.0),
                        two_stage=False,
                        smoothing=0.2,
                        max_iter=args.max_iter,
                        time_limit=args.time_limit,
                        tol=1e-6,
                        smo_options={"working_set_size": size},
                    )
                    if case == "nu_svr":
                        # delta > epsilon creates overlapping dual copies even for a good fit;
                        # this small bound remains active in the final model.
                        kw.update(epsilon=0.05, max_asum=0.08 * args.n, normalize_y=False)
                    model = cls(**kw)
                    start = time.perf_counter()
                    model.fit(x, y)
                    elapsed = time.perf_counter() - start
                    result = model.optimize_result_
                    info = model.training_info_
                    # Outside timing: the rough start must not decide incorrectly that the
                    # norm bound is inactive in the Newton refinement.
                    folds = model._folds(x, y, None)
                    theta = np.log(np.r_[model.lam_, model.scaling_opt_])
                    check = cv_error_and_gradient(
                        theta,
                        x,
                        y,
                        np.ones(args.n),
                        folds,
                        model._loss(),
                        model._validation(),
                        max_asum=kw.get("max_asum"),
                        smo_options={"working_set_size": size},
                    )
                    difference = abs(check.error - model.cv_error_)
                    excess = (
                        max(
                            (abs(fold.solution.a).sum() - kw["max_asum"] for fold in check.folds),
                            default=0.0,
                        )
                        if "max_asum" in kw
                        else 0.0
                    )
                    verified = difference <= 1e-7 and excess <= 1e-7
                    row = dict(
                        case=case,
                        n=args.n,
                        working_set_size=size,
                        repeat=repeat,
                        seconds=elapsed,
                        hpo_seconds=model.time_,
                        cv_error=model.cv_error_,
                        cv_error_recomputed=check.error,
                        cv_error_difference=difference,
                        cv_max_norm_excess=max(0.0, float(excess)),
                        cv_verified=verified,
                        lam=model.lam_,
                        scaling=json.dumps(model.scaling_opt_.tolist()),
                        evaluations=len(model.history_),
                        iterations=model.n_iter_,
                        optimizer_success=False if result is None else bool(result.success),
                        optimizer_status="time_limit" if result is None else result.status,
                        optimizer_message="time_limit" if result is None else str(result.message),
                        max_asum=kw.get("max_asum"),
                        final_c=info["c"],
                        bound_active=bool(info["c"] > 1e-8),
                        final_residual=info["residual"],
                        final_smo_steps=info["smo_steps"],
                        final_newton_steps=info["newton_steps"],
                    )
                    rows.append(row)
                    print(
                        f"{case} size{size} repeat{repeat}: {elapsed:.5f}s "
                        f"CV {model.cv_error_:.9g}, evaluations{len(model.history_)}, "
                        f"success={row['optimizer_success']}, c={info['c']:.5g}",
                        flush=True,
                    )
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    with args.out.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(rows)
                    assert verified, (
                        f"HPO verification failed: reported CV differs by {difference:g}, "
                        f"maximum norm-bound excess {excess:g}"
                    )
    args.out.with_suffix(".txt").write_text(
        f"Python {platform.python_version()}, optimized-svm {optimized_svm.__version__}\n"
        f"Optimized sources: {optimized_svm.__file__}\n"
        f"Arguments: {vars(args)}\n"
        "Sequential calls, BLAS threads=1, deterministic data seed7 and CV seed0.\n"
        "Timed total fit includes all CV evaluations and final full-data training.\n"
        "Fresh CV verification and norm-bound checks are outside timed regions.\n"
        + "Optimized training.py SHA256: "
        + hashlib.sha256(
            (Path(optimized_svm.__file__).parent / "training.py").read_bytes()
        ).hexdigest()
        + "\n"
        + provenance()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=180)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "results/block_hpo.csv"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
