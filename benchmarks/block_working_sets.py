"""Sequential timing/accuracy study of joint working sets of sizes 2, 4, 8 and 16.

Run with a release-built extension exposing ``working_set_size``::

    python benchmarks/block_working_sets.py --repeats 3 --time-limit 10

The exact saved proptest regression is included as ``saved_nu_svr``. Timed runs are strictly
sequential. A stopped run is reported as such; speedups are only meaningful when both compared
runs converge to the requested violation, with comparable gap/objective/predictions. Reference
predictions use the most accurate converged result among the working-set sizes (after timing).
The measured solver time includes final restoration/measurement and may exceed its main-loop
time limit. Cache counters count accesses, not bytes or direct dense-kernel evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import smo_solver


class Lcg:
    """Exact Rust test generator, including wrapping u64 arithmetic."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.state >> 11) / (1 << 53)


def rust_dataset(n, seed):
    rng = Lcg(seed)
    x, target = [], []
    for i in range(n):
        label = 1.0 if i % 2 == 0 else -1.0
        row = [rng.next() * 2 - 1 + 0.4 * label for _ in range(2)]
        x.append(row)
        target.append(sum(np.sin(3 * np.array(row))) + 0.1 * (rng.next() - 0.5))
    rng = Lcg(seed ^ 0xDEAD)
    weights = np.array([0.2 + rng.next() * 2 for _ in range(n)])
    bound = 0.5 + rng.next() * n
    return np.array(x), np.array(target), weights, bound


def cases(n):
    x, y, weights, bound = rust_dataset(5, 1519)
    yield (
        "saved_nu_svr",
        "regression",
        x,
        y,
        dict(
            lam=10**-3.9700411425769886,
            gamma=4.863234423924022,
            smoothing=0.09944635523522093,
            epsilon=0.05,
            weights=weights,
            max_asum=bound,
            shrinking=False,
        ),
    )
    x, y, weights, bound = rust_dataset(n, 1519)
    yield (
        "nu_svr",
        "regression",
        x,
        y,
        dict(
            lam=1e-4,
            gamma=1.0,
            smoothing=0.1,
            epsilon=0.05,
            weights=weights,
            max_asum=0.3 * bound,
            shrinking=True,
        ),
    )
    yield (
        "nu_svr_moderate",
        "regression",
        x,
        y,
        dict(
            lam=0.01,
            gamma=1.0,
            smoothing=0.1,
            epsilon=0.05,
            weights=weights,
            max_asum=0.3 * bound,
            shrinking=True,
        ),
    )
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 10))
    f = np.sin(x[:, 0]) + x[:, 1] * x[:, 2] - 0.5 * x[:, 3]
    labels = np.where(f + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    targets = f + 0.1 * rng.normal(size=n)
    yield "svc", "classification", x, labels, dict(lam=0.01, gamma=0.1, smoothing=0.0)
    yield (
        "svr",
        "regression",
        x,
        targets,
        dict(
            lam=0.01,
            gamma=0.1,
            smoothing=0.1,
            epsilon=0.1,
        ),
    )


FIELDS = [
    "case",
    "n",
    "working_set_size",
    "repeat",
    "status",
    "tol",
    "seconds",
    "solver_seconds",
    "steps",
    "violation",
    "gap",
    "dual_objective",
    "relative_gap",
    "recomputed_violation",
    "recomputed_gap",
    "recomputed_dual",
    "max_gradient_drift",
    "equality_residual",
    "asum",
    "max_asum",
    "cache_hits",
    "cache_misses",
    "shrinks",
    "unshrinks",
    "reference_size",
    "dual_difference",
    "max_training_difference",
    "max_prediction_difference",
]


def provenance():
    """Identify an editable native build even before its source changes are committed."""
    extension = Path(smo_solver._core.__file__)
    lines = [
        f"Extension: {extension}",
        f"Extension SHA256: {hashlib.sha256(extension.read_bytes()).hexdigest()}",
    ]
    source = Path(smo_solver.__file__).resolve().parents[2]
    if not (source / "crates/smo-solver/src").is_dir():
        return "\n".join(lines + ["Editable source provenance unavailable (installed wheel).\n"])
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source, stderr=subprocess.DEVNULL, text=True
        ).strip()
        lines.append(f"Native source base commit: {head}; dirty={bool(dirty)}")
        for name in ["solver.rs", "block_qp.rs", "problem.rs"]:
            path = source / "crates/smo-solver/src" / name
            if path.exists():
                lines.append(
                    f"Source {name} SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}"
                )
    except (OSError, subprocess.SubprocessError):
        lines.append("Editable source provenance unavailable (installed wheel).")
    return "\n".join(lines) + "\n"


def independent_measures(result, kernel, y, kind, kw):
    """Rebuild all kernel products, KKT violation and primal/dual values after timed calls."""
    lam = kw["lam"]
    delta = kw.get("smoothing", 0.0)
    weights = kw.get("weights", np.ones(len(y)))
    inv_weight = np.divide(1.0, weights, out=np.zeros_like(weights), where=weights > 0)
    sample_ka = kernel @ result.beta / lam
    alpha = result.alpha
    if kind == "classification":
        sign = y
        lo, hi = np.minimum(0.0, y) * weights, np.maximum(0.0, y) * weights
        q, p, ka = 2 * delta * inv_weight, -(1.0 + delta) * y, sample_ka
        losses = weights * smo_solver.smooth_max(1.0 - y * (sample_ka + result.b) - result.c, delta)
    else:
        sign = np.repeat([1.0, -1.0], len(y))
        lo, hi = np.r_[np.zeros(len(y)), -weights], np.r_[weights, np.zeros(len(y))]
        q = np.tile(2 * delta * inv_weight, 2)
        p = -np.tile(y, 2) + sign * (kw["epsilon"] - delta)
        ka = np.tile(sample_ka, 2)
        prediction = sample_ka + result.b
        losses = weights * (
            smo_solver.smooth_max(y - prediction - kw["epsilon"] - result.c, delta)
            + smo_solver.smooth_max(prediction - y - kw["epsilon"] - result.c, delta)
        )
    gradient = ka + q * alpha + p
    bound = kw.get("max_asum")
    if bound is not None and sign @ alpha >= bound - 1e-10 * max(1.0, bound):
        groups = [(sign > 0, sign > 0), (sign < 0, sign < 0), (sign > 0, sign < 0)]
    else:
        groups = [(np.ones(len(alpha), dtype=bool), np.ones(len(alpha), dtype=bool))]
    violation = 0.0
    for down_group, up_group in groups:
        down = gradient[(alpha > lo) & down_group]
        up = gradient[(alpha < hi) & up_group]
        if len(down) and len(up):
            violation = max(violation, float(down.max() - up.min()))
    norm_term = 0.5 * alpha @ ka
    dual = norm_term + 0.5 * (q * alpha) @ alpha + p @ alpha
    primal = norm_term + losses.sum() + (bound or 0.0) * result.c
    return dict(
        recomputed_violation=violation,
        recomputed_gap=float(primal + dual),
        recomputed_dual=float(dual),
        max_gradient_drift=float(np.max(abs(ka - result.ka))),
        equality_residual=float(abs(alpha.sum())),
    ), sample_ka + result.b


def run(args):
    rows = []
    selected = None if args.cases is None else set(args.cases.split(","))
    for name, kind, x, y, kw in cases(args.n):
        if selected is not None and name not in selected:
            continue
        results = []
        print(f"{name}: n={len(x)}, lambda={kw['lam']}, bound={kw.get('max_asum')}", flush=True)
        for size in args.sizes:
            for rep in range(args.repeats):
                start = time.perf_counter()
                result = getattr(smo_solver, f"solve_{kind}")(
                    x,
                    y,
                    **kw,
                    working_set_size=size,
                    tol=args.tol,
                    time_limit=args.time_limit,
                    max_steps=args.max_steps,
                )
                seconds = time.perf_counter() - start
                row = dict(
                    case=name,
                    n=len(x),
                    working_set_size=size,
                    repeat=rep,
                    status=result.status,
                    tol=args.tol,
                    seconds=seconds,
                    solver_seconds=result.elapsed,
                    steps=result.steps,
                    violation=result.violation,
                    gap=result.gap,
                    dual_objective=result.dual_objective,
                    relative_gap=result.gap / max(1.0, abs(result.dual_objective)),
                    asum=float(abs(result.alpha).sum()),
                    max_asum=kw.get("max_asum"),
                    cache_hits=result.cache_hits,
                    cache_misses=result.cache_misses,
                    shrinks=result.shrinks,
                    unshrinks=result.unshrinks,
                )
                results.append((result, row))
                print(
                    f"  size {size:2d} run {rep}: {result.status:10s} {seconds:.6f}s "
                    f"steps {result.steps}, r {result.violation:.3g}, gap {result.gap:.3g}",
                    flush=True,
                )
        # Independent dense products are outside every timed call, never between repetitions.
        kernel = smo_solver.kernel_matrix(x, x, gamma=kw["gamma"])
        heldout = x[: min(200, len(x))] + np.random.default_rng(91).normal(
            scale=0.2, size=x[: min(200, len(x))].shape
        )
        heldout_kernel = smo_solver.kernel_matrix(heldout, x, gamma=kw["gamma"])
        predictions = []
        for result, row in results:
            measures, training = independent_measures(result, kernel, y, kind, kw)
            row.update(measures)
            predictions.append((training, heldout_kernel @ result.beta / kw["lam"] + result.b))
        converged = [
            (index, result, row)
            for index, (result, row) in enumerate(results)
            if result.converged and row["recomputed_violation"] <= args.tol * 1.01
        ]
        if converged:
            ref_index, ref, refrow = min(converged, key=lambda rr: rr[2]["recomputed_dual"])
            for index, (_result, row) in enumerate(results):
                row.update(
                    reference_size=refrow["working_set_size"],
                    dual_difference=row["recomputed_dual"] - refrow["recomputed_dual"],
                    max_training_difference=float(
                        np.max(abs(predictions[index][0] - predictions[ref_index][0]))
                    ),
                    max_prediction_difference=float(
                        np.max(abs(predictions[index][1] - predictions[ref_index][1]))
                    ),
                )
        rows.extend(row for _, row in results)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    metadata = args.out.with_suffix(".txt")
    metadata.write_text(
        f"Python {platform.python_version()}\n{platform.platform()}\n"
        f"smo-solver {smo_solver.__version__}\n"
        f"Logical CPUs: {os.cpu_count()}, RAYON_NUM_THREADS={os.getenv('RAYON_NUM_THREADS')}\n"
        f"Command arguments: {vars(args)}\n"
        "All measured calls run sequentially. No CPU affinity was set.\n" + provenance()
    )
    print(f"Wrote {len(rows)} records to {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--cases", help="Comma-separated names; default all")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results/block_working_sets.csv",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
