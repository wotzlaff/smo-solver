# Changelog

## Unreleased

## 0.2.0 – 2026-11-17

### Added
- Optional joint block updates via `working_set_size` (2–64, default 2), in Rust, all Python
  solve functions and the scikit-learn estimators. A dense active-set QP handles the sum,
  box and optional 1-norm constraints together. The existing working-set rule chooses a
  seed pair; larger blocks include matching regression copies and additional candidates.
  Singular or rejected block steps fall back to the original pair update. Steps, callback
  periods and iteration limits count outer iterations; the default two-variable path is unchanged.
- The progress callback of the Python functions receives `alpha` (a copy of the current iterate, original order).
- `benchmarks/working_set_study.py` (thesis Sec. 10.2.1–10.2.3): MVP, baMVP, MuDP, MDP and tMDP compared by the
  development of the optimality measures; baMVP and tMDP in a NumPy reference SMO validated against the solver.
- `benchmarks/convergence_study.py` (Sec. 10.2.4–10.2.7, 10.2.9): relations between the measures, their λ factors,
  the smoothed hinge loss, support vectors over λ and scaling over the problem size.

### Fixed
- An infinite `max_asum` now contributes zero to the primal objective when its multiplier is
  zero, so the returned objective and duality gap remain finite instead of forming `inf * 0`.

## 0.1.3 – 2026-10-29

### Changed
- Shrinking defaults from a study (`benchmarks/shrinking_study.py`): `shrinking_exponent = 1` (was 0.5)
  and a new option `shrinking_min_fraction = 0.1` that skips attempts removing at most 10 % of the active
  set. Geometric mean speed-up over 24 settings against no shrinking 1.56 (was 1.25), worst case 0.97 (was
  0.62, narrow kernels); ε-regression is up to 1.9 times faster than before. Results are unchanged.

### Added
- `shrinking_factor`, `shrinking_exponent` and `shrinking_min_fraction` in the Python functions; the result
  reports `shrinks`, `unshrinks`, `min_active` and `bound_aware` (the bound-aware violation of the final point).
- `benchmarks/stopping_study.py`: which optimality measure predicts the error of the solution best.

## 0.1.2 – 2026-10-29

### Added
- Bound-aware stopping criterion (`stopping="bound_aware"`, `rho`; thesis Def. 7.21): stops when
  `r◇(α, ρ)/ρ ≤ tol`, evaluated in `O(n log n)` every `shrinking_period` steps. Available in the Rust
  crate (`StoppingCriterion`, `SmoOptions::{stopping, rho}`), the Python functions and the estimators.
  Not available together with `max_asum`; the default criterion is unchanged.

## 0.1.1 – 2026-10-29

### Changed
- Faster SMO (same iterates): distances to the bounds are kept as arrays, extremal-value scans
  and the second-order search work block-wise with a division-free filter, AVX2 kernels with
  run-time detection (scalar fallback), lazy compaction of cached rows on shrinking.
  ε-regression n = 5000: 38 s → 11 s, classification n = 5000: 1.3 s → 0.6 s.

### Documentation
- The extra SMO steps of ε-regression compared with LIBSVM at `n = 5000` are a stopping artefact
  (different duality gap at the same tolerance), see the performance section of the README.

## 0.1.0 – 2026-10-28

Complete rewrite of the former Nim package `nimsmo` (preserved as git tag `legacy-nim`).

### Added
- Rust crate `smo-solver`: SMO for a general separable quadratic dual (hinge, ε-insensitive,
  quadratically smoothed variants, least squares) with sample weights and an optional 1-norm
  constraint; working set rules MVP / maximal unbounded decrease / maximal decrease
  (one- or two-sided); shrinking; LRU kernel row cache; warm start; callbacks; time limit;
  duality gap and bound-aware violation measure.
- Kernels: RBF, linear, polynomial, precomputed. Optional rayon parallelism; the core crate
  builds for `wasm32-unknown-unknown`.
- `smo-solver-wasm`: 66 KB WebAssembly module with a plain C ABI (no JS glue) for browser demos.
- Python package `smo_solver` (PyO3, abi3 wheels for Python ≥ 3.10): zero-copy NumPy input,
  GIL released during training, `Ctrl-C` support, type stubs, scikit-learn estimators
  `SmoSVC` (incl. one-vs-rest), `SmoSVR`, `SmoLSSVR`.
- Tests against duality gap, closed-form LS-SVM, LIBSVM, SciPy and cvxpy; property tests;
  criterion and LIBSVM benchmarks; GitHub Actions for CI and wheel releases.

### Fixed (compared to `nimsmo`)
- step counter and status are reported correctly on time limit / callback termination
- time limit uses wall-clock time instead of CPU time
- pairs that decrease the 1-norm are no longer excluded while the constraint is active
- consistent meaning of `shrinking_period = 0` for all problem types
- bias estimated from free variables instead of the midpoint of the violating pair
