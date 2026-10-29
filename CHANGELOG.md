# Changelog

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
