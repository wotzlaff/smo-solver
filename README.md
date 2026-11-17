# smo-solver

[![PyPI](https://img.shields.io/pypi/v/smo-solver)](https://pypi.org/project/smo-solver/)
[![CI](https://github.com/wotzlaff/smo-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/wotzlaff/smo-solver/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-demo%20%26%20benchmarks-1d4ed8)](https://wotzlaff.github.io/svm-site/)

**➜ [Try the solver in your browser (WebAssembly demo), see the benchmarks and a summary of the thesis on the project website](https://wotzlaff.github.io/svm-site/)**

**Fast Sequential Minimal Optimization (SMO) for smoothed, weighted and ν-constrained support vector machines — written in Rust, usable from Python.**

`smo-solver` solves the dual training problem of kernel support vector machines for a whole
family of loss functions: the classical hinge and ε-insensitive losses, their *quadratically
smoothed* versions, the squared loss (LS-SVM), arbitrary sample weights and an optional bound on
the 1-norm of the dual variables (a generalised ν-SVM). It implements the methods analysed in

> N. Strasdat, *Duality, Derivative-Based Training Methods and Hyperparameter Optimization for
> Support Vector Machines*, Dissertation, Technische Universität Dresden, 2023.
> [urn:nbn:de:bsz:14-qucosa2-874731](https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731)

and is the training backend of [`optimized-svm`](https://github.com/wotzlaff/optimized-svm),
which tunes SVM hyperparameters with exact gradients.

## Features

- one solver for a **general separable quadratic dual** (hinge, ε-insensitive, smoothed variants,
  least squares, custom problems via `solve_dual`)
- **working set selection**: maximal violating pair, second-order (maximal unbounded decrease)
  and bound-aware maximal decrease (default)
- optional **joint block updates** (`working_set_size=4`, or larger, up to 64);
  default 2 retains classic SMO
- **shrinking** with the fixed-variable detection rule of the thesis (tuned defaults, see below), **LRU kernel row cache**
- **warm starts**, time limits, progress callbacks, `Ctrl-C` support
- Python: **zero-copy NumPy input**, GIL released while solving, scikit-learn compatible
  estimators `SmoSVC` (binary and one-vs-rest), `SmoSVR`, `SmoLSSVR`
- every solution reports its **duality gap**, so optimality can be checked independently

## Installation

```bash
pip install smo-solver              # solver only (NumPy)
pip install "smo-solver[sklearn]"   # + scikit-learn estimators
```

Building from source requires a Rust toolchain (≥ 1.75):

```bash
git clone https://github.com/wotzlaff/smo-solver && cd smo-solver
uv venv && uv pip install maturin && uv run maturin develop --release
```

## Quick start (Python)

```python
import numpy as np
from smo_solver import SmoSVC, solve_classification

rng = np.random.default_rng(0)
x = rng.normal(size=(500, 2))
y = np.where(x[:, 0] * x[:, 1] > 0, 1.0, -1.0)

# scikit-learn estimator (lam = 1 / C)
clf = SmoSVC(lam=0.01, gamma=1.0, smoothing=0.1).fit(x, y)
print(clf.score(x, y), clf.n_support_)

# low-level solver: full access to the dual solution
res = solve_classification(x, y, lam=0.01, gamma=1.0, smoothing=0.1, tol=1e-6)
print(res)            # SmoResult(status="converged", steps=..., gap=...)
res.alpha, res.b      # dual variables and bias
res.beta / res.lam    # kernel expansion coefficients per sample
```

## Quick start (Rust)

```rust
use smo_solver::{solve, DualProblem, FeatureKernel, KernelFunction, Model, SmoOptions};

let x = [0.0, 0.0, 0.2, 0.1, 1.0, 1.0, 0.9, 1.2]; // 4 samples, row-major
let y = [-1.0, -1.0, 1.0, 1.0];
let f = KernelFunction::Rbf { gamma: 1.0 };
let kernel = FeatureKernel::new(&x, 2, f);
let problem = DualProblem::classification(&y, 0.1, None, 1.0, 0.0, None)?;
let result = solve(&kernel, &problem, &SmoOptions::default())?;
let model = Model::new(f, &x, 2, &problem, &result);
assert!(model.decision(&[1.0, 1.1]) > 0.0);
```

## The problem class

For a kernel matrix `K`, regularization `λ > 0` and per-variable data `q_i ≥ 0`, `p_i`,
`lo_i ≤ 0 ≤ hi_i`, the solver computes

```
min_α  1/(2λ) αᵀKα + Σ_i (½ q_i α_i² + p_i α_i)
s.t.   Σ_i α_i = 0,   lo_i ≤ α_i ≤ hi_i,   [ Σ_i s_i α_i ≤ A ]
```

which is the dual of `min_{w,b} λ/2 ‖w‖² + Σ_i ℓ_i(⟨w, φ(x_i)⟩ + b)` whenever the conjugate of the
loss is quadratic on an interval. The decision function is `f(x) = 1/λ Σ_k β_k K(x, x_k) + b`.

| Problem | loss `ℓ_i(t)` | constructor |
|---|---|---|
| SVC | `w_i · m_δ(shift − y_i t)` | `solve_classification` / `DualProblem::classification` |
| ε-SVR | `w_i · (m_δ(t − y_i − ε) + m_δ(y_i − t − ε))` | `solve_regression` / `DualProblem::regression` |
| LS-SVM | `w_i/2 · (t − y_i)²` | `solve_least_squares` / `DualProblem::least_squares` |

`m_δ` is the quadratically smoothed maximum (`max(0, t)` for `δ = 0`). The stopping tolerance
uses the same scaling as LIBSVM; with `δ = 0` and `λ = 1/C` the solutions coincide with
`sklearn.svm.SVC`/`SVR`.

### Larger working sets (0.2.0)

`working_set_size` selects how many dual variables are jointly optimized per outer iteration:

```python
from smo_solver import SmoSVR, solve_regression

res = solve_regression(
    x, targets, lam=1e-4, smoothing=0.1, max_asum=10,
    working_set_size=4, tol=1e-6,
)
# Also available on SmoSVC, SmoSVR and SmoLSSVR:
reg = SmoSVR(working_set_size=8)
```

The accepted range is 2–64, including odd sizes; a small problem or shrunk active set may use
fewer variables. `working_set` still chooses the seed pair. A larger block adds matching
regression copies and gradient candidates, then solves the dense local quadratic program with
the sum constraint, box bounds and the optional 1-norm inequality together. The inequality can
become active or be released. This can address zig-zagging between the two sign groups when
the norm bound is active and lambda is small.

An active-set method solves the block problem, with a pair-step fallback if a block cannot
be accepted safely. The default remains 2, preserving the existing fast SMO path. Larger
blocks do more work per iteration and need not be faster. `steps`, `max_steps`, shrinking
periods and callback periods count **outer iterations**, not pair-equivalent work. The
stopping measure and tolerance retain their meaning. Benchmark wall time at comparable final
violation and duality gap rather than comparing iteration counts alone.

`benchmarks/block_working_sets.py` compares sizes 2, 4, 8 and 16 with independently rebuilt
kernel products, KKT violations and duality gaps. Release extension, one Rayon thread,
`tol=1e-6`, minimum of three sequential runs on one shared machine:

| problem | size 2 | size 4 | size 8 | size 16 |
|---|---:|---:|---:|---:|
| saved difficult ν-SVR, n = 5 | 0.386 s | 0.000174 s | 0.000200 s | 0.000026 s |
| ν-SVR, n = 1000, λ = 0.01 | 9.33 s | 10.83 s† | 4.36 s | 3.38 s |
| ordinary SVC, n = 1000 | 0.031 s | 0.102 s | 0.114 s | 0.189 s |
| ordinary smoothed SVR, n = 1000 | 0.448 s | 1.987 s | 2.102 s | 2.933 s |

† One additional run with a 20 s limit, after all three 10 s runs stopped just short of the
target. All displayed runs converged at the same requested tolerance. In the n = 5 example
there are ten dual variables, so size 16 solves the entire problem in one outer iteration;
size 4 uses 33 iterations versus 1,744,616 pair steps. Recomputed gaps are below `6e-12`.
For ν-SVR at n = 1000, sizes 8 and 16 are about 2.1× and 2.8× faster than pairs, with
recomputed gaps below `3e-10`. On ordinary problems the block overhead outweighs the reduced
iteration count. At the more difficult λ = 1e-4, every tested size stopped at the 10 s limit;
size 16 had a smaller remaining gap, but none reached the target. These results support an
explicit option, rather than a new default.

Raw records and build provenance are in `benchmarks/results/block_working_sets*.{csv,txt}`.
Reproduce the ordinary/hard cases and the converged moderate case with:

```bash
RAYON_NUM_THREADS=1 python benchmarks/block_working_sets.py --n 1000 \
    --cases saved_nu_svr,nu_svr,svc,svr
RAYON_NUM_THREADS=1 python benchmarks/block_working_sets.py --n 1000 \
    --cases nu_svr_moderate --out benchmarks/results/block_working_sets_moderate.csv
RAYON_NUM_THREADS=1 python benchmarks/block_working_sets.py --n 1000 \
    --cases nu_svr_moderate --sizes 4 --repeats 1 --time-limit 20 \
    --out benchmarks/results/block_working_sets_moderate_size4.csv
```

`benchmarks/block_hpo.py` measures complete `optimized-svm` fits (180 samples, three CV folds,
SMO warm starts plus Newton and adjoint gradients). After correcting the Newton norm-bound
activity check in `optimized-svm`, all sizes give the same CV error to roundoff. Minimum of
two sequential fits, one BLAS/Rayon thread:

| complete HPO fit | size 2 | size 4 | size 8 |
|---|---:|---:|---:|
| SVC | 0.226 s | 0.228 s | 0.233 s |
| ν-SVR, active norm bound | 0.450 s | 0.306 s | 0.270 s |

All optimizations converged; each best CV error was recomputed from cold fold solutions and
each fold's norm bound checked outside the timed region. Run with the development sources:

```bash
PYTHONPATH=../optimized-svm/src:python RAYON_NUM_THREADS=1 python benchmarks/block_hpo.py
```

`block_hpo_before_norm_fix.*` contains diagnostic records from the previously incorrect
Newton active-set handling. Those runs are not a valid performance baseline.

### Stopping criterion

By default the solver stops when the violation measure `r(α) = max_{α_i > lo_i} g_i − min_{α_j < hi_j} g_j`
drops below `tol`. With `stopping="bound_aware"` it also stops when the bound-aware violation measure of
the thesis (Def. 7.21), normalized by `ρ`, is below `tol`:

```
r◇(α, ρ) / ρ = max_{i,j} min(ε_ij, ρ)/ρ · (g_j − g_i)  ≤  tol,   ε_ij = min(hi_i − α_i, α_j − lo_j)
```

Because `min(ε_ij, ρ) ≤ ρ` this measure never exceeds `r(α)`, so the criterion stops no later than the
default one; it discounts violating pairs that can only move a tiny distance. It costs `O(n log n)` and is
checked every `shrinking_period` steps. `rho` defaults to `λ` (the value used in the thesis, Sec. 10.2.2);
the criterion is not available together with `max_asum`.

Which measure is the right one? `benchmarks/stopping_study.py` follows 12 runs (classification and
ε-regression, with and without smoothing, three λ, n = 1500) and compares the measures of every iterate with the
actual error against a tight solution. The violation `r` and the bound-aware `r◇/λ` are interchangeable, and both
track the largest deviation of the decision function from the exact one almost one-to-one (80 % of the iterates
within a factor of 0.4 to 1.5): `r < τ` means `max |f − f*| ≲ τ`, so the tolerance is the accuracy of the
predictions. The duality gap is a poorer criterion: it grows with the number of samples, its relation to the
decision error varies over 2.7 decades (0.6 for `r`), and a gap threshold that is safe for an accuracy of `1e-3`
(2·10⁻⁵) is far stricter than `r < 1e-3`. The dual objective converges quadratically, so objective-based criteria
stop much too early or too late. The default stays as it is.

In our experiments the bound-aware criterion rarely differs from the default: with `ρ = λ` the two measures agree up to
pairs at a bound, in line with the observation `r ≈ r◇/λ` of the thesis (Sec. 10.2.2). It stops
earlier only for ε-regression at `n = 5000` (292 000 instead of 387 707 steps at `tol = 1e-3`, with a
duality gap of 0.32 instead of 0.08). Larger `ρ` stops earlier still, but trades accuracy for steps
about like a larger `tol` does.

## Performance

Training time against LIBSVM (charts and a comparison of hyperparameter search methods: [Benchmarks](https://wotzlaff.github.io/svm-site/benchmarks/)) (`sklearn.svm.SVC`/`SVR`, same `C = 1/λ`, `γ = 0.1`, tolerance
`1e-3`) on synthetic data with 10 features (`benchmarks/bench_vs_libsvm.py`, single machine,
indicative only):

| task | n | smo-solver | LIBSVM | SMO steps (smo-solver / LIBSVM) |
|---|---:|---:|---:|---:|
| classification | 2000 | 0.08 s | 0.10 s | 13 996 / 14 125 |
| classification | 5000 | 0.60 s | 0.73 s | 51 759 / 51 754 |
| ε-regression | 2000 | 0.51 s | 0.55 s | 42 591 / 44 255 |
| ε-regression | 5000 | 11.1 s | 9.6 s | 387 707 / 304 960 |

Classification needs as many steps as LIBSVM. The higher step count for ε-regression at
`n = 5000` is an effect of the stopping point, not of the working set rule: with LIBSVM's rule
(`working_set="mud"`) the solver stops after 281 751 steps, but with a duality gap of 0.35
instead of 0.08. At equal gap both rules need the same number of steps (`n = 3000`, tolerances
`1e-2` … `1e-4`). Almost all steps are unconstrained two-variable steps on the free variables,
so the convergence speed is set by the conditioning of the problem, not by the selection rule.
The loops over the active set use AVX2 when available (detected at run time).
Decision values agree to about `1e-3`.

## Working set rule and convergence

`benchmarks/working_set_study.py` repeats Sec. 10.2.1–10.2.3 of the thesis (checkerboard data, n = 1000, four
settings of λ and γ, hinge loss). The second-order rules (`md`, `mud`) need 4–15 times fewer steps than the
maximal violating pair (`mvp`) and are practically equal to each other (within about 10 %, in no consistent
order). The bound-aware pair (thesis Def. 7.21) behaves like `mvp`; the pair of maximal decrease over all pairs
saves only 1.2–1.3 times the steps at `O(n²)` cost per step. Both exist only in a NumPy reference SMO inside the
script, validated against the solver.

`benchmarks/convergence_study.py` repeats Sec. 10.2.4–10.2.7 and 10.2.9. Along an `md` run the duality gap,
the step length and the distance to the solution are proportional to the violation `r`, the decrease of the
objective to `λ r²`, with the same λ exponents as in the thesis; the gap is about `4.4 r` (thesis `5.8 r`). With
the smoothed hinge loss the gap falls like `r²`. The number of free variables hardly changes with λ or with n
(n = 1000 … 50 000), and the time to tolerance `1e-8` grows like `n^0.9 … n^1.4` (thesis `n^1.6`); raw data in
`benchmarks/results/`.

## Shrinking

Variables that sit at a bound and whose shifted gradient exceeds `shrinking_factor · r^shrinking_exponent`
(`r`: the violation measure) are removed from the active set; before termination the full problem is
restored and checked again, so the result does not depend on the heuristic. The defaults
(`shrinking_exponent = 1`, `shrinking_factor = 1`, `shrinking_min_fraction = 0.1`) come from a study
(`benchmarks/shrinking_study.py`, classification with n = 5000 and ε-regression with n = 2000, three
kernel widths, two regularization parameters, two tolerances, plus an active 1-norm bound):

| | geometric mean speed-up vs. no shrinking | worst case |
|---|---:|---:|
| earlier default (`exponent = 0.5`, no minimal fraction) | 1.25 | 0.62 |
| current default | 1.56 | 0.97 |

The speed-up grows with the problem size and with the requested accuracy (classification, n = 10 000: 1.6× at
tolerance `1e-2`, 2.5× at `1e-5`; ε-regression, n = 3000: 1.8–2.2×). A factor below 1 removes variables too early (twice as many steps because
variables have to return), a larger factor shrinks too late. An attempt that would remove at most 10 % of the
active set is skipped: with a narrow kernel almost every sample is a support vector, nothing can be shrunk and
the restoration of the full problem used to cost up to 60 % extra time. The result reports `shrinks`,
`unshrinks` (restorations before termination) and `min_active`.

## Testing

```bash
cargo test --workspace --release      # unit, integration and property tests
uv run pytest                         # Python tests (incl. scikit-learn checks)
```

Correctness is verified against independent references: zero duality gap, closed-form LS-SVM
solutions, LIBSVM (via scikit-learn), SciPy's SLSQP and primal formulations solved with cvxpy.

## Nonlinear optimization in the browser

The [standalone L-BFGS-B prototype](tools/browser-optimizer/) compiles to WebAssembly.
It takes a smooth objective and gradient from JavaScript and supports finite box
bounds. The included browser example optimizes a bounded Rosenbrock problem.

## Known limitations

- With an *active* 1-norm constraint and a very small `λ`, two-variable steps can zigzag between
  the sign groups (for ε-SVR, both copies of a sample share one kernel row); convergence then
  needs very many steps. The optional larger working sets address this case; see the
  comparison above and `benchmarks/block_working_sets.py`.
- Only quadratic conjugate losses are supported by SMO; smooth non-quadratic losses
  (e.g. logistic, Poisson) are handled by the Newton solver in `optimized-svm`.

## Citation

If you use this software, please cite the dissertation:

```bibtex
@phdthesis{strasdat2023,
  author = {Nico Strasdat},
  title  = {Duality, Derivative-Based Training Methods and
            Hyperparameter Optimization for Support Vector Machines},
  school = {Technische Universit{\"a}t Dresden},
  type   = {Dissertation},
  year   = {2023},
  url    = {https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731},
  note   = {urn:nbn:de:bsz:14-qucosa2-874731}
}
```

## License

BSD-3-Clause. The predecessor implementation in Nim is preserved in the git tag `legacy-nim`.
