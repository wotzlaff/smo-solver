# smo-solver

[![PyPI](https://img.shields.io/pypi/v/smo-solver)](https://pypi.org/project/smo-solver/)
[![CI](https://github.com/wotzlaff/smo-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/wotzlaff/smo-solver/actions/workflows/ci.yml)

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
- **shrinking** with the fixed-variable detection rule of the thesis, **LRU kernel row cache**
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

## Performance

Training time against LIBSVM (`sklearn.svm.SVC`/`SVR`, same `C = 1/λ`, `γ = 0.1`, tolerance
`1e-3`) on synthetic data with 10 features (`benchmarks/bench_vs_libsvm.py`, single machine,
indicative only):

| task | n | smo-solver | LIBSVM | SMO steps (smo-solver / LIBSVM) |
|---|---:|---:|---:|---:|
| classification | 2000 | 0.12–0.18 s | 0.12 s | 13 996 / 14 125 |
| classification | 5000 | 1.2–1.4 s | 0.71 s | 51 759 / 51 754 |
| ε-regression | 2000 | 1.2–1.4 s | 0.67 s | 42 591 / 44 255 |
| ε-regression | 5000 | 24 s | 9.8 s | 286 566 / 304 960 |

The number of steps matches LIBSVM; the remaining factor of about 1.5–2.5 per step is a known
optimisation target (see the roadmap). Decision values agree to about `1e-3`.

## Testing

```bash
cargo test --workspace --release      # unit, integration and property tests
uv run pytest                         # Python tests (incl. scikit-learn checks)
```

Correctness is verified against independent references: zero duality gap, closed-form LS-SVM
solutions, LIBSVM (via scikit-learn), SciPy's SLSQP and primal formulations solved with cvxpy.

## Known limitations

- With an *active* 1-norm constraint and a very small `λ`, two-variable steps can zigzag between
  the sign groups (for ε-SVR, both copies of a sample share one kernel row); convergence then
  needs very many steps.
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
