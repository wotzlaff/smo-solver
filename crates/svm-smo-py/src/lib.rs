//! Python bindings of `svm-smo` (module `svm_smo._core`).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use svm_smo::{
    solve_with, DualProblem, FeatureKernel, Kernel, KernelFunction, PrecomputedKernel, SmoOptions,
    Status, WorkingSetSelection,
};

/// Result of an SMO run.
#[pyclass(get_all, frozen, module = "svm_smo._core")]
struct SmoResult {
    /// Dual variables (length ``2 n`` for regression).
    alpha: Py<PyArray1<f64>>,
    /// Dual coefficient per training sample, ``beta_k = sum_{i: k(i) = k} alpha_i``.
    beta: Py<PyArray1<f64>>,
    /// Bias of the decision function.
    b: f64,
    /// Multiplier of the 1-norm constraint.
    c: f64,
    /// Decision values without bias, ``ka_i = 1/lam sum_j alpha_j K[k(i), k(j)]``.
    ka: Py<PyArray1<f64>>,
    /// One of ``"converged"``, ``"max_steps"``, ``"time_limit"``, ``"interrupted"``.
    status: &'static str,
    steps: u64,
    elapsed: f64,
    violation: f64,
    dual_objective: f64,
    primal_objective: f64,
    gap: f64,
    n_active: usize,
    cache_hits: u64,
    cache_misses: u64,
    lam: f64,
}

#[pymethods]
impl SmoResult {
    /// ``True`` if the solver reached the requested tolerance.
    #[getter]
    fn converged(&self) -> bool {
        self.status == "converged"
    }

    fn __repr__(&self) -> String {
        format!(
            "SmoResult(status={:?}, steps={}, violation={:.3e}, gap={:.3e}, b={:.6}, elapsed={:.3}s)",
            self.status, self.steps, self.violation, self.gap, self.b, self.elapsed
        )
    }
}

fn status_name(status: Status) -> &'static str {
    match status {
        Status::Converged => "converged",
        Status::MaxSteps => "max_steps",
        Status::TimeLimit => "time_limit",
        Status::Interrupted => "interrupted",
    }
}

fn working_set(name: &str) -> PyResult<WorkingSetSelection> {
    match name {
        "mvp" | "max_violating_pair" => Ok(WorkingSetSelection::MaximalViolatingPair),
        "mud" | "max_unbounded_decrease" => Ok(WorkingSetSelection::MaximalUnboundedDecrease),
        "md" | "max_decrease" => Ok(WorkingSetSelection::MaximalDecrease),
        _ => Err(PyValueError::new_err(format!(
            "unknown working set selection {name:?} (use 'mvp', 'mud' or 'md')"
        ))),
    }
}

/// Kernel specification shared by all solver entry points.
#[derive(Clone, Copy)]
enum KernelSpec {
    Function(KernelFunction),
    Precomputed,
}

fn kernel_spec(name: &str, gamma: f64, coef0: f64, degree: i32) -> PyResult<KernelSpec> {
    Ok(match name {
        "rbf" | "gaussian" => KernelSpec::Function(KernelFunction::Rbf { gamma }),
        "linear" => KernelSpec::Function(KernelFunction::Linear),
        "poly" | "polynomial" => KernelSpec::Function(KernelFunction::Polynomial {
            gamma,
            coef0,
            degree,
        }),
        "precomputed" => KernelSpec::Precomputed,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown kernel {name:?} (use 'rbf', 'linear', 'poly' or 'precomputed')"
            )))
        }
    })
}

/// Options and callback shared by all entry points.
struct Run<'py> {
    options: SmoOptions,
    alpha0: Option<PyReadonlyArray1<'py, f64>>,
    callback: Option<Py<PyAny>>,
}

fn solve_problem(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    spec: KernelSpec,
    problem: DualProblem,
    run: Run<'_>,
) -> PyResult<SmoResult> {
    let x = x.as_array();
    let (n, dim) = x.dim();
    let data = x
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("x must be a C-contiguous float64 array"))?;
    let alpha0 = match &run.alpha0 {
        Some(a) => Some(
            a.as_slice()
                .map_err(|_| PyValueError::new_err("alpha0 must be contiguous"))?,
        ),
        None => None,
    };
    let options = run.options;
    let callback = run.callback;
    let mut error: Option<PyErr> = None;

    let outcome = py.detach(|| {
        let on_progress = |progress: &svm_smo::Progress| -> bool {
            Python::attach(|py| {
                if let Err(e) = py.check_signals() {
                    error = Some(e);
                    return true;
                }
                let Some(cb) = &callback else { return false };
                let info = PyDict::new(py);
                let filled = info
                    .set_item("step", progress.step)
                    .and_then(|_| info.set_item("elapsed", progress.elapsed))
                    .and_then(|_| info.set_item("violation", progress.violation))
                    .and_then(|_| info.set_item("dual_objective", progress.dual_objective))
                    .and_then(|_| info.set_item("n_active", progress.n_active));
                match filled
                    .and_then(|_| cb.call1(py, (info,)))
                    .and_then(|r| r.is_truthy(py))
                {
                    Ok(stop) => stop,
                    Err(e) => {
                        error = Some(e);
                        true
                    }
                }
            })
        };
        let solve =
            |kernel: &dyn Kernel| solve_with(kernel, &problem, &options, alpha0, on_progress);
        match spec {
            KernelSpec::Function(f) => solve(&FeatureKernel::new(data, dim.max(1), f)),
            KernelSpec::Precomputed => {
                if n != dim {
                    return Err(svm_smo::ProblemError::LengthMismatch {
                        what: "precomputed kernel columns",
                        expected: n,
                        got: dim,
                    });
                }
                solve(&PrecomputedKernel::new(data, n))
            }
        }
    });
    // KeyboardInterrupt or an exception raised by the user callback
    if let Some(e) = error {
        return Err(e);
    }
    let res = outcome.map_err(|e| PyValueError::new_err(e.to_string()))?;
    let beta = problem.sample_coefficients(&res.alpha);
    Ok(SmoResult {
        alpha: res.alpha.into_pyarray(py).unbind(),
        beta: beta.into_pyarray(py).unbind(),
        b: res.b,
        c: res.c,
        ka: res.ka.into_pyarray(py).unbind(),
        status: status_name(res.status),
        steps: res.steps,
        elapsed: res.elapsed,
        violation: res.violation,
        dual_objective: res.dual_objective,
        primal_objective: res.primal_objective,
        gap: res.gap,
        n_active: res.n_active,
        cache_hits: res.cache_hits,
        cache_misses: res.cache_misses,
        lam: problem.lambda,
    })
}

fn n_rows(x: &PyReadonlyArray2<'_, f64>) -> usize {
    x.as_array().dim().0
}

fn check_samples(x: &PyReadonlyArray2<'_, f64>, len: usize) -> PyResult<()> {
    if n_rows(x) != len {
        return Err(PyValueError::new_err(format!(
            "x has {} rows but y has {} entries",
            n_rows(x),
            len
        )));
    }
    Ok(())
}

fn slice<'a>(a: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float64 array")))
}

macro_rules! run_args {
    ($tol:expr, $max_steps:expr, $time_limit:expr, $working_set:expr, $shrinking:expr,
     $shrinking_period:expr, $cache_size_mb:expr, $callback_period:expr, $alpha0:expr, $callback:expr) => {
        Run {
            options: SmoOptions {
                tol: $tol,
                max_steps: $max_steps,
                time_limit: $time_limit,
                working_set: working_set($working_set)?,
                shrinking: $shrinking,
                shrinking_period: $shrinking_period,
                cache_size_mb: $cache_size_mb,
                callback_period: $callback_period,
                ..SmoOptions::default()
            },
            alpha0: $alpha0,
            callback: $callback,
        }
    };
}

/// Binary classification with the (smoothed) hinge loss
/// ``w_i * m_delta(shift - y_i f(x_i))``; labels must be -1 or +1.
#[pyfunction]
#[pyo3(signature = (
    x, y, lam, *, kernel="rbf", gamma=1.0, coef0=0.0, degree=3, weights=None, shift=1.0,
    smoothing=0.0, max_asum=None, tol=1e-3, max_steps=100_000_000, time_limit=None,
    working_set="md", shrinking=true, shrinking_period=0, cache_size_mb=256.0,
    alpha0=None, callback=None, callback_period=1000
))]
#[allow(clippy::too_many_arguments)]
fn solve_classification(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    lam: f64,
    kernel: &str,
    gamma: f64,
    coef0: f64,
    degree: i32,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    shift: f64,
    smoothing: f64,
    max_asum: Option<f64>,
    tol: f64,
    max_steps: u64,
    time_limit: Option<f64>,
    working_set: &str,
    shrinking: bool,
    shrinking_period: u64,
    cache_size_mb: f64,
    alpha0: Option<PyReadonlyArray1<'_, f64>>,
    callback: Option<Py<PyAny>>,
    callback_period: u64,
) -> PyResult<SmoResult> {
    let ys = slice(&y, "y")?;
    check_samples(&x, ys.len())?;
    let w = weights.as_ref().map(|w| slice(w, "weights")).transpose()?;
    let problem = DualProblem::classification(ys, lam, w, shift, smoothing, max_asum)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let spec = kernel_spec(kernel, gamma, coef0, degree)?;
    let run = run_args!(
        tol,
        max_steps,
        time_limit,
        working_set,
        shrinking,
        shrinking_period,
        cache_size_mb,
        callback_period,
        alpha0,
        callback
    );
    solve_problem(py, x, spec, problem, run)
}

/// epsilon-insensitive regression with the (smoothed) loss
/// ``w_i * (m_delta(f(x_i) - y_i - epsilon) + m_delta(y_i - f(x_i) - epsilon))``.
#[pyfunction]
#[pyo3(signature = (
    x, y, lam, *, epsilon=0.1, kernel="rbf", gamma=1.0, coef0=0.0, degree=3, weights=None,
    smoothing=0.0, max_asum=None, tol=1e-3, max_steps=100_000_000, time_limit=None,
    working_set="md", shrinking=true, shrinking_period=0, cache_size_mb=256.0,
    alpha0=None, callback=None, callback_period=1000
))]
#[allow(clippy::too_many_arguments)]
fn solve_regression(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    lam: f64,
    epsilon: f64,
    kernel: &str,
    gamma: f64,
    coef0: f64,
    degree: i32,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    smoothing: f64,
    max_asum: Option<f64>,
    tol: f64,
    max_steps: u64,
    time_limit: Option<f64>,
    working_set: &str,
    shrinking: bool,
    shrinking_period: u64,
    cache_size_mb: f64,
    alpha0: Option<PyReadonlyArray1<'_, f64>>,
    callback: Option<Py<PyAny>>,
    callback_period: u64,
) -> PyResult<SmoResult> {
    let ys = slice(&y, "y")?;
    check_samples(&x, ys.len())?;
    let w = weights.as_ref().map(|w| slice(w, "weights")).transpose()?;
    let problem = DualProblem::regression(ys, lam, w, epsilon, smoothing, max_asum)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let spec = kernel_spec(kernel, gamma, coef0, degree)?;
    let run = run_args!(
        tol,
        max_steps,
        time_limit,
        working_set,
        shrinking,
        shrinking_period,
        cache_size_mb,
        callback_period,
        alpha0,
        callback
    );
    solve_problem(py, x, spec, problem, run)
}

/// Least-squares SVM with loss ``w_i / 2 * (f(x_i) - y_i)^2``.
#[pyfunction]
#[pyo3(signature = (
    x, y, lam, *, kernel="rbf", gamma=1.0, coef0=0.0, degree=3, weights=None,
    tol=1e-3, max_steps=100_000_000, time_limit=None, working_set="md",
    cache_size_mb=256.0, alpha0=None, callback=None, callback_period=1000
))]
#[allow(clippy::too_many_arguments)]
fn solve_least_squares(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    lam: f64,
    kernel: &str,
    gamma: f64,
    coef0: f64,
    degree: i32,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    tol: f64,
    max_steps: u64,
    time_limit: Option<f64>,
    working_set: &str,
    cache_size_mb: f64,
    alpha0: Option<PyReadonlyArray1<'_, f64>>,
    callback: Option<Py<PyAny>>,
    callback_period: u64,
) -> PyResult<SmoResult> {
    let ys = slice(&y, "y")?;
    check_samples(&x, ys.len())?;
    let w = weights.as_ref().map(|w| slice(w, "weights")).transpose()?;
    let problem =
        DualProblem::least_squares(ys, lam, w).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let spec = kernel_spec(kernel, gamma, coef0, degree)?;
    let run = run_args!(
        tol,
        max_steps,
        time_limit,
        working_set,
        false,
        0,
        cache_size_mb,
        callback_period,
        alpha0,
        callback
    );
    solve_problem(py, x, spec, problem, run)
}

/// General separable quadratic dual problem
///
/// ``min 1/(2 lam) a^T K a + sum(q/2 a^2 + p a)``
/// s.t. ``sum(a) = 0``, ``lo <= a <= hi`` and optionally ``sum(sign * a) <= max_asum``.
/// Variable ``i`` uses kernel sample ``kernel_index[i]`` (default: ``i``).
#[pyfunction]
#[pyo3(signature = (
    x, lam, q, p, lo, hi, *, kernel_index=None, sign=None, max_asum=None, kernel="rbf",
    gamma=1.0, coef0=0.0, degree=3, tol=1e-3, max_steps=100_000_000, time_limit=None,
    working_set="md", shrinking=true, shrinking_period=0, cache_size_mb=256.0,
    alpha0=None, callback=None, callback_period=1000
))]
#[allow(clippy::too_many_arguments)]
fn solve_dual(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    lam: f64,
    q: PyReadonlyArray1<'_, f64>,
    p: PyReadonlyArray1<'_, f64>,
    lo: PyReadonlyArray1<'_, f64>,
    hi: PyReadonlyArray1<'_, f64>,
    kernel_index: Option<Vec<usize>>,
    sign: Option<PyReadonlyArray1<'_, f64>>,
    max_asum: Option<f64>,
    kernel: &str,
    gamma: f64,
    coef0: f64,
    degree: i32,
    tol: f64,
    max_steps: u64,
    time_limit: Option<f64>,
    working_set: &str,
    shrinking: bool,
    shrinking_period: u64,
    cache_size_mb: f64,
    alpha0: Option<PyReadonlyArray1<'_, f64>>,
    callback: Option<Py<PyAny>>,
    callback_period: u64,
) -> PyResult<SmoResult> {
    let q = slice(&q, "q")?.to_vec();
    let n = q.len();
    let problem = DualProblem {
        lambda: lam,
        kernel_index: kernel_index.unwrap_or_else(|| (0..n).collect()),
        q,
        p: slice(&p, "p")?.to_vec(),
        lo: slice(&lo, "lo")?.to_vec(),
        hi: slice(&hi, "hi")?.to_vec(),
        sign: match &sign {
            Some(s) => slice(s, "sign")?.to_vec(),
            None => vec![1.0; n],
        },
        max_asum,
    };
    let spec = kernel_spec(kernel, gamma, coef0, degree)?;
    let run = run_args!(
        tol,
        max_steps,
        time_limit,
        working_set,
        shrinking,
        shrinking_period,
        cache_size_mb,
        callback_period,
        alpha0,
        callback
    );
    solve_problem(py, x, spec, problem, run)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SmoResult>()?;
    m.add_function(wrap_pyfunction!(solve_classification, m)?)?;
    m.add_function(wrap_pyfunction!(solve_regression, m)?)?;
    m.add_function(wrap_pyfunction!(solve_least_squares, m)?)?;
    m.add_function(wrap_pyfunction!(solve_dual, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
