//! WebAssembly interface of `smo-solver` for the browser demo.
//!
//! The interface is a plain C ABI: JavaScript allocates `f64` buffers with [`alloc`], writes the
//! data into `memory`, calls [`train`] and evaluates the resulting model with [`decision`].
//! The model is kept in a global slot (the demo trains one model at a time).

mod cv;
mod linalg;
mod logistic;
mod trace;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

use smo_solver::{solve, DualProblem, FeatureKernel, KernelFunction, Model, SmoOptions, SmoResult};

/// The model of the last [`train`] call with the statistics reported by [`stat`]
/// (`[bias, gap, objective, seconds, violation]`).
struct Trained {
    model: Model,
    stats: [f64; 5],
}

impl Trained {
    fn smo(model: Model, r: &SmoResult) -> Self {
        Self {
            model,
            stats: [r.b, r.gap, r.dual_objective, r.elapsed, r.violation],
        }
    }

    fn newton(model: Model, fit: &logistic::Fit, seconds: f64) -> Self {
        Self {
            model,
            stats: [fit.b, fit.residual, fit.objective, seconds, fit.residual],
        }
    }
}

/// Wall-clock seconds since `start` (zero on wasm32 where `std::time` is unavailable).
#[cfg(not(target_arch = "wasm32"))]
fn seconds_since(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn seconds_since(_: ()) -> f64 {
    0.0
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> std::time::Instant {
    std::time::Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn now() {}

static STATE: Mutex<Option<Trained>> = Mutex::new(None);
static WARM: Mutex<Option<cv::WarmStart>> = Mutex::new(None);
static TRACE: Mutex<Option<trace::Trace>> = Mutex::new(None);

/// Locks a global slot, ignoring poisoning (a panic must not disable later calls).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Allocates a buffer for `len` `f64` values and returns its address.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut f64 {
    let mut v = vec![0.0f64; len];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

/// Frees a buffer obtained from [`alloc`].
///
/// # Safety
/// `ptr` and `len` must come from a previous call of [`alloc`].
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut f64, len: usize) {
    drop(Vec::from_raw_parts(ptr, len, len));
}

/// Trains a model on `n` samples with `dim` features (row-major `x`) and targets `y`.
///
/// `kind = 0`: classification (labels ±1), `kind = 1`: ε-regression, `kind = 2`: kernel logistic
/// regression (labels ±1, Strasdat Sec. 4.1.3) trained by a dense damped Newton method.
/// Returns the number of SMO steps (Newton iterations for `kind = 2`), or a negative value on
/// invalid input (`-2` unknown kind, `-3` invalid problem or labels, `-4` solver failure).
///
/// For `kind = 2` the regularisation is the same as for the SVM (`λ/2 |w|² + Σ log(1 + e^{-y f})`),
/// `smoothing` and `epsilon` are ignored and `tol` is the stopping tolerance on the optimality
/// norm `sqrt(mean((a + φ')²) + (Σ φ')²)` (use about `1e-8`); at most 100 iterations. The model
/// in the global slot has all samples with nonzero coefficient as [`support`] (generically all
/// of them). [`decision`] returns the logit `f`; the class probability of `+1` is `σ(f)` =
/// `1 / (1 + e^{-f})`. [`stat`]: 1 = final optimality norm (in place of the duality gap),
/// 2 = primal objective, 4 = optimality norm, 3 = seconds (0 in WebAssembly).
///
/// # Safety
/// `x` must point to `n * dim` and `y` to `n` valid `f64` values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn train(
    x: *const f64,
    n: usize,
    dim: usize,
    y: *const f64,
    kind: u32,
    lambda: f64,
    gamma: f64,
    smoothing: f64,
    epsilon: f64,
    tol: f64,
) -> i64 {
    if n == 0 || dim == 0 {
        return -1;
    }
    let x = std::slice::from_raw_parts(x, n * dim);
    let y = std::slice::from_raw_parts(y, n);
    if kind == 2 {
        return train_logistic(x, dim, y, lambda, gamma, tol, None);
    }
    let problem = match kind {
        0 => DualProblem::classification(y, lambda, None, 1.0, smoothing, None),
        1 => DualProblem::regression(y, lambda, None, epsilon, smoothing, None),
        _ => return -2,
    };
    let Ok(problem) = problem else { return -3 };
    let function = KernelFunction::Rbf { gamma };
    let kernel = FeatureKernel::new(x, dim, function);
    let options = SmoOptions {
        tol,
        max_steps: 2_000_000,
        ..SmoOptions::default()
    };
    let Ok(result) = solve(&kernel, &problem, &options) else {
        return -4;
    };
    let model = Model::new(function, x, dim, &problem, &result);
    let steps = result.steps as i64;
    *lock(&STATE) = Some(Trained::smo(model, &result));
    steps
}

/// Newton training for `kind = 2`; with `record = Some((period, max_frames))` also fills the trace.
fn train_logistic(
    x: &[f64],
    dim: usize,
    y: &[f64],
    lambda: f64,
    gamma: f64,
    tol: f64,
    record: Option<(u64, usize)>,
) -> i64 {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !(positive(lambda) && positive(gamma) && positive(tol)) || !logistic::valid_labels(y) {
        return -3;
    }
    let start = now();
    let run = catch_unwind(AssertUnwindSafe(|| {
        logistic::train(x, dim, y, lambda, gamma, tol, record)
    }));
    let Ok(Ok((fit, trace))) = run else { return -4 };
    let function = KernelFunction::Rbf { gamma };
    let model = logistic::model(function, x, dim, lambda, &fit);
    if trace.is_some() {
        *lock(&TRACE) = trace;
    }
    *lock(&STATE) = Some(Trained::newton(model, &fit, seconds_since(start)));
    fit.steps as i64
}

/// Evaluates the decision function at `m` points (row-major, same dimension) into `out`.
///
/// # Safety
/// `points` must point to `m * dim` and `out` to `m` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn decision(points: *const f64, m: usize, out: *mut f64) -> i32 {
    let guard = lock(&STATE);
    let Some(trained) = guard.as_ref() else {
        return -1;
    };
    let dim = trained.model.dim;
    let points = std::slice::from_raw_parts(points, m * dim);
    let out = std::slice::from_raw_parts_mut(out, m);
    for (o, p) in out.iter_mut().zip(points.chunks_exact(dim)) {
        *o = trained.model.decision(p);
    }
    0
}

/// Writes the indices of the support vectors into `out` (as `f64`) and returns their number.
/// With `out = null` only the number is returned.
///
/// # Safety
/// `out` must be null or point to at least `n_support` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn support(out: *mut f64) -> i64 {
    let guard = lock(&STATE);
    let Some(trained) = guard.as_ref() else {
        return -1;
    };
    let s = &trained.model.support;
    if !out.is_null() {
        let out = std::slice::from_raw_parts_mut(out, s.len());
        for (o, &k) in out.iter_mut().zip(s) {
            *o = k as f64;
        }
    }
    s.len() as i64
}

/// Statistics of the last run: 0 = bias, 1 = duality gap, 2 = dual objective, 3 = seconds,
/// 4 = final violation (see [`train`] for `kind = 2`).
#[no_mangle]
pub extern "C" fn stat(which: u32) -> f64 {
    let guard = lock(&STATE);
    let Some(t) = guard.as_ref() else {
        return f64::NAN;
    };
    t.stats.get(which as usize).copied().unwrap_or(f64::NAN)
}

/// Cross-validation objective and its exact hypergradient for an RBF-kernel SVM.
///
/// Data layout as in [`train`]: `x` is row-major `n * dim`, `y` has `n` values (labels ±1 for
/// `kind = 0`). `fold_of` holds the fold id `0..k-1` of every sample (as `f64`); fold `i` is
/// validated on its samples and trained on all others. `kind`, `lambda`, `gamma`, `smoothing`
/// (training loss `δ`, `0` = plain hinge/ε-loss) and `epsilon` are as in [`train`]; `tol` is the
/// SMO tolerance (use about `1e-8` for gradients accurate to ~1e-6).
///
/// Writes `out[0]` = Σ fold validation errors / `n`, `out[1]` = d/d log λ, `out[2]` = d/d log γ
/// (log of the RBF `gamma`; the Python reference differentiates w.r.t. `log s` with
/// `γ = s²`, which is exactly twice this value). The validation losses are the defaults of
/// `SmoothHingeValidation()` (kind 0) and `SmoothAbsoluteValidation()` (kind 1). With
/// `smoothing = 0` only the explicit part of the gradient (dual variables and bias held fixed)
/// is returned, as in the Python reference.
///
/// Dual variables of the last call are kept as warm start while `y`, `fold_of` and `kind` are
/// unchanged. Returns 0, or a negative status: -1 invalid input, -2 unknown kind, -3 invalid
/// problem, -4 SMO did not converge, -5 linear algebra failure or non-finite result.
///
/// `kind = 2` is kernel logistic regression: every fold is trained by Newton's method (`tol` =
/// tolerance on the optimality norm, `smoothing` and `epsilon` ignored, labels must be ±1), the
/// validation loss is `LogisticValidation()` (`Σ log(1 + e^{-y f})`) and the gradient is the exact
/// hypergradient (adjoint system with `h = σ(1 - σ)` on all training samples). The warm start of
/// a fold is its previous Newton solution.
///
/// # Safety
/// `x` must point to `n * dim`, `y` and `fold_of` to `n` and `out` to 3 valid `f64` values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cv_objective(
    x: *const f64,
    n: usize,
    dim: usize,
    y: *const f64,
    fold_of: *const f64,
    k: usize,
    kind: u32,
    log_lambda: f64,
    log_gamma: f64,
    smoothing: f64,
    epsilon: f64,
    tol: f64,
    out: *mut f64,
) -> i32 {
    if x.is_null() || y.is_null() || fold_of.is_null() || out.is_null() || n == 0 || dim == 0 {
        return cv::ERR_INPUT;
    }
    let x = std::slice::from_raw_parts(x, n * dim);
    let y = std::slice::from_raw_parts(y, n);
    let fold_of = std::slice::from_raw_parts(fold_of, n);
    let params = cv::Params {
        kind,
        lambda: log_lambda.exp(),
        gamma: log_gamma.exp(),
        smoothing,
        epsilon,
        tol,
        ard: false,
    };
    let mut res = [0.0; 3];
    let status = cv_run(x, dim, y, fold_of, k, &params, &mut res);
    if status == 0 {
        std::slice::from_raw_parts_mut(out, 3).copy_from_slice(&res);
    }
    status
}

/// Runs [`cv::cv`] with the global warm start, converting panics into a status code.
fn cv_run(
    x: &[f64],
    dim: usize,
    y: &[f64],
    fold_of: &[f64],
    k: usize,
    params: &cv::Params,
    res: &mut [f64],
) -> i32 {
    let run = catch_unwind(AssertUnwindSafe(|| {
        let mut slot = lock(&WARM);
        let warm = slot.get_or_insert_with(Default::default);
        let mut tmp = vec![0.0; res.len()];
        let status = cv::cv(x, dim, y, fold_of, k, params, warm, &mut tmp);
        (status, tmp)
    }));
    match run {
        Ok((Ok(()), tmp)) => {
            res.copy_from_slice(&tmp);
            0
        }
        Ok((Err(code), _)) => code,
        Err(_) => cv::ERR_LINALG,
    }
}

/// Like [`cv_objective`], but with one RBF width per feature (automatic relevance
/// determination): `K(x, z) = exp(-Σ_d γ_d (x_d - z_d)²)` with `γ_d = exp(log_gammas[d])`.
///
/// `log_gammas` holds `dim` values; `out` receives `2 + dim` values: the CV error, `d/d log λ`
/// and `d/d log γ_d` for `d = 0..dim-1` (the Python reference's `d/d log s_d` with `γ_d = s_d²`
/// is twice these). With all `γ_d` equal the value equals that of [`cv_objective`] and the
/// width entries sum to its `d/d log γ`. Warm starts, status codes and all other arguments are as
/// in [`cv_objective`], for `kind` 0, 1 and 2.
///
/// Training at given `γ_d` needs no new export: scale feature `d` of `x` by `sqrt(γ_d)` in
/// JavaScript and call [`train`] with `gamma = 1`. (This is also how the CV is evaluated here.)
/// Memory is `O(n²)`, time `O(n² · dim)` per fold.
///
/// # Safety
/// `x` must point to `n * dim`, `y` and `fold_of` to `n`, `log_gammas` to `dim` and `out` to
/// `2 + dim` valid `f64` values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cv_objective_ard(
    x: *const f64,
    n: usize,
    dim: usize,
    y: *const f64,
    fold_of: *const f64,
    k: usize,
    kind: u32,
    log_lambda: f64,
    log_gammas: *const f64,
    smoothing: f64,
    epsilon: f64,
    tol: f64,
    out: *mut f64,
) -> i32 {
    if x.is_null()
        || y.is_null()
        || fold_of.is_null()
        || log_gammas.is_null()
        || out.is_null()
        || n == 0
        || dim == 0
    {
        return cv::ERR_INPUT;
    }
    let x = std::slice::from_raw_parts(x, n * dim);
    let y = std::slice::from_raw_parts(y, n);
    let fold_of = std::slice::from_raw_parts(fold_of, n);
    let log_gammas = std::slice::from_raw_parts(log_gammas, dim);
    let scale: Vec<f64> = log_gammas.iter().map(|l| (0.5 * l).exp()).collect();
    if scale.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return cv::ERR_INPUT;
    }
    let xs: Vec<f64> = x
        .chunks(dim)
        .flat_map(|r| r.iter().zip(&scale).map(|(v, s)| v * s))
        .collect();
    let params = cv::Params {
        kind,
        lambda: log_lambda.exp(),
        gamma: 1.0,
        smoothing,
        epsilon,
        tol,
        ard: true,
    };
    let mut res = vec![0.0; 2 + dim];
    let status = cv_run(&xs, dim, y, fold_of, k, &params, &mut res);
    if status == 0 {
        std::slice::from_raw_parts_mut(out, 2 + dim).copy_from_slice(&res);
    }
    status
}

/// Like [`train`] (same arguments), but also records the SMO iteration: an initial frame
/// (`α = 0`), one frame every `period` steps (at most `max_frames - 1` of them, so a long run
/// is recorded only at its beginning) and always the final state as the last frame. The final
/// model is stored in the global slot as by [`train`], so [`decision`], [`support`] and [`stat`]
/// keep working. Shrinking is disabled while recording. Returns the number of SMO steps or a
/// negative value on invalid input (`-5`: `period = 0` or `max_frames < 2`).
///
/// `kind = 2` records the Newton iteration instead: one frame at the start (`a = 0`, `b = 0`),
/// one every `period` iterations, and the final state; a frame holds the optimality norm in the
/// `gap` and `violation` slots (the initial violation is `NaN`) and `coef_k = a_k / λ`. Returns the
/// number of Newton iterations.
///
/// # Safety
/// `x` must point to `n * dim` and `y` to `n` valid `f64` values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn train_trace(
    x: *const f64,
    n: usize,
    dim: usize,
    y: *const f64,
    kind: u32,
    lambda: f64,
    gamma: f64,
    smoothing: f64,
    epsilon: f64,
    tol: f64,
    period: usize,
    max_frames: usize,
) -> i64 {
    if n == 0 || dim == 0 {
        return -1;
    }
    if period == 0 || max_frames < 2 {
        return -5;
    }
    let x = std::slice::from_raw_parts(x, n * dim);
    let y = std::slice::from_raw_parts(y, n);
    if kind == 2 {
        return train_logistic(
            x,
            dim,
            y,
            lambda,
            gamma,
            tol,
            Some((period as u64, max_frames)),
        );
    }
    let problem = match kind {
        0 => DualProblem::classification(y, lambda, None, 1.0, smoothing, None),
        1 => DualProblem::regression(y, lambda, None, epsilon, smoothing, None),
        _ => return -2,
    };
    let Ok(problem) = problem else { return -3 };
    let function = KernelFunction::Rbf { gamma };
    let Ok((trace, result, model)) =
        trace::record(x, dim, &problem, function, tol, period as u64, max_frames)
    else {
        return -4;
    };
    let steps = result.steps as i64;
    *lock(&TRACE) = Some(trace);
    *lock(&STATE) = Some(Trained::smo(model, &result));
    steps
}

/// Number of frames of the last [`train_trace`] run (`-1` if there is none).
#[no_mangle]
pub extern "C" fn trace_len() -> i64 {
    lock(&TRACE).as_ref().map_or(-1, |t| t.frames.len() as i64)
}

/// Writes frame `i` into `out` as `[step, duality gap, violation, bias, coef_0, ..., coef_{n-1}]`
/// (`4 + n` values; `coef_k = β_k / λ` is the coefficient of training sample `k`, so the decision
/// function is `Σ coef_k K(x, x_k) + bias`). The initial frame has violation `NaN`.
/// Returns the number of values written or a negative value for an invalid index.
///
/// # Safety
/// `out` must point to `4 + n` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn trace_frame(i: usize, out: *mut f64) -> i64 {
    let guard = lock(&TRACE);
    let Some(frame) = guard.as_ref().and_then(|t| t.frames.get(i)) else {
        return -1;
    };
    let out = std::slice::from_raw_parts_mut(out, 4 + frame.coef.len());
    out[..4].copy_from_slice(&[frame.step, frame.gap, frame.violation, frame.b]);
    out[4..].copy_from_slice(&frame.coef);
    out.len() as i64
}

/// Evaluates the decision function of frame `i` at `m` points (row-major) into `out`.
///
/// # Safety
/// `points` must point to `m * dim` and `out` to `m` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn trace_decision(
    i: usize,
    points: *const f64,
    m: usize,
    out: *mut f64,
) -> i32 {
    let guard = lock(&TRACE);
    let Some(trace) = guard.as_ref() else {
        return -1;
    };
    let Some(frame) = trace.frames.get(i) else {
        return -2;
    };
    let points = std::slice::from_raw_parts(points, m * trace.dim);
    let out = std::slice::from_raw_parts_mut(out, m);
    for (o, p) in out.iter_mut().zip(points.chunks_exact(trace.dim)) {
        *o = trace.decision(frame, p);
    }
    0
}

#[cfg(test)]
mod tests;
