//! WebAssembly interface of `smo-solver` for the browser demo.
//!
//! The interface is a plain C ABI: JavaScript allocates `f64` buffers with [`alloc`], writes the
//! data into `memory`, calls [`train`] and evaluates the resulting model with [`decision`].
//! The model is kept in a global slot (the demo trains one model at a time).

mod cv;
mod trace;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

use smo_solver::{solve, DualProblem, FeatureKernel, KernelFunction, Model, SmoOptions, SmoResult};

struct Trained {
    model: Model,
    result: SmoResult,
}

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
/// `kind = 0`: classification (labels ±1), `kind = 1`: ε-regression.
/// Returns the number of SMO steps, or a negative value on invalid input.
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
    *STATE.lock().unwrap() = Some(Trained { model, result });
    steps
}

/// Evaluates the decision function at `m` points (row-major, same dimension) into `out`.
///
/// # Safety
/// `points` must point to `m * dim` and `out` to `m` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn decision(points: *const f64, m: usize, out: *mut f64) -> i32 {
    let guard = STATE.lock().unwrap();
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
    let guard = STATE.lock().unwrap();
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
/// 4 = final violation.
#[no_mangle]
pub extern "C" fn stat(which: u32) -> f64 {
    let guard = STATE.lock().unwrap();
    let Some(t) = guard.as_ref() else {
        return f64::NAN;
    };
    match which {
        0 => t.result.b,
        1 => t.result.gap,
        2 => t.result.dual_objective,
        3 => t.result.elapsed,
        4 => t.result.violation,
        _ => f64::NAN,
    }
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
    };
    let run = catch_unwind(AssertUnwindSafe(|| {
        let mut slot = lock(&WARM);
        let warm = slot.get_or_insert_with(Default::default);
        let mut res = [0.0; 3];
        let status = cv::cv(x, dim, y, fold_of, k, &params, warm, &mut res);
        (status, res)
    }));
    match run {
        Ok((Ok(()), res)) => {
            std::slice::from_raw_parts_mut(out, 3).copy_from_slice(&res);
            0
        }
        Ok((Err(code), _)) => code,
        Err(_) => cv::ERR_LINALG,
    }
}

/// Like [`train`] (same arguments), but also records the SMO iteration: an initial frame
/// (`α = 0`), one frame every `period` steps (at most `max_frames - 1` of them, so a long run
/// is recorded only at its beginning) and always the final state as the last frame. The final
/// model is stored in the global slot as by [`train`], so [`decision`], [`support`] and [`stat`]
/// keep working. Shrinking is disabled while recording. Returns the number of SMO steps or a
/// negative value on invalid input (`-5`: `period = 0` or `max_frames < 2`).
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
    *lock(&STATE) = Some(Trained { model, result });
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
