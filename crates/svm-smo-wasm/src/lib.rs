//! WebAssembly interface of `svm-smo` for the browser demo.
//!
//! The interface is a plain C ABI: JavaScript allocates `f64` buffers with [`alloc`], writes the
//! data into `memory`, calls [`train`] and evaluates the resulting model with [`decision`].
//! The model is kept in a global slot (the demo trains one model at a time).

use std::sync::Mutex;

use svm_smo::{solve, DualProblem, FeatureKernel, KernelFunction, Model, SmoOptions, SmoResult};

struct Trained {
    model: Model,
    result: SmoResult,
}

static STATE: Mutex<Option<Trained>> = Mutex::new(None);

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
