//! Kernel logistic regression (Strasdat, Sec. 4.1.3) trained by a damped Newton method.
//!
//! Mirrors `optimized_svm.training.newton` with `LogisticLoss` (unit weights, no 1-norm
//! constraint): the primal problem `min ½ aᵀK a/λ + Σ log(1 + exp(-y_i (K a/λ + b)_i))` has the
//! optimality conditions `a_i + φ'_i = 0` with `φ'_i = -y_i σ(-y_i f_i)` and `Σ a_i = 0`; the
//! decision function (the logit) is `f(x) = Σ_i a_i/λ K(x, x_i) + b`. The hypergradient is the
//! same adjoint bordered system as for the SMO losses, here on all terms with `h = σ(1 - σ) > 0`.

use crate::cv::{add_dgamma, d2, row, Params, ERR_INPUT, ERR_LINALG, ERR_SOLVER};
use crate::linalg::{cholesky, cholesky_solve};
use crate::trace::{Frame, Trace};
use smo_solver::{KernelFunction, Model};

/// Newton iteration limit.
pub const MAX_STEPS: u64 = 100;
const MAX_BACKTRACKS: u32 = 20;
const SIGMA: f64 = 1e-4;

/// `σ(t) = 1 / (1 + e^{-t})` without overflow.
pub fn sigmoid(t: f64) -> f64 {
    if t >= 0.0 {
        1.0 / (1.0 + (-t).exp())
    } else {
        let e = t.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + e^t)` without overflow.
pub fn softplus(t: f64) -> f64 {
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// Labels must be exactly ±1.
pub fn valid_labels(y: &[f64]) -> bool {
    y.iter().all(|&v| v == 1.0 || v == -1.0)
}

/// RBF kernel matrix (row-major `n × n`).
pub fn kernel_matrix(x: &[f64], dim: usize, gamma: f64) -> Vec<f64> {
    let n = x.len() / dim;
    let mut k = vec![1.0; n * n];
    for i in 0..n {
        for j in 0..i {
            let v = (-gamma * d2(row(x, i, dim), row(x, j, dim))).exp();
            k[i * n + j] = v;
            k[j * n + i] = v;
        }
    }
    k
}

fn matvec(k: &[f64], v: &[f64], scale: f64) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| {
            k[i * n..(i + 1) * n]
                .iter()
                .zip(v)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                * scale
        })
        .collect()
}

fn objective(a: &[f64], ka: &[f64], b: f64, y: &[f64]) -> f64 {
    let quad: f64 = a.iter().zip(ka).map(|(p, q)| p * q).sum();
    0.5 * quad
        + ka.iter()
            .zip(y)
            .map(|(&f, &yi)| softplus(-yi * (f + b)))
            .sum::<f64>()
}

/// Result of [`newton`].
pub struct Fit {
    /// Dual coefficients `a` (the model coefficients are `a / λ`).
    pub a: Vec<f64>,
    pub b: f64,
    /// Newton iterations.
    pub steps: u64,
    /// Final optimality norm `sqrt(mean((a + φ')²) + (Σ φ')²)`.
    pub residual: f64,
    /// Final primal objective.
    pub objective: f64,
}

/// Full Newton direction `(da, db)` (the step is `a - t da`, `b - t db`), `None` if the system
/// is not positive definite. Terms with vanishing curvature are handled as in the Python code.
fn direction(k: &[f64], lam: f64, a: &[f64], g: &[f64], h: &[f64]) -> Option<(Vec<f64>, f64)> {
    let n = a.len();
    let p: Vec<usize> = (0..n).filter(|&i| h[i] > 0.0).collect();
    let mut da = vec![0.0; n];
    let mut sum_gz = 0.0;
    for i in (0..n).filter(|&i| h[i] <= 0.0) {
        da[i] = a[i] + g[i];
        sum_gz += g[i];
    }
    let m = p.len();
    if m == 0 {
        return None;
    }
    let mut q = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (r, &i) in p.iter().enumerate() {
        for (c, &j) in p.iter().enumerate().take(r + 1) {
            q[r * m + c] = k[i * n + j] / lam + if r == c { 1.0 / h[i] } else { 0.0 };
        }
        let kz: f64 = (0..n)
            .filter(|&j| h[j] <= 0.0)
            .map(|j| k[i * n + j] * da[j])
            .sum::<f64>()
            / lam;
        rhs[r] = (a[i] + g[i]) / h[i] - kz;
    }
    if !cholesky(&mut q, m) {
        return None;
    }
    let rhs0: f64 = p.iter().map(|&i| a[i]).sum::<f64>() - sum_gz;
    let mut qia = vec![1.0; m];
    cholesky_solve(&q, m, &mut qia);
    cholesky_solve(&q, m, &mut rhs);
    let v = (rhs.iter().sum::<f64>() - rhs0) / qia.iter().sum::<f64>();
    for (r, &i) in p.iter().enumerate() {
        da[i] = rhs[r] - qia[r] * v;
    }
    Some((da, v))
}

/// Damped Newton method with Armijo backtracking. `start` is an optional `(a, b)`, otherwise
/// `a = 0`, `b = 0`. `tol` bounds the final optimality norm. `observe(step, a, b, norm)` is called
/// at the start of every iteration (step 0 is the starting point), including the last one.
pub fn newton(
    k: &[f64],
    y: &[f64],
    lam: f64,
    start: Option<(&[f64], f64)>,
    tol: f64,
    mut observe: impl FnMut(u64, &[f64], f64, f64),
) -> Result<Fit, ()> {
    let n = y.len();
    let (mut a, mut b) = match start {
        Some((a0, b0)) if a0.len() == n => (a0.to_vec(), b0),
        _ => (vec![0.0; n], 0.0),
    };
    let mut ka = matvec(k, &a, 1.0 / lam);
    let mut obj = objective(&a, &ka, b, y);
    let mut step = 0;
    loop {
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n];
        for i in 0..n {
            let t = y[i] * (ka[i] + b);
            let s = sigmoid(-t);
            g[i] = -y[i] * s;
            h[i] = s * sigmoid(t);
        }
        let r: Vec<f64> = a.iter().zip(&g).map(|(p, q)| p + q).collect();
        let sum_g: f64 = g.iter().sum();
        let norm = (r.iter().map(|v| v * v).sum::<f64>() / n as f64 + sum_g * sum_g).sqrt();
        if !norm.is_finite() || !obj.is_finite() {
            return Err(());
        }
        observe(step, &a, b, norm);
        if norm < tol {
            break;
        }
        if step == MAX_STEPS {
            return Err(());
        }
        step += 1;
        let mut directions = Vec::new();
        if let Some(d) = direction(k, lam, &a, &g, &h) {
            directions.push(d);
        }
        directions.push((r.clone(), lam * sum_g)); // preconditioned gradient as fallback
        let mut accepted = false;
        for (da, db) in directions {
            let kda = matvec(k, &da, 1.0 / lam);
            let slope: f64 = r.iter().zip(&kda).map(|(p, q)| p * q).sum::<f64>() + db * sum_g;
            if slope.is_nan() || slope <= 0.0 {
                continue;
            }
            let mut t = 1.0;
            for _ in 0..MAX_BACKTRACKS {
                let a_new: Vec<f64> = a.iter().zip(&da).map(|(p, d)| p - t * d).collect();
                let ka_new: Vec<f64> = ka.iter().zip(&kda).map(|(p, d)| p - t * d).collect();
                let b_new = b - t * db;
                let obj_new = objective(&a_new, &ka_new, b_new, y);
                if obj_new <= obj - SIGMA * t * slope + 1e-15 * obj.abs() {
                    (a, ka, b, obj) = (a_new, ka_new, b_new, obj_new);
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if accepted {
                break;
            }
        }
        if !accepted {
            // rounding limits the objective decrease close to the solution
            if norm < 1e3 * tol {
                break;
            }
            return Err(());
        }
    }
    // remove accumulated rounding errors of the incremental updates
    let ka = matvec(k, &a, 1.0 / lam);
    let mut res = 0.0;
    let mut sum_g = 0.0;
    for i in 0..n {
        let g = -y[i] * sigmoid(-y[i] * (ka[i] + b));
        res += (a[i] + g) * (a[i] + g);
        sum_g += g;
    }
    let residual = (res / n as f64 + sum_g * sum_g).sqrt();
    let objective = objective(&a, &ka, b, y);
    Ok(Fit {
        a,
        b,
        steps: step,
        residual,
        objective,
    })
}

/// Fitted model for the global slot of `train`: all samples with a nonzero coefficient.
pub fn model(function: KernelFunction, x: &[f64], dim: usize, lam: f64, fit: &Fit) -> Model {
    let mut model = Model {
        function,
        dim,
        support_vectors: Vec::new(),
        support: Vec::new(),
        coef: Vec::new(),
        b: fit.b,
    };
    for (i, &ai) in fit.a.iter().enumerate() {
        if ai != 0.0 {
            model.support.push(i);
            model.coef.push(ai / lam);
            model.support_vectors.extend_from_slice(row(x, i, dim));
        }
    }
    model
}

/// Trains on the full data, optionally recording frames (`period`, `max_frames`; `None` = no
/// trace). Returns the fit and the trace.
pub fn train(
    x: &[f64],
    dim: usize,
    y: &[f64],
    lam: f64,
    gamma: f64,
    tol: f64,
    record: Option<(u64, usize)>,
) -> Result<(Fit, Option<Trace>), ()> {
    let function = KernelFunction::Rbf { gamma };
    let k = kernel_matrix(x, dim, gamma);
    let mut trace = record.map(|_| Trace {
        x: x.to_vec(),
        dim,
        function,
        frames: Vec::new(),
    });
    let fit = newton(&k, y, lam, None, tol, |step, a, b, norm| {
        if let (Some(t), Some((period, max_frames))) = (trace.as_mut(), record) {
            if step % period == 0 && t.frames.len() + 1 < max_frames {
                t.frames.push(frame(step, a, b, norm, lam));
            }
        }
    })?;
    if let Some(t) = trace.as_mut() {
        if t.frames.last().map(|f| f.step) != Some(fit.steps as f64) {
            t.frames
                .push(frame(fit.steps, &fit.a, fit.b, fit.residual, lam));
        }
    }
    Ok((fit, trace))
}

fn frame(step: u64, a: &[f64], b: f64, norm: f64, lam: f64) -> Frame {
    Frame {
        step: step as f64,
        gap: norm,
        violation: if step == 0 { f64::NAN } else { norm },
        b,
        coef: a.iter().map(|v| v / lam).collect(),
    }
}

/// One cross-validation fold of kernel logistic regression: returns `(Σ validation error,
/// gradient of the sum in (log λ, log γ))`. `warm` holds `[a, b]` of the previous evaluation.
pub fn fold(
    xt: &[f64],
    yt: &[f64],
    xv: &[f64],
    yv: &[f64],
    dim: usize,
    p: &Params,
    warm: &mut Vec<f64>,
) -> Result<(f64, Vec<f64>), i32> {
    let nt = yt.len();
    let nv = yv.len();
    let (lam, g) = (p.lambda, p.gamma);
    let lam_inv = 1.0 / lam;
    let k = kernel_matrix(xt, dim, g);
    let run = |start: Option<(&[f64], f64)>| newton(&k, yt, lam, start, p.tol, |_, _, _, _| {});
    let fit = if warm.len() == nt + 1 {
        run(Some((&warm[..nt], warm[nt])))
            .or_else(|_| run(None))
            .map_err(|_| ERR_SOLVER)?
    } else {
        run(None).map_err(|_| ERR_SOLVER)?
    };
    warm.clear();
    warm.extend_from_slice(&fit.a);
    warm.push(fit.b);
    let beta: Vec<f64> = fit.a.iter().map(|v| v * lam_inv).collect();
    let sv: Vec<usize> = (0..nt).filter(|&j| beta[j] != 0.0).collect();

    // validation predictions and the explicit derivatives (a and b fixed)
    let mut ka_val = vec![0.0; nv];
    let mut d_err = vec![0.0; nv];
    let mut err = 0.0;
    let mut grad = vec![0.0; 1 + p.n_gamma(dim)];
    for v in 0..nv {
        let mut ka = 0.0;
        let mut kv_j = Vec::with_capacity(sv.len());
        for &j in &sv {
            let kv = (-g * d2(row(xv, v, dim), row(xt, j, dim))).exp();
            ka += beta[j] * kv;
            kv_j.push(kv);
        }
        let f = ka + fit.b;
        ka_val[v] = ka;
        err += softplus(-yv[v] * f);
        d_err[v] = -yv[v] * sigmoid(-yv[v] * f);
        grad[0] -= d_err[v] * ka;
        for (&j, &kv) in sv.iter().zip(&kv_j) {
            add_dgamma(
                &mut grad,
                d_err[v] * beta[j],
                kv,
                row(xv, v, dim),
                row(xt, j, dim),
                g,
                p.ard,
            );
        }
    }

    // implicit part: adjoint system on the terms with positive curvature
    let ka_train = matvec(&k, &beta, 1.0);
    let terms: Vec<usize> = (0..nt)
        .filter(|&i| {
            let t = yt[i] * (ka_train[i] + fit.b);
            sigmoid(-t) * sigmoid(t) > 0.0
        })
        .collect();
    let m = terms.len();
    if m > 0 {
        let mut q = vec![0.0; m * m];
        for (r, &i) in terms.iter().enumerate() {
            for (c, &j) in terms.iter().enumerate().take(r + 1) {
                q[r * m + c] = k[i * nt + j] * lam_inv;
            }
            let t = yt[i] * (ka_train[i] + fit.b);
            q[r * m + r] += 1.0 / (sigmoid(-t) * sigmoid(t));
        }
        if !cholesky(&mut q, m) {
            return Err(ERR_LINALG);
        }
        let mut qia = vec![1.0; m];
        cholesky_solve(&q, m, &mut qia);
        let mut u: Vec<f64> = terms
            .iter()
            .map(|&i| {
                (0..nv)
                    .map(|v| d_err[v] * (-g * d2(row(xv, v, dim), row(xt, i, dim))).exp())
                    .sum::<f64>()
                    * lam_inv
            })
            .collect();
        cholesky_solve(&q, m, &mut u);
        let r0: f64 = d_err.iter().sum();
        let v = (u.iter().sum::<f64>() - r0) / qia.iter().sum::<f64>();
        for (ui, qi) in u.iter_mut().zip(&qia) {
            *ui -= qi * v;
        }
        for (&i, &ui) in terms.iter().zip(&u) {
            // d ka_P / d log λ = -ka_P
            grad[0] += ui * ka_train[i];
            for &j in &sv {
                let (a, b) = (row(xt, i, dim), row(xt, j, dim));
                let kv = (-g * d2(a, b)).exp();
                add_dgamma(&mut grad, -ui * beta[j], kv, a, b, g, p.ard);
            }
        }
    }
    if !err.is_finite() {
        return Err(ERR_LINALG);
    }
    Ok((err, grad))
}

/// Input check shared by the exports: labels ±1.
pub fn check_labels(y: &[f64]) -> Result<(), i32> {
    if valid_labels(y) {
        Ok(())
    } else {
        Err(ERR_INPUT)
    }
}
