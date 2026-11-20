//! k-fold cross-validation objective and its exact hypergradient for the RBF-kernel SVM.
//!
//! Mirrors `optimized_svm.hypergradient.cv_error_and_gradient` (blocks = None, unit weights,
//! no `max_asum`, no loss parameters): every fold is trained by SMO, then the derivative is the
//! explicit part (dual variables and bias fixed) plus the adjoint bordered system on the loss
//! terms with positive curvature `h = ℓ''`, see Strasdat, Sec. 9.2.

use smo_solver::{
    solve_with, DualProblem, FeatureKernel, KernelFunction, SmoOptions, SmoResult, Status,
};

/// Status codes returned by the exports (negative values are errors).
pub const ERR_INPUT: i32 = -1;
pub const ERR_KIND: i32 = -2;
pub const ERR_PROBLEM: i32 = -3;
pub const ERR_SOLVER: i32 = -4;
pub const ERR_LINALG: i32 = -5;

/// Hyperparameters and solver settings of one evaluation.
pub struct Params {
    /// 0: classification, 1: ε-regression.
    pub kind: u32,
    pub lambda: f64,
    pub gamma: f64,
    /// Smoothing `δ` of the training loss.
    pub smoothing: f64,
    pub epsilon: f64,
    pub tol: f64,
}

/// Dual variables of the last evaluation, reused as warm start while the data layout is unchanged.
#[derive(Default)]
pub struct WarmStart {
    y: Vec<f64>,
    fold_of: Vec<f64>,
    kind: u32,
    alphas: Vec<Vec<f64>>,
}

/// Smoothed maximum `m_δ` and its derivative (`degree` 2 or 6, `δ > 0`) at `t`.
fn smooth_max_d(t: f64, d: f64, degree: u32) -> (f64, f64) {
    if t >= d {
        return (t, 1.0);
    }
    if t <= -d {
        return (0.0, 0.0);
    }
    if degree == 2 {
        (0.25 / d * (t + d) * (t + d), 0.5 / d * (t + d))
    } else {
        let u = t / d;
        let m = d * (5.0 + u * (16.0 + u * (15.0 + u * u * (-5.0 + u * u)))) / 32.0;
        let dm = (8.0 + u * (15.0 + u * u * (-10.0 + 3.0 * u * u))) / 16.0;
        (m, dm)
    }
}

/// Validation error of the predictions `f` and its derivative w.r.t. `f` (summed, not averaged).
///
/// Classification: `Σ m_0.5^(6)(1 - y f)` (`SmoothHingeValidation()`), regression:
/// `Σ m_0.1^(2)(f - y) + m_0.1^(2)(y - f)` (`SmoothAbsoluteValidation()`).
fn validation(kind: u32, f: &[f64], y: &[f64], d_err: &mut [f64]) -> f64 {
    let mut err = 0.0;
    for ((&fv, &yv), g) in f.iter().zip(y).zip(d_err.iter_mut()) {
        if kind == 0 {
            let (m, dm) = smooth_max_d(1.0 - yv * fv, 0.5, 6);
            err += m;
            *g = -yv * dm;
        } else {
            let e = fv - yv;
            let (m1, d1) = smooth_max_d(e, 0.1, 2);
            let (m2, d2) = smooth_max_d(-e, 0.1, 2);
            err += m1 + m2;
            *g = d1 - d2;
        }
    }
    err
}

fn row(x: &[f64], i: usize, dim: usize) -> &[f64] {
    &x[i * dim..(i + 1) * dim]
}

#[inline]
fn d2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum()
}

/// In-place Cholesky `Q = L Lᵀ` (lower triangle of the row-major `m × m` matrix).
fn cholesky(q: &mut [f64], m: usize) -> bool {
    for j in 0..m {
        let mut s = q[j * m + j];
        for k in 0..j {
            s -= q[j * m + k] * q[j * m + k];
        }
        if s <= 0.0 || !s.is_finite() {
            return false;
        }
        let l = s.sqrt();
        q[j * m + j] = l;
        for i in j + 1..m {
            let mut s = q[i * m + j];
            for k in 0..j {
                s -= q[i * m + k] * q[j * m + k];
            }
            q[i * m + j] = s / l;
        }
    }
    true
}

/// Solves `L Lᵀ x = b` in place.
fn cholesky_solve(l: &[f64], m: usize, b: &mut [f64]) {
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * m + k] * b[k];
        }
        b[i] = s / l[i * m + i];
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= l[k * m + i] * b[k];
        }
        b[i] = s / l[i * m + i];
    }
}

/// Trains one fold and returns `(Σ validation error, gradient of the sum, validation count)`
/// with the gradient in `(log λ, log γ)` coordinates; `alpha` is updated to the new solution.
fn fold(
    xt: &[f64],
    yt: &[f64],
    xv: &[f64],
    yv: &[f64],
    dim: usize,
    p: &Params,
    alpha: &mut Vec<f64>,
) -> Result<(f64, [f64; 2]), i32> {
    let nt = yt.len();
    let nv = yv.len();
    let problem = if p.kind == 0 {
        DualProblem::classification(yt, p.lambda, None, 1.0, p.smoothing, None)
    } else {
        DualProblem::regression(yt, p.lambda, None, p.epsilon, p.smoothing, None)
    }
    .map_err(|_| ERR_PROBLEM)?;
    let function = KernelFunction::Rbf { gamma: p.gamma };
    let kernel = FeatureKernel::new(xt, dim, function);
    let options = SmoOptions {
        tol: p.tol,
        max_steps: 2_000_000,
        cache_size_mb: 16.0,
        ..SmoOptions::default()
    };
    let warm = (alpha.len() == problem.len()).then_some(alpha.as_slice());
    let result: SmoResult = match solve_with(&kernel, &problem, &options, warm, |_| false) {
        Ok(r) => r,
        // an infeasible warm start (changed data) falls back to the cold start
        Err(_) if warm.is_some() => {
            solve_with(&kernel, &problem, &options, None, |_| false).map_err(|_| ERR_SOLVER)?
        }
        Err(_) => return Err(ERR_SOLVER),
    };
    if result.status != Status::Converged {
        return Err(ERR_SOLVER);
    }
    alpha.clone_from(&result.alpha);
    let lam_inv = 1.0 / p.lambda;
    let beta: Vec<f64> = problem
        .sample_coefficients(&result.alpha)
        .iter()
        .map(|b| b * lam_inv)
        .collect();
    let sv: Vec<usize> = (0..nt).filter(|&j| beta[j] != 0.0).collect();
    let g = p.gamma;

    // validation predictions and the explicit derivatives (dual variables and bias fixed)
    let mut ka_val = vec![0.0; nv];
    let mut f_val = vec![0.0; nv];
    let mut d_err = vec![0.0; nv];
    for v in 0..nv {
        ka_val[v] = sv
            .iter()
            .map(|&j| beta[j] * (-g * d2(row(xv, v, dim), row(xt, j, dim))).exp())
            .sum();
        f_val[v] = ka_val[v] + result.b;
    }
    let err = validation(p.kind, &f_val, yv, &mut d_err);
    let mut grad = [0.0; 2];
    for v in 0..nv {
        grad[0] -= d_err[v] * ka_val[v];
        for &j in &sv {
            let dd = d2(row(xv, v, dim), row(xt, j, dim));
            // d K / d log γ = -γ |x - z|² K
            grad[1] -= d_err[v] * beta[j] * g * dd * (-g * dd).exp();
        }
    }

    // implicit part: adjoint system on the terms with positive curvature
    let delta = p.smoothing;
    if delta > 0.0 {
        let ka_train: Vec<f64> = (0..nt)
            .map(|i| {
                sv.iter()
                    .map(|&j| beta[j] * (-g * d2(row(xt, i, dim), row(xt, j, dim))).exp())
                    .sum()
            })
            .collect();
        // terms in the smoothing zone |arg| < δ, where h = ℓ'' = 1 / (2δ)
        let mut terms: Vec<usize> = Vec::new(); // sample index of each term in P
        for (t, &k) in problem.kernel_index.iter().enumerate() {
            let dec = ka_train[k] + result.b;
            let arg = if p.kind == 0 {
                1.0 - yt[k] * dec
            } else {
                -problem.sign[t] * (dec - yt[k]) - p.epsilon
            };
            if arg.abs() < delta {
                terms.push(k);
            }
        }
        let m = terms.len();
        if m > 0 {
            let h_inv = 2.0 * delta;
            let mut q = vec![0.0; m * m];
            for a in 0..m {
                for b in 0..=a {
                    let kab = (-g * d2(row(xt, terms[a], dim), row(xt, terms[b], dim))).exp();
                    q[a * m + b] = kab * lam_inv + if a == b { h_inv } else { 0.0 };
                }
            }
            if !cholesky(&mut q, m) {
                return Err(ERR_LINALG);
            }
            let mut qia = vec![1.0; m];
            cholesky_solve(&q, m, &mut qia);
            let mut u: Vec<f64> = terms
                .iter()
                .map(|&k| {
                    (0..nv)
                        .map(|v| d_err[v] * (-g * d2(row(xv, v, dim), row(xt, k, dim))).exp())
                        .sum::<f64>()
                        * lam_inv
                })
                .collect();
            cholesky_solve(&q, m, &mut u); // u = Q⁻¹ r
            let r0: f64 = d_err.iter().sum();
            let schur: f64 = qia.iter().sum();
            let v = (u.iter().sum::<f64>() - r0) / schur;
            for (ui, qi) in u.iter_mut().zip(&qia) {
                *ui -= qi * v;
            }
            for (&k, &ui) in terms.iter().zip(&u) {
                // d ka_P / d log λ = -ka_P
                grad[0] += ui * ka_train[k];
                // d ka_P / d log γ = (dK_{P,SV} / d log γ) β_SV
                for &j in &sv {
                    let dd = d2(row(xt, k, dim), row(xt, j, dim));
                    grad[1] -= ui * beta[j] * (-g * dd) * (-g * dd).exp();
                }
            }
        }
    }
    Ok((err, grad))
}

/// Cross-validation error `Σ_folds E_fold / n` and its gradient w.r.t. `(log λ, log γ)`.
#[allow(clippy::too_many_arguments)]
pub fn cv(
    x: &[f64],
    dim: usize,
    y: &[f64],
    fold_of: &[f64],
    k: usize,
    p: &Params,
    warm: &mut WarmStart,
    out: &mut [f64; 3],
) -> Result<(), i32> {
    let n = y.len();
    if p.kind > 1 || k < 2 || n < k || dim == 0 || x.len() != n * dim || fold_of.len() != n {
        return Err(if p.kind > 1 { ERR_KIND } else { ERR_INPUT });
    }
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !(positive(p.lambda) && positive(p.gamma) && positive(p.tol))
        || !(p.smoothing.is_finite() && p.smoothing >= 0.0)
        || !(p.epsilon.is_finite() && p.epsilon >= 0.0)
        || x.iter().chain(y).any(|v| !v.is_finite())
    {
        return Err(ERR_INPUT);
    }
    let mut ids = Vec::with_capacity(n);
    for &f in fold_of {
        if !(f >= 0.0 && f < k as f64 && f.fract() == 0.0) {
            return Err(ERR_INPUT);
        }
        ids.push(f as usize);
    }
    if warm.kind != p.kind || warm.y != y || warm.fold_of != fold_of || warm.alphas.len() != k {
        *warm = WarmStart {
            y: y.to_vec(),
            fold_of: fold_of.to_vec(),
            kind: p.kind,
            alphas: vec![Vec::new(); k],
        };
    }
    let (mut total, mut grad) = (0.0, [0.0; 2]);
    for (fi, alpha) in warm.alphas.iter_mut().enumerate() {
        let (mut xt, mut yt, mut xv, mut yv) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let (xs, ys) = if ids[i] == fi {
                (&mut xv, &mut yv)
            } else {
                (&mut xt, &mut yt)
            };
            xs.extend_from_slice(&x[i * dim..(i + 1) * dim]);
            ys.push(y[i]);
        }
        if yt.is_empty() || yv.is_empty() {
            return Err(ERR_INPUT);
        }
        let (e, g) = fold(&xt, &yt, &xv, &yv, dim, p, alpha)?;
        total += e;
        grad[0] += g[0];
        grad[1] += g[1];
    }
    let nf = n as f64;
    *out = [total / nf, grad[0] / nf, grad[1] / nf];
    if out.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(ERR_LINALG)
    }
}
