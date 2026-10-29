//! Optimality measures for the dual problem (Strasdat, Sec. 7.4–7.6).
//!
//! In the sign convention of this crate, a dual point `α` is optimal iff there is a bias `b` with
//! `b ≥ -g_i` for all `i` with `α_i < hi_i` and `b ≤ -g_j` for all `j` with `α_j > lo_j`.
//! A pair `(i, j)` with `-g_i > -g_j` is a *violating pair*: moving mass from `j` to `i`
//! decreases the objective.

use crate::problem::DualProblem;

/// Dual gradient `g_i = ka_i + q_i α_i + p_i`.
pub fn dual_gradient(problem: &DualProblem, alpha: &[f64], ka: &[f64]) -> Vec<f64> {
    (0..problem.len())
        .map(|i| ka[i] + problem.q[i] * alpha[i] + problem.p[i])
        .collect()
}

/// Violation measure `r(α) = max_{α_i > lo_i} g_i - min_{α_j < hi_j} g_j`, clipped at zero.
///
/// The optional 1-norm constraint is ignored.
pub fn violation(problem: &DualProblem, alpha: &[f64], grad: &[f64]) -> f64 {
    let mut down = f64::NEG_INFINITY;
    let mut up = f64::INFINITY;
    for i in 0..problem.len() {
        if alpha[i] > problem.lo[i] {
            down = down.max(grad[i]);
        }
        if alpha[i] < problem.hi[i] {
            up = up.min(grad[i]);
        }
    }
    (down - up).max(0.0)
}

/// Bound-aware violation measure (Strasdat, Def. 7.21)
///
/// `r◇(α, ρ) = max_{i,j} min(ε_ij, ρ) · (g_j - g_i)` over pairs where `α_i` can increase and
/// `α_j` can decrease, with the maximal feasible step `ε_ij = min(hi_i - α_i, α_j - lo_j)`.
/// Computed in `O(n log n)` as described in Remark 7.22. The optional 1-norm constraint is ignored.
pub fn bound_aware_violation(problem: &DualProblem, alpha: &[f64], grad: &[f64], rho: f64) -> f64 {
    bound_aware_violation_slices(alpha, &problem.lo, &problem.hi, grad, rho)
}

/// [`bound_aware_violation`] for explicit bounds (e.g. the active prefix of the solver's arrays).
pub fn bound_aware_violation_slices(
    alpha: &[f64],
    lo: &[f64],
    hi: &[f64],
    grad: &[f64],
    rho: f64,
) -> f64 {
    // candidates to increase: (δ⁺_i, -g_i); candidates to decrease: (δ⁻_j, g_j)
    let mut inc: Vec<(f64, f64)> = Vec::new();
    let mut dec: Vec<(f64, f64)> = Vec::new();
    for i in 0..alpha.len() {
        let up = (hi[i] - alpha[i]).min(rho);
        if up > 0.0 {
            inc.push((up, -grad[i]));
        }
        let down = (alpha[i] - lo[i]).min(rho);
        if down > 0.0 {
            dec.push((down, grad[i]));
        }
    }
    let best = one_sided(&inc, &dec).max(one_sided(&dec, &inc));
    best.max(0.0)
}

/// `max_{a ∈ fixed} a.0 · (max_{b ∈ other, b.0 ≥ a.0} b.1 + a.1)`.
fn one_sided(fixed: &[(f64, f64)], other: &[(f64, f64)]) -> f64 {
    let mut sorted: Vec<(f64, f64)> = other.to_vec();
    sorted.sort_by(|x, y| y.0.total_cmp(&x.0)); // descending step size
    let mut prefix_max = Vec::with_capacity(sorted.len());
    let mut m = f64::NEG_INFINITY;
    for &(_, v) in &sorted {
        m = m.max(v);
        prefix_max.push(m);
    }
    let mut best = f64::NEG_INFINITY;
    for &(step, v) in fixed {
        // number of entries with step size ≥ `step`
        let cnt = sorted.partition_point(|x| x.0 >= step);
        if cnt > 0 {
            best = best.max(step * (prefix_max[cnt - 1] + v));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(problem: &DualProblem, alpha: &[f64], grad: &[f64], rho: f64) -> f64 {
        let n = problem.len();
        let mut best: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let eps = (problem.hi[i] - alpha[i])
                    .min(alpha[j] - problem.lo[j])
                    .min(rho);
                if eps > 0.0 {
                    best = best.max(eps * (grad[j] - grad[i]));
                }
            }
        }
        best
    }

    #[test]
    fn bound_aware_matches_brute_force() {
        let y: Vec<f64> = (0..12)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let prob = DualProblem::classification(&y, 0.5, None, 1.0, 0.0, None).unwrap();
        let mut state = 12345u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..50 {
            let alpha: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let r = rnd();
                    let v = if r < 0.3 {
                        0.0
                    } else if r < 0.6 {
                        1.0
                    } else {
                        rnd()
                    };
                    yi * v
                })
                .collect();
            let grad: Vec<f64> = (0..12).map(|_| rnd() * 4.0 - 2.0).collect();
            for rho in [0.1, 0.5, 10.0] {
                let fast = bound_aware_violation(&prob, &alpha, &grad, rho);
                let slow = brute_force(&prob, &alpha, &grad, rho);
                assert!((fast - slow).abs() < 1e-12, "{fast} vs {slow}");
            }
        }
    }
}
