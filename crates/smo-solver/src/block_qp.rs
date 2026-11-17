//! Small feasible active-set QP solver for joint working-set updates.
//!
//! The Hessian is never modified in objective/decrease checks. A tiny diagonal regularization
//! may be used to compute a direction in a singular reduced system; all accepted iterates
//! satisfy the original constraints. Failure returns the best feasible displacement so far.

#![allow(clippy::needless_range_loop)] // Small dense linear algebra follows matrix notation.

#[derive(Clone)]
struct Constraint {
    normal: Vec<f64>,
    rhs: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Orthogonalizes a vector twice (important with nearly dependent active constraints).
fn orthogonalize(v: &mut [f64], basis: &[Vec<f64>]) {
    for _ in 0..2 {
        for q in basis {
            let projection = dot(v, q);
            for (v, q) in v.iter_mut().zip(q) {
                *v -= projection * q;
            }
        }
    }
}

/// Cholesky solve, with a tiny regularizer if the reduced Hessian is singular.
fn positive_solve(h: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    let scale = (0..n).map(|i| h[i][i].abs()).fold(1.0, f64::max);
    for ridge in [0.0, 1e-14, 1e-12, 1e-10] {
        let mut l = vec![vec![0.0; n]; n];
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut value = h[i][j];
                if i == j {
                    value += ridge * scale;
                }
                for k in 0..j {
                    value -= l[i][k] * l[j][k];
                }
                if i == j {
                    if value <= 1e-16 * scale || !value.is_finite() {
                        ok = false;
                        break;
                    }
                    l[i][j] = value.sqrt();
                } else {
                    l[i][j] = value / l[j][j];
                }
            }
            if !ok {
                break;
            }
        }
        if !ok {
            continue;
        }
        let mut x = rhs.to_vec();
        for i in 0..n {
            for j in 0..i {
                x[i] -= l[i][j] * x[j];
            }
            x[i] /= l[i][i];
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= l[j][i] * x[j];
            }
            x[i] /= l[i][i];
        }
        if x.iter().all(|x| x.is_finite()) {
            return Some(x);
        }
    }
    None
}

/// Minimize `gᵀd + dᵀHd/2`, starting at the feasible `d = 0`.
///
/// Subject to `sum(d)=0`, `lo<=d<=hi`, and optionally `signᵀd<=slack`.
/// `lo`/`hi` are distances from the current outer iterate, not absolute alpha bounds.
/// Returns a joint displacement; the caller independently checks feasibility and decrease.
pub(crate) fn solve(
    h: &[Vec<f64>],
    g: &[f64],
    lo: &[f64],
    hi: &[f64],
    signed: Option<(&[f64], f64)>,
    tolerance: f64,
) -> Vec<f64> {
    let n = g.len();
    let mut equality = vec![vec![1.0; n]];
    let mut inequalities = Vec::new();
    for i in 0..n {
        if lo[i] == hi[i] {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            equality.push(row);
        } else {
            if lo[i].is_finite() {
                let mut normal = vec![0.0; n];
                normal[i] = -1.0;
                inequalities.push(Constraint {
                    normal,
                    rhs: -lo[i],
                });
            }
            if hi[i].is_finite() {
                let mut normal = vec![0.0; n];
                normal[i] = 1.0;
                inequalities.push(Constraint { normal, rhs: hi[i] });
            }
        }
    }
    if let Some((sign, slack)) = signed {
        inequalities.push(Constraint {
            normal: sign.to_vec(),
            rhs: slack.max(0.0),
        });
    }
    let mut active: Vec<usize> = Vec::new();
    let mut d = vec![0.0; n];
    let tolerance = tolerance.max(1e-12);
    for _ in 0..(8 * n + 64).min(512) {
        let gradient: Vec<f64> = (0..n).map(|i| g[i] + dot(&h[i], &d)).collect();
        // QR of independent active normals. Dependent inequalities are implied by these
        // normals and need not take part in the multiplier solve.
        let mut normals: Vec<Vec<f64>> = Vec::new();
        let mut triangular: Vec<Vec<f64>> = Vec::new();
        let mut independent: Vec<Option<usize>> = Vec::new();
        for (row, index) in equality
            .iter()
            .map(|r| (r, None))
            .chain(active.iter().map(|&i| (&inequalities[i].normal, Some(i))))
        {
            let mut q = row.clone();
            orthogonalize(&mut q, &normals);
            let length = norm(&q);
            if length <= 1e-10 * norm(row).max(1.0) {
                continue;
            }
            let mut coefficients: Vec<f64> = normals.iter().map(|q| dot(row, q)).collect();
            coefficients.push(length);
            for x in &mut q {
                *x /= length;
            }
            normals.push(q);
            triangular.push(coefficients);
            independent.push(index);
        }
        let mut nullspace = Vec::new();
        for i in 0..n {
            let mut z = vec![0.0; n];
            z[i] = 1.0;
            orthogonalize(&mut z, &normals);
            orthogonalize(&mut z, &nullspace);
            let length = norm(&z);
            if length > 1e-10 {
                for x in &mut z {
                    *x /= length;
                }
                nullspace.push(z);
            }
        }
        let projected: Vec<f64> = nullspace.iter().map(|z| dot(z, &gradient)).collect();
        if norm(&projected) <= tolerance {
            // gradient + B^T lambda = 0; B = L Q, so solve L^T lambda = -Q gradient.
            let r = normals.len();
            let mut multiplier: Vec<f64> = normals.iter().map(|q| -dot(q, &gradient)).collect();
            for i in (0..r).rev() {
                for j in i + 1..r {
                    multiplier[i] -= triangular[j][i] * multiplier[j];
                }
                multiplier[i] /= triangular[i][i];
            }
            let release = (0..r)
                .filter(|&i| independent[i].is_some() && multiplier[i] < -tolerance)
                .min_by(|&i, &j| multiplier[i].total_cmp(&multiplier[j]));
            if let Some(i) = release {
                let index = independent[i].unwrap();
                active.retain(|&j| j != index);
                continue;
            }
            break;
        }
        let hz: Vec<Vec<f64>> = nullspace
            .iter()
            .map(|z| h.iter().map(|row| dot(row, z)).collect())
            .collect();
        let reduced: Vec<Vec<f64>> = nullspace
            .iter()
            .map(|z| hz.iter().map(|hz| dot(z, hz)).collect())
            .collect();
        let rhs: Vec<f64> = projected.iter().map(|g| -g).collect();
        let Some(step) = positive_solve(&reduced, &rhs) else {
            break;
        };
        let mut direction = vec![0.0; n];
        for (z, step) in nullspace.iter().zip(step) {
            for i in 0..n {
                direction[i] += z[i] * step;
            }
        }
        let slope = dot(&gradient, &direction);
        if !slope.is_finite() || slope >= 0.0 {
            break;
        }
        let mut length = 1.0;
        let mut blocker = None;
        for (i, c) in inequalities.iter().enumerate() {
            if active.contains(&i) {
                continue;
            }
            let speed = dot(&c.normal, &direction);
            if speed > 1e-14 * norm(&direction).max(1e-30) {
                let allowed = ((c.rhs - dot(&c.normal, &d)) / speed).max(0.0);
                if allowed < length {
                    length = allowed;
                    blocker = Some(i);
                }
            }
        }
        for i in 0..n {
            d[i] += length * direction[i];
        }
        if let Some(index) = blocker {
            active.push(index);
        }
        if !d.iter().all(|x| x.is_finite()) {
            return vec![0.0; n];
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_four_variable_problem_needs_a_joint_direction() {
        // Two sum-zero pair directions are nearly collinear in the Hessian metric.
        // Joint optimization solves both, instead of alternating small coordinate corrections.
        let h = vec![
            vec![2.0, 0.0, 0.9, -0.9],
            vec![0.0, 2.0, -0.9, 0.9],
            vec![0.9, -0.9, 2.0, 0.0],
            vec![-0.9, 0.9, 0.0, 2.0],
        ];
        let g = [-1.0, 1.0, 1.0, -1.0];
        let d = solve(
            &h,
            &g,
            &[-100.0; 4],
            &[100.0; 4],
            Some((&[1.0, 1.0, -1.0, -1.0], 0.0)),
            1e-10,
        );
        for i in 0..4 {
            assert!((d[i] - g[i] * -5.0).abs() < 1e-8, "{d:?}");
        }
    }

    #[test]
    fn inequality_can_activate_and_release() {
        let h = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let sign = [1.0, 1.0, -1.0, -1.0];
        let active = solve(
            &h,
            &[-2.0, -1.0, 2.0, 1.0],
            &[-10.0; 4],
            &[10.0; 4],
            Some((&sign, 1.0)),
            1e-10,
        );
        assert!((dot(&sign, &active) - 1.0).abs() < 1e-9);
        let release = solve(
            &h,
            &[2.0, 1.0, -2.0, -1.0],
            &[-10.0; 4],
            &[10.0; 4],
            Some((&sign, 0.0)),
            1e-10,
        );
        assert!(dot(&sign, &release) < -1.0);
    }

    #[test]
    fn fixed_variables_and_singular_hessian() {
        let h = vec![vec![1.0; 4]; 4];
        let d = solve(
            &h,
            &[-1.0, 1.0, 0.0, 0.0],
            &[0.0, -1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            None,
            1e-10,
        );
        assert!((d[0] - 1.0).abs() < 1e-9 && (d[1] + 1.0).abs() < 1e-9);
        assert!(d[2].abs() < 1e-12);
        assert!(d[3].abs() < 1e-12);
    }
}
