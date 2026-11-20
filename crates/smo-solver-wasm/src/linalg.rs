//! Small dense linear algebra shared by the cross-validation and the Newton solver.

/// In-place Cholesky `Q = L Lᵀ` (lower triangle of the row-major `m × m` matrix).
pub fn cholesky(q: &mut [f64], m: usize) -> bool {
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
pub fn cholesky_solve(l: &[f64], m: usize, b: &mut [f64]) {
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
