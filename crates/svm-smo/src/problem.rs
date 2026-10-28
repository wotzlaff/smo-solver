//! The separable quadratic dual training problem and constructors for common SVMs.
//!
//! All problems solved by this crate have the form
//!
//! ```text
//!   min_α  1/(2λ) Σ_ij α_i α_j K[k(i), k(j)]  +  Σ_i (½ q_i α_i² + p_i α_i)
//!   s.t.   Σ_i α_i = 0,
//!          lo_i ≤ α_i ≤ hi_i               for all i,
//!          Σ_i s_i α_i ≤ A                 (optional, s_i ∈ {-1, +1})
//! ```
//!
//! where `k(i)` maps a solver variable to a sample of the kernel matrix. This is the
//! dual of the *standard training problem* (Strasdat, Cor. 3.11)
//!
//! ```text
//!   min_{w,b}  λ/2 |w|² + Σ_i ℓ_i(<w, φ(x_{k(i)})> + b)
//! ```
//!
//! for loss functions whose conjugates are quadratic on an interval, i.e.
//! `ℓ_i*(-α) = ½ q_i α² + p_i α` for `α ∈ [lo_i, hi_i]`. This covers hinge and
//! ε-insensitive losses, their quadratically smoothed versions and the squared loss.
//! The optional constraint bounds the 1-norm of `α` (for the constructors below,
//! `s_i α_i = |α_i|`), which yields a generalised ν-SVM (Strasdat, Sec. 5.2); its
//! multiplier `c ≥ 0` enters the decision values of the training points as `s_i c`.

use std::fmt;

/// Errors raised when a problem is inconsistent.
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemError {
    /// Two input vectors have different lengths.
    LengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A parameter is outside its admissible range.
    InvalidParameter { what: &'static str, value: f64 },
    /// `lo_i ≤ 0 ≤ hi_i` is violated, so `α = 0` is not feasible.
    InfeasibleStart { index: usize },
    /// A kernel index is out of range.
    KernelIndex { index: usize, kernel_size: usize },
}

impl fmt::Display for ProblemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                what,
                expected,
                got,
            } => {
                write!(f, "length of `{what}` is {got}, expected {expected}")
            }
            Self::InvalidParameter { what, value } => {
                write!(f, "invalid value {value} for `{what}`")
            }
            Self::InfeasibleStart { index } => {
                write!(
                    f,
                    "bounds of variable {index} do not contain zero (lo <= 0 <= hi required)"
                )
            }
            Self::KernelIndex { index, kernel_size } => {
                write!(
                    f,
                    "kernel index {index} out of range for kernel of size {kernel_size}"
                )
            }
        }
    }
}

impl std::error::Error for ProblemError {}

/// Separable quadratic dual problem (see module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct DualProblem {
    /// Regularization parameter `λ > 0`.
    pub lambda: f64,
    /// Kernel sample `k(i)` of each variable.
    pub kernel_index: Vec<usize>,
    /// Quadratic coefficients `q_i ≥ 0` of the conjugate loss.
    pub q: Vec<f64>,
    /// Linear coefficients `p_i` of the conjugate loss.
    pub p: Vec<f64>,
    /// Lower bounds (may be `-inf`).
    pub lo: Vec<f64>,
    /// Upper bounds (may be `+inf`).
    pub hi: Vec<f64>,
    /// Signs `s_i ∈ {-1, +1}` of the optional 1-norm constraint.
    pub sign: Vec<f64>,
    /// Right-hand side `A` of `Σ s_i α_i ≤ A`, `None` if the constraint is absent.
    pub max_asum: Option<f64>,
}

/// Quadratically smoothed maximum function `m_δ(t)` (Strasdat, Example 4.12 with degree 2).
///
/// `m_δ(t) = max(0, t)` for `|t| ≥ δ` and `(t + δ)² / (4δ)` otherwise; `δ = 0` gives `max(0, t)`.
pub fn smooth_max(t: f64, delta: f64) -> f64 {
    if t >= delta {
        t
    } else if t <= -delta {
        0.0
    } else {
        0.25 / delta * (t + delta) * (t + delta)
    }
}

fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), ProblemError> {
    if expected == got {
        Ok(())
    } else {
        Err(ProblemError::LengthMismatch {
            what,
            expected,
            got,
        })
    }
}

fn check_param(what: &'static str, value: f64, ok: bool) -> Result<(), ProblemError> {
    if ok {
        Ok(())
    } else {
        Err(ProblemError::InvalidParameter { what, value })
    }
}

/// `a / w`, or `0` for `w = 0` (the variable is then fixed at zero by its bounds).
fn scaled_inverse(a: f64, w: f64) -> f64 {
    if w > 0.0 {
        a / w
    } else {
        0.0
    }
}

fn weights_or_ones(weights: Option<&[f64]>, n: usize) -> Result<Vec<f64>, ProblemError> {
    match weights {
        None => Ok(vec![1.0; n]),
        Some(w) => {
            check_len("weights", n, w.len())?;
            for &wi in w {
                check_param("weights", wi, wi >= 0.0 && wi.is_finite())?;
            }
            Ok(w.to_vec())
        }
    }
}

impl DualProblem {
    /// Number of solver variables.
    pub fn len(&self) -> usize {
        self.q.len()
    }

    /// Returns `true` if the problem has no variables.
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// Checks consistency of all fields for a kernel with `kernel_size` samples.
    pub fn validate(&self, kernel_size: usize) -> Result<(), ProblemError> {
        let n = self.len();
        check_param(
            "lambda",
            self.lambda,
            self.lambda > 0.0 && self.lambda.is_finite(),
        )?;
        check_len("kernel_index", n, self.kernel_index.len())?;
        check_len("p", n, self.p.len())?;
        check_len("lo", n, self.lo.len())?;
        check_len("hi", n, self.hi.len())?;
        check_len("sign", n, self.sign.len())?;
        for i in 0..n {
            if self.kernel_index[i] >= kernel_size {
                return Err(ProblemError::KernelIndex {
                    index: self.kernel_index[i],
                    kernel_size,
                });
            }
            check_param("q", self.q[i], self.q[i] >= 0.0 && self.q[i].is_finite())?;
            check_param("p", self.p[i], self.p[i].is_finite())?;
            if !(self.lo[i] <= 0.0 && 0.0 <= self.hi[i]) {
                return Err(ProblemError::InfeasibleStart { index: i });
            }
            if self.max_asum.is_some() {
                check_param(
                    "sign",
                    self.sign[i],
                    self.sign[i] == 1.0 || self.sign[i] == -1.0,
                )?;
            }
        }
        if let Some(a) = self.max_asum {
            check_param("max_asum", a, a >= 0.0)?;
        }
        Ok(())
    }

    /// Weighted binary classification with the (smoothed) hinge loss
    /// `ℓ_i(t) = w_i m_δ(shift - y_i t)` (Strasdat, Prop. 4.1 and Obs. 4.16).
    ///
    /// Labels must be `±1`. `smoothing = δ = 0` gives the classical SVM with `C = 1/λ`.
    pub fn classification(
        y: &[f64],
        lambda: f64,
        weights: Option<&[f64]>,
        shift: f64,
        smoothing: f64,
        max_asum: Option<f64>,
    ) -> Result<Self, ProblemError> {
        let n = y.len();
        let w = weights_or_ones(weights, n)?;
        check_param(
            "smoothing",
            smoothing,
            smoothing >= 0.0 && smoothing.is_finite(),
        )?;
        check_param("shift", shift, shift.is_finite())?;
        let mut prob = Self {
            lambda,
            kernel_index: (0..n).collect(),
            q: Vec::with_capacity(n),
            p: Vec::with_capacity(n),
            lo: Vec::with_capacity(n),
            hi: Vec::with_capacity(n),
            sign: Vec::with_capacity(n),
            max_asum,
        };
        for (&yi, &wi) in y.iter().zip(&w) {
            check_param("y", yi, yi == 1.0 || yi == -1.0)?;
            prob.q.push(scaled_inverse(2.0 * smoothing, wi));
            prob.p.push(-(shift + smoothing) * yi);
            prob.lo.push(if yi > 0.0 { 0.0 } else { -wi });
            prob.hi.push(if yi > 0.0 { wi } else { 0.0 });
            prob.sign.push(yi);
        }
        Ok(prob)
    }

    /// Weighted ε-insensitive regression with the (smoothed) loss
    /// `ℓ_i(t) = w_i (m_δ(t - y_i - ε) + m_δ(y_i - t - ε))` (Strasdat, Prop. 4.5).
    ///
    /// Uses `2n` variables: `α_i ∈ [0, w_i]` for `i < n` and `α_{n+i} ∈ [-w_i, 0]`,
    /// both mapped to kernel sample `i`. The dual coefficient of sample `i` is
    /// `α_i + α_{n+i}` (see [`DualProblem::sample_coefficients`]).
    pub fn regression(
        y: &[f64],
        lambda: f64,
        weights: Option<&[f64]>,
        epsilon: f64,
        smoothing: f64,
        max_asum: Option<f64>,
    ) -> Result<Self, ProblemError> {
        let n = y.len();
        let w = weights_or_ones(weights, n)?;
        check_param("epsilon", epsilon, epsilon >= 0.0 && epsilon.is_finite())?;
        check_param(
            "smoothing",
            smoothing,
            smoothing >= 0.0 && smoothing.is_finite(),
        )?;
        let mut prob = Self {
            lambda,
            kernel_index: (0..n).chain(0..n).collect(),
            q: Vec::with_capacity(2 * n),
            p: Vec::with_capacity(2 * n),
            lo: Vec::with_capacity(2 * n),
            hi: Vec::with_capacity(2 * n),
            sign: Vec::with_capacity(2 * n),
            max_asum,
        };
        for s in [1.0, -1.0] {
            for (&yi, &wi) in y.iter().zip(&w) {
                check_param("y", yi, yi.is_finite())?;
                prob.q.push(scaled_inverse(2.0 * smoothing, wi));
                prob.p.push(-yi + s * (epsilon - smoothing));
                prob.lo.push(if s > 0.0 { 0.0 } else { -wi });
                prob.hi.push(if s > 0.0 { wi } else { 0.0 });
                prob.sign.push(s);
            }
        }
        Ok(prob)
    }

    /// Weighted least-squares SVM `ℓ_i(t) = w_i/2 (t - y_i)²` (Strasdat, Prop. 4.4).
    pub fn least_squares(
        y: &[f64],
        lambda: f64,
        weights: Option<&[f64]>,
    ) -> Result<Self, ProblemError> {
        let n = y.len();
        let w = weights_or_ones(weights, n)?;
        for &yi in y {
            check_param("y", yi, yi.is_finite())?;
        }
        Ok(Self {
            lambda,
            kernel_index: (0..n).collect(),
            q: w.iter().map(|&wi| scaled_inverse(1.0, wi)).collect(),
            p: y.iter().map(|yi| -yi).collect(),
            // a sample with weight zero has no loss: its dual variable is fixed at zero
            lo: w
                .iter()
                .map(|&wi| if wi > 0.0 { f64::NEG_INFINITY } else { 0.0 })
                .collect(),
            hi: w
                .iter()
                .map(|&wi| if wi > 0.0 { f64::INFINITY } else { 0.0 })
                .collect(),
            sign: vec![1.0; n],
            max_asum: None,
        })
    }

    /// Number of kernel samples referenced by the problem (`max k(i) + 1`).
    pub fn n_samples(&self) -> usize {
        self.kernel_index.iter().max().map_or(0, |m| m + 1)
    }

    /// Sums the variables belonging to the same kernel sample: `β_k = Σ_{i: k(i)=k} α_i`.
    ///
    /// The decision function is `f(x) = 1/λ Σ_k β_k K(x, x_k) + b`.
    pub fn sample_coefficients(&self, alpha: &[f64]) -> Vec<f64> {
        let mut beta = vec![0.0; self.n_samples()];
        for (&k, &a) in self.kernel_index.iter().zip(alpha) {
            beta[k] += a;
        }
        beta
    }

    /// Value of the conjugate loss term `½ q_i α² + p_i α` (without the box indicator).
    #[inline]
    pub fn conjugate_loss(&self, i: usize, alpha: f64) -> f64 {
        (0.5 * self.q[i] * alpha + self.p[i]) * alpha
    }

    /// Loss `ℓ_i(t) = sup_{α ∈ [lo_i, hi_i]} -α t - ½ q_i α² - p_i α`, recovered from its conjugate.
    pub fn loss(&self, i: usize, t: f64) -> f64 {
        let slope = -(t + self.p[i]);
        let a = if self.q[i] > 0.0 {
            (slope / self.q[i]).clamp(self.lo[i], self.hi[i])
        } else if slope > 0.0 {
            self.hi[i]
        } else if slope < 0.0 {
            self.lo[i]
        } else {
            0.0
        };
        if a.is_infinite() {
            return f64::INFINITY;
        }
        slope * a - 0.5 * self.q[i] * a * a
    }

    /// Dual objective for given `α` and `ka_i = 1/λ Σ_j α_j K[k(i), k(j)]`.
    pub fn dual_objective(&self, alpha: &[f64], ka: &[f64]) -> f64 {
        (0..self.len())
            .map(|i| 0.5 * alpha[i] * ka[i] + self.conjugate_loss(i, alpha[i]))
            .sum()
    }

    /// Primal objective `λ/2 |w|² + Σ_i ℓ_i(ka_i + b + s_i c) + A c` for the primal point
    /// induced by `α` (with `w = 1/λ Σ α_j φ_j`).
    pub fn primal_objective(&self, alpha: &[f64], ka: &[f64], b: f64, c: f64) -> f64 {
        let mut obj = 0.0;
        for i in 0..self.len() {
            obj += 0.5 * alpha[i] * ka[i] + self.loss(i, ka[i] + b + self.sign[i] * c);
        }
        if let Some(a) = self.max_asum {
            obj += a * c;
        }
        obj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_conjugate(prob: &DualProblem, i: usize, alpha: f64) -> f64 {
        // ℓ*(-α) = sup_t (-α t - ℓ(t)), evaluated on a fine grid
        (-4000..=4000)
            .map(|k| {
                let t = k as f64 * 1e-3;
                -alpha * t - prob.loss(i, t)
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn classification_loss_matches_smoothed_hinge() {
        let y = [1.0, -1.0];
        let w = [2.0, 0.5];
        let delta = 0.3;
        let prob = DualProblem::classification(&y, 1.0, Some(&w), 1.0, delta, None).unwrap();
        for i in 0..2 {
            for k in -30..30 {
                let t = k as f64 * 0.1;
                let expected = w[i] * smooth_max(1.0 - y[i] * t, delta);
                assert!((prob.loss(i, t) - expected).abs() < 1e-12, "i={i} t={t}");
            }
            // conjugate is consistent on the box
            for k in 0..=10 {
                let a = prob.lo[i] + (prob.hi[i] - prob.lo[i]) * k as f64 / 10.0;
                let num = numeric_conjugate(&prob, i, a);
                assert!(
                    (num - prob.conjugate_loss(i, a)).abs() < 1e-5,
                    "i={i} a={a}"
                );
            }
        }
    }

    #[test]
    fn regression_loss_matches_epsilon_insensitive() {
        let y = [0.5];
        let (eps, delta) = (0.2, 0.1);
        let prob = DualProblem::regression(&y, 1.0, None, eps, delta, None).unwrap();
        assert_eq!(prob.len(), 2);
        for k in -30..30 {
            let t = k as f64 * 0.1;
            let expected = smooth_max(t - 0.5 - eps, delta) + smooth_max(0.5 - t - eps, delta);
            let got = prob.loss(0, t) + prob.loss(1, t);
            assert!((got - expected).abs() < 1e-12, "t={t}");
        }
    }

    #[test]
    fn least_squares_loss() {
        let prob = DualProblem::least_squares(&[1.5], 1.0, Some(&[4.0])).unwrap();
        assert!((prob.loss(0, 0.5) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn zero_weights_fix_variables() {
        let p = DualProblem::classification(&[1.0, -1.0], 1.0, Some(&[0.0, 1.0]), 1.0, 0.5, None)
            .unwrap();
        assert_eq!((p.lo[0], p.hi[0], p.q[0]), (0.0, 0.0, 0.0));
        let p = DualProblem::least_squares(&[1.0], 1.0, Some(&[0.0])).unwrap();
        assert_eq!((p.lo[0], p.hi[0], p.q[0]), (0.0, 0.0, 0.0));
        assert_eq!(p.loss(0, 3.0), 0.0);
    }

    #[test]
    fn validation_errors() {
        assert!(DualProblem::classification(&[1.0, 0.0], 1.0, None, 1.0, 0.0, None).is_err());
        assert!(
            DualProblem::classification(&[1.0], 1.0, Some(&[1.0, 2.0]), 1.0, 0.0, None).is_err()
        );
        let mut p = DualProblem::least_squares(&[1.0, 2.0], 1.0, None).unwrap();
        assert!(p.validate(2).is_ok());
        assert!(p.validate(1).is_err());
        p.lambda = 0.0;
        assert!(p.validate(2).is_err());
        let mut p = DualProblem::least_squares(&[1.0], 1.0, None).unwrap();
        p.lo[0] = 0.5;
        assert_eq!(
            p.validate(1),
            Err(ProblemError::InfeasibleStart { index: 0 })
        );
    }

    #[test]
    fn sample_coefficients_merge_regression_copies() {
        let prob = DualProblem::regression(&[0.0, 1.0], 1.0, None, 0.1, 0.0, None).unwrap();
        let beta = prob.sample_coefficients(&[0.5, 0.0, 0.0, -0.25]);
        assert_eq!(beta, vec![0.5, -0.25]);
    }
}
