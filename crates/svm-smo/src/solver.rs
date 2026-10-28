//! Sequential Minimal Optimization for [`DualProblem`]s.
//!
//! In every step two variables `(i, j)` are updated along `α ← α + t (e_j - e_i)`, which keeps
//! `Σ α = 0`. The step size `t` minimises the (quadratic) objective along this direction
//! subject to the bounds and, if present, the 1-norm constraint.
//!
//! *Optimality.* With the dual gradient `g_i = ka_i + q_i α_i + p_i` and
//! `I_down = {i : α_i > lo_i}`, `I_up = {j : α_j < hi_j}` the violation measure is
//! `r(α) = max_{i ∈ I_down} g_i - min_{j ∈ I_up} g_j` (Strasdat, Sec. 7.2/7.3); `α` is optimal
//! iff `r(α) ≤ 0`. While the 1-norm constraint is active, pairs that would increase
//! `Σ s_i α_i` are excluded. The solver stops once `r(α) ≤ tol`. The scaling of `r` matches
//! the gradient used by LIBSVM, hence `tol = 1e-3` is a reasonable default.
//!
//! *Working set selection* (Strasdat, Sec. 7.7): maximal violating pair, maximal unbounded
//! decrease pair (second-order rule of Fan, Chen & Lin) or maximal decrease pair, which also
//! takes the bounds into account when predicting the decrease (default).
//!
//! *Shrinking* (Strasdat, Sec. 7.10): a variable at a bound is removed from the active set if
//! its shifted gradient `g_i + b + s_i c` exceeds `factor · r(α)^η` with the wrong sign for
//! leaving the bound. Before termination the full problem is restored and checked again.

use web_time::Instant;

use crate::cache::RowCache;
use crate::kernel::Kernel;
use crate::problem::{DualProblem, ProblemError};

/// Lower bound for the curvature along a pair direction (guards against non-PSD kernels).
const TAU: f64 = 1e-12;

/// Working set selection rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkingSetSelection {
    /// First-order rule: the maximal violating pair.
    MaximalViolatingPair,
    /// Second-order rule predicting the decrease without bounds (Fan, Chen & Lin 2005).
    MaximalUnboundedDecrease,
    /// Second-order rule predicting the decrease of the bound-constrained step.
    #[default]
    MaximalDecrease,
}

/// Options of the SMO solver.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoOptions {
    /// Stop as soon as the violation measure is at most `tol`.
    pub tol: f64,
    /// Maximal number of SMO steps.
    pub max_steps: u64,
    /// Wall-clock time limit in seconds.
    pub time_limit: Option<f64>,
    /// Working set selection rule.
    pub working_set: WorkingSetSelection,
    /// Enables shrinking of the active set.
    pub shrinking: bool,
    /// Number of steps between two shrinking attempts (`0`: `min(n, 1000)`).
    pub shrinking_period: u64,
    /// Exponent `η ∈ (0, 1)` of the shrinking threshold `factor · r^η`.
    pub shrinking_exponent: f64,
    /// Factor of the shrinking threshold.
    pub shrinking_factor: f64,
    /// Memory budget of the kernel row cache in megabytes.
    pub cache_size_mb: f64,
    /// Number of steps between two callback invocations.
    pub callback_period: u64,
}

impl Default for SmoOptions {
    fn default() -> Self {
        Self {
            tol: 1e-3,
            max_steps: 100_000_000,
            time_limit: None,
            working_set: WorkingSetSelection::default(),
            shrinking: true,
            shrinking_period: 0,
            shrinking_exponent: 0.5,
            shrinking_factor: 1.0,
            cache_size_mb: 256.0,
            callback_period: 1000,
        }
    }
}

/// Reason for the termination of the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The violation measure dropped below the tolerance.
    Converged,
    /// The maximal number of steps was reached.
    MaxSteps,
    /// The time limit was reached.
    TimeLimit,
    /// The callback requested termination.
    Interrupted,
}

/// Snapshot passed to the progress callback.
#[derive(Debug)]
pub struct Progress<'a> {
    pub step: u64,
    pub elapsed: f64,
    pub violation: f64,
    pub dual_objective: f64,
    pub n_active: usize,
    pub alpha: &'a [f64],
}

/// Result of the SMO solver.
#[derive(Clone, Debug)]
pub struct SmoResult {
    /// Dual variables.
    pub alpha: Vec<f64>,
    /// Bias of the decision function.
    pub b: f64,
    /// Multiplier of the 1-norm constraint (`0` if absent or inactive).
    pub c: f64,
    /// `ka_i = 1/λ Σ_j α_j K[k(i), k(j)]`, i.e. the decision values without bias.
    pub ka: Vec<f64>,
    pub status: Status,
    pub steps: u64,
    /// Elapsed wall-clock time in seconds.
    pub elapsed: f64,
    /// Final violation measure `r(α)`.
    pub violation: f64,
    /// Dual objective value (the solver minimises it).
    pub dual_objective: f64,
    /// Primal objective value of the induced primal point `(w(α), b, c)`.
    pub primal_objective: f64,
    /// Duality gap `primal + dual ≥ 0`.
    pub gap: f64,
    /// Size of the active set when the main loop ended.
    pub n_active: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Solves `problem` with kernel `kernel` starting from `α = 0`.
pub fn solve<K: Kernel + ?Sized>(
    kernel: &K,
    problem: &DualProblem,
    options: &SmoOptions,
) -> Result<SmoResult, ProblemError> {
    solve_with(kernel, problem, options, None, |_| false)
}

/// Solves `problem`, optionally warm-started at `alpha0`, calling `callback` every
/// `options.callback_period` steps. The callback returns `true` to stop the solver.
pub fn solve_with<K, F>(
    kernel: &K,
    problem: &DualProblem,
    options: &SmoOptions,
    alpha0: Option<&[f64]>,
    callback: F,
) -> Result<SmoResult, ProblemError>
where
    K: Kernel + ?Sized,
    F: FnMut(&Progress) -> bool,
{
    problem.validate(kernel.len())?;
    let mut smo = Smo::new(kernel, problem, options, alpha0)?;
    Ok(smo.run(callback))
}

/// Extremal gradients per sign group (`0`: `s ≥ 0` or no constraint, `1`: `s < 0`).
struct Extremes {
    down_max: [f64; 2],
    down_arg: [usize; 2],
    up_min: [f64; 2],
    up_arg: [usize; 2],
    free_sum: [f64; 2],
    free_cnt: [usize; 2],
}

struct Smo<'a, K: Kernel + ?Sized> {
    kernel: &'a K,
    prob: &'a DualProblem,
    opt: &'a SmoOptions,
    alpha: Vec<f64>,
    g: Vec<f64>,
    ka: Vec<f64>,
    kdiag: Vec<f64>,
    /// Active variables; cache columns are `kernel_index[active[p]]`.
    active: Vec<usize>,
    cache: RowCache,
    constrained: bool,
    max_asum: f64,
    asum: f64,
    violation: f64,
    value: f64,
    start: Instant,
}

impl<'a, K: Kernel + ?Sized> Smo<'a, K> {
    fn new(
        kernel: &'a K,
        prob: &'a DualProblem,
        opt: &'a SmoOptions,
        alpha0: Option<&[f64]>,
    ) -> Result<Self, ProblemError> {
        let n = prob.len();
        let active: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = prob.kernel_index.clone();
        let capacity = RowCache::capacity_for(opt.cache_size_mb, n);
        let mut smo = Self {
            kernel,
            prob,
            opt,
            alpha: vec![0.0; n],
            g: prob.p.clone(),
            ka: vec![0.0; n],
            kdiag: prob.kernel_index.iter().map(|&k| kernel.diag(k)).collect(),
            active,
            cache: RowCache::new(kernel.len(), cols, capacity),
            constrained: prob.max_asum.is_some(),
            max_asum: prob.max_asum.unwrap_or(f64::INFINITY),
            asum: 0.0,
            violation: f64::INFINITY,
            value: 0.0,
            start: Instant::now(),
        };
        if let Some(a0) = alpha0 {
            smo.warm_start(a0)?;
        }
        Ok(smo)
    }

    fn warm_start(&mut self, alpha0: &[f64]) -> Result<(), ProblemError> {
        let p = self.prob;
        let n = p.len();
        if alpha0.len() != n {
            return Err(ProblemError::LengthMismatch {
                what: "alpha0",
                expected: n,
                got: alpha0.len(),
            });
        }
        let scale: f64 = alpha0.iter().map(|a| a.abs()).sum::<f64>().max(1.0);
        let sum: f64 = alpha0.iter().sum();
        if sum.abs() > 1e-9 * scale {
            return Err(ProblemError::InvalidParameter {
                what: "sum(alpha0)",
                value: sum,
            });
        }
        for ((&a, &lo), &hi) in alpha0.iter().zip(&p.lo).zip(&p.hi) {
            if !(lo <= a && a <= hi) {
                return Err(ProblemError::InvalidParameter {
                    what: "alpha0",
                    value: a,
                });
            }
        }
        let asum: f64 = alpha0.iter().zip(&p.sign).map(|(a, s)| a * s).sum();
        if self.constrained && asum > self.max_asum * (1.0 + 1e-12) {
            return Err(ProblemError::InvalidParameter {
                what: "asum(alpha0)",
                value: asum,
            });
        }
        self.alpha.copy_from_slice(alpha0);
        self.asum = asum.min(self.max_asum);
        self.recompute_ka();
        self.value = p.dual_objective(&self.alpha, &self.ka);
        Ok(())
    }

    #[inline]
    fn at_asum_bound(&self) -> bool {
        self.constrained && self.asum >= self.max_asum
    }

    /// Updates the gradient on the active set and collects extremal values.
    fn scan(&mut self) -> Extremes {
        let p = self.prob;
        let mut ex = Extremes {
            down_max: [f64::NEG_INFINITY; 2],
            down_arg: [usize::MAX; 2],
            up_min: [f64::INFINITY; 2],
            up_arg: [usize::MAX; 2],
            free_sum: [0.0; 2],
            free_cnt: [0; 2],
        };
        for (pos, &v) in self.active.iter().enumerate() {
            let a = self.alpha[v];
            let gv = self.ka[v] + p.q[v] * a + p.p[v];
            self.g[v] = gv;
            let grp = usize::from(self.constrained && p.sign[v] < 0.0);
            let can_down = a > p.lo[v];
            let can_up = a < p.hi[v];
            if can_down && gv > ex.down_max[grp] {
                ex.down_max[grp] = gv;
                ex.down_arg[grp] = pos;
            }
            if can_up && gv < ex.up_min[grp] {
                ex.up_min[grp] = gv;
                ex.up_arg[grp] = pos;
            }
            if can_down && can_up {
                ex.free_sum[grp] += gv;
                ex.free_cnt[grp] += 1;
            }
        }
        ex
    }

    /// Maximal violating pair `(i, j)` (positions in the active set) and the violation.
    fn maximal_violating_pair(&self, ex: &Extremes) -> (f64, usize, usize) {
        let mut best = (f64::NEG_INFINITY, usize::MAX, usize::MAX);
        let candidates: &[(usize, usize)] = if !self.constrained {
            &[(0, 0)]
        } else if self.at_asum_bound() {
            &[(0, 0), (1, 1), (0, 1)]
        } else {
            &[(0, 0), (1, 1), (0, 1), (1, 0)]
        };
        for &(gi, gj) in candidates {
            let d = ex.down_max[gi] - ex.up_min[gj];
            if ex.down_arg[gi] != usize::MAX && ex.up_arg[gj] != usize::MAX && d > best.0 {
                best = (d, ex.down_arg[gi], ex.up_arg[gj]);
            }
        }
        best
    }

    /// Maximal step for moving from variable `i` to variable `j`.
    #[inline]
    fn max_step(&self, i: usize, j: usize) -> f64 {
        let p = self.prob;
        let mut t = (self.alpha[i] - p.lo[i]).min(p.hi[j] - self.alpha[j]);
        if self.constrained && p.sign[j] > p.sign[i] {
            t = t.min(0.5 * (self.max_asum - self.asum).max(0.0));
        }
        t
    }

    #[inline]
    fn curvature(&self, i: usize, j: usize, kij: f64) -> f64 {
        let p = self.prob;
        ((self.kdiag[i] + self.kdiag[j] - 2.0 * kij) / p.lambda + p.q[i] + p.q[j]).max(TAU)
    }

    #[inline]
    fn predicted_decrease(&self, i: usize, j: usize, grad_diff: f64, eta: f64) -> f64 {
        match self.opt.working_set {
            WorkingSetSelection::MaximalUnboundedDecrease => 0.5 * grad_diff * grad_diff / eta,
            _ => {
                let t = (grad_diff / eta).min(self.max_step(i, j));
                t * (grad_diff - 0.5 * eta * t)
            }
        }
    }

    /// Second-order working set selection (Strasdat, Def. 7.27/7.28).
    fn second_order_pair(&mut self, i0: usize, j1: usize) -> (usize, usize) {
        let si = self
            .cache
            .ensure(self.kernel, self.prob.kernel_index[self.active[i0]]);
        let sj = self
            .cache
            .ensure(self.kernel, self.prob.kernel_index[self.active[j1]]);
        let (vi0, vj1) = (self.active[i0], self.active[j1]);
        let (gi0, gj1) = (self.g[vi0], self.g[vj1]);
        let block_up = self.at_asum_bound();
        let p = self.prob;
        let mut best = (f64::NEG_INFINITY, i0, j1);
        for (pos, &v) in self.active.iter().enumerate() {
            let a = self.alpha[v];
            let gv = self.g[v];
            // partner j = v for fixed i0
            if a < p.hi[v] && gi0 > gv && !(block_up && p.sign[v] > p.sign[vi0]) {
                let eta = self.curvature(vi0, v, self.cache.row(si)[pos]);
                let dec = self.predicted_decrease(vi0, v, gi0 - gv, eta);
                if dec > best.0 {
                    best = (dec, i0, pos);
                }
            }
            // partner i = v for fixed j1
            if a > p.lo[v] && gv > gj1 && !(block_up && p.sign[vj1] > p.sign[v]) {
                let eta = self.curvature(v, vj1, self.cache.row(sj)[pos]);
                let dec = self.predicted_decrease(v, vj1, gv - gj1, eta);
                if dec > best.0 {
                    best = (dec, pos, j1);
                }
            }
        }
        (best.1, best.2)
    }

    /// Exact minimisation along `e_j - e_i` for active positions `ip`, `jp`.
    fn update(&mut self, ip: usize, jp: usize) {
        let p = self.prob;
        let (vi, vj) = (self.active[ip], self.active[jp]);
        let si = self.cache.ensure(self.kernel, p.kernel_index[vi]);
        let sj = self.cache.ensure(self.kernel, p.kernel_index[vj]);
        let eta = self.curvature(vi, vj, self.cache.row(si)[jp]);
        let grad_diff = self.g[vi] - self.g[vj];
        let box_i = self.alpha[vi] - p.lo[vi];
        let box_j = p.hi[vj] - self.alpha[vj];
        let half_slack = if self.constrained && p.sign[vj] > p.sign[vi] {
            0.5 * (self.max_asum - self.asum).max(0.0)
        } else {
            f64::INFINITY
        };
        let t = (grad_diff / eta).min(box_i).min(box_j).min(half_slack);
        if t <= 0.0 {
            return;
        }
        self.alpha[vi] = if t == box_i {
            p.lo[vi]
        } else {
            self.alpha[vi] - t
        };
        self.alpha[vj] = if t == box_j {
            p.hi[vj]
        } else {
            self.alpha[vj] + t
        };
        if self.constrained {
            if t == half_slack {
                self.asum = self.max_asum;
            } else {
                self.asum += t * (p.sign[vj] - p.sign[vi]);
            }
        }
        self.value -= t * (grad_diff - 0.5 * eta * t);
        let tl = t / p.lambda;
        let Self {
            cache, ka, active, ..
        } = self;
        let (ri, rj) = (cache.row(si), cache.row(sj));
        for (pos, &v) in active.iter().enumerate() {
            ka[v] += tl * (rj[pos] - ri[pos]);
        }
    }

    /// Estimates `b` and `c` from the gradient (averages over free variables).
    fn bias(&self, ex: &Extremes) -> (f64, f64) {
        let level = |sum: f64, cnt: usize, dmax: f64, umin: f64| -> Option<f64> {
            if cnt > 0 {
                Some(-sum / cnt as f64)
            } else if dmax.is_finite() && umin.is_finite() {
                Some(-0.5 * (dmax + umin))
            } else if dmax.is_finite() {
                Some(-dmax)
            } else if umin.is_finite() {
                Some(-umin)
            } else {
                None
            }
        };
        let combined = level(
            ex.free_sum[0] + ex.free_sum[1],
            ex.free_cnt[0] + ex.free_cnt[1],
            ex.down_max[0].max(ex.down_max[1]),
            ex.up_min[0].min(ex.up_min[1]),
        )
        .unwrap_or(0.0);
        if !self.at_asum_bound() {
            return (combined, 0.0);
        }
        let m0 = level(ex.free_sum[0], ex.free_cnt[0], ex.down_max[0], ex.up_min[0]);
        let m1 = level(ex.free_sum[1], ex.free_cnt[1], ex.down_max[1], ex.up_min[1]);
        match (m0, m1) {
            (Some(m0), Some(m1)) if m0 > m1 => (0.5 * (m0 + m1), 0.5 * (m0 - m1)),
            _ => (combined, 0.0),
        }
    }

    fn shrink(&mut self, ex: &Extremes) {
        let p = self.prob;
        let (b, c) = self.bias(ex);
        let threshold =
            self.opt.shrinking_factor * self.violation.max(0.0).powf(self.opt.shrinking_exponent);
        let keep: Vec<bool> = self
            .active
            .iter()
            .map(|&v| {
                let shifted = self.g[v] + b + p.sign[v] * c;
                let a = self.alpha[v];
                let fixed_lo = a <= p.lo[v] && shifted > threshold;
                let fixed_hi = a >= p.hi[v] && shifted < -threshold;
                !(fixed_lo || fixed_hi)
            })
            .collect();
        if keep.iter().all(|&k| k) {
            return;
        }
        let mut it = keep.iter();
        self.active.retain(|_| *it.next().unwrap());
        self.cache.restrict(&keep);
    }

    /// Recomputes `ka` for all variables from scratch.
    fn recompute_ka(&mut self) {
        let p = self.prob;
        let beta = p.sample_coefficients(&self.alpha);
        let support: Vec<(usize, f64)> = beta
            .iter()
            .enumerate()
            .filter(|(_, &b)| b != 0.0)
            .map(|(k, &b)| (k, b / p.lambda))
            .collect();
        let kernel = self.kernel;
        let compute =
            |m: usize| -> f64 { support.iter().map(|&(k, b)| b * kernel.eval(m, k)).sum() };
        let n_samples = beta.len();
        #[cfg(feature = "parallel")]
        let kam: Vec<f64> = {
            use rayon::prelude::*;
            (0..n_samples).into_par_iter().map(compute).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let kam: Vec<f64> = (0..n_samples).map(compute).collect();
        for (ka, &k) in self.ka.iter_mut().zip(&p.kernel_index) {
            *ka = kam[k];
        }
    }

    fn unshrink(&mut self) {
        self.active = (0..self.prob.len()).collect();
        self.cache.reset(self.prob.kernel_index.clone());
        self.recompute_ka();
    }

    fn run<F: FnMut(&Progress) -> bool>(&mut self, mut callback: F) -> SmoResult {
        let n = self.prob.len();
        let opt = self.opt;
        let shrinking_period = match opt.shrinking_period {
            0 => (n as u64).clamp(1, 1000),
            k => k,
        };
        let mut status = Status::MaxSteps;
        let mut steps = 0;
        while steps < opt.max_steps {
            let elapsed = self.start.elapsed().as_secs_f64();
            if opt.time_limit.is_some_and(|limit| elapsed >= limit) {
                status = Status::TimeLimit;
                break;
            }
            if opt.callback_period > 0 && steps % opt.callback_period == 0 && steps > 0 {
                let progress = Progress {
                    step: steps,
                    elapsed,
                    violation: self.violation,
                    dual_objective: self.value,
                    n_active: self.active.len(),
                    alpha: &self.alpha,
                };
                if callback(&progress) {
                    status = Status::Interrupted;
                    break;
                }
            }

            let ex = self.scan();
            let (violation, i0, j1) = self.maximal_violating_pair(&ex);
            self.violation = violation.max(0.0);
            if violation <= opt.tol {
                if self.active.len() < n {
                    self.unshrink();
                    continue;
                }
                status = Status::Converged;
                break;
            }
            if opt.shrinking && steps > 0 && steps % shrinking_period == 0 {
                let before = self.active.len();
                self.shrink(&ex);
                if self.active.len() < before {
                    steps += 1;
                    continue;
                }
            }
            let (i, j) = match opt.working_set {
                WorkingSetSelection::MaximalViolatingPair => (i0, j1),
                _ => self.second_order_pair(i0, j1),
            };
            self.update(i, j);
            steps += 1;
        }
        self.finish(status, steps)
    }

    fn finish(&mut self, status: Status, steps: u64) -> SmoResult {
        let n_active = self.active.len();
        if n_active < self.prob.len() {
            self.unshrink();
        }
        let ex = self.scan();
        let (violation, _, _) = self.maximal_violating_pair(&ex);
        let (b, c) = self.bias(&ex);
        let p = self.prob;
        let dual = p.dual_objective(&self.alpha, &self.ka);
        let primal = p.primal_objective(&self.alpha, &self.ka, b, c);
        SmoResult {
            alpha: std::mem::take(&mut self.alpha),
            b,
            c,
            ka: std::mem::take(&mut self.ka),
            status,
            steps,
            elapsed: self.start.elapsed().as_secs_f64(),
            violation: violation.max(0.0),
            dual_objective: dual,
            primal_objective: primal,
            gap: primal + dual,
            n_active,
            cache_hits: self.cache.hits,
            cache_misses: self.cache.misses,
        }
    }
}
