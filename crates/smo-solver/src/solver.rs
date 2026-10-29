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
//! the gradient used by LIBSVM, hence `tol = 1e-3` is a reasonable default. Optionally the
//! bound-aware measure `r◇(α, ρ) / ρ` (Sec. 7.6) can be used as well, see [`StoppingCriterion`].
//!
//! *Working set selection* (Strasdat, Sec. 7.7): maximal violating pair, maximal unbounded
//! decrease pair (second-order rule of Fan, Chen & Lin) or maximal decrease pair, which also
//! takes the bounds into account when predicting the decrease (default).
//!
//! *Shrinking* (Strasdat, Sec. 7.10): a variable at a bound is removed from the active set if
//! its shifted gradient `g_i + b + s_i c` exceeds `factor · r(α)^η` with the wrong sign for
//! leaving the bound. Before termination the full problem is restored and checked again.

use crate::clock::Instant;

use crate::cache::RowCache;
use crate::kernel::Kernel;
use crate::measures;
use crate::problem::{DualProblem, ProblemError};
use crate::simd::{self, Block, Candidates};

/// Lower bound for the curvature along a pair direction (guards against non-PSD kernels).
pub(crate) const TAU: f64 = 1e-12;

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

/// Stopping criterion of the solver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StoppingCriterion {
    /// Stop as soon as the violation measure `r(α)` is at most `tol`.
    #[default]
    Violation,
    /// Stop as soon as `r(α) ≤ tol` or the bound-aware violation measure of Strasdat, Def. 7.21,
    /// normalised by `ρ`, is at most `tol`: `r◇(α, ρ) / ρ ≤ tol`.
    ///
    /// Since `min(ε_ij, ρ) ≤ ρ` one always has `r◇(α, ρ) / ρ ≤ r(α)`, so this criterion stops
    /// no later than [`StoppingCriterion::Violation`]; it ignores violating pairs that can only
    /// move a tiny distance because one of their variables is almost at its bound. The measure
    /// costs `O(m log m)` and is evaluated every `shrinking_period` steps (`min(n, 1000)` by default). Not available together
    /// with the 1-norm constraint.
    BoundAware,
}

/// Options of the SMO solver.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoOptions {
    /// Stop as soon as the stopping measure (see `stopping`) is at most `tol`.
    pub tol: f64,
    /// Stopping criterion.
    pub stopping: StoppingCriterion,
    /// Step size bound `ρ` of the bound-aware measure (`0`: `λ`, the value used in the
    /// computational study of the thesis, Sec. 10.2.2).
    pub rho: f64,
    /// Maximal number of SMO steps.
    pub max_steps: u64,
    /// Wall-clock time limit in seconds.
    pub time_limit: Option<f64>,
    /// Working set selection rule.
    pub working_set: WorkingSetSelection,
    /// Second-order rules: search a partner for both indices of the maximal violating pair
    /// (Strasdat, Def. 7.27/7.28) instead of only for the first one (as in LIBSVM).
    pub two_sided: bool,
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
            stopping: StoppingCriterion::default(),
            rho: 0.0,
            max_steps: 100_000_000,
            time_limit: None,
            working_set: WorkingSetSelection::default(),
            two_sided: true,
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

impl Extremes {
    fn new() -> Self {
        Self {
            down_max: [f64::NEG_INFINITY; 2],
            down_arg: [usize::MAX; 2],
            up_min: [f64::INFINITY; 2],
            up_arg: [usize::MAX; 2],
            free_sum: [0.0; 2],
            free_cnt: [0; 2],
        }
    }

    /// Accounts for the variable at `pos` with value `a`, bounds `lo`, `hi` and gradient `gv`.
    #[inline(always)]
    fn visit(&mut self, pos: usize, grp: usize, a: f64, lo: f64, hi: f64, gv: f64) {
        let can_down = a > lo;
        let can_up = a < hi;
        if can_down && gv > self.down_max[grp] {
            self.down_max[grp] = gv;
            self.down_arg[grp] = pos;
        }
        if can_up && gv < self.up_min[grp] {
            self.up_min[grp] = gv;
            self.up_arg[grp] = pos;
        }
        if can_down && can_up {
            self.free_sum[grp] += gv;
            self.free_cnt[grp] += 1;
        }
    }
}

impl Extremes {
    /// Like [`Extremes::visit`] without the statistics of the free variables.
    #[inline(always)]
    fn visit_lite(&mut self, pos: usize, grp: usize, a: f64, lo: f64, hi: f64, gv: f64) {
        if a > lo && gv > self.down_max[grp] {
            self.down_max[grp] = gv;
            self.down_arg[grp] = pos;
        }
        if a < hi && gv < self.up_min[grp] {
            self.up_min[grp] = gv;
            self.up_arg[grp] = pos;
        }
    }

    /// Accounts for a block of variables of group `0` starting at `base`, given their distances
    /// to the bounds. The position is only located if the block improves an extremum.
    #[inline(always)]
    fn visit_block(&mut self, base: usize, g: &[f64], room_down: &[f64], room_up: &[f64]) {
        let (bmax, bmin) = simd::extremal_values(g, room_down, room_up);
        if bmax > self.down_max[0] {
            let k = g
                .iter()
                .zip(room_down)
                .position(|(&gv, &rd)| rd > 0.0 && gv == bmax);
            self.down_max[0] = bmax;
            self.down_arg[0] = base + k.unwrap_or(0);
        }
        if bmin < self.up_min[0] {
            let k = g
                .iter()
                .zip(room_up)
                .position(|(&gv, &ru)| ru > 0.0 && gv == bmin);
            self.up_min[0] = bmin;
            self.up_arg[0] = base + k.unwrap_or(0);
        }
    }
}

/// Block length of the vectorisable passes over the active set.
const BLOCK: usize = 64;

/// Solver state. All per-variable arrays are stored in a permuted order such that the active
/// variables form the prefix `0..n_active` (as in LIBSVM); `perm[pos]` is the original index.
struct Smo<'a, K: Kernel + ?Sized> {
    kernel: &'a K,
    prob: &'a DualProblem,
    opt: &'a SmoOptions,
    perm: Vec<usize>,
    kidx: Vec<usize>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    q: Vec<f64>,
    p: Vec<f64>,
    sign: Vec<f64>,
    /// `K[k(i), k(i)] / λ + q_i`
    cdiag: Vec<f64>,
    alpha: Vec<f64>,
    /// `alpha - lo`: how far a variable can move down (`0` at the bound).
    room_down: Vec<f64>,
    /// `hi - alpha`: how far a variable can move up (`0` at the bound).
    room_up: Vec<f64>,
    /// Dual gradient `ka_i + q_i α_i + p_i`, updated incrementally on the active prefix.
    g: Vec<f64>,
    /// Extremal gradients computed by the last [`Smo::update`].
    pending: Option<Extremes>,
    n_active: usize,
    cache: RowCache,
    inv_lambda: f64,
    constrained: bool,
    max_asum: f64,
    asum: f64,
    violation: f64,
    value: f64,
    start: Instant,
}

/// Reorders `v` such that `v_new[pos] = v[order[pos]]`.
fn permute<T: Copy>(v: &mut [T], order: &[usize], scratch: &mut Vec<T>) {
    scratch.clear();
    scratch.extend(order.iter().map(|&o| v[o]));
    v.copy_from_slice(scratch);
}

impl<'a, K: Kernel + ?Sized> Smo<'a, K> {
    fn new(
        kernel: &'a K,
        prob: &'a DualProblem,
        opt: &'a SmoOptions,
        alpha0: Option<&[f64]>,
    ) -> Result<Self, ProblemError> {
        let n = prob.len();
        if !(opt.rho >= 0.0 && opt.rho.is_finite()) {
            return Err(ProblemError::InvalidParameter {
                what: "rho",
                value: opt.rho,
            });
        }
        if opt.stopping == StoppingCriterion::BoundAware {
            if let Some(a) = prob.max_asum {
                return Err(ProblemError::InvalidParameter {
                    what: "max_asum (not supported with the bound-aware stopping criterion)",
                    value: a,
                });
            }
        }
        let inv_lambda = 1.0 / prob.lambda;
        let capacity = RowCache::capacity_for(opt.cache_size_mb, n);
        let mut smo = Self {
            kernel,
            prob,
            opt,
            perm: (0..n).collect(),
            kidx: prob.kernel_index.clone(),
            lo: prob.lo.clone(),
            hi: prob.hi.clone(),
            q: prob.q.clone(),
            p: prob.p.clone(),
            sign: prob.sign.clone(),
            cdiag: (0..n)
                .map(|i| kernel.diag(prob.kernel_index[i]) * inv_lambda + prob.q[i])
                .collect(),
            alpha: vec![0.0; n],
            room_down: vec![0.0; n],
            room_up: prob.hi.clone(),
            g: prob.p.clone(),
            pending: None,
            n_active: n,
            cache: RowCache::new(kernel.len(), prob.kernel_index.clone(), capacity),
            inv_lambda,
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
        for pos in 0..n {
            smo.set_rooms(pos);
        }
        Ok(smo)
    }

    /// Refreshes the distances to the bounds of the variable at `pos` after a change of its value.
    #[inline]
    fn set_rooms(&mut self, pos: usize) {
        self.room_down[pos] = self.alpha[pos] - self.lo[pos];
        self.room_up[pos] = self.hi[pos] - self.alpha[pos];
    }

    /// Sets the initial point (called before any permutation took place).
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
        let ka = self.compute_ka();
        self.value = p.dual_objective(&self.alpha, &ka);
        self.set_gradient(&ka);
        Ok(())
    }

    #[inline]
    fn at_asum_bound(&self) -> bool {
        self.constrained && self.asum >= self.max_asum
    }

    /// Sets `g = ka + q α + p` for all variables.
    fn set_gradient(&mut self, ka: &[f64]) {
        for (pos, g) in self.g.iter_mut().enumerate() {
            *g = ka[pos] + self.q[pos] * self.alpha[pos] + self.p[pos];
        }
    }

    /// Collects the extremal gradients on the active set.
    #[inline(never)]
    fn scan(&self) -> Extremes {
        let m = self.n_active;
        let mut ex = Extremes::new();
        let (alpha, lo, hi, sign, g) = (
            &self.alpha[..m],
            &self.lo[..m],
            &self.hi[..m],
            &self.sign[..m],
            &self.g[..m],
        );
        for pos in 0..m {
            let grp = usize::from(self.constrained && sign[pos] < 0.0);
            ex.visit(pos, grp, alpha[pos], lo[pos], hi[pos], g[pos]);
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

    /// Second-order working set selection (Strasdat, Def. 7.27/7.28): the MVP partner of
    /// either `i0` or `j1` is replaced by the variable with the largest predicted decrease.
    #[inline(never)]
    fn second_order_pair(&mut self, i0: usize, j1: usize) -> (usize, usize) {
        let si = self.cache.ensure(self.kernel, self.kidx[i0]);
        let sj = self.cache.ensure(self.kernel, self.kidx[j1]);
        let m = self.n_active;
        let (row_i, row_j) = (&self.cache.row(si)[..m], &self.cache.row(sj)[..m]);
        let (alpha, g, cdiag) = (&self.alpha[..m], &self.g[..m], &self.cdiag[..m]);
        let (lo, hi, sign) = (&self.lo[..m], &self.hi[..m], &self.sign[..m]);
        let (gi0, gj1) = (g[i0], g[j1]);
        let (ci, cj) = (cdiag[i0], cdiag[j1]);
        let room_i = alpha[i0] - lo[i0];
        let room_j = hi[j1] - alpha[j1];
        let two_inv_lambda = 2.0 * self.inv_lambda;
        let unbounded = self.opt.working_set == WorkingSetSelection::MaximalUnboundedDecrease;
        let constrained = self.constrained;
        let half_slack = if constrained {
            0.5 * (self.max_asum - self.asum).max(0.0)
        } else {
            f64::INFINITY
        };
        let block_up = self.at_asum_bound();
        let two_sided = self.opt.two_sided;
        // predicted decrease for gradient difference `d`, curvature `eta` and maximal step `tmax`
        let decrease = |d: f64, eta: f64, tmax: f64| -> f64 {
            if unbounded {
                0.5 * d * d / eta
            } else {
                let t = (d / eta).min(tmax);
                t * (d - 0.5 * eta * t)
            }
        };
        let (room_down, room_up) = (&self.room_down[..m], &self.room_up[..m]);
        let cand = Candidates {
            gi0,
            gj1,
            ci,
            cj,
            two_inv_lambda,
            room_i,
            room_j,
            unbounded,
            two_sided,
        };
        let mut best = (f64::NEG_INFINITY, i0, j1);
        for start in (0..m).step_by(BLOCK) {
            let end = (start + BLOCK).min(m);
            let block = Block {
                g: &g[start..end],
                cdiag: &cdiag[start..end],
                row_i: &row_i[start..end],
                row_j: &row_j[start..end],
                room_down: &room_down[start..end],
                room_up: &room_up[start..end],
            };
            if !cand.may_improve(best.0, &block) {
                continue;
            }
            for pos in start..end {
                let a = alpha[pos];
                let gv = g[pos];
                // partner j = pos for fixed i0
                if a < hi[pos] && gi0 > gv {
                    let cross = constrained && sign[pos] > sign[i0];
                    if !(block_up && cross) {
                        let eta = (ci + cdiag[pos] - two_inv_lambda * row_i[pos]).max(TAU);
                        let mut tmax = room_i.min(hi[pos] - a);
                        if cross {
                            tmax = tmax.min(half_slack);
                        }
                        let d = gi0 - gv;
                        // `d²/(2 eta)` bounds the decrease from above: skip the division
                        if d * d > 2.0 * eta * best.0 {
                            let dec = decrease(d, eta, tmax);
                            if dec > best.0 {
                                best = (dec, i0, pos);
                            }
                        }
                    }
                }
                // partner i = pos for fixed j1
                if two_sided && a > lo[pos] && gv > gj1 {
                    let cross = constrained && sign[j1] > sign[pos];
                    if !(block_up && cross) {
                        let eta = (cdiag[pos] + cj - two_inv_lambda * row_j[pos]).max(TAU);
                        let mut tmax = (a - lo[pos]).min(room_j);
                        if cross {
                            tmax = tmax.min(half_slack);
                        }
                        let d = gv - gj1;
                        if d * d > 2.0 * eta * best.0 {
                            let dec = decrease(d, eta, tmax);
                            if dec > best.0 {
                                best = (dec, pos, j1);
                            }
                        }
                    }
                }
            }
        }
        (best.1, best.2)
    }

    /// Exact minimisation along `e_j - e_i` for active positions `i`, `j`.
    #[inline(never)]
    fn update(&mut self, i: usize, j: usize) {
        let si = self.cache.ensure(self.kernel, self.kidx[i]);
        let sj = self.cache.ensure(self.kernel, self.kidx[j]);
        let kij = self.cache.row(si)[j];
        let eta = (self.cdiag[i] + self.cdiag[j] - 2.0 * kij * self.inv_lambda).max(TAU);
        let grad_diff = self.g[i] - self.g[j];
        let box_i = self.alpha[i] - self.lo[i];
        let box_j = self.hi[j] - self.alpha[j];
        let half_slack = if self.constrained && self.sign[j] > self.sign[i] {
            0.5 * (self.max_asum - self.asum).max(0.0)
        } else {
            f64::INFINITY
        };
        let t = (grad_diff / eta).min(box_i).min(box_j).min(half_slack);
        if t <= 0.0 {
            return;
        }
        let (old_i, old_j) = (self.alpha[i], self.alpha[j]);
        self.alpha[i] = if t == box_i {
            self.lo[i]
        } else {
            self.alpha[i] - t
        };
        self.alpha[j] = if t == box_j {
            self.hi[j]
        } else {
            self.alpha[j] + t
        };
        if self.constrained {
            if t == half_slack {
                self.asum = self.max_asum;
            } else {
                self.asum += t * (self.sign[j] - self.sign[i]);
            }
        }
        self.value -= t * (grad_diff - 0.5 * eta * t);
        // the quadratic term only changes the gradient of the two moved variables
        self.g[i] += self.q[i] * (self.alpha[i] - old_i);
        self.g[j] += self.q[j] * (self.alpha[j] - old_j);
        // fused pass: kernel part of the gradient and its extremal values
        self.set_rooms(i);
        self.set_rooms(j);
        let tl = t * self.inv_lambda;
        let m = self.n_active;
        let (ri, rj) = (&self.cache.row(si)[..m], &self.cache.row(sj)[..m]);
        let (alpha, lo, hi, sign) = (
            &self.alpha[..m],
            &self.lo[..m],
            &self.hi[..m],
            &self.sign[..m],
        );
        let mut ex = Extremes::new();
        if self.constrained {
            for ((g, &kj), &ki) in self.g[..m].iter_mut().zip(rj).zip(ri) {
                *g += tl * (kj - ki);
            }
            for pos in 0..m {
                let grp = usize::from(sign[pos] < 0.0);
                ex.visit_lite(pos, grp, alpha[pos], lo[pos], hi[pos], self.g[pos]);
            }
        } else {
            let (rd, ru) = (&self.room_down[..m], &self.room_up[..m]);
            for start in (0..m).step_by(BLOCK) {
                let end = (start + BLOCK).min(m);
                let g = &mut self.g[start..end];
                simd::add_scaled_diff(g, &ri[start..end], &rj[start..end], tl);
                ex.visit_block(start, g, &rd[start..end], &ru[start..end]);
            }
        }
        self.pending = Some(ex);
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

    /// Moves variables detected as fixed behind the active prefix (stable partition).
    fn shrink(&mut self, ex: &Extremes) {
        let (b, c) = self.bias(ex);
        let threshold =
            self.opt.shrinking_factor * self.violation.max(0.0).powf(self.opt.shrinking_exponent);
        let m = self.n_active;
        let keep: Vec<bool> = (0..m)
            .map(|pos| {
                let shifted = self.g[pos] + b + self.sign[pos] * c;
                let a = self.alpha[pos];
                let fixed_lo = a <= self.lo[pos] && shifted > threshold;
                let fixed_hi = a >= self.hi[pos] && shifted < -threshold;
                !(fixed_lo || fixed_hi)
            })
            .collect();
        let n_keep = keep.iter().filter(|&&k| k).count();
        if n_keep == m {
            return;
        }
        let order: Vec<usize> = (0..m)
            .filter(|&pos| keep[pos])
            .chain((0..m).filter(|&pos| !keep[pos]))
            .chain(m..self.perm.len())
            .collect();
        let mut fs = Vec::new();
        for v in [
            &mut self.lo,
            &mut self.hi,
            &mut self.q,
            &mut self.p,
            &mut self.sign,
            &mut self.cdiag,
            &mut self.alpha,
            &mut self.room_down,
            &mut self.room_up,
            &mut self.g,
        ] {
            permute(v, &order, &mut fs);
        }
        let mut us = Vec::new();
        permute(&mut self.perm, &order, &mut us);
        permute(&mut self.kidx, &order, &mut us);
        self.n_active = n_keep;
        self.cache.restrict(&keep);
    }

    /// Computes `ka` for all variables from scratch.
    fn compute_ka(&self) -> Vec<f64> {
        let mut beta = vec![0.0; self.kernel.len()];
        for (&k, &a) in self.kidx.iter().zip(&self.alpha) {
            beta[k] += a;
        }
        let support: Vec<(usize, f64)> = beta
            .iter()
            .enumerate()
            .filter(|(_, &b)| b != 0.0)
            .map(|(k, &b)| (k, b * self.inv_lambda))
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
        self.kidx.iter().map(|&k| kam[k]).collect()
    }

    fn unshrink(&mut self) {
        self.n_active = self.perm.len();
        self.cache.reset(self.kidx.clone());
        let ka = self.compute_ka();
        self.set_gradient(&ka);
    }

    /// `v` in the original variable order.
    fn unpermuted(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; v.len()];
        for (&o, &x) in self.perm.iter().zip(v) {
            out[o] = x;
        }
        out
    }

    /// `r◇(α, ρ) / ρ` on the active set (see [`StoppingCriterion::BoundAware`]).
    fn bound_aware_measure(&self) -> f64 {
        let rho = if self.opt.rho > 0.0 {
            self.opt.rho
        } else {
            self.prob.lambda
        };
        let m = self.n_active;
        let r = measures::bound_aware_violation_slices(
            &self.alpha[..m],
            &self.lo[..m],
            &self.hi[..m],
            &self.g[..m],
            rho,
        );
        r / rho
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
                let alpha = self.unpermuted(&self.alpha);
                let progress = Progress {
                    step: steps,
                    elapsed,
                    violation: self.violation,
                    dual_objective: self.value,
                    n_active: self.n_active,
                    alpha: &alpha,
                };
                if callback(&progress) {
                    status = Status::Interrupted;
                    break;
                }
            }

            let ex = match self.pending.take() {
                Some(ex) => ex,
                None => self.scan(),
            };
            let (violation, i0, j1) = self.maximal_violating_pair(&ex);
            self.violation = violation.max(0.0);
            let converged = violation <= opt.tol
                || (opt.stopping == StoppingCriterion::BoundAware
                    && steps > 0
                    && steps % shrinking_period == 0
                    && self.bound_aware_measure() <= opt.tol);
            if converged {
                if self.n_active < n {
                    self.unshrink();
                    continue;
                }
                status = Status::Converged;
                break;
            }
            if opt.shrinking && steps > 0 && steps % shrinking_period == 0 {
                let before = self.n_active;
                let full = self.scan();
                self.shrink(&full);
                if self.n_active < before {
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
        let n_active = self.n_active;
        if n_active < self.prob.len() {
            self.unshrink();
        }
        let ex = self.scan();
        let (violation, _, _) = self.maximal_violating_pair(&ex);
        let (b, c) = self.bias(&ex);
        let alpha = self.unpermuted(&self.alpha);
        let ka_perm: Vec<f64> = (0..self.g.len())
            .map(|pos| self.g[pos] - self.q[pos] * self.alpha[pos] - self.p[pos])
            .collect();
        let ka = self.unpermuted(&ka_perm);
        let p = self.prob;
        let dual = p.dual_objective(&alpha, &ka);
        let primal = p.primal_objective(&alpha, &ka, b, c);
        SmoResult {
            alpha,
            b,
            c,
            ka,
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
