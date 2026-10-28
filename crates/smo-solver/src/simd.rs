//! Block kernels of the SMO loops over the active set.
//!
//! On x86_64 with AVX2 (detected at run time) the kernels use four-lane intrinsics, otherwise a
//! scalar implementation. Both evaluate the same IEEE operations in the same order (no fused
//! multiply-add), so the results are bit-identical. The scalar code is written such that the
//! compiler can use packed instructions where it is able to.

use crate::solver::TAU;

#[inline]
fn use_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `g[k] += tl * (rj[k] - ri[k])`.
pub(crate) fn add_scaled_diff(g: &mut [f64], ri: &[f64], rj: &[f64], tl: f64) {
    assert!(ri.len() >= g.len() && rj.len() >= g.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        // SAFETY: AVX2 is available and the slice lengths were checked above.
        return unsafe { avx2::add_scaled_diff(g, ri, rj, tl) };
    }
    for ((g, &kj), &ki) in g.iter_mut().zip(rj).zip(ri) {
        *g += tl * (kj - ki);
    }
}

/// Largest `g[k]` with `room_down[k] > 0` and smallest `g[k]` with `room_up[k] > 0`
/// (`-∞`/`+∞` if there is none).
pub(crate) fn extremal_values(g: &[f64], room_down: &[f64], room_up: &[f64]) -> (f64, f64) {
    assert!(room_down.len() >= g.len() && room_up.len() >= g.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        // SAFETY: AVX2 is available and the slice lengths were checked above.
        return unsafe { avx2::extremal_values(g, room_down, room_up) };
    }
    let (mut bmax, mut bmin) = (f64::NEG_INFINITY, f64::INFINITY);
    for ((&gv, &rd), &ru) in g.iter().zip(room_down).zip(room_up) {
        let down = if rd > 0.0 { gv } else { f64::NEG_INFINITY };
        let up = if ru > 0.0 { gv } else { f64::INFINITY };
        bmax = if down > bmax { down } else { bmax };
        bmin = if up < bmin { up } else { bmin };
    }
    (bmax, bmin)
}

/// Data of the second-order working set search for the partners of `i0` (as `j`) and `j1`
/// (as `i`); see `Smo::second_order_pair`.
pub(crate) struct Candidates {
    pub gi0: f64,
    pub gj1: f64,
    pub ci: f64,
    pub cj: f64,
    pub two_inv_lambda: f64,
    pub room_i: f64,
    pub room_j: f64,
    pub unbounded: bool,
    pub two_sided: bool,
}

/// Per-variable slices of a block for [`Candidates::may_improve`].
pub(crate) struct Block<'a> {
    pub g: &'a [f64],
    pub cdiag: &'a [f64],
    pub row_i: &'a [f64],
    pub row_j: &'a [f64],
    pub room_down: &'a [f64],
    pub room_up: &'a [f64],
}

impl Candidates {
    /// Division-free test whether some candidate of the block could have a predicted decrease
    /// larger than `best`: the decrease is at most `d²/(2 eta)` and (for bounded steps) at most
    /// `d tmax`.
    pub(crate) fn may_improve(&self, best: f64, b: &Block) -> bool {
        let n = b.g.len();
        assert!([b.cdiag, b.row_i, b.row_j, b.room_down, b.room_up]
            .iter()
            .all(|s| s.len() >= n));
        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            // SAFETY: AVX2 is available and the slice lengths were checked above.
            return unsafe { avx2::may_improve(self, best, b) };
        }
        let mut any = false;
        for k in 0..n {
            any |= self.test_up(best, b, k) | (self.two_sided & self.test_down(best, b, k));
        }
        any
    }

    #[inline(always)]
    fn test_up(&self, best: f64, b: &Block, k: usize) -> bool {
        let ru = b.room_up[k];
        let d = self.gi0 - b.g[k];
        let eta = self.ci + b.cdiag[k] - self.two_inv_lambda * b.row_i[k];
        let eta = if eta > TAU { eta } else { TAU };
        let tmax = if self.room_i < ru { self.room_i } else { ru };
        (ru > 0.0) & (d > 0.0) & (d * d > 2.0 * eta * best) & (self.unbounded | (d * tmax > best))
    }

    #[inline(always)]
    fn test_down(&self, best: f64, b: &Block, k: usize) -> bool {
        let rd = b.room_down[k];
        let d = b.g[k] - self.gj1;
        let eta = b.cdiag[k] + self.cj - self.two_inv_lambda * b.row_j[k];
        let eta = if eta > TAU { eta } else { TAU };
        let tmax = if rd < self.room_j { rd } else { self.room_j };
        (rd > 0.0) & (d > 0.0) & (d * d > 2.0 * eta * best) & (self.unbounded | (d * tmax > best))
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Block, Candidates};
    use crate::solver::TAU;
    use std::arch::x86_64::*;

    const W: usize = 4;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_scaled_diff(g: &mut [f64], ri: &[f64], rj: &[f64], tl: f64) {
        let n = g.len();
        let t = _mm256_set1_pd(tl);
        let mut k = 0;
        while k + W <= n {
            let gv = _mm256_loadu_pd(g.as_ptr().add(k));
            let d = _mm256_sub_pd(
                _mm256_loadu_pd(rj.as_ptr().add(k)),
                _mm256_loadu_pd(ri.as_ptr().add(k)),
            );
            _mm256_storeu_pd(
                g.as_mut_ptr().add(k),
                _mm256_add_pd(gv, _mm256_mul_pd(t, d)),
            );
            k += W;
        }
        while k < n {
            g[k] += tl * (rj[k] - ri[k]);
            k += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn extremal_values(g: &[f64], rd: &[f64], ru: &[f64]) -> (f64, f64) {
        let n = g.len();
        let zero = _mm256_setzero_pd();
        let (ninf, pinf) = (
            _mm256_set1_pd(f64::NEG_INFINITY),
            _mm256_set1_pd(f64::INFINITY),
        );
        let (mut vmax, mut vmin) = (ninf, pinf);
        let mut k = 0;
        while k + W <= n {
            let gv = _mm256_loadu_pd(g.as_ptr().add(k));
            let can_down = _mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_loadu_pd(rd.as_ptr().add(k)), zero);
            let can_up = _mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_loadu_pd(ru.as_ptr().add(k)), zero);
            vmax = _mm256_max_pd(_mm256_blendv_pd(ninf, gv, can_down), vmax);
            vmin = _mm256_min_pd(_mm256_blendv_pd(pinf, gv, can_up), vmin);
            k += W;
        }
        let (mut a, mut b) = ([0.0; W], [0.0; W]);
        _mm256_storeu_pd(a.as_mut_ptr(), vmax);
        _mm256_storeu_pd(b.as_mut_ptr(), vmin);
        let (mut bmax, mut bmin) = (f64::NEG_INFINITY, f64::INFINITY);
        for l in 0..W {
            bmax = if a[l] > bmax { a[l] } else { bmax };
            bmin = if b[l] < bmin { b[l] } else { bmin };
        }
        while k < n {
            let down = if rd[k] > 0.0 { g[k] } else { f64::NEG_INFINITY };
            let up = if ru[k] > 0.0 { g[k] } else { f64::INFINITY };
            bmax = if down > bmax { down } else { bmax };
            bmin = if up < bmin { up } else { bmin };
            k += 1;
        }
        (bmax, bmin)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn may_improve(c: &Candidates, best: f64, b: &Block) -> bool {
        let n = b.g.len();
        let zero = _mm256_setzero_pd();
        let two = _mm256_set1_pd(2.0);
        let tau = _mm256_set1_pd(TAU);
        let best_v = _mm256_set1_pd(best);
        let tl2 = _mm256_set1_pd(c.two_inv_lambda);
        let unbounded = if c.unbounded {
            _mm256_castsi256_pd(_mm256_set1_epi64x(-1))
        } else {
            zero
        };
        let (gi0, gj1) = (_mm256_set1_pd(c.gi0), _mm256_set1_pd(c.gj1));
        let (ci, cj) = (_mm256_set1_pd(c.ci), _mm256_set1_pd(c.cj));
        let (room_i, room_j) = (_mm256_set1_pd(c.room_i), _mm256_set1_pd(c.room_j));
        // `(room > 0) & (d > 0) & (d² > 2 eta best) & (unbounded | (d tmax > best))`
        let test = |room: __m256d, d: __m256d, eta: __m256d, tmax: __m256d| -> __m256d {
            let eta = _mm256_max_pd(eta, tau);
            let ok = _mm256_and_pd(
                _mm256_cmp_pd::<_CMP_GT_OQ>(room, zero),
                _mm256_cmp_pd::<_CMP_GT_OQ>(d, zero),
            );
            let dd = _mm256_mul_pd(d, d);
            let bound = _mm256_mul_pd(_mm256_mul_pd(two, eta), best_v);
            let lim = _mm256_or_pd(
                unbounded,
                _mm256_cmp_pd::<_CMP_GT_OQ>(_mm256_mul_pd(d, tmax), best_v),
            );
            _mm256_and_pd(
                _mm256_and_pd(ok, _mm256_cmp_pd::<_CMP_GT_OQ>(dd, bound)),
                lim,
            )
        };
        let mut acc = zero;
        let mut k = 0;
        while k + W <= n {
            let gv = _mm256_loadu_pd(b.g.as_ptr().add(k));
            let cd = _mm256_loadu_pd(b.cdiag.as_ptr().add(k));
            let ru = _mm256_loadu_pd(b.room_up.as_ptr().add(k));
            let ri = _mm256_loadu_pd(b.row_i.as_ptr().add(k));
            let eta = _mm256_sub_pd(_mm256_add_pd(ci, cd), _mm256_mul_pd(tl2, ri));
            acc = _mm256_or_pd(
                acc,
                test(ru, _mm256_sub_pd(gi0, gv), eta, _mm256_min_pd(room_i, ru)),
            );
            if c.two_sided {
                let rd = _mm256_loadu_pd(b.room_down.as_ptr().add(k));
                let rj = _mm256_loadu_pd(b.row_j.as_ptr().add(k));
                let eta = _mm256_sub_pd(_mm256_add_pd(cd, cj), _mm256_mul_pd(tl2, rj));
                acc = _mm256_or_pd(
                    acc,
                    test(rd, _mm256_sub_pd(gv, gj1), eta, _mm256_min_pd(rd, room_j)),
                );
            }
            k += W;
        }
        let mut any = _mm256_movemask_pd(acc) != 0;
        while k < n {
            any |= c.test_up(best, b, k) | (c.two_sided & c.test_down(best, b, k));
            k += 1;
        }
        any
    }
}
