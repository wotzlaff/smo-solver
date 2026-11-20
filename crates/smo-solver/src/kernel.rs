//! Kernel functions and kernel matrices.
//!
//! The solver only needs access to single entries and rows of the kernel matrix,
//! which is expressed by the [`Kernel`] trait. Two implementations are provided:
//!
//! * [`FeatureKernel`] evaluates a [`KernelFunction`] on a row-major feature matrix
//!   that is *borrowed* (e.g. directly from a NumPy array, without copying),
//! * [`PrecomputedKernel`] wraps an explicitly given square matrix.

use crate::simd;

/// Access to a (symmetric, positive semi-definite) kernel matrix of size `len() × len()`.
pub trait Kernel: Sync {
    /// Number of samples, i.e. the size of the kernel matrix.
    fn len(&self) -> usize;

    /// Returns `true` if the kernel matrix is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Kernel matrix entry `K[i, j]`.
    fn eval(&self, i: usize, j: usize) -> f64;

    /// Diagonal entry `K[i, i]`.
    fn diag(&self, i: usize) -> f64 {
        self.eval(i, i)
    }

    /// Writes `K[i, cols[p]]` into `out[p]` for all `p`.
    fn row_into(&self, i: usize, cols: &[usize], out: &mut [f64]) {
        generic_row_into(self, i, cols, out);
    }
}

/// Entry-wise implementation of [`Kernel::row_into`].
fn generic_row_into<K: Kernel + ?Sized>(kernel: &K, i: usize, cols: &[usize], out: &mut [f64]) {
    debug_assert_eq!(cols.len(), out.len());
    #[cfg(feature = "parallel")]
    if cols.len() >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(cols.par_iter())
            .for_each(|(o, &j)| *o = kernel.eval(i, j));
        return;
    }
    for (o, &j) in out.iter_mut().zip(cols) {
        *o = kernel.eval(i, j);
    }
}

/// Rows longer than this are computed in parallel (feature `parallel`).
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 4096;

/// Chunk size of the parallel row computation of [`FeatureKernel`].
#[cfg(feature = "parallel")]
const PARALLEL_CHUNK: usize = 512;

/// A kernel function `k(x, z)` acting on feature vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelFunction {
    /// `k(x, z) = <x, z>`
    Linear,
    /// Gaussian kernel `k(x, z) = exp(-gamma * |x - z|^2)`
    Rbf { gamma: f64 },
    /// `k(x, z) = (gamma * <x, z> + coef0)^degree`
    Polynomial { gamma: f64, coef0: f64, degree: i32 },
}

impl KernelFunction {
    /// Evaluates the kernel function for two feature vectors of equal length.
    #[inline]
    pub fn eval(&self, x: &[f64], z: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), z.len());
        match *self {
            KernelFunction::Linear => dot(x, z),
            KernelFunction::Rbf { gamma } => {
                let d2: f64 = x.iter().zip(z).map(|(a, b)| (a - b) * (a - b)).sum();
                (-gamma * d2).exp()
            }
            KernelFunction::Polynomial {
                gamma,
                coef0,
                degree,
            } => (gamma * dot(x, z) + coef0).powi(degree),
        }
    }
}

#[inline]
fn dot(x: &[f64], z: &[f64]) -> f64 {
    x.iter().zip(z).map(|(a, b)| a * b).sum()
}

/// Kernel matrix defined by a kernel function and a borrowed row-major feature matrix.
#[derive(Clone, Copy, Debug)]
pub struct FeatureKernel<'a> {
    data: &'a [f64],
    n: usize,
    dim: usize,
    function: KernelFunction,
}

impl<'a> FeatureKernel<'a> {
    /// Creates a kernel for `n = data.len() / dim` samples with `dim` features each.
    ///
    /// # Panics
    /// If `dim == 0` or `data.len()` is not a multiple of `dim`.
    pub fn new(data: &'a [f64], dim: usize, function: KernelFunction) -> Self {
        assert!(dim > 0, "feature dimension must be positive");
        assert_eq!(data.len() % dim, 0, "data length must be a multiple of dim");
        Self {
            data,
            n: data.len() / dim,
            dim,
            function,
        }
    }

    /// Feature vector of sample `i`.
    #[inline]
    pub fn sample(&self, i: usize) -> &'a [f64] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Number of features per sample.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The underlying kernel function.
    pub fn function(&self) -> KernelFunction {
        self.function
    }
}

impl Kernel for FeatureKernel<'_> {
    fn len(&self) -> usize {
        self.n
    }

    #[inline]
    fn eval(&self, i: usize, j: usize) -> f64 {
        self.function.eval(self.sample(i), self.sample(j))
    }

    fn diag(&self, i: usize) -> f64 {
        match self.function {
            KernelFunction::Rbf { .. } => 1.0,
            _ => self.eval(i, i),
        }
    }

    fn row_into(&self, i: usize, cols: &[usize], out: &mut [f64]) {
        let KernelFunction::Rbf { gamma } = self.function else {
            return generic_row_into(self, i, cols, out);
        };
        assert_eq!(cols.len(), out.len());
        let x = self.sample(i);
        let block = |cols: &[usize], out: &mut [f64]| {
            simd::sq_dist_to_samples(self.data, self.dim, x, cols, out);
            for o in out.iter_mut() {
                *o = (-gamma * *o).exp();
            }
        };
        #[cfg(feature = "parallel")]
        if cols.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            out.par_chunks_mut(PARALLEL_CHUNK)
                .zip(cols.par_chunks(PARALLEL_CHUNK))
                .for_each(|(o, c)| block(c, o));
            return;
        }
        block(cols, out);
    }
}

/// Explicitly given (row-major, square) kernel matrix.
#[derive(Clone, Copy, Debug)]
pub struct PrecomputedKernel<'a> {
    data: &'a [f64],
    n: usize,
}

impl<'a> PrecomputedKernel<'a> {
    /// # Panics
    /// If `data.len() != n * n`.
    pub fn new(data: &'a [f64], n: usize) -> Self {
        assert_eq!(data.len(), n * n, "precomputed kernel must be n x n");
        Self { data, n }
    }
}

impl Kernel for PrecomputedKernel<'_> {
    fn len(&self) -> usize {
        self.n
    }

    #[inline]
    fn eval(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    fn row_into(&self, i: usize, cols: &[usize], out: &mut [f64]) {
        let row = &self.data[i * self.n..(i + 1) * self.n];
        for (o, &j) in out.iter_mut().zip(cols) {
            *o = row[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_is_one_on_diagonal_and_symmetric() {
        let x = [0.0, 1.0, 2.0, -1.0, 0.5, 0.5];
        let k = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 0.7 });
        assert_eq!(k.len(), 3);
        for i in 0..3 {
            assert!((k.eval(i, i) - 1.0).abs() < 1e-15);
            assert_eq!(k.diag(i), 1.0);
            for j in 0..3 {
                assert_eq!(k.eval(i, j), k.eval(j, i));
            }
        }
        let expected = (-0.7f64 * (4.0 + 4.0)).exp();
        assert!((k.eval(0, 1) - expected).abs() < 1e-15);
    }

    #[test]
    fn polynomial_and_linear() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let lin = FeatureKernel::new(&x, 2, KernelFunction::Linear);
        assert_eq!(lin.eval(0, 1), 11.0);
        let poly = FeatureKernel::new(
            &x,
            2,
            KernelFunction::Polynomial {
                gamma: 0.5,
                coef0: 1.0,
                degree: 2,
            },
        );
        assert_eq!(poly.eval(0, 1), 6.5 * 6.5);
    }

    #[test]
    fn row_into_matches_eval() {
        let x: Vec<f64> = (0..20).map(|v| (v as f64 * 0.37).sin()).collect();
        let k = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.3 });
        let cols = [9, 0, 4, 4];
        let mut out = [0.0; 4];
        k.row_into(3, &cols, &mut out);
        for (o, &j) in out.iter().zip(&cols) {
            assert_eq!(*o, k.eval(3, j));
        }
        let km: Vec<f64> = (0..100).map(|v| v as f64).collect();
        let pk = PrecomputedKernel::new(&km, 10);
        pk.row_into(2, &cols, &mut out);
        assert_eq!(out, [29.0, 20.0, 24.0, 24.0]);
    }

    #[test]
    fn rbf_row_into_matches_eval_bitwise() {
        let dim = 10;
        let n = 9000;
        let x: Vec<f64> = (0..n * dim)
            .map(|v| (v as f64 * 0.37).sin() * 2.0)
            .collect();
        let k = FeatureKernel::new(&x, dim, KernelFunction::Rbf { gamma: 0.1 });
        // short (sequential) and long (parallel chunks) rows with repeated columns
        for len in [0, 5, 100, 5000, 9000] {
            let cols: Vec<usize> = (0..len).map(|p| (p * 7919 + 3) % n).collect();
            let mut out = vec![0.0; len];
            k.row_into(11, &cols, &mut out);
            for (p, &j) in cols.iter().enumerate() {
                assert_eq!(out[p].to_bits(), k.eval(11, j).to_bits());
            }
        }
    }
}
