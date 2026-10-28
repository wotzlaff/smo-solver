//! Decision functions built from a dual solution.

use crate::kernel::KernelFunction;
use crate::problem::DualProblem;
use crate::solver::SmoResult;

/// Kernel expansion `f(x) = Σ_k coef_k K(x, sv_k) + b` over the support vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub function: KernelFunction,
    pub dim: usize,
    /// Row-major support vectors.
    pub support_vectors: Vec<f64>,
    /// Indices of the support vectors in the training data.
    pub support: Vec<usize>,
    /// Coefficients `β_k / λ`.
    pub coef: Vec<f64>,
    pub b: f64,
}

impl Model {
    /// Builds the model from a solution of `problem` trained on row-major `data` with `dim` features.
    pub fn new(
        function: KernelFunction,
        data: &[f64],
        dim: usize,
        problem: &DualProblem,
        result: &SmoResult,
    ) -> Self {
        let beta = problem.sample_coefficients(&result.alpha);
        let mut model = Self {
            function,
            dim,
            support_vectors: Vec::new(),
            support: Vec::new(),
            coef: Vec::new(),
            b: result.b,
        };
        for (k, &bk) in beta.iter().enumerate() {
            if bk != 0.0 {
                model.support.push(k);
                model.coef.push(bk / problem.lambda);
                model
                    .support_vectors
                    .extend_from_slice(&data[k * dim..(k + 1) * dim]);
            }
        }
        model
    }

    /// Number of support vectors.
    pub fn n_support(&self) -> usize {
        self.coef.len()
    }

    /// Decision value for a single sample.
    pub fn decision(&self, x: &[f64]) -> f64 {
        self.coef
            .iter()
            .zip(self.support_vectors.chunks_exact(self.dim))
            .map(|(c, sv)| c * self.function.eval(x, sv))
            .sum::<f64>()
            + self.b
    }

    /// Decision values for row-major samples.
    pub fn decision_batch(&self, data: &[f64]) -> Vec<f64> {
        data.chunks_exact(self.dim)
            .map(|x| self.decision(x))
            .collect()
    }
}
