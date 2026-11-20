//! Recording of the SMO iteration for the browser animation.

use smo_solver::{
    solve_with, DualProblem, FeatureKernel, KernelFunction, Model, SmoOptions, SmoResult,
};

/// One snapshot: dual coefficients of all samples (`β_k / λ`) with the bias and progress measures.
pub struct Frame {
    pub step: f64,
    pub gap: f64,
    pub violation: f64,
    pub b: f64,
    pub coef: Vec<f64>,
}

/// The recorded run: the training data (for evaluating frames) and the frames.
pub struct Trace {
    pub x: Vec<f64>,
    pub dim: usize,
    pub function: KernelFunction,
    pub frames: Vec<Frame>,
}

impl Trace {
    /// Decision value of frame `frame` at `point`.
    pub fn decision(&self, frame: &Frame, point: &[f64]) -> f64 {
        frame
            .coef
            .iter()
            .zip(self.x.chunks_exact(self.dim))
            .filter(|(c, _)| **c != 0.0)
            .map(|(c, xj)| c * self.function.eval(point, xj))
            .sum::<f64>()
            + frame.b
    }
}

/// Dual coefficients per sample and duality gap of the dual variables `alpha` with bias `b`.
fn snapshot(
    problem: &DualProblem,
    x: &[f64],
    dim: usize,
    function: KernelFunction,
    alpha: &[f64],
    b: f64,
) -> (Vec<f64>, f64) {
    let coef: Vec<f64> = problem
        .sample_coefficients(alpha)
        .iter()
        .map(|v| v / problem.lambda)
        .collect();
    let ka_sample: Vec<f64> = x
        .chunks_exact(dim)
        .map(|xi| {
            coef.iter()
                .zip(x.chunks_exact(dim))
                .filter(|(c, _)| **c != 0.0)
                .map(|(c, xj)| c * function.eval(xi, xj))
                .sum()
        })
        .collect();
    let ka: Vec<f64> = problem.kernel_index.iter().map(|&k| ka_sample[k]).collect();
    let gap = problem.primal_objective(alpha, &ka, b, 0.0) + problem.dual_objective(alpha, &ka);
    (coef, gap)
}

/// Solves `problem` and records the initial state, one frame every `period` steps (at most
/// `max_frames - 1` of them) and the final state. Returns the trace and the solver result.
pub fn record(
    x: &[f64],
    dim: usize,
    problem: &DualProblem,
    function: KernelFunction,
    tol: f64,
    period: u64,
    max_frames: usize,
) -> Result<(Trace, SmoResult, Model), ()> {
    let kernel = FeatureKernel::new(x, dim, function);
    let options = SmoOptions {
        tol,
        max_steps: 2_000_000,
        // the bias estimate of a snapshot is only exact on the full problem
        shrinking: false,
        callback_period: period,
        ..SmoOptions::default()
    };
    let mut trace = Trace {
        x: x.to_vec(),
        dim,
        function,
        frames: Vec::new(),
    };
    let zeros = vec![0.0; problem.len()];
    let (coef, gap) = snapshot(problem, x, dim, function, &zeros, 0.0);
    trace.frames.push(Frame {
        step: 0.0,
        gap,
        violation: f64::NAN,
        b: 0.0,
        coef,
    });
    let result = solve_with(&kernel, problem, &options, None, |p| {
        if trace.frames.len() + 1 < max_frames {
            let (coef, gap) = snapshot(problem, x, dim, function, p.alpha, p.b);
            trace.frames.push(Frame {
                step: p.step as f64,
                gap,
                violation: p.violation,
                b: p.b,
                coef,
            });
        }
        false
    })
    .map_err(|_| ())?;
    let model = Model::new(function, x, dim, problem, &result);
    let (coef, _) = snapshot(problem, x, dim, function, &result.alpha, result.b);
    trace.frames.push(Frame {
        step: result.steps as f64,
        gap: result.gap,
        violation: result.violation,
        b: result.b,
        coef,
    });
    Ok((trace, result, model))
}
