//! # smo-solver
//!
//! Sequential Minimal Optimization (SMO) for a general class of support vector machine
//! dual problems, including smoothed loss functions, sample weights and an optional
//! 1-norm constraint (generalised ν-SVM). The algorithms follow
//!
//! > N. Strasdat, *Duality, Derivative-Based Training Methods and Hyperparameter
//! > Optimization for Support Vector Machines*, Dissertation, Technische Universität Dresden,
//! > 2023, <https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-874731>.
//!
//! ```
//! use smo_solver::{solve, DualProblem, FeatureKernel, KernelFunction, Model, SmoOptions, Status};
//!
//! // four points in the plane, two per class
//! let x = [0.0, 0.0, 0.2, 0.1, 1.0, 1.0, 0.9, 1.2];
//! let y = [-1.0, -1.0, 1.0, 1.0];
//! let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.0 });
//! let problem = DualProblem::classification(&y, 0.1, None, 1.0, 0.0, None).unwrap();
//! let result = solve(&kernel, &problem, &SmoOptions::default()).unwrap();
//! assert_eq!(result.status, Status::Converged);
//! let model = Model::new(KernelFunction::Rbf { gamma: 1.0 }, &x, 2, &problem, &result);
//! assert!(model.decision(&[1.0, 1.1]) > 0.0);
//! ```

mod cache;
mod clock;
pub mod kernel;
pub mod measures;
pub mod model;
pub mod problem;
mod simd;
pub mod solver;

pub use kernel::{FeatureKernel, Kernel, KernelFunction, PrecomputedKernel};
pub use model::Model;
pub use problem::{smooth_max, DualProblem, ProblemError};
pub use solver::{solve, solve_with, Progress, SmoOptions, SmoResult, Status, WorkingSetSelection};
