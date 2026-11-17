//! Bounded nonlinear optimization for browser experiments.

#[cfg(any(test, target_arch = "wasm32"))]
mod optimizer {
    use std::cell::RefCell;

    use basin::{
        BoxBounds, ConstraintDescription, ConstraintSet, ConstraintSolveError, Constraints,
        CostFunction, Executor, FirstOrderState, Gradient, Lbfgsb, Termination,
    };

    #[derive(Debug, PartialEq, Eq)]
    pub enum Error {
        InvalidInput,
        Objective,
        Solver,
    }

    #[derive(Debug)]
    pub struct Solution {
        pub x: Vec<f64>,
        pub cost: f64,
        pub iterations: u64,
        pub evaluations: usize,
        pub status: i32,
    }

    struct Evaluation {
        point: Vec<f64>,
        cost: f64,
        gradient: Vec<f64>,
    }

    struct Evaluator<'a, F> {
        evaluate: &'a mut F,
        cached: Option<Evaluation>,
    }

    impl<F> Evaluator<'_, F>
    where
        F: FnMut(&[f64], &mut [f64]) -> f64,
    {
        fn evaluate(&mut self, point: &[f64]) -> Result<&Evaluation, ()> {
            if !point.iter().all(|value| value.is_finite()) {
                return Err(());
            }
            // Separate Basin cost and gradient calls can share one host evaluation.
            if self
                .cached
                .as_ref()
                .is_none_or(|entry| entry.point != point)
            {
                let mut gradient = vec![f64::NAN; point.len()];
                let cost = (self.evaluate)(point, &mut gradient);
                if !cost.is_finite() || !gradient.iter().all(|value| value.is_finite()) {
                    return Err(());
                }
                self.cached = Some(Evaluation {
                    point: point.to_vec(),
                    cost,
                    gradient,
                });
            }
            Ok(self.cached.as_ref().unwrap())
        }
    }

    struct Objective<'a, F> {
        lower: Vec<f64>,
        upper: Vec<f64>,
        evaluator: RefCell<Evaluator<'a, F>>,
    }

    impl<F> Constraints for Objective<'_, F> {
        type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

        fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
            ConstraintSet::new().with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
        }
    }

    impl<F: FnMut(&[f64], &mut [f64]) -> f64> CostFunction for Objective<'_, F> {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = ();

        fn cost(&self, point: &Vec<f64>) -> Result<f64, ()> {
            Ok(self.evaluator.borrow_mut().evaluate(point)?.cost)
        }
    }

    impl<F: FnMut(&[f64], &mut [f64]) -> f64> Gradient for Objective<'_, F> {
        type Gradient = Vec<f64>;

        fn gradient(&self, point: &Vec<f64>) -> Result<Vec<f64>, ()> {
            Ok(self
                .evaluator
                .borrow_mut()
                .evaluate(point)?
                .gradient
                .clone())
        }

        fn cost_and_gradient(&self, point: &Vec<f64>) -> Result<(f64, Vec<f64>), ()> {
            let mut evaluator = self.evaluator.borrow_mut();
            let result = evaluator.evaluate(point)?;
            Ok((result.cost, result.gradient.clone()))
        }
    }

    fn within_bounds(point: &[f64], lower: &[f64], upper: &[f64]) -> bool {
        point.len() == lower.len()
            && point.len() == upper.len()
            && point.iter().zip(lower).zip(upper).all(|((&x, &lo), &hi)| {
                lo.is_finite() && hi.is_finite() && x.is_finite() && lo <= x && x <= hi
            })
    }

    pub fn minimize(
        start: &[f64],
        lower: &[f64],
        upper: &[f64],
        max_iter: u32,
        gtol: f64,
        mut evaluate: impl FnMut(&[f64], &mut [f64]) -> f64,
    ) -> Result<Solution, Error> {
        if start.is_empty()
            || !within_bounds(start, lower, upper)
            || max_iter == 0
            || !gtol.is_finite()
            || gtol <= 0.0
        {
            return Err(Error::InvalidInput);
        }
        let mut evaluations = 0;
        let mut counted = |point: &[f64], gradient: &mut [f64]| {
            evaluations += 1;
            evaluate(point, gradient)
        };
        let objective = Objective {
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            evaluator: RefCell::new(Evaluator {
                evaluate: &mut counted,
                cached: None,
            }),
        };
        let solver = Lbfgsb::new()
            .with_absolute_projected_gradient_tolerance(gtol)
            .with_relative_cost_reduction_tolerance(None);
        let result = Executor::new(objective, solver, FirstOrderState::new(start.to_vec()))
            .max_iter(u64::from(max_iter))
            .run()
            .map_err(|err| match err {
                ConstraintSolveError::Evaluation(()) => Error::Objective,
                _ => Error::Solver,
            })?;
        if !result.cost().is_finite() || !within_bounds(result.param(), lower, upper) {
            return Err(Error::Solver);
        }
        let status = match result.report.termination {
            Termination::Converged(_) => 0,
            Termination::Limit(_) => 1,
            _ => 2,
        };
        Ok(Solution {
            cost: result.cost(),
            x: result.param().clone(),
            iterations: result.iter(),
            evaluations,
            status,
        })
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::optimizer;
    use js_sys::{Error, Float64Array, Function};
    use wasm_bindgen::prelude::*;

    /// Minimize a box-constrained objective. The synchronous callback receives
    /// a copy of the candidate vector and returns a Float64Array containing
    /// the cost followed by the gradient. The returned Float64Array contains
    /// status, cost, iterations, callback evaluations and the final point.
    #[wasm_bindgen]
    pub fn minimize(
        start: &[f64],
        lower: &[f64],
        upper: &[f64],
        max_iter: u32,
        gtol: f64,
        evaluate: &Function,
    ) -> Result<Vec<f64>, JsValue> {
        let mut thrown = None;
        let solution =
            optimizer::minimize(start, lower, upper, max_iter, gtol, |point, gradient| {
                let evaluated =
                    match evaluate.call1(&JsValue::UNDEFINED, &Float64Array::from(point)) {
                        Ok(value) => value,
                        Err(error) => {
                            thrown = Some(error);
                            return f64::NAN;
                        }
                    };
                let Ok(values) = evaluated.dyn_into::<Float64Array>() else {
                    return f64::NAN;
                };
                if values.length() as usize != point.len() + 1 {
                    return f64::NAN;
                }
                values.subarray(1, values.length()).copy_to(gradient);
                values.get_index(0)
            });
        if let Some(error) = thrown {
            return Err(error);
        }
        let solution = solution.map_err(|error| {
            Error::new(match error {
                optimizer::Error::InvalidInput => "Invalid optimization parameters",
                optimizer::Error::Objective => {
                    "Expected a finite cost and gradient in a Float64Array"
                }
                optimizer::Error::Solver => {
                    "The optimizer could not produce a finite feasible result"
                }
            })
        })?;
        let mut output = Vec::with_capacity(solution.x.len() + 4);
        output.extend([
            f64::from(solution.status),
            solution.cost,
            solution.iterations as f64,
            solution.evaluations as f64,
        ]);
        output.extend(solution.x);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::optimizer::{Error, minimize};

    fn rosenbrock(x: &[f64], gradient: &mut [f64]) -> f64 {
        let d = x[1] - x[0] * x[0];
        gradient[0] = 2.0 * (x[0] - 1.0) - 400.0 * x[0] * d;
        gradient[1] = 200.0 * d;
        (x[0] - 1.0).powi(2) + 100.0 * d * d
    }

    #[test]
    fn solves_coupled_nonlinear_problem() {
        let mut host_calls = 0;
        let result = minimize(&[-1.2, 1.0], &[-2.0; 2], &[2.0; 2], 200, 1e-7, |x, g| {
            host_calls += 1;
            rosenbrock(x, g)
        })
        .unwrap();
        assert_eq!(result.status, 0, "{result:?}");
        assert!(result.x.iter().all(|x| (x - 1.0).abs() < 1e-6));
        assert!(result.cost < 1e-12);
        assert!(result.iterations > 0 && result.iterations < 200);
        assert_eq!(result.evaluations, host_calls);
    }

    #[test]
    fn respects_box_and_fixed_coordinates() {
        let result = minimize(&[0.0, 0.0], &[-1.0, 0.0], &[1.0, 0.0], 50, 1e-8, |x, g| {
            assert!((-1.0..=1.0).contains(&x[0]));
            assert_eq!(x[1], 0.0);
            g.copy_from_slice(&[2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)]);
            (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2)
        })
        .unwrap();
        assert_eq!(result.status, 0, "{result:?}");
        assert_eq!(result.x, vec![1.0, 0.0]);
        assert_eq!(result.cost, 10.0);
    }

    #[test]
    fn distinguishes_iteration_limit_and_invalid_callback() {
        let result = minimize(&[-1.2, 1.0], &[-2.0; 2], &[2.0; 2], 1, 1e-7, rosenbrock).unwrap();
        assert_eq!(result.status, 1, "{result:?}");
        assert_eq!(result.iterations, 1);

        let bad = minimize(&[0.0], &[-1.0], &[1.0], 50, 1e-8, |_, _| 0.0);
        assert_eq!(bad.unwrap_err(), Error::Objective);

        let bad = minimize(&[2.0], &[-1.0], &[1.0], 50, 1e-8, |_, _| unreachable!());
        assert_eq!(bad.unwrap_err(), Error::InvalidInput);
    }
}
