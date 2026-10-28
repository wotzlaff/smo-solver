//! Integration tests of the SMO solver against optimality conditions and exact solutions.

#![allow(clippy::needless_range_loop)] // index loops mirror the matrix notation

use proptest::prelude::*;
use svm_smo::measures::{dual_gradient, violation};
use svm_smo::*;

/// Deterministic pseudo-random numbers in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Two noisy, overlapping clusters in `dim` dimensions.
fn dataset(n: usize, dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut x = Vec::with_capacity(n * dim);
    let mut y = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let label = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mut s = 0.0;
        for _ in 0..dim {
            let v = rng.next() * 2.0 - 1.0 + 0.4 * label;
            s += (3.0 * v).sin();
            x.push(v);
        }
        y.push(label);
        t.push(s + 0.1 * (rng.next() - 0.5));
    }
    (x, y, t)
}

fn tight() -> SmoOptions {
    SmoOptions {
        tol: 1e-9,
        ..SmoOptions::default()
    }
}

fn assert_optimal(problem: &DualProblem, res: &SmoResult, gap_tol: f64) {
    assert_eq!(res.status, Status::Converged);
    let sum: f64 = res.alpha.iter().sum();
    assert!(sum.abs() < 1e-9, "sum(alpha) = {sum}");
    for i in 0..problem.len() {
        assert!(problem.lo[i] <= res.alpha[i] && res.alpha[i] <= problem.hi[i]);
    }
    if let Some(a) = problem.max_asum {
        let asum: f64 = res
            .alpha
            .iter()
            .zip(&problem.sign)
            .map(|(a, s)| a * s)
            .sum();
        assert!(asum <= a * (1.0 + 1e-9) + 1e-12, "asum {asum} > {a}");
        assert!(res.c >= 0.0);
    }
    assert!(
        res.gap.abs() < gap_tol,
        "duality gap {} (primal {}, dual {})",
        res.gap,
        res.primal_objective,
        res.dual_objective
    );
}

#[test]
fn classification_reaches_zero_duality_gap() {
    let (x, y, _) = dataset(120, 3, 1);
    let kernel = FeatureKernel::new(&x, 3, KernelFunction::Rbf { gamma: 0.8 });
    for smoothing in [0.0, 0.5] {
        let problem = DualProblem::classification(&y, 0.05, None, 1.0, smoothing, None).unwrap();
        let res = solve(&kernel, &problem, &tight()).unwrap();
        assert_optimal(&problem, &res, 1e-6);
        let g = dual_gradient(&problem, &res.alpha, &res.ka);
        assert!(violation(&problem, &res.alpha, &g) <= 1e-9);
    }
}

#[test]
fn all_working_set_rules_and_shrinking_agree() {
    let (x, y, _) = dataset(150, 2, 7);
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 2.0 });
    let problem = DualProblem::classification(&y, 0.01, None, 1.0, 0.1, None).unwrap();
    let mut objectives = Vec::new();
    for ws in [
        WorkingSetSelection::MaximalViolatingPair,
        WorkingSetSelection::MaximalUnboundedDecrease,
        WorkingSetSelection::MaximalDecrease,
    ] {
        for shrinking in [false, true] {
            let opt = SmoOptions {
                working_set: ws,
                shrinking,
                shrinking_period: 20,
                ..tight()
            };
            let res = solve(&kernel, &problem, &opt).unwrap();
            assert_optimal(&problem, &res, 1e-6);
            objectives.push(res.dual_objective);
        }
    }
    let first = objectives[0];
    for o in objectives {
        assert!(
            (o - first).abs() < 1e-7 * first.abs().max(1.0),
            "{o} vs {first}"
        );
    }
}

#[test]
fn second_order_rule_needs_fewer_steps() {
    let (x, y, _) = dataset(200, 2, 3);
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.0 });
    let problem = DualProblem::classification(&y, 0.01, None, 1.0, 0.0, None).unwrap();
    let steps = |ws| {
        let opt = SmoOptions {
            working_set: ws,
            tol: 1e-6,
            ..SmoOptions::default()
        };
        solve(&kernel, &problem, &opt).unwrap().steps
    };
    assert!(
        steps(WorkingSetSelection::MaximalDecrease)
            < steps(WorkingSetSelection::MaximalViolatingPair)
    );
}

/// Solves a dense linear system by Gaussian elimination with partial pivoting.
fn linsolve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let piv = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, piv);
        b.swap(c, piv);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}

#[test]
fn least_squares_matches_closed_form() {
    let (x, _, t) = dataset(60, 2, 11);
    let n = t.len();
    let lambda = 0.1;
    let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 3) as f64 * 0.5).collect();
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 0.5 });
    let problem = DualProblem::least_squares(&t, lambda, Some(&w)).unwrap();
    let res = solve(&kernel, &problem, &tight()).unwrap();
    assert_optimal(&problem, &res, 1e-6);
    // [K/λ + W⁻¹  1; 1ᵀ 0] [α; b] = [y; 0]
    let mut a = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = kernel.eval(i, j) / lambda;
        }
        a[i][i] += 1.0 / w[i];
        a[i][n] = 1.0;
        a[n][i] = 1.0;
    }
    let mut rhs = t.clone();
    rhs.push(0.0);
    let sol = linsolve(a, rhs);
    for i in 0..n {
        assert!(
            (sol[i] - res.alpha[i]).abs() < 1e-6,
            "alpha[{i}]: {} vs {}",
            sol[i],
            res.alpha[i]
        );
    }
    assert!((sol[n] - res.b).abs() < 1e-6, "b: {} vs {}", sol[n], res.b);
}

#[test]
fn regression_reaches_zero_duality_gap() {
    let (x, _, t) = dataset(100, 2, 5);
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.0 });
    for (epsilon, smoothing) in [(0.1, 0.0), (0.05, 0.2), (0.0, 0.1)] {
        let problem = DualProblem::regression(&t, 0.02, None, epsilon, smoothing, None).unwrap();
        let res = solve(&kernel, &problem, &tight()).unwrap();
        assert_optimal(&problem, &res, 1e-6);
        // both copies of a sample are never active at the same time when ε > 0
        if epsilon > 0.0 && smoothing == 0.0 {
            let n = t.len();
            for i in 0..n {
                assert!(res.alpha[i] == 0.0 || res.alpha[n + i] == 0.0);
            }
        }
    }
}

#[test]
fn norm_constraint_is_respected_and_optimal() {
    let (x, y, t) = dataset(100, 2, 9);
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.0 });
    for max_asum in [5.0, 20.0, 1000.0] {
        let problem =
            DualProblem::classification(&y, 0.01, None, 1.0, 0.1, Some(max_asum)).unwrap();
        let res = solve(&kernel, &problem, &tight()).unwrap();
        assert_optimal(&problem, &res, 1e-6);
        let problem = DualProblem::regression(&t, 0.01, None, 0.05, 0.1, Some(max_asum)).unwrap();
        let res = solve(&kernel, &problem, &tight()).unwrap();
        assert_optimal(&problem, &res, 1e-6);
    }
    // a tight constraint must be active
    let problem = DualProblem::classification(&y, 0.01, None, 1.0, 0.0, Some(5.0)).unwrap();
    let res = solve(&kernel, &problem, &tight()).unwrap();
    let asum: f64 = res.alpha.iter().map(|a| a.abs()).sum();
    assert!((asum - 5.0).abs() < 1e-9);
    assert!(res.c > 0.0);
}

#[test]
fn warm_start_from_solution_terminates_immediately() {
    let (x, y, _) = dataset(80, 2, 4);
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.0 });
    let problem = DualProblem::classification(&y, 0.05, None, 1.0, 0.2, None).unwrap();
    let opt = SmoOptions {
        tol: 1e-6,
        ..SmoOptions::default()
    };
    let first = solve(&kernel, &problem, &opt).unwrap();
    let second = solve_with(&kernel, &problem, &opt, Some(&first.alpha), |_| false).unwrap();
    assert_eq!(second.steps, 0);
    assert!((second.dual_objective - first.dual_objective).abs() < 1e-10);
    assert!(solve_with(&kernel, &problem, &opt, Some(&[1.0]), |_| false).is_err());
}

#[test]
fn limits_and_callback() {
    let (x, y, _) = dataset(200, 2, 2);
    let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma: 1.0 });
    let problem = DualProblem::classification(&y, 0.001, None, 1.0, 0.0, None).unwrap();
    let opt = SmoOptions {
        max_steps: 10,
        ..tight()
    };
    let res = solve(&kernel, &problem, &opt).unwrap();
    assert_eq!((res.status, res.steps), (Status::MaxSteps, 10));
    assert!(res.gap > 0.0);
    let opt = SmoOptions {
        time_limit: Some(0.0),
        ..tight()
    };
    assert_eq!(
        solve(&kernel, &problem, &opt).unwrap().status,
        Status::TimeLimit
    );
    let mut calls = 0;
    let opt = SmoOptions {
        callback_period: 5,
        ..tight()
    };
    let res = solve_with(&kernel, &problem, &opt, None, |p| {
        calls += 1;
        assert_eq!(p.step % 5, 0);
        p.step >= 15
    })
    .unwrap();
    assert_eq!((res.status, res.steps, calls), (Status::Interrupted, 15, 3));
}

#[test]
fn precomputed_kernel_gives_same_solution() {
    let (x, y, _) = dataset(50, 2, 8);
    let fk = FeatureKernel::new(
        &x,
        2,
        KernelFunction::Polynomial {
            gamma: 0.5,
            coef0: 1.0,
            degree: 3,
        },
    );
    let km: Vec<f64> = (0..50 * 50).map(|v| fk.eval(v / 50, v % 50)).collect();
    let pk = PrecomputedKernel::new(&km, 50);
    let problem = DualProblem::classification(&y, 0.1, None, 1.0, 0.0, None).unwrap();
    let a = solve(&fk, &problem, &tight()).unwrap();
    let b = solve(&pk, &problem, &tight()).unwrap();
    assert!((a.dual_objective - b.dual_objective).abs() < 1e-10);
}

#[test]
fn model_reproduces_training_decision_values() {
    let (x, y, _) = dataset(60, 2, 12);
    let f = KernelFunction::Rbf { gamma: 1.5 };
    let kernel = FeatureKernel::new(&x, 2, f);
    let problem = DualProblem::classification(&y, 0.05, None, 1.0, 0.0, None).unwrap();
    let res = solve(&kernel, &problem, &tight()).unwrap();
    let model = Model::new(f, &x, 2, &problem, &res);
    let dec = model.decision_batch(&x);
    for i in 0..60 {
        assert!((dec[i] - res.ka[i] - res.b).abs() < 1e-10);
    }
    assert!(model.n_support() < 60);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn random_problems_are_solved(
        seed in 0u64..10_000,
        n in 2usize..40,
        log_lambda in -4.0f64..1.0,
        gamma in 0.1f64..5.0,
        smoothing in prop_oneof![Just(0.0), 0.01f64..1.0],
        kind in 0usize..3,
        constrained in any::<bool>(),
        shrinking in any::<bool>(),
    ) {
        let (x, mut y, t) = dataset(n, 2, seed);
        // occasionally use a single class
        if seed % 17 == 0 { y.iter_mut().for_each(|v| *v = 1.0); }
        let mut rng = Lcg(seed ^ 0xdead);
        let w: Vec<f64> = (0..n).map(|_| 0.2 + rng.next() * 2.0).collect();
        let lambda = 10f64.powf(log_lambda);
        let max_asum = if constrained && kind != 2 { Some(0.5 + rng.next() * n as f64) } else { None };
        let problem = match kind {
            0 => DualProblem::classification(&y, lambda, Some(&w), 1.0, smoothing, max_asum),
            1 => DualProblem::regression(&t, lambda, Some(&w), 0.05, smoothing, max_asum),
            _ => DualProblem::least_squares(&t, lambda, Some(&w)),
        }.unwrap();
        let kernel = FeatureKernel::new(&x, 2, KernelFunction::Rbf { gamma });
        let opt = SmoOptions { tol: 1e-8, shrinking, shrinking_period: 7, max_steps: 1_000_000, ..SmoOptions::default() };
        let res = solve(&kernel, &problem, &opt).unwrap();
        let scale = res.primal_objective.abs().max(1.0);
        assert_optimal(&problem, &res, 1e-5 * scale);
    }
}
