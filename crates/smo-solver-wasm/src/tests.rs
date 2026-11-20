use super::*;
use serde_json::Value;

const FIXTURE: &str = include_str!("../tests/cv_fixture.json");

/// The exports share global state (model slot, warm starts): run the tests one at a time.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_f64().unwrap())
        .collect()
}

struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
    fold_of: Vec<f64>,
    dim: usize,
    k: usize,
    kind: u32,
    smoothing: f64,
    epsilon: f64,
}

impl Data {
    fn cv(&self, ll: f64, lg: f64, tol: f64) -> [f64; 3] {
        let mut out = [0.0; 3];
        let status = unsafe {
            cv_objective(
                self.x.as_ptr(),
                self.y.len(),
                self.dim,
                self.y.as_ptr(),
                self.fold_of.as_ptr(),
                self.k,
                self.kind,
                ll,
                lg,
                self.smoothing,
                self.epsilon,
                tol,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        out
    }

    fn cv_ard(&self, ll: f64, lgs: &[f64], tol: f64) -> Vec<f64> {
        let mut out = vec![0.0; 2 + self.dim];
        let status = unsafe {
            cv_objective_ard(
                self.x.as_ptr(),
                self.y.len(),
                self.dim,
                self.y.as_ptr(),
                self.fold_of.as_ptr(),
                self.k,
                self.kind,
                ll,
                lgs.as_ptr(),
                self.smoothing,
                self.epsilon,
                tol,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        out
    }
}

/// Small deterministic generator (xorshift) for the random test data.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_data(kind: u32, n: usize, k: usize, smoothing: f64) -> Data {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let (a, b) = (rng.next() * 2.0 - 1.0, rng.next() * 2.0 - 1.0);
        x.extend([a, b]);
        y.push(if kind != 1 {
            // noisy circle, so that both classes appear in every fold
            if a * a + b * b + 0.2 * (rng.next() - 0.5) < 0.4 {
                1.0
            } else {
                -1.0
            }
        } else {
            (2.0 * a).sin() + 0.5 * b + 0.05 * (rng.next() - 0.5)
        });
        let _ = i;
    }
    Data {
        x,
        y,
        fold_of: (0..n).map(|i| (i % k) as f64).collect(),
        dim: 2,
        k,
        kind,
        smoothing,
        epsilon: 0.1,
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-6)
}

#[test]
fn matches_python_reference() {
    let _serial = serial();
    let cases: Value = serde_json::from_str(FIXTURE).unwrap();
    for case in cases.as_array().unwrap() {
        if case.get("ard").is_some() {
            continue; // see `ard_matches_python_reference`
        }
        let data = Data {
            x: floats(&case["x"]),
            y: floats(&case["y"]),
            fold_of: floats(&case["fold_of"]),
            dim: case["dim"].as_u64().unwrap() as usize,
            k: case["k"].as_u64().unwrap() as usize,
            kind: case["kind"].as_u64().unwrap() as u32,
            smoothing: case["smoothing"].as_f64().unwrap(),
            epsilon: case["epsilon"].as_f64().unwrap(),
        };
        let mut worst = 0.0f64;
        for pt in case["points"].as_array().unwrap() {
            let get = |k: &str| pt[k].as_f64().unwrap();
            let start = std::time::Instant::now();
            let tol = if data.kind == 2 { 1e-10 } else { 1e-8 };
            let out = data.cv(get("log_lambda"), get("log_gamma"), tol);
            println!(
                "  cv_objective took {:.1} ms",
                start.elapsed().as_secs_f64() * 1e3
            );
            let errs = [
                rel(out[0], get("value")),
                rel(out[1], get("grad_log_lambda")),
                rel(out[2], get("grad_log_gamma")),
            ];
            worst = errs.iter().fold(worst, |a, &e| a.max(e));
            assert!(
                errs.iter().all(|&e| e <= 1e-5),
                "{}: {errs:?}",
                case["name"]
            );
        }
        if let Some(fits) = case.get("fits") {
            let test = floats(&case["test_points"]);
            let m = test.len() / data.dim;
            for fit in fits.as_array().unwrap() {
                let get = |k: &str| fit[k].as_f64().unwrap();
                let lam = get("log_lambda").exp();
                let gamma = get("log_gamma").exp();
                let n = data.y.len();
                let steps = unsafe {
                    train(
                        data.x.as_ptr(),
                        n,
                        data.dim,
                        data.y.as_ptr(),
                        2,
                        lam,
                        gamma,
                        0.0,
                        0.0,
                        1e-10,
                    )
                };
                assert!(steps > 0, "train failed: {steps}");
                let mut dec = vec![0.0; m];
                unsafe { decision(test.as_ptr(), m, dec.as_mut_ptr()) };
                let want = floats(&fit["decision"]);
                let e = dec
                    .iter()
                    .zip(&want)
                    .map(|(&a, &b)| rel(a, b))
                    .fold(rel(stat(0), get("bias")), f64::max);
                println!("  train: {steps} Newton steps, max rel. error {e:.2e}");
                assert!(e <= 1e-5, "{}: train {e}", case["name"]);
                worst = worst.max(e);
            }
        }
        println!("oracle {}: max rel. error {worst:.2e}", case["name"]);
    }
}

fn check_gradient(kind: u32, smoothing: f64, ll: f64, lg: f64) {
    let data = random_data(kind, 60, 3, smoothing);
    let tol = 1e-10;
    let out = data.cv(ll, lg, tol);
    let h = 1e-4;
    let fd_l = (data.cv(ll + h, lg, tol)[0] - data.cv(ll - h, lg, tol)[0]) / (2.0 * h);
    let fd_g = (data.cv(ll, lg + h, tol)[0] - data.cv(ll, lg - h, tol)[0]) / (2.0 * h);
    println!("kind {kind} delta {smoothing}: {out:?} fd ({fd_l}, {fd_g})");
    if smoothing > 0.0 || kind == 2 {
        let scale = out[1].abs().max(out[2].abs()).max(1e-3);
        assert!(
            (out[1] - fd_l).abs() <= 1e-4 * scale,
            "{} vs {fd_l}",
            out[1]
        );
        assert!(
            (out[2] - fd_g).abs() <= 1e-4 * scale,
            "{} vs {fd_g}",
            out[2]
        );
    } else {
        // pure hinge: only the explicit part is returned (as in the Python reference)
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn gradient_matches_finite_differences() {
    let _serial = serial();
    check_gradient(0, 0.3, -1.0, 0.0);
    check_gradient(0, 0.3, -2.5, 0.8);
    check_gradient(1, 0.2, -3.0, 0.5);
    check_gradient(1, 0.2, -2.0, -0.5);
    check_gradient(2, 0.0, -1.0, 0.0);
    check_gradient(2, 0.0, -3.0, 0.8);
    check_gradient(0, 0.0, -1.0, 0.0);
    check_gradient(1, 0.0, -3.0, 0.5);
}

#[test]
fn invalid_input_is_reported() {
    let _serial = serial();
    let data = random_data(0, 12, 3, 0.3);
    let mut out = [0.0; 3];
    let call = |kind: u32, k: usize, lg: f64, out: &mut [f64; 3]| unsafe {
        cv_objective(
            data.x.as_ptr(),
            12,
            2,
            data.y.as_ptr(),
            data.fold_of.as_ptr(),
            k,
            kind,
            -1.0,
            lg,
            0.3,
            0.1,
            1e-8,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(call(0, 3, 0.0, &mut out), 0);
    assert_eq!(call(5, 3, 0.0, &mut out), -2);
    assert_eq!(call(0, 1, 0.0, &mut out), -1);
    assert_eq!(call(0, 3, f64::NAN, &mut out), -1);
    assert_eq!(call(0, 5, 0.0, &mut out), -1); // folds 3 and 4 do not exist / are empty
}

#[test]
fn trace_is_monotone_and_ends_at_the_final_model() {
    let _serial = serial();
    for kind in [0u32, 1] {
        let data = random_data(kind, 80, 4, 0.2);
        let n = data.y.len();
        let steps = unsafe {
            train_trace(
                data.x.as_ptr(),
                n,
                2,
                data.y.as_ptr(),
                kind,
                0.1,
                2.0,
                0.2,
                0.1,
                1e-6,
                5,
                1000,
            )
        };
        assert!(steps > 5);
        let len = trace_len() as usize;
        assert!(len >= 3);
        let mut buf = vec![0.0; 4 + n];
        let mut last_step = -1.0;
        let mut first_gap = 0.0;
        for i in 0..len {
            assert_eq!(unsafe { trace_frame(i, buf.as_mut_ptr()) }, 4 + n as i64);
            assert!(buf[0] > last_step, "steps must increase");
            last_step = buf[0];
            if i == 0 {
                first_gap = buf[1];
            }
        }
        assert_eq!(last_step, steps as f64);
        assert!(buf[1] < first_gap && buf[1] >= -1e-9);
        // the last frame reproduces the decision function of the global model
        let pts = [0.1, -0.3, 0.7, 0.2, -0.9, 0.5];
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        unsafe {
            assert_eq!(decision(pts.as_ptr(), 3, a.as_mut_ptr()), 0);
            assert_eq!(trace_decision(len - 1, pts.as_ptr(), 3, b.as_mut_ptr()), 0);
        }
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-12);
        }
        assert_eq!(buf[3], stat(0));
        assert_eq!(buf[1], stat(1));
        assert_eq!(unsafe { trace_frame(len, buf.as_mut_ptr()) }, -1);
    }
}

#[test]
fn logistic_newton_satisfies_the_optimality_conditions() {
    let _serial = serial();
    for (n, lam, gamma) in [(60usize, 0.1, 2.0), (120, 1e-3, 5.0), (80, 10.0, 0.5)] {
        let data = random_data(2, n, 3, 0.0);
        let start = std::time::Instant::now();
        let steps = unsafe {
            train(
                data.x.as_ptr(),
                n,
                2,
                data.y.as_ptr(),
                2,
                lam,
                gamma,
                0.0,
                0.0,
                1e-10,
            )
        };
        println!(
            "n={n} lam={lam}: {steps} Newton steps, {:.2} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        assert!(steps > 0 && steps < 100);
        assert!(stat(1) < 1e-10 && stat(1) == stat(4));
        // KKT: a_i = y_i sigma(-y_i f_i) with a = lam * coef, and sum(a) = 0
        let mut dec = vec![0.0; n];
        unsafe { decision(data.x.as_ptr(), n, dec.as_mut_ptr()) };
        let k = unsafe { support(std::ptr::null_mut()) } as usize;
        let mut idx = vec![0.0; k];
        unsafe { support(idx.as_mut_ptr()) };
        assert_eq!(k, n);
        for (i, f) in dec.iter().enumerate() {
            let want = data.y[i] / (1.0 + (data.y[i] * f).exp());
            let guard = lock(&STATE);
            let coef = guard.as_ref().unwrap().model.coef[i];
            assert!((lam * coef - want).abs() < 1e-9, "{} vs {want}", lam * coef);
        }
        let guard = lock(&STATE);
        let sum: f64 = guard.as_ref().unwrap().model.coef.iter().sum::<f64>() * lam;
        assert!(sum.abs() < 1e-9);
    }
}

#[test]
fn logistic_rejects_bad_labels_and_parameters() {
    let _serial = serial();
    let data = random_data(2, 20, 2, 0.0);
    let mut y = data.y.clone();
    y[3] = 0.5;
    let call = |y: &[f64], lam: f64| unsafe {
        train(
            data.x.as_ptr(),
            20,
            2,
            y.as_ptr(),
            2,
            lam,
            1.0,
            0.0,
            0.0,
            1e-8,
        )
    };
    assert_eq!(call(&y, 0.1), -3);
    assert_eq!(call(&data.y, -1.0), -3);
    assert!(call(&data.y, 0.1) > 0);
}

#[test]
fn logistic_trace_is_monotone_and_ends_at_the_final_model() {
    let _serial = serial();
    let data = random_data(2, 80, 4, 0.0);
    let n = 80;
    let steps = unsafe {
        train_trace(
            data.x.as_ptr(),
            n,
            2,
            data.y.as_ptr(),
            2,
            0.05,
            2.0,
            0.0,
            0.0,
            1e-10,
            1,
            1000,
        )
    };
    assert!(steps >= 3);
    let len = trace_len() as usize;
    assert_eq!(len as i64, steps + 1); // period 1: every iteration plus the start
    let mut buf = vec![0.0; 4 + n];
    let (mut last_step, mut last_gap) = (-1.0, f64::INFINITY);
    for i in 0..len {
        assert_eq!(unsafe { trace_frame(i, buf.as_mut_ptr()) }, 4 + n as i64);
        assert!(buf[0] > last_step, "steps must increase");
        // the optimality norm of damped Newton need not drop every step, but must end tiny
        last_step = buf[0];
        last_gap = buf[1];
    }
    assert_eq!(last_step, steps as f64);
    assert!(last_gap < 1e-10);
    // decision values of the last frame match the global model
    let pts = [0.1, -0.3, 0.7, 0.2, -0.9, 0.5];
    let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
    unsafe {
        assert_eq!(decision(pts.as_ptr(), 3, a.as_mut_ptr()), 0);
        assert_eq!(trace_decision(len - 1, pts.as_ptr(), 3, b.as_mut_ptr()), 0);
    }
    for (u, v) in a.iter().zip(&b) {
        assert!((u - v).abs() < 1e-12);
    }
    assert_eq!(buf[3], stat(0));
    // sparse recording keeps the first frame and always the final one
    let s2 = unsafe {
        train_trace(
            data.x.as_ptr(),
            n,
            2,
            data.y.as_ptr(),
            2,
            0.05,
            2.0,
            0.0,
            0.0,
            1e-10,
            2,
            1000,
        )
    };
    assert_eq!(s2, steps);
    let len2 = trace_len() as usize;
    unsafe { trace_frame(len2 - 1, buf.as_mut_ptr()) };
    assert_eq!(buf[0], steps as f64);
}

#[test]
#[ignore = "timing; run with --ignored --nocapture in release"]
fn logistic_timings() {
    let _serial = serial();
    for n in [120usize, 300] {
        let data = random_data(2, n, 4, 0.0);
        let t = std::time::Instant::now();
        let steps = unsafe {
            train(
                data.x.as_ptr(),
                n,
                2,
                data.y.as_ptr(),
                2,
                0.05,
                2.0,
                0.0,
                0.0,
                1e-8,
            )
        };
        let t_train = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let out = data.cv(-3.0, 0.7, 1e-8);
        let t_cv = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        data.cv(-3.0, 0.72, 1e-8);
        println!(
            "n={n}: train {t_train:.1} ms ({steps} steps), cv cold {t_cv:.1} ms, warm {:.1} ms, value {}",
            t.elapsed().as_secs_f64() * 1e3,
            out[0]
        );
    }
}

/// Data with `dim` features: the first three drive the target (Friedman #1 type for regression,
/// a noisy sphere for classification), the others are noise.
fn ard_data(kind: u32, n: usize, k: usize, dim: usize, smoothing: f64) -> Data {
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..n {
        let row: Vec<f64> = (0..dim).map(|_| rng.next()).collect();
        let (a, b, c) = (row[0], row[1], row[2]);
        y.push(if kind == 1 {
            10.0 * (std::f64::consts::PI * a * b).sin()
                + 20.0 * (c - 0.5).powi(2)
                + 0.5 * (rng.next() - 0.5)
        } else if (a - 0.5).powi(2) + (b - 0.5).powi(2) + 0.1 * (rng.next() - 0.5) < 0.1 {
            1.0
        } else {
            -1.0
        });
        x.extend(row);
    }
    Data {
        x,
        y,
        fold_of: (0..n).map(|i| (i % k) as f64).collect(),
        dim,
        k,
        kind,
        smoothing,
        epsilon: 0.1,
    }
}

#[test]
fn ard_with_equal_widths_reproduces_the_isotropic_objective() {
    let _serial = serial();
    for (kind, smoothing, ll, lg) in [
        (0, 0.3, -1.0, 0.5),
        (1, 0.2, -3.0, 0.8),
        (2, 0.0, -2.0, 0.3),
    ] {
        let data = ard_data(kind, 60, 3, 5, smoothing);
        let tol = if kind == 2 { 1e-10 } else { 1e-9 };
        let iso = data.cv(ll, lg, tol);
        let ard = data.cv_ard(ll, &[lg; 5], tol);
        let sum: f64 = ard[2..].iter().sum();
        let errs = [rel(ard[0], iso[0]), rel(ard[1], iso[1]), rel(sum, iso[2])];
        println!("kind {kind}: equal-width ARD vs isotropic, rel. errors {errs:?}");
        assert!(errs.iter().all(|&e| e <= 1e-9), "kind {kind}: {errs:?}");
    }
}

#[test]
fn ard_gradient_matches_finite_differences() {
    let _serial = serial();
    for (kind, smoothing, ll) in [(0, 0.3, -1.0), (1, 0.2, -3.0), (2, 0.0, -2.0)] {
        let data = ard_data(kind, 60, 3, 4, smoothing);
        let lgs = [1.0, -0.3, 0.6, -1.2];
        let tol = 1e-10;
        let out = data.cv_ard(ll, &lgs, tol);
        let h = 1e-4;
        let scale = out[1..].iter().fold(1e-3f64, |a, v| a.max(v.abs()));
        let fd_l =
            (data.cv_ard(ll + h, &lgs, tol)[0] - data.cv_ard(ll - h, &lgs, tol)[0]) / (2.0 * h);
        assert!(
            (out[1] - fd_l).abs() <= 1e-4 * scale,
            "kind {kind}: {} vs {fd_l}",
            out[1]
        );
        for d in 0..4 {
            let (mut p, mut m) = (lgs, lgs);
            p[d] += h;
            m[d] -= h;
            let fd = (data.cv_ard(ll, &p, tol)[0] - data.cv_ard(ll, &m, tol)[0]) / (2.0 * h);
            println!("kind {kind} d={d}: {} fd {fd}", out[2 + d]);
            assert!(
                (out[2 + d] - fd).abs() <= 1e-4 * scale,
                "kind {kind} d={d}: {} vs {fd}",
                out[2 + d]
            );
        }
    }
}

#[test]
fn ard_matches_python_reference() {
    let _serial = serial();
    let cases: Value = serde_json::from_str(FIXTURE).unwrap();
    let mut seen = 0;
    for case in cases.as_array().unwrap() {
        if case.get("ard").is_none() {
            continue;
        }
        seen += 1;
        let data = Data {
            x: floats(&case["x"]),
            y: floats(&case["y"]),
            fold_of: floats(&case["fold_of"]),
            dim: case["dim"].as_u64().unwrap() as usize,
            k: case["k"].as_u64().unwrap() as usize,
            kind: case["kind"].as_u64().unwrap() as u32,
            smoothing: case["smoothing"].as_f64().unwrap(),
            epsilon: case["epsilon"].as_f64().unwrap(),
        };
        let mut worst = 0.0f64;
        for pt in case["points"].as_array().unwrap() {
            let tol = if data.kind == 2 { 1e-10 } else { 1e-8 };
            let out = data.cv_ard(
                pt["log_lambda"].as_f64().unwrap(),
                &floats(&pt["log_gammas"]),
                tol,
            );
            let mut want = vec![
                pt["value"].as_f64().unwrap(),
                pt["grad_log_lambda"].as_f64().unwrap(),
            ];
            want.extend(floats(&pt["grad_log_gammas"]));
            let errs: Vec<f64> = out.iter().zip(&want).map(|(&a, &b)| rel(a, b)).collect();
            worst = errs.iter().fold(worst, |a, &e| a.max(e));
            assert!(
                errs.iter().all(|&e| e <= 1e-5),
                "{}: {errs:?}",
                case["name"]
            );
        }
        println!("oracle {}: max rel. error {worst:.2e}", case["name"]);
    }
    assert_eq!(seen, 3);
}

#[test]
fn ard_rejects_bad_widths() {
    let _serial = serial();
    let data = ard_data(0, 12, 3, 3, 0.3);
    let mut out = [0.0; 5];
    for lgs in [
        [0.0, f64::NAN, 0.0],
        [0.0, f64::INFINITY, 0.0],
        [0.0, -2000.0, 0.0],
    ] {
        let status = unsafe {
            cv_objective_ard(
                data.x.as_ptr(),
                12,
                3,
                data.y.as_ptr(),
                data.fold_of.as_ptr(),
                3,
                0,
                -1.0,
                lgs.as_ptr(),
                0.3,
                0.1,
                1e-8,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, -1, "{lgs:?}");
    }
}

#[test]
#[ignore = "timing; run with --ignored --nocapture in release"]
fn ard_timings() {
    let _serial = serial();
    for (kind, n, dim, k) in [(1, 400, 8, 5), (0, 300, 6, 5)] {
        let data = ard_data(kind, n, k, dim, if kind == 1 { 0.2 } else { 0.3 });
        let lgs: Vec<f64> = (0..dim).map(|d| if d < 3 { 0.5 } else { -2.0 }).collect();
        let ll = if kind == 1 { -3.0 } else { -4.0 };
        let t = std::time::Instant::now();
        let out = data.cv_ard(ll, &lgs, 1e-8);
        let cold = t.elapsed().as_secs_f64() * 1e3;
        let mut lgs2 = lgs.clone();
        lgs2[0] += 0.02;
        let t = std::time::Instant::now();
        data.cv_ard(ll, &lgs2, 1e-8);
        println!(
            "ard kind {kind} n={n} dim={dim}: cold {cold:.0} ms, warm {:.0} ms, value {}",
            t.elapsed().as_secs_f64() * 1e3,
            out[0]
        );
    }
}
