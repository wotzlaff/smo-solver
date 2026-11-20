use super::*;
use serde_json::Value;

const FIXTURE: &str = include_str!("../tests/cv_fixture.json");

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
        y.push(if kind == 0 {
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
    let cases: Value = serde_json::from_str(FIXTURE).unwrap();
    for case in cases.as_array().unwrap() {
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
            let out = data.cv(get("log_lambda"), get("log_gamma"), 1e-8);
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
    if smoothing > 0.0 {
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
    check_gradient(0, 0.3, -1.0, 0.0);
    check_gradient(0, 0.3, -2.5, 0.8);
    check_gradient(1, 0.2, -3.0, 0.5);
    check_gradient(1, 0.2, -2.0, -0.5);
    check_gradient(0, 0.0, -1.0, 0.0);
    check_gradient(1, 0.0, -3.0, 0.5);
}

#[test]
fn invalid_input_is_reported() {
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
