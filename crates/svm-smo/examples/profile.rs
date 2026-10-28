//! Profiling driver: `cargo run --release --example profile -- [reg|cls] [n]`.

use svm_smo::{solve, DualProblem, FeatureKernel, KernelFunction, SmoOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let task = args.get(1).map_or("reg", String::as_str);
    let n: usize = args.get(2).map_or(5000, |s| s.parse().unwrap());
    let dim = 10;
    let mut state = 42u64;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let x: Vec<f64> = (0..n * dim).map(|_| 2.0 * rnd()).collect();
    let f: Vec<f64> = x
        .chunks_exact(dim)
        .map(|r| r[0].sin() + r[1] * r[2] - 0.5 * r[3])
        .collect();
    let kernel = FeatureKernel::new(&x, dim, KernelFunction::Rbf { gamma: 0.1 });
    let prob = if task == "cls" {
        let y: Vec<f64> = f
            .iter()
            .map(|&v| if v + 0.3 * rnd() > 0.0 { 1.0 } else { -1.0 })
            .collect();
        DualProblem::classification(&y, 0.01, None, 1.0, 0.0, None).unwrap()
    } else {
        let t: Vec<f64> = f.iter().map(|&v| v + 0.1 * rnd()).collect();
        DualProblem::regression(&t, 0.01, None, 0.1, 0.0, None).unwrap()
    };
    let r = solve(&kernel, &prob, &SmoOptions::default()).unwrap();
    println!(
        "{task} n={n}: {:.3}s steps={} hits={} misses={} gap={:.2e}",
        r.elapsed, r.steps, r.cache_hits, r.cache_misses, r.gap
    );
}
