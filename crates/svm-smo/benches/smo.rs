//! Benchmarks of the SMO solver on synthetic data (`cargo bench -p svm-smo`).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use svm_smo::{solve, DualProblem, FeatureKernel, KernelFunction, SmoOptions};

fn dataset(n: usize, dim: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let y = f
        .iter()
        .map(|&v| if v + 0.3 * rnd() > 0.0 { 1.0 } else { -1.0 })
        .collect();
    let t = f.iter().map(|&v| v + 0.1 * rnd()).collect();
    (x, y, t)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("smo");
    group.sample_size(10);
    for n in [1000, 2000] {
        let (x, y, t) = dataset(n, 10);
        let kernel = FeatureKernel::new(&x, 10, KernelFunction::Rbf { gamma: 0.1 });
        let opt = SmoOptions::default();
        let one_sided = SmoOptions {
            two_sided: false,
            ..SmoOptions::default()
        };
        let cls = DualProblem::classification(&y, 0.01, None, 1.0, 0.0, None).unwrap();
        group.bench_with_input(BenchmarkId::new("classification", n), &n, |b, _| {
            b.iter(|| solve(&kernel, &cls, &opt).unwrap())
        });
        group.bench_with_input(
            BenchmarkId::new("classification-one-sided", n),
            &n,
            |b, _| b.iter(|| solve(&kernel, &cls, &one_sided).unwrap()),
        );
        let reg = DualProblem::regression(&t, 0.01, None, 0.1, 0.0, None).unwrap();
        group.bench_with_input(BenchmarkId::new("regression", n), &n, |b, _| {
            b.iter(|| solve(&kernel, &reg, &opt).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("regression-one-sided", n), &n, |b, _| {
            b.iter(|| solve(&kernel, &reg, &one_sided).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
