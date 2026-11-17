# Raw results of the studies

| file | script | content |
|---|---|---|
| `shrinking_scaling.csv` | `shrinking_study.py scaling` | time with (`default`) and without (`off`) shrinking over n and tolerance |
| `shrinking_lambda.csv` | `shrinking_study.py lambda` | the same over the regularization parameter, with the number of support vectors |
| `shrinking_params.csv` | `shrinking_study.py params` | `shrinking_exponent` (η) × `shrinking_factor` (c) and the shrinking period |
| `shrinking_robust.csv` | `shrinking_study.py robust` | earlier default vs. η = 0.75 and η = 1 over kernel width, λ, tolerance, 1-norm bound |
| `shrinking_tune.csv` | `shrinking_study.py tune` | factor c and minimal removed fraction for η = 1 |
| `stopping.csv` | `stopping_study.py run` | measures and actual errors along 12 runs |
| `convergence_relations.csv.gz` | `convergence_study.py relations` | per-step r, τ̃, Δψ, r̃ (every 25th step) and gap, D_k, V_k (thinned steps) for 5 (λ, γ) settings, hinge loss, MDP |
| `convergence_factor.csv.gz` | `convergence_study.py factor` | the same traces for γ = 30, λ = 1e-5 … 1e-1 (half decades), basis of the c(λ) fits |
| `convergence_smooth.csv.gz` | `convergence_study.py smooth` | traces for the smoothed hinge (δ = 1/4), λ = 1e-3, 1e-2, 1e-1 |
| `convergence_lambda.csv` | `convergence_study.py lambda` | support vectors, free variables, steps to r < 1e-10 over λ (hinge and δ = 1/4), n = 1000 |
| `convergence_size.csv` | `convergence_study.py size --reps 3` | steps, time (3 repetitions), free variables and support vectors over n = 1000 … 50 000 for λ = 1e-1 … 1e-4, tol 1e-8, shrinking on |
| `working_set_traj.csv.gz` | `working_set_study.py run` | thinned per-step traces (all steps ≤ 1000, then log-spaced) of r, r◇(α, λ), gap, D_k, V_k with running minima and r_k, r̃_k, τ̃_k, decrease for MVP, baMVP, MuDP, MDP, tMDP in 4 settings |
| `working_set_hits.csv` | `working_set_study.py run` | exact first step at which r, r◇/λ, gap, D_k, V_k drop below 1e-1 … 1e-10 |
| `working_set_timing.csv` | `working_set_study.py run` | Rust solver: time to r ≤ 1e-3, 1e-5, 1e-8 (min of 3); NumPy reference SMO: time per step |
| `working_set_validation.csv` | `working_set_study.py run` | NumPy reference SMO against the Rust iterates (first 2000 steps of MVP, MuDP, MDP) |

The files `shrinking_scaling`, `shrinking_lambda` and `shrinking_params` were produced with the current defaults
(`shrinking_exponent = 1`, `shrinking_factor = 1`, `shrinking_min_fraction = 0.1`), `shrinking_robust` and
`shrinking_tune` before the change, where `default` means the earlier defaults (`exponent = 0.5`, no minimal
fraction) and `min0.1` the current ones. Times are wall-clock seconds on a shared machine (minimum of two runs).

The `convergence_*` files repeat the figures of Sec. 10.2.4-10.2.7 of the thesis on the checkerboard problem of
Sec. 10.1 (n = 1000, γ = 30, c_i = 1, MDP, no shrinking, solver stopped at r < 1e-10 or after 500,000 steps). The
trace files have one row per recorded step (`step` = k): `r`, `tau` (τ̃_k), `dpsi` (decrease ψ(α^k) − ψ(α^{k+1}) > 0),
`rt` (r̃_k) are available for every 25th step, `gap` (minimised over the bias), `D` and `V` only at about 1000
logarithmically/uniformly thinned steps (empty otherwise). They are computed from consecutive iterates of the
callback, the gap as a sum of Fenchel-Young terms (no cancellation), V_k via (g\* + b\*)ᵀd + dᵀHd/2 with d = α^k − α\*
(reference solution with r < 1e-12). `convergence_study.py summary benchmarks/results` prints the fitted constants.
`convergence_size.csv` (Sec. 10.2.9) was measured alone, with other users' jobs on the machine (load ≈ 5 of 12
cores; the solver runs on one core); the minimum of three repetitions is the time.
The large traces are gzipped (`*.csv.gz`); the scripts read both forms.

The `working_set_*` files repeat Sec. 10.2.1-10.2.3 (n = 1000 uniform points, checkerboard labels, hinge loss, c_i = 1, seed 0).
Settings: `base` (λ = 1e-3, γ = 30), `lam1e-2`, `lam1e-4` (γ = 30), `gamma10` (λ = 1e-3, γ = 10). MVP, MuDP and MDP are the
solver's rules (`shrinking=False`, per-step callback); baMVP (pair realizing r◇(α, λ)) and tMDP (best of all pairs, O(n²))
run in a NumPy reference SMO with the same clipped two-variable step. Runs stop at r ≤ 1e-8 (tMDP: r ≤ 1e-6 or 20,000
steps, which was hit for `base` and `gamma10`). Reference solution: MDP with tol 1e-12; the gap Δ\* uses the midpoint bias of
the thesis (Obs. 7.4) and is about 1e-10 at the reference solution. r_k = min(ε, 1)·(g_i − g_j) (ρ₀ = 1), r̃_k = √2 r_k/‖d^k‖
is the violation of the chosen pair. The validation file shows identical iterates (≤ 1e-10) for MVP over 2000 steps and for
MuDP/MDP until the first step with a different pair (46 … 1591), caused by ties in the gradient of the order of 1e-13.
Timings in `working_set_timing.csv` come from a shared, heavily loaded machine (two runs in parallel) and are indicative only.
`working_set_study.py summary` prints the key numbers.
