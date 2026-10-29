# Raw results of the studies

| file | script | content |
|---|---|---|
| `shrinking_scaling.csv` | `shrinking_study.py scaling` | time with (`default`) and without (`off`) shrinking over n and tolerance |
| `shrinking_lambda.csv` | `shrinking_study.py lambda` | the same over the regularization parameter, with the number of support vectors |
| `shrinking_params.csv` | `shrinking_study.py params` | `shrinking_exponent` (η) × `shrinking_factor` (c) and the shrinking period |
| `shrinking_robust.csv` | `shrinking_study.py robust` | earlier default vs. η = 0.75 and η = 1 over kernel width, λ, tolerance, 1-norm bound |
| `shrinking_tune.csv` | `shrinking_study.py tune` | factor c and minimal removed fraction for η = 1 |
| `stopping.csv` | `stopping_study.py run` | measures and actual errors along 12 runs |

The files `shrinking_scaling`, `shrinking_lambda` and `shrinking_params` were produced with the current defaults
(`shrinking_exponent = 1`, `shrinking_factor = 1`, `shrinking_min_fraction = 0.1`), `shrinking_robust` and
`shrinking_tune` before the change, where `default` means the earlier defaults (`exponent = 0.5`, no minimal
fraction) and `min0.1` the current ones. Times are wall-clock seconds on a shared machine (minimum of two runs).
