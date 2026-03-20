# DWCM Benchmark — N=5,000 nodes

**Configuration:** N=5,000 · 5 seeds (start\_seed=0) · tol=1e-5 · `k_s_generator_pl(N=5000, rho=1e-3)` · `--fast` flag

## Per-seed results

| Seed | Method | Conv | Iters | Time (s) | Max rel. err |
|------|--------|:----:|------:|--------:|------------:|
| 0 | FP-GS Anderson(10) | ✓ | 55 | 45.23 | 4.31e-10 |
| 0 | θ-Newton Anderson(10) | ✓ | 18 | 13.86 | 3.11e-08 |
| 1 | FP-GS Anderson(10) | ✓ | 28 | 13.27 | 1.94e-09 |
| 1 | θ-Newton Anderson(10) | ✓ | 18 | 13.86 | 2.73e-09 |
| 2 | FP-GS Anderson(10) | ✓ | 14 | 7.01 | 5.48e-09 |
| 2 | θ-Newton Anderson(10) | ✓ | 14 | 10.86 | 1.62e-08 |
| 3 | FP-GS Anderson(10) | ✓ | 13 | 6.41 | 2.12e-09 |
| 3 | θ-Newton Anderson(10) | ✓ | 10 | 7.89 | 3.51e-08 |
| 4 | FP-GS Anderson(10) | ✓ | 11 | 5.63 | 3.77e-08 |
| 4 | θ-Newton Anderson(10) | ✓ | 12 | 9.39 | 3.76e-08 |

## Aggregate statistics (converged runs only)

| Method | Conv% | Time mean±2σ (s) | RAM mean±2σ (MB) | Iters mean±2σ | MaxRelErr mean±2σ |
|--------|:-----:|----------------:|----------------:|-------------:|------------------:|
| FP-GS Anderson(10) | **100%** | 15.51 ± 33.78 | 0.0 ± 0.0 | 24 ± 37 | 9.53e-09 ± 3.17e-08 |
| θ-Newton Anderson(10) | **100%** | 11.17 ± 5.34 | 0.0 ± 0.0 | 14 ± 7 | 2.46e-08 ± 2.95e-08 |

## Bad-seed validation (seeds that previously failed)

Seeds `[181998678, 181998679, 181998681, 181998682, 181998683, 181998684]` generated with
`k_s_generator_pl(N=5000, rho=1e-3)`, budget=120 s per solver:

| Seed | FP-GS And(10) | θ-Newton And(10) |
|------|:-------------:|:----------------:|
| 181998678 | ✓ 151i 70.7s | ✓ 21i 15.2s |
| 181998679 | ✓ 93i 66.6s | ✓ 21i 15.2s |
| 181998681 | ✓ 73i 45.4s | ✓ 30i 21.7s |
| 181998682 | ✓ 85i 54.0s | ✓ 18i 13.1s |
| 181998683 | ✓ 71i 50.9s | ✓ 25i 18.1s |
| 181998684 | ✓ 93i 78.3s | ✓ 79i 57.1s |

## Key fixes applied (`src/solvers/fixed_point_dwcm.py`)

| Fix | Description |
|-----|-------------|
| **Anderson blowup reset** | When residual jumps >100× above best, clear Anderson history and reset θ to `best_theta` |
| **Weighted Anderson mixing** | Per-component row-normalisation of the residual matrix prevents hub-dominated ill-conditioning |
| **Universal θ floor** | After every FP/Newton step, enforce `θ_new ≥ 0.1 × θ_current` to stop θ → ETA\_MIN blowups |
| **Node-level Newton fallback** | When `|Δθ_FP| > 0.1` for a node, replace FP step with exact diagonal Newton step |
| **FP-GS Newton-Anderson mini-loop** | When FP-GS Anderson stagnates for 30 iters (residual unchanged by ≥1%), run a θ-Newton Anderson(5) mini-loop of up to 30 steps from `best_theta`; placed post-Anderson so blowup recovery cannot discard it |
| **best\_theta tracking** | `SolverResult.theta` always returns the lowest-residual iterate, not the final one |
