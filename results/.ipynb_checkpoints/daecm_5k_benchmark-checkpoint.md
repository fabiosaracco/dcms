# DaECM N=5,000 Benchmark

**Command:**
```
python -m src.benchmarks.daecm_comparison --sizes 5000 --n_seeds 5 --start_seed 0 --timeout 0 --fast
```

**Network generator:** `k_s_generator_pl(N=5000, rho=1e-3)` (power-law degree/strength sequences)  
**Tolerance:** `1e-5`  
**Solver cap (fast mode):** 150 s per solver  

---

## Per-seed results

| Seed | FP-GS Anderson(10) | θ-Newton Anderson(10) | L-BFGS |
|------|-------------------|----------------------|--------|
| 0 | ✗ TIMEOUT (150s) | ✓ err=3.19e-08, 39 it, 28.4s | ✗ TIMEOUT (150s) |
| 1 | ✗ TIMEOUT (150s) | ✓ err=1.88e-07, 50 it, 49.8s | ✗ TIMEOUT (150s) |
| 2 | ✗ NO-CONV err=8.17e-01, 111 it, 125.6s | ✓ err=6.86e-08, 47 it, 39.2s | ✗ TIMEOUT (150s) |
| 3 | ✗ NO-CONV err=7.56e-01, 108 it, 126.8s | ✓ err=8.31e-08, 47 it, 32.4s | ✗ TIMEOUT (150s) |
| 4 | ✗ TIMEOUT (150s) | ✓ err=6.06e-09, 39 it, 30.7s | ✗ TIMEOUT (150s) |

---

## Aggregate statistics (converged runs only)

| Method | Conv% | Time (s) mean±2σ | RAM (MB) mean±2σ | Iters mean±2σ | MaxRelErr mean±2σ |
|--------|-------|-------------------|------------------|---------------|-------------------|
| FP-GS Anderson(10) multi-init | 0% | — | — | — | — |
| **θ-Newton Anderson(10) multi-init** | **100%** | **36.1 ± 17.3** | **0.0 ± 0.0** | **44 ± 10** | **7.55e-08 ± 1.39e-07** |
| L-BFGS (multi-start) | 0% | — | — | — | — |

---

## Notes

- **θ-Newton Anderson(10)** is the only reliable solver at N=5,000 within a 150 s budget.
- **FP-GS Anderson(10)** fails because the fixed-point map is not a contraction at this scale
  (spectral radius > 1 for power-law hub nodes with high s/k ratio).
- **L-BFGS** is too slow: O(N²) cost per gradient evaluation (~370 ms/call at N=5,000)
  and hundreds of iterations needed → always exceeds the 150 s cap.
- The weight solver fixes (PR#12 continuation + bad-seed fixes in `fixed_point_daecm.py`):
  - `_Z_G_CLAMP = 1e-8`: accurate G near z=0, enables self-escape from z→0 deadlock
  - `_Z_NEWTON_FLOOR = 1e-8`: hard floor = z_clamp, enables O(log N) doubling recovery
  - `_Z_NEWTON_FRAC = 0.5`: relative downward step limit, prevents period-2 oscillation
  - `_ANDERSON_BLOWUP_FACTOR = 5000`: allows Anderson to accumulate and average oscillating iterates
- All 10 originally failing seeds (448470081–448470090) now converge with θ-Newton Anderson(10),
  taking 24–159 iterations (15–111 s).
