# DCMS — Maximum-Entropy Solvers for Directed Networks

This project implements numerical solvers for **maximum-entropy** (MaxEnt) models of directed networks.  Given an observed graph, the models find the probability distribution over all directed graphs that maximises entropy subject to reproducing a chosen set of topological constraints (degree and/or strength sequences).

Every model is solved by the same method, a **θ-Newton iteration with Anderson(10) acceleration on the degeneracy-reduced system** (nodes with identical constraints share their parameters, so the solver works on M ≤ N groups and expands the solution back to the N nodes — §2.5). The reduction is exact, not an approximation. Earlier versions also offered a per-node solver, a Gauss-Seidel fixed-point variant, a Numba backend and a backtracking line search: they were removed on 2026-10-27 because the reduced θ-Newton solver is what every result of this project uses (see §2.4).

Park, J. & Newman, M.E.J. (2004). Statistical mechanics of networks. *Physical Review E*, **70**, 066117.

Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, 13, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

Mastrandrea, R., Squartini, T., Fagiolo G., and Garlaschelli, D. (2014). Enhanced reconstruction of weighted networks from strengths and degrees. *New Journal of Physics*, 16 043022
https://iopscience.iop.org/article/10.1088/1367-2630/16/4/043022

Gabrielli, A, Mastrandrea, R., Caldarelli, G. and Cimini, G. (2019) Grand canonical ensemble of weighted networks. *Phys. Rev. E* 99, 030301(R) 
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.030301

Parisi, F., Squartini, T. and Garlaschelli, D. (2020). A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks. *New Journal of Physics*, 22 053053
https://iopscience.iop.org/article/10.1088/1367-2630/ab74a7


---

## 0. Installation

Install from GitHub (the package is not yet on PyPI):

```bash
pip install git+https://github.com/fabiosaracco/dcms.git
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy ≥ 1.24, SciPy ≥ 1.10. GPU acceleration (optional, §2.7) uses whatever PyTorch backend is already installed — Apple Silicon MPS or an NVIDIA CUDA GPU — no extra package.

---

## 1. Models

### 1.1 DCM — Directed Configuration Model (binary)

The DCM constrains the **out-degree** and **in-degree** of every node.  Given observed sequences `k_out` and `k_in`, it finds `2N` Lagrange multipliers `(θ_out, θ_in)` such that

```
k_out_i = Σ_{j≠i}  x_i · y_j / (1 + x_i · y_j)
k_in_i  = Σ_{j≠i}  x_j · y_i / (1 + x_j · y_i)
```

where `x_i = exp(-θ_out_i)` and `y_i = exp(-θ_in_i)`.  The link probability is then `p_ij = x_i y_j / (1 + x_i y_j)`.

**Implementation:** `dcms/models/dcm.py` — `DCMModel`

### 1.2 DWCM — Directed Weighted Configuration Model (weighted)

The DWCM constrains the **out-strength** and **in-strength** of every node.  Weights are geometrically distributed (integer-valued), leading to

```
s_out_i = Σ_{j≠i}  β_out_i · β_in_j / (1 − β_out_i · β_in_j)
s_in_i  = Σ_{j≠i}  β_out_j · β_in_i / (1 − β_out_j · β_in_i)
```

where `β = exp(-θ)`.  **Feasibility constraint:** `β_out_i · β_in_j < 1` for all `i ≠ j` (i.e. `θ > 0` for all multipliers).

**Implementation:** `dcms/models/dwcm.py` — `DWCMModel`

### 1.3 qDECM — Quasi Directed Enhanced Configuration Model (binary + weighted)

The qDECM constrains *four* sequences per node: **out-degree**, **in-degree**, **out-strength** and **in-strength**.  It is solved in two sequential steps:

1. **Topology step** — solve the DCM to find `2N` multipliers `(x_i, y_i)` reproducing the degree sequences.  The resulting link probability is `p_ij = x_i · y_j / (1 + x_i · y_j)`.

2. **Weight step** — solve a DWCM conditioned on the DCM topology to find `2N` additional multipliers `(β_out_i, β_in_i)` reproducing the strength sequences:

```
s_out_i = Σ_{j≠i} p_ij / (1 − β_out_i · β_in_j)
s_in_i  = Σ_{j≠i} p_ji / (1 − β_out_j · β_in_i)
```

The expected weight of arc i→j conditioned on the DCM topology is `E[w_ij] = p_ij · E[w_ij | a_ij=1]` where the conditional mean weight is `E[w_ij | a_ij=1] = 1 / (1 − β_out_i · β_in_j)`.  Hence the numerator is `p_ij`, not `p_ij · β_out_i · β_in_j`.

The total number of unknowns is `4N`: `2N` topology multipliers + `2N` weight multipliers.

**Feasibility constraint:** `β_out_i · β_in_j < 1` for all `i ≠ j`.

**Implementation:** `dcms/models/qdecm.py` — `qDECMModel`

### 1.4 DECM — Directed Enhanced Configuration Model (binary + weighted, fully coupled)

The DECM constrains the same four sequences as the qDECM but is the **exact** maximum-entropy model: the weight multipliers `(β_out_i, β_in_i)` enter directly into the connection probability, making all four constraint equations **coupled**.

For each directed pair `(i,j)`, `i ≠ j`, the partition function is:

```
Z_ij = 1 + x_i · y_j · q_ij      where  q_ij = z_ij / (1 − z_ij),  z_ij = β_out_i · β_in_j
```

**Connection probability (coupled to weight parameters):**
```
p_ij = x_i · y_j · q_ij / (1 + x_i · y_j · q_ij)
     = sigmoid(−θ_out_i − θ_in_j − log(expm1(η_out_i + η_in_j)))
```

where `x_i = exp(−θ_out_i)`, `y_j = exp(−θ_in_j)`, `β_out_i = exp(−η_out_i)`, `β_in_j = exp(−η_in_j)`.

**Expected weight:** `E[w_ij] = p_ij · G_ij` where `G_ij = 1/(1 − z_ij)`.

**4N coupled equations:**
```
k_out_i = Σ_{j≠i} p_ij
k_in_i  = Σ_{j≠i} p_ji
s_out_i = Σ_{j≠i} p_ij · G_ij
s_in_i  = Σ_{j≠i} p_ji · G_ji
```

**Feasibility constraint:** `η_out_i + η_in_j > 0` for all `i ≠ j`.

**Key difference from qDECM:** in the qDECM approximation, `p_ij = x_i y_j/(1+x_i y_j)` is decoupled from `β`; in the exact DECM, `p_ij` depends on both `(θ, η)` simultaneously.

**Implementation:** `dcms/models/decm.py` — `DECMModel`

---


## 2. Solver Methods

All solvers return a `SolverResult` dataclass with fields `theta`, `converged`, `iterations`, `residuals`, `elapsed_time`, `peak_ram_bytes`, and `message`.

---

### 2.1 θ-Newton step

#### Rationale

The constraints are treated directly as a **nonlinear system** `F(θ) = 0` and solved with a **coordinate-wise Newton step** in θ-space:

```
Δθ_out_g = -F_out_g(θ) / (∂F_out_g / ∂θ_out_g)
```

where, for the DCM, `F_out_g = k_hat_out_g - k_out_g` is the residual of the out-degree constraint of group `g`, and the denominator is the diagonal element of the Jacobian:

```
∂F_out_g / ∂θ_out_g = -(Σ_h m_h p_gh (1 - p_gh) - p_gg (1 - p_gg))
```

(`m_h` = number of nodes of group `h`; "weighted sum minus one diagonal term" is the reduction of the per-node `Σ_{j≠i}`). Out-parameters are updated first and the fresh values are immediately used for the in-parameters (a Gauss-Seidel *ordering* of the Newton steps). The step is clipped to `[-max_step, +max_step]` to prevent large excursions near hubs. Because the update is additive in θ, the iteration never leaves the feasible domain (β > 1 is never produced), and hubs — whose large residual is divided by a large denominator — are handled gracefully.

For the DWCM/qDECM weight step, the coordinate Newton formula becomes:

```
Δη_out_g = (s_hat_out_g - s_out_g) / (Σ_h m_h p_gh · G_gh · (G_gh - 1) - diagonal term)
```

where `G = 1/(1 - β_g β_h)` is the geometric-distribution correction factor and `p` is the topology probability from the DCM step (for the qDECM, **including its diagonal**: the probability between two distinct members of the same group). The denominator is always negative (the Jacobian diagonal is negative-definite), so the step is in the correct descent direction.

For the DECM, the coupling between degree and strength equations modifies the strength Jacobian diagonal:

```
∂F_s_out_i / ∂η_out_i = −Σ_{j≠i} p_ij · G_ij² · (1 − p_ij + z_ij)
```

which equals the qDECM diagonal plus a correction `Σ p_ij · (1 − p_ij) · G_ij²` reflecting the dependence of `p_ij` on `η`. The DECM solver therefore uses alternating out-group / in-group GS-Newton passes that update both topology (θ) and weight (η) multipliers simultaneously: pass 1 updates (θ_out, η_out) from row sums; pass 2 updates (θ_in, η_in) from col sums.

**The z-floor mechanism:** define `z_ij = θ_out_i + θ_in_j`. When `z_ij → 0`, `G_ij → ∞` and the residual blows up. The solver maintains per-node floors `z_min_out[i]` and `z_min_in[j]` and applies a global floor from `min(θ_in)` over non-zero-strength nodes. This guarantees `z_ij > _Z_G_CLAMP` for all pairs after every Newton step.

**Numerical constants (tuneable):**

| Constant | Default | Role |
|----------|---------|------|
| `_Z_G_CLAMP` | `1e-8` (`1e-6` for DECM) | Minimum `z = θ_out + θ_in` before clamping |
| `_Z_NEWTON_FLOOR` | same as `_Z_G_CLAMP` | Hard floor on `z` after each Newton step |
| `_Z_NEWTON_FRAC` | `0.5` | Max fractional decrease of `z` per step (prevents period-2 oscillation) |
| `max_step` | `1.0` (DCM/DWCM/qDECM), `0.5` (DECM) | Max `|Δθ|` per coordinate per step (reduces for heterogeneous hubs) |
| `_ANDERSON_BLOWUP_FACTOR` | `100` (DCM), `5000` (qDECM), scale-adaptive (DECM) | Residual-jump ratio that triggers history clear |

#### Implementation

- `dcms/solvers/fixed_point_dcm.py` — `solve_fixed_point_dcm(theta0, k_out, k_in, ...)`
- `dcms/solvers/fixed_point_dwcm.py` — `solve_fixed_point_dwcm(theta0, s_out, s_in, ...)`
- `dcms/solvers/fixed_point_qdecm.py` — `solve_fixed_point_qdecm(theta_topo0, theta_weight0, k_out, k_in, s_out, s_in, ...)` (topology + weight) and `solve_fixed_point_qdecm_weight(...)` (weight step given a topology)
- `dcms/solvers/fixed_point_decm.py` — `solve_fixed_point_decm(theta0, k_out, k_in, s_out, s_in, ...)` (alternating out/in GS-Newton on the 4N vector)

Each takes the per-node sequences, groups the nodes internally, iterates on the M groups and returns a `SolverResult` expanded back to N nodes. The dense/chunked choice applies to the DECM reduced step only (chunked once `M` exceeds `DECM_LARGE_N_THRESHOLD = 2000`, so memory stays O(chunk × M)).

**Literature:**
Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM. Chapter 5.

---

### 2.2 Anderson acceleration

Anderson mixing (depth m = 10) transforms the plain Newton sequence into a quasi-Newton method by finding the linear combination of the last `m` residuals `r_k = g(θ_k) - θ_k` (here `g` is one Newton step) that minimises the mixed residual norm:

```
min_{c, Σc=1}  ‖Σ_k c_k · r_k‖²
```

The coefficients `c` are found by a small `m×m` least-squares system (O(m²) per step). The acceleration can reduce iteration counts by 5–50× on well-conditioned problems.

**Blowup protection:** if the Anderson iterate produces a residual jump > `_ANDERSON_BLOWUP_FACTOR × best_residual`, the history is cleared and the plain Newton step is used instead. This prevents one bad linear combination from ruining the run. When the Anderson mix violates the z-floor (`min(θ_out) + min(θ_in) < 0`), the mix is rejected, the plain Newton step is used and the history is cleared.

**Weighted mixing:** residuals are row-normalised by their component-wise maximum before solving the least-squares problem, so hub nodes (large absolute residuals) do not dominate the mixing coefficients; in the reduced system the rows are also weighted by the group multiplicities (a group of `m` identical nodes counts `m` times, exactly as in the per-node system).

**Best-θ tracking:** the result always returns the lowest-residual iterate seen, not the final one.

**Literature:**
Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735. https://doi.org/10.1137/10078356X

---

### 2.3 Hub bisection — exact 1D solve for high s/k nodes

For networks with a few nodes whose **strength-to-degree ratio** `s_i / k_i ≫ 1` (e.g. a node with 2 out-links but total out-strength of 150), the global Newton step tends to drive `β_i → 1` (qDECM) or `η_i → 0` (DECM) and becomes numerically unstable.  Mathematically a solution always exists; the issue is purely numerical.

The `hub_sk_threshold` parameter activates an exact per-node solver for those nodes.  At each outer iteration, before or after the global Newton step:

1. **Identify hub nodes**: any node `i` with `s_i / k_i > hub_sk_threshold` (independently for out and in directions).
2. **1D bisection** (60 steps, precision ≈ 2⁻⁶⁰): for each hub out-node `i`, find `β_i` (qDECM) or `η_i` (DECM) exactly such that the strength equation is satisfied, treating all other parameters as fixed.
   - *qDECM*: `f(β_out_i) = Σ_j p_ij / (1 − β·β_in_j) = s_out_i`.  `f` is strictly increasing in `β` → unique root on `[0, 1/max(β_in_j))`.
   - *DECM*: `f(η_out_i) = Σ_j p_ij(η) · G_ij(η) = s_out_i`.  `f` is strictly decreasing in `η` (both `p_ij → 0` and `G_ij → 1` as `η → ∞`) → unique root on `[_ETA_MIN, _ETA_MAX]`.
3. **3-sweep Gauss-Seidel consistency**: after bisecting all out-hubs with current `β_in` / `η_in`, bisect all in-hubs with the fresh `β_out` / `η_out`, then repeat for a total of 3 passes.  This ensures that nodes appearing in *both* the out-hub and in-hub lists have a fully consistent solution.
4. **Anderson interaction guard**: hub components of the Anderson residual `r_k` are zeroed (so mixing weights focus on non-hub convergence), and after mixing the hub components of `θ_next` are overwritten with the bisection values (preventing Anderson from corrupting them).

**When to use:** networks where `max(s/k)` exceeds ~5 and the solver stagnates at MRE around 0.5–0.9.  `hub_sk_threshold=5` captures all meaningful hubs on real-world networks tested.  The default `hub_sk_threshold=0.0` disables the feature entirely, leaving the solver unchanged for standard cases.

**Performance note:** each bisection call is O(N) per hub node; with 3 sweeps and typically O(1–100) hubs the overhead is negligible compared to the main O(N²) matrix computation.  Recommended `tol=1e-4` (rather than the default `1e-6`) since oscillation near the solution is expected at very fine scales.

**Example — real-world network with high s/k hubs:**

```python
model = qDECMModel(k_out, k_in, s_out, s_in)
converged = model.solve_tool(
    hub_sk_threshold=5,   # activate bisection for nodes with s/k > 5
    tol=1e-4,             # looser tolerance; bisection converges to ~1e-4
    max_time=600,         # hard time limit
    verbose=True,
)
print(model.sol.mre)
```

---

### 2.4 Options removed on 2026-10-27

The following existed in earlier versions and are gone (passing them is a `TypeError`):

| Removed | Why |
|---|---|
| per-node (unreduced) solver, `reduce_degeneracy` | The degeneracy reduction is analytically exact (§2.5) and was cross-checked against the per-node solver when it was introduced; nothing in this project used the per-node path. Consistency is now tested against the per-node *residuals*, see `tests/test_reduction_consistency.py`. |
| `variant="gauss-seidel"` (β-space fixed point) | Not implemented for the reduced system; the θ-Newton iteration is what all results use. |
| `backend="numba"`, `num_threads` | Never used (the reduced systems have M ≤ ~7 000 groups, far below the old N > 100 000 switch). |
| `backtracking_gamma` (line search) | Available only on the per-node path. |
| `bisection_kick_*`, `stall_block_newton_first` | The structured "kick" tier of the stagnation ladder never helped (an 8 h run on one of the hardest networks gained ~1%). |

---

### 2.5 Degeneracy reduction — collapsing symmetric nodes into groups

Two nodes that share the exact same **sufficient statistics** — the observed sequences that pin down the model's Lagrange multipliers — provably have **identical multipliers at the true solution**. This follows from a label-swap symmetry argument: swapping the labels of two such nodes leaves the log-likelihood unchanged, and since the log-likelihood is strictly concave, its unique maximiser must be symmetric under that swap (Vallarano, N. et al. (2021), *Scientific Reports*, 11, 15227).

| Model | Sufficient statistics (degeneracy key) |
|---|---|
| DCM | `(k_out, k_in)` |
| DWCM | `(s_out, s_in)` |
| qDECM (conditioned-weight step only — the topology step is a plain DCM) | `(k_out, k_in, s_out, s_in)` |
| DECM | `(k_out, k_in, s_out, s_in)` |

On real-world networks with heavy-tailed degree/strength distributions, most of the tail shares a small number of distinct low-degree/low-strength value combinations, so the number of **groups** `M` (unique keys) is often far smaller than the number of nodes `N`. Solving for `M` group-level unknowns instead of `N` per-node unknowns turns the O(N²) pairwise compute into O(M²) — measured **10–67× faster** on real networks (see table below), and the group-level residual equals the per-node residual of the expanded parameters **identically** (checked at random points, with groups of multiplicity > 1, in `tests/test_reduction_consistency.py`).

> **All four models have a genuine gauge freedom**, not just DCM. `p_ij`/`W_ij` always enter through the additive combination `θ_out_i + θ_in_j` (degree part) and/or `η_out_i + η_in_j` (strength part, DWCM/qDECM-weight/DECM), so `θ_out += c, θ_in -= c` — and, separately, `η_out += c, η_in -= c` where a strength part exists — leaves every `p_ij`/`W_ij` unchanged. **DECM has two independent gauge directions** (θ-shift and η-shift). Two solutions of the same problem are therefore not expected to coincide in raw θ: always compare via a gauge-invariant quantity (`p_ij` and/or `E[w_ij]`), never raw θ.

> **Correctness note (2026-10-27).** Until this date the qDECM weight step dropped the pairs between two *distinct members of the same group* (the group-level topology matrix had its diagonal zeroed instead of keeping the real probability), so for groups with more than one node the solver converged on a slightly wrong system: its own (group-level) MRE reached 1e-5 while the per-node strength MRE of the expanded solution was ~3e-3–6e-3 on the real networks tried. This had been attributed to "gauge freedom" in an earlier version of this section; it was found by the per-node residual check above. DCM, DWCM and DECM were checked the same way and were correct.

**Measured on empirical data from online social networks** (`tol≈1e-9/1e-10`, `theta-newton`, `anderson_depth=10`; network labels below are anonymized, distinct letters = distinct networks):

| Model | Network | N → M | Speedup |
|---|---|---|---|
| DCM | online social network A | 1304 → 61 | 24.2× |
| DCM | online social network B | 20914 → 2310 | 43.5× |
| DWCM | online social network C | 15168 → 1860 | 47.1× |
| DWCM | online social network D | 31874 → 3083 | 66.8× |
| qDECM (weight step) | online social network B | 20914 → 4070 | 23.5× |
| DECM | online social network C | 15168 → 3003 | ~10–25× |

**Status:** always on — there is no other path. The standalone functions group the nodes themselves:

```python
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm

model = DCMModel(k_out, k_in)
theta0 = model.initial_theta("degrees")

result = solve_fixed_point_dcm(theta0, k_out, k_in, tol=1e-9, max_iter=2000, anderson_depth=10)
# result.theta / result.best_theta are expanded back to shape (2N,)
print(result.converged, result.mre, result.message)  # message reports "N=... -> M=..."
```

`solve_fixed_point_dwcm(theta0, s_out, s_in, ...)`, `solve_fixed_point_qdecm(theta_topo0, theta_weight0, k_out, k_in, s_out, s_in, ...)` and `solve_fixed_point_decm(theta0, k_out, k_in, s_out, s_in, ...)` follow the same pattern. `solve_fixed_point_decm` additionally supports `hub_sk_threshold` (§2.3) and `weight_anderson` (multiplicity-aware Anderson mixing, on by default — a degeneracy class of `mult` identical nodes must count `mult` times in the Anderson least-squares fit).

### 2.6 Perturbed restart on stagnation / repeated blowup (DECM)

Even with Anderson acceleration and the blowup guard (§2.2), some hub-heavy instances get stuck at a **quasi-fixed point**: progress halts for hundreds of iterations, or the Anderson blowup guard keeps tripping and rolling back to the same point without ever improving on it. Plain rollback-and-retry alone cannot escape a genuine trap — the deterministic Newton/Anderson dynamics just reproduce the same trajectory. The fix is to occasionally add noise:

- **Stagnation:** if `best_theta_res` has not set a *significant* record (more than `patience_rel_tol` better than the reference) for `patience` consecutive iterations, the solver climbs a stall ladder instead of continuing from the stuck iterate: soft reset from `best_theta`, then block-Newton ON, then restarts from `best_theta` plus Gaussian noise (see the *Stall ladder* paragraph below).
- **Repeated blowup:** the existing blowup guard's plain rollback-to-`best_theta` is left untouched for an *isolated* blowup (common and self-resolving on hub-heavy networks — see `blowup_factor` below). Only if the guard trips **again with no record improvement in between** does it escalate to the same noisy restart — this is the "quasi-fixed-point trap" case, not routine Anderson housekeeping.

Both triggers share **one** escalation counter: noise scale starts at `noise_base` and doubles on each consecutive restart that fails to improve the record (`noise_base × min(2^(restarts-1), noise_cap_mult)`), capped at `noise_base × noise_cap_mult`, and resets to `noise_base` only on a genuine improvement. (An earlier, external version of this mechanism kept blowup-retries and stagnation-retries on separate counters, one of them with a *fixed*, non-escalating noise scale — that combination produced a real infinite loop, 59 consecutive chunks converging on the exact same stuck point, on the network below. A single shared, always-escalating counter is what actually fixed it.) Give up (`converged=False`) after `max_stalls` restarts *at the noise cap* in a row without improving the record — i.e. only once the strongest available kick has been tried repeatedly and still hasn't helped; restarts made while still escalating don't count toward this limit.

**The restart noise is multiplicative (log-scale), not additive, on the *entire* theta vector**: every component of `[θ_out|θ_in|η_out|η_in]` is scaled as `x_i *= exp(N(0, noise_base))`. This matters on hub-heavy networks: a topological hub's `|θ_out_i|` can sit near the `±_THETA_MAX` boundary (fixed-scale additive noise too weak to matter there), while a strength hub's `η_i` can sit near `_ETA_MIN` — and the model is extremely sensitive to *absolute* changes in η near zero (`G ≈ 1/η`, `log_q ≈ -log(η)`), so fixed-scale additive noise there is catastrophic instead. (An earlier version of this mechanism used plain additive noise on the whole vector; on empirical online social network data (hub-heavy, tens of thousands of nodes — see §2.5's network A-D for size references), 2026-07-24, this made `MRE_weights` jump to ~1×10⁵ on the very first post-restart step — bug found and fixed together with the point below.)

**The blowup guard gives every restart a one-iteration grace period.** Right after a perturbed restart, the very first evaluated residual is the raw noisy point itself — deliberately far from the record by design — and must not be judged as a "blowup" against the pre-restart optimum. Without this grace, the guard's own trigger (calibrated for small steady-state Anderson excursions) could kill a restart's recovery before Newton/Anderson got even one real chance to reconverge, escalating straight to a bigger-noise restart instead: observed on an empirical online social network (N≈15 168, several nodes with a very high strength-to-degree ratio — hub nodes; the "network C" of §2.5's table), where two genuine post-restart Newton steps brought the residual down 6.6× (0.397 → 0.060) before the guard, still anchored to the ancient record, discarded that progress and forced restart #2. The fix re-anchors the guard's reference (`_best_res_for_anderson`) to the restart's own residual for that one grace iteration, so later checks judge progress *since the restart*, not distance from history.

**The isolated-blowup rollback applies the same hub-eta correction a normal step gets.** On hub-heavy networks (`hub_sk_threshold > 0`), every regular iteration corrects hub nodes' η exactly via 1D bisection instead of the (less stable) global Newton step. The rollback branch used to skip this — its one-step `theta_rb = _step(best_theta)` bypassed the hub-bisection block entirely, leaving hub η only as accurate as one global Newton step right at the moment recovery needs it most. Fixed 2026-07-24 by factoring hub-bisection into a shared helper applied in both places.

**A restart that actually fires resets the "isolated vs. repeated" clock.** The distinction between an isolated blowup (cheap rollback) and a repeated one (escalate) is tracked by "has there been a record improvement since the last intervention"; a successful perturbed restart used to *not* count as an intervention for this purpose, so it never reset the clock — meaning the very next blowup after a restart, however far downstream and however much clean progress happened first, was always misread as "still the same trap" and escalated immediately, regardless of how well that restart had actually gone. Observed on the same network, 2026-07-24: restart #1 ran cleanly for 95 iterations with zero blowups, yet the blowup that eventually occurred was still classified "repeated" and jumped straight to restart #2 with doubled noise — with every later restart's recovery ceiling landing further from the record than the last (3.6×10⁻⁴ → 4.0×10⁻⁴ → ... → 1.4×10⁻³), never beating it. Fixed by resetting the "no progress" flag on a restart that actually fires, giving the resulting trajectory its own fresh isolated-blowup grace instead of inheriting the trap's history. Together with the hub-bisection fix above, this took the same reproduction from 9 cascading restarts giving up in 24 iterations down to 1 restart followed by 479 iterations of clean, continuously record-improving recovery.

**The blowup guard's sensitivity is tunable via `blowup_factor`.** A rollback triggers when the current iteration's residual exceeds `blowup_factor` times the best residual ever seen this call (a *running minimum*, so this catches a slow multi-hundred-iteration drift away from the record just as well as a sudden spike).

Default `None` uses the built-in scale-adaptive formula `eff_blowup = max(200, min(5000, 200_000/N))`:

- **N ≤ 40:** `eff_blowup = 5000`, the cap. On tiny networks a wild Anderson excursion is cheap to recover from and the search space is small enough that Anderson can still land on a good iterate after one — being very permissive lets it keep exploring instead of rolling back prematurely.
- **40 < N < 1000:** `eff_blowup` slides from 5000 down to 200, inversely proportional to N.
- **N ≥ 1000:** `eff_blowup = 200`, the floor. On large networks a residual cascade corrupts the Anderson history and can take hundreds of iterations to work itself out on its own, so intervention should trigger sooner — but 200, not lower, is itself an empirical finding, not a guess (see below).

Pass an explicit value to override this — but the two directions call for opposite instincts, and picking the wrong one defeats this whole section's mechanism:

- **Lower it (e.g. `20`-`50`)** only for instances that visibly *wander* far from their best point over many iterations without any single jump large enough to trip the default floor. There, more frequent, cheap (noise-free) rollbacks beat waiting for `patience` iterations to force a (noisy) restart.
- **Do not reflexively lower it on a network that's stuck in a quasi-fixed-point trap** (§ above) — that's the opposite failure mode, and a low `blowup_factor` makes it *worse*: it rolls back so eagerly that the trajectory never gets the room to cross the unstable region and escape on its own via plain Newton/Anderson dynamics, forcing every recovery attempt through the noisy-restart path instead (which then has its own chance to fail and escalate, §§ above). Concretely, on the same network as above (N=15 168, several hub nodes with very high strength-to-degree ratio; trapped `best_theta=3.436×10⁻⁴`), `blowup_factor=20` — a value carried over from an earlier, different instance that needed frequent cheap rollbacks — ran 2 000 iterations without escaping the trap (best MRE stuck at 2.42×10⁻⁴, repeated blowups throughout). Removing the override (`blowup_factor=None`, i.e. the scale-adaptive default of 200 for this N) converged the *identical* run in 1 205 iterations to MRE=9.24×10⁻⁶ — no restart of any kind fired; the extra headroom alone let the trajectory cross the unstable region it had been bouncing off of.

**With `verbose=True`, every intervention prints a message** — `[blowup] ...` when the guard trips (one line for an isolated rollback, another when it escalates), and `[perturbed-restart] restart #N ...` for both stagnation- and blowup-triggered noisy restarts — so an unattended run's log shows exactly when and why the trajectory was redirected, not just silent MRE fluctuation.

**Stall ladder (solve_tool defaults, 2026-10):** a *significant* record = beating the reference by more than `patience_rel_tol` (0.1). After `patience` (250) iterations without one: (1) soft reset from `best_theta` with Anderson cleared; (2) block-Newton switched ON (`stall_block_newton_gate`, 0.05; switched OFF again once the record is below `stall_block_newton_off_mre`, 2e-4, if it was switched on above it; and reverted OFF if its own patience window yields no significant record, `stall_block_newton_revert=True`); (3) noisy restarts, then STOP after `max_stalls`. The standalone solver functions keep the old defaults (patience 750, `patience_rel_tol=0`, block-Newton tier off). Validated from scratch with one recipe on a set of real directed weighted networks (N from ~10³ to ~10⁵): four networks that the old defaults never converged within 6-10 h now converge in 1.2-3.1k iterations, four more need 2-8x fewer iterations, and the rest are unchanged. A hard case stalls at a 3.3e-3 plateau with the stall ladder alone (a handful of out-side rows with `s ≈ k` and `k_in ≈ 0`, one `(node, side)` holding the maximum for hundreds of iterations), but converges from zero adding `streak_fix_threshold=10` (723 it, 3.5 min, 2 threads) or `20` (1752 it); `50` fires too late and does not.

**Status:** on by default (`DECMModel.solve_tool()` and the standalone `solve_fixed_point_decm`). Inert for well-behaved instances that never stagnate or blow up — no default-value tuning is needed to get the old behaviour on easy networks. Set `patience<=0` to fully disable and restore the old fail-fast-on-stagnation behaviour.

```python
model = DECMModel(k_out, k_in, s_out, s_in)
model.solve_tool(
    max_iter=20000,
    blowup_factor=None,   # None = scale-adaptive default; lower (e.g. 20-50) to catch slow drift sooner
    patience=250,        # stall ladder after 250 iterations without a significant record (solve_tool default; solver default 750)
    noise_base=1e-2,      # first restart's noise scale (multiplicative, see above)
    noise_cap_mult=16.0,  # noise saturates at noise_base * 16
    noise_growth=2.0,     # noise doubles per consecutive failed restart; lower (e.g. 1.2-1.5) if escalation overshoots
    max_stalls=5,         # give up after 5 restarts at max noise with no improvement
    seed=None,             # set an int for reproducible restarts (irrelevant if none fire)
)
```

**Validated on a real hard instance:** DECM on the same empirical online social network as above (N=15 168, M=3 003 after degeneracy reduction, §2.5, several hub nodes), run unattended from scratch with `anderson_depth=10`, `hub_sk_threshold=5.0`, `patience=750` (the then-default, translated from the value validated in the external checkpointed-runner prototype, §3.10.1; `solve_tool` now defaults to `patience=250` with `patience_rel_tol=0.1` and the block-Newton stall tier, which was added later and was not part of this run) — one repeated-blowup restart and four stagnation restarts fired, and the run converged after 9 460 iterations to MRE=9.45×10⁻⁶, with no manual intervention.

---

### 2.7 GPU acceleration — the `device` parameter

Every model's `solve_tool()` (and the underlying `solve_fixed_point_*` functions, §3.8) accepts `device: str = "cpu"`. Passing a torch device string (`"mps"` on Apple Silicon, `"cuda"` on an NVIDIA GPU) runs the entire Newton/Anderson iteration — dense **and** chunked (§2.1's crossover) — on that device instead of the CPU.

**Precision:** CPU always uses float64 (unchanged, exact). Non-CPU devices use **float32**: MPS has no float64 support at all (a hard PyTorch limitation, not a choice made here), and CUDA float64 throughput is slow enough that float32 is the practical choice there too. `SolverResult` (`self.sol`) is always converted back to CPU/float64 before being returned, regardless of `device` — the float32 arithmetic is purely an internal, iteration-time detail.

**What this means in practice:** GPU runs will not reach the same `tol` as CPU — float32 has roughly 7 significant decimal digits, so expect convergence to plateau somewhere in the `1e-6`–`1e-7` MRE range even on an easy network, rather than continuing down to `1e-9`/`1e-10`. This is normal, not a bug. GPU is a speed trade for networks large enough that the O(M²) (dense) or O(chunk×M) (chunked) per-iteration cost dominates — for small/medium `M` the fixed overhead of moving data to the device usually outweighs any benefit, and CPU is both faster and more precise.

**Requirements:** none beyond a torch device — every solver runs the reduced system, which is the only one supported on a non-CPU device.

```python
model = DECMModel(k_out, k_in, s_out, s_in)
model.solve_tool(
    tol=1e-6,             # loosen vs. the CPU default -- float32 won't reach 1e-9
    device="mps",         # or "cuda"; default "cpu"
)
```

**What has been tested:** `device="mps"` on Apple Silicon was run during development and is exercised by `tests/test_device.py` (all four models, small problem, `tol=1e-3`, compared with the CPU solution). `device="cuda"` is the same torch code path (a device string, float32) and is **potentially working but has never been run**: no CUDA hardware was available, so its tests are skipped here and are the first thing to run on a machine with an NVIDIA GPU. Note for qDECM: its topology stage uses `tol * 1e-3`, which for `tol < ~1e-3` is below the float32 plateau (~2e-7), so on a non-CPU device it reports `converged=False` (with a good solution) unless `tol >= 1e-3`.

**Status:** available on all four models (DCM, DWCM, qDECM, DECM), including DECM's chunked degeneracy-reduced step (used automatically once `M` exceeds `DECM_LARGE_N_THRESHOLD=2000` — i.e. every large real network). DECM's stagnation-recovery mechanisms (hub bisection, perturbed restart, §2.6) remain internally CPU-only by design and interoperate transparently with a non-CPU main-loop `device` — theta is moved to/from the right device automatically at each boundary, with no action needed from the caller.

---

### 2.8 Argmax-streak targeted fix — breaking a single equation's monopoly on the residual (DECM)

On some networks, once every other equation is well below tolerance, the same single `(node, side)` pair stays the argmax of the residual for many iterations in a row — the global Newton/Anderson step keeps nudging it, but too slowly, because that pair's own `(θ, η)` equations are ill-conditioned relative to the rest of the network (typically a node paired with a hub, or otherwise near a boundary). `streak_fix_threshold` breaks this monopoly by solving that one pair directly instead of waiting for the global step to finish it off.

**Mechanism:**
1. Each iteration, track which `(node, side)` (`side` = out or in — a node's `k` and `s` equations on one side share the same `(θ, η)` pair, so they're keyed together, not separately) is the argmax of the relative residual.
2. If the *same* key has been the argmax for `streak_fix_threshold` consecutive iterations, run `_targeted_bisection_fix`: solve the `k`-equation for `θ` and the `s`-equation for `η` **at fixed `φ = θ + η`**, via nested bisection (`streak_fix_n_sweeps` outer sweeps × `streak_fix_n_bisect` bisection halvings each), for both the offending side and — since fixing one side can shift the other's balance — the opposite side too.
3. Anderson history is cleared afterward (the fix is a discontinuous jump, not a step the mixing history should extrapolate from).

**Why `φ`-fixed, not `θ`-fixed:** an earlier version alternated solving `θ` with `η` fixed, then `η` with `θ` fixed — this failed to converge whenever `η` was large (the two 1D searches fight each other along the same effective direction). Solving at fixed `φ = θ + η` instead decouples the pair cleanly and converges in a handful of sweeps regardless of scale.

**Negative-`η` support:** the bisection bracket for `η` includes negative values for nodes that are "unreachable" under the plain `η ≥ 0` rule (see §2.3's hub bisection) but satisfy the true constraint `η_out_i + η_in_j > 0` for every existing pair — the same relaxation the hub bisection uses, applied here to non-hub nodes that get stuck.

**When to use:** `streak_fix_threshold=0` (default) disables it — inert on networks that don't have this failure mode. When a network's residual visibly stalls with one equation dominating for a long stretch, values from `1` (fix immediately, every time the same pair repeats) to `50` (fix only after a long monopoly) have all been the deciding factor on different real networks; there is no single value that's right for every network, and this is exactly the kind of knob §3.4's `solve_tool()` now exposes so it can be tuned per network rather than guessed.

**Combine with:** `hub_bisect_max_sweeps` (raise from the default 3 only when a relaxed hub sits next to a very tight partner, `z = η_out+η_in ~ 1e-4` — the out/in alternation of the ordinary hub bisection converges very slowly there; values above ~10 risk an unsafe extrapolation of the sweep sequence, don't raise it blindly) and `patience_rel_tol` (> 0 makes the stagnation-recovery ladder of §2.6 treat a record that's creeping down by a tiny amount every iteration — e.g. by ~1e-8 per step over thousands of iterations, which never triggers the *default* `patience_rel_tol=0`'s "any improvement resets the counter" rule — as a genuine stall, entering recovery instead of silently crawling forever).

```python
model = DECMModel(k_out, k_in, s_out, s_in)
converged = model.solve_tool(
    streak_fix_threshold=50,   # fix a (node, side) pair directly after it dominates the argmax for 50 iterations
    streak_fix_n_sweeps=5,     # phi-fixed bisection sweeps per fix (default)
    streak_fix_n_bisect=60,    # bisection halvings per sweep (default)
    hub_bisect_max_sweeps=10,  # raise only if a relaxed hub / tight-partner pair converges slowly
    patience_rel_tol=0.1,      # only a >10% record improvement resets the stagnation counter (solve_tool default)
)
```

---

### 2.9 Joint 2×2 block-Newton step for nearly saturated rows (DECM)

Even with every mechanism above, a handful of real networks still plateaued: after degeneracy reduction, a family of nodes with `s ≈ k` (every out-link carries essentially weight 1) kept the residual pinned at a fixed floor no amount of extra time moved. Tracing one of these plateaus down to the group level found the actual cause: the Newton step solves `θ` from the `k`-equation and `η` from the `s`-equation as two **decoupled** scalar Newton steps (ignoring the cross-derivative between them). For a row with `s ≈ k` — where the weight factor `G = 1/(1-β_out β_in) ≈ 1` — the two equations are *almost the same function*, so the exact 2×2 Hessian block `[[A,B],[B,C]]` of that row is nearly singular: `det/(A·C) = 1 - ρ² ~ 1e-4`–`1e-5` on the affected rows. The decoupled scalar iteration only contracts by `ρ ≈ 1 - 1e-5` per step along that row's own soft direction — thousands of iterations per e-fold of progress, which looks exactly like a stall.

**Mechanism (`block_newton_gate`):** for a row whose `det/(A·C)` falls below `block_newton_gate`, solve the exact joint 2×2 Newton step for `(θ, η)` instead of the two decoupled scalar ones (written in the numerically stable `g = G-1` form to avoid cancellation in `A·C - B²`; jointly clipped to `max_step` and sharing the pair-sum `η ≥ z_clamp` floor's scaling, so it can't push the pair below the feasibility floor either). `block_newton_gate=0.0` (default) reproduces the old decoupled step bit-for-bit. Hub-owned rows (§2.3) are always excluded — their `η` already belongs to the hub bisection.

**The failure mode this alone does NOT fix — and why `block_newton_min_rel` exists:** a row with `s == k` *exactly* has no finite solution — its true fixed point lies at `θ → -∞, η → +∞` along the soft direction `(θ - c, η + c)`, with the residual only ever approaching zero in the limit. Once such a row's residual is already small, the joint step has nothing stopping it from continuing to march that row along the soft direction at `max_step` per iteration, forever, long past any useful precision — and a handful of rows doing this simultaneously destabilizes their neighbours (`θ` reaching -30, `η` reaching +37 was traced directly on one real network before the residual exploded past 1). `block_newton_min_rel` (default `0.1 * tol` whenever the gate is active, `0.0` disables the restriction) excludes rows whose relative residual is already below that value from the joint step, leaving them to the harmless (if slow) scalar step once they're that close.

**⚠️ Honest status — this is opt-in per network, not a default, and `block_newton_gate` alone (without `min_rel`) is not safe:**
- On a 30-iteration stability check from several already-converged real networks, a bare `block_newton_gate=0.05` (no `min_rel`) made 4 of them **worse**, some substantially (one network's residual rose from converged to a transient peak of 8.8×10³ before settling back down; another rose to 4.9×10⁻¹ and had *not* recovered by iteration 30) — and diverged a different, previously-easy network within a 10-minute fresh-start run that converges cleanly with the gate off.
- With `min_rel` added, the two plateaued real networks this was built for both converged cleanly (see below) — but the full stability check above has **not** been re-run with `min_rel` active, so a blanket default is not yet justified either way.
- **Recommendation:** treat `block_newton_gate`/`block_newton_min_rel` as a targeted intervention for a network you've confirmed is stuck on this specific failure mode (a `hub_sk_threshold`-independent residual plateau dominated by `s ≈ k` rows), not something to turn on everywhere. Always pair a non-zero `block_newton_gate` with `block_newton_min_rel` (its default already does this). The exclusion can also hurt: on one network the default `0.1·tol` kept rows that still needed the joint step out of it and the run blew up from zero, while `block_newton_min_rel=0.0` converged in a few thousand iterations; if a run with the gate on fails this way, try `0.0`.

**Validated on two real networks that had resisted every other mechanism in this README** (2026-09-28):
- Network A (N=22 754): plateaued at MRE=3.5175×10⁻⁵. `block_newton_gate=0.05` alone (converged too fast, 46 iterations, to observe the runaway-row failure mode above) reached **MRE=9.35×10⁻⁶**.
- Network B (N=107 056): plateaued at MRE=1.8513×10⁻⁴ — a larger, more dominant `s≈k` family than network A's. `block_newton_gate=0.05` alone repeatedly diverged via the runaway-row mode within ~35-100 iterations (traced and confirmed identical at `anderson_depth=1` and `anderson_depth=3`, ruling out an Anderson-mixing artifact). Adding `block_newton_min_rel=1e-6` (the same value the default formula gives at `tol=1e-5`) converged cleanly in 94 iterations to **MRE=9.96×10⁻⁶**.

```python
model = DECMModel(k_out, k_in, s_out, s_in)
converged = model.solve_tool(
    hub_sk_threshold=2.0,        # this network's other stuck rows are hub-shaped too; tune independently
    streak_fix_threshold=1,      # often combined with the streak-fix of §2.8
    block_newton_gate=0.05,      # opt in only once you've confirmed a s~k plateau -- see the caveat above
    block_newton_min_rel=None,   # None = 0.1*tol whenever the gate is active (recommended); 0.0 = no restriction (try it if None stalls, see above)
)
```

---

## 3. API Reference

All three models expose a unified `solve_tool()` method.  Instantiate with the observed sequences, call `solve_tool()`, and inspect the stored result.

### 3.1 DCM — `DCMModel`

```python
from dcms.models.dcm import DCMModel

model = DCMModel(k_out, k_in)
converged = model.solve_tool(
    ic="degrees",           # "degrees" (default), "random" — or pass an array (warm start)
    tol=1e-6,               # convergence tolerance (ℓ∞ relative residual MRE)
    max_iter=2000,
    max_time=0,             # wall-clock timeout in seconds (0 = no limit)
    anderson_depth=10,
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    device="cpu",           # "cpu" (default, float64) or "mps"/"cuda" (float32; see §2.7)
)
theta = model.sol.theta     # converged parameters, shape (2N,) -- always CPU/float64 regardless of device
```

Additional model methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `model.pij_matrix(theta)` | `(N, N)` tensor | Link-probability matrix `p_ij = x_i y_j / (1 + x_i y_j)` |
| `model.residual(theta)` | `(2N,)` tensor | Constraint violation `F(θ)` |
| `model.neg_log_likelihood(theta)` | float | Negative log-likelihood `−L(θ)` |
| `model.bic(theta)` | float | Bayesian Information Criterion, `2N·ln(N(N−1)) − 2·ln L` |
| `model.constraint_error(theta)` | float | `max‖F(θ)‖` |
| `model.initial_theta(method)` | `(2N,)` tensor | Initial guess: `"degrees"` (default) or `"random"` |
| `model.sample(seed, chunk_size)` | `list[[i,j]]` | Sample a binary network from the fitted DCM (see §3.5) |

### 3.2 DWCM — `DWCMModel`

```python
from dcms.models.dwcm import DWCMModel

model = DWCMModel(s_out, s_in)
converged = model.solve_tool(
    ic="strengths",         # "strengths" (default), "random" — or pass an array (warm start)
    tol=1e-6,
    max_iter=2000,
    max_time=0,
    anderson_depth=10,
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    device="cpu",           # "cpu" (default, float64) or "mps"/"cuda" (float32; see §2.7)
)
theta = model.sol.theta     # converged parameters, shape (2N,) -- always CPU/float64 regardless of device
```

Additional model methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `model.wij_matrix(theta)` | `(N, N)` tensor | Expected weight matrix `w_ij = β_i β_j / (1 − β_i β_j)` |
| `model.pij_matrix(theta)` | `(N, N)` tensor | Link-existence probability `p_ij = P(w_ij > 0) = β_i β_j` |
| `model.residual(theta)` | `(2N,)` tensor | Constraint violation `F(θ)` |
| `model.neg_log_likelihood(theta)` | float | Negative log-likelihood `−L(θ)` |
| `model.bic(theta)` | float | Bayesian Information Criterion, `2N·ln(N(N−1)) − 2·ln L` |
| `model.constraint_error(theta)` | float | `max‖F(θ)‖` |
| `model.max_relative_error(theta)` | float | `max‖F_i‖ / s_i` |
| `model.initial_theta(method)` | `(2N,)` tensor | Initial guess (see below) |
| `model.sample(seed, chunk_size)` | `list[[i,j,w]]` | Sample a weighted network from the fitted DWCM (see §3.5) |
| `model.p_value_calculator(edgelist)` | structured array `[(source_id, target_id, p_value)]` | Edge-weight p-values under the fitted DWCM (see §3.6) |

`initial_theta` methods for DWCM:

| Method | Description |
|--------|-------------|
| `"strengths"` (default) | `β ≈ sqrt(s / (s + N − 1))`, mean-field approximation |
| `"normalized"` | `β_out_i ∝ s_out_i / Σ_j s_out_j` (fractional share of total weight) |
| `"uniform"` | All β equal to the median of the `"strengths"` approximation |
| `"random"` | Uniform random `θ ∈ [0.1, 2.0]` |

### 3.3 qDECM — `qDECMModel`

```python
from dcms.models.qdecm import qDECMModel

model = qDECMModel(k_out, k_in, s_out, s_in)
converged = model.solve_tool(
    ic_topo="degrees",      # topology init: "degrees" (default) or "random" — or pass an array
    ic_wei="topology",      # weight init: "topology" (default) or "random" — or pass an array
    tol=1e-6,
    max_iter=2000,
    max_time=0,
    anderson_depth=10,
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    hub_sk_threshold=0.0,   # >0: use 1D bisection for nodes with s/k > threshold (see §2.3)
    z_clamp=1e-8,           # floor on the weight step's z; exposed for symmetry with DECM's own z_clamp, don't change lightly
    device="cpu",           # "cpu" (default, float64) or "mps"/"cuda" (float32; see §2.7)
)
# solve_tool returns True if *both* topology and weight steps converged
# sol.theta has shape (4N,): [θ_out_topo, θ_in_topo, θ_β_out, θ_β_in]
theta_topo   = model.sol.theta[:2*N]   # topology parameters
theta_weight = model.sol.theta[2*N:]   # weight parameters
```

Additional model methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `model.pij_matrix(theta_topo)` | `(N, N)` tensor | DCM link-probability matrix |
| `model.wij_matrix_conditioned(theta_topo, theta_weight)` | `(N, N)` tensor | Expected weight matrix |
| `model.residual_strength(theta_topo, theta_weight)` | `(2N,)` tensor | Strength constraint violation `F_w(θ)` |
| `model.neg_log_likelihood_strength(theta_topo, theta_weight)` | float | Negative log-likelihood of the weight model |
| `model.neg_log_likelihood(theta_topo, theta_weight)` | float | Generalized log-likelihood, −L_topo + −L_w (see docstring; comparable to DECM's) |
| `model.bic(theta_topo, theta_weight)` | float | Bayesian Information Criterion on the generalized likelihood, `4N·ln(N(N−1)) − 2·ln L_generalized` |
| `model.constraint_error_topology(theta_topo)` | float | Max-abs degree constraint error |
| `model.constraint_error_strength(theta_topo, theta_weight)` | float | Max-abs strength constraint error |
| `model.max_relative_error(theta_topo, theta_weight)` | float | Max relative error over all 4N constraints |
| `model.initial_theta_topo(method)` | `(2N,)` tensor | Topology initial guess (`"degrees"` or `"random"`) |
| `model.initial_theta_weight(theta_topo, method)` | `(2N,)` tensor | Weight initial guess (see below) |
| `model.sample(seed, chunk_size)` | `list[[i,j,w]]` | Sample a weighted network from the fitted qDECM (see §3.5) |
| `model.p_value_calculator(edgelist)` | structured array `[(source_id, target_id, p_value)]` | Edge-weight p-values under the fitted qDECM (see §3.6) |

`initial_theta_weight` methods for qDECM:

| Method | Description |
|--------|-------------|
| `"topology"` (default) | `β = sqrt(1 − k/s)`, mean-field inversion of `s = k / (1 − β²)` |
| `"topology_node"` | Per-node Newton solve (5 iterations, chunked); uses p_ij from DCM to give the most accurate starting point |

### 3.4 DECM — `DECMModel`

```python
from dcms.models.decm import DECMModel

model = DECMModel(k_out, k_in, s_out, s_in)
converged = model.solve_tool(
    ic="degrees",           # "degrees" (default), "random", "qdecm" — or pass an array (warm start)
    tol=1e-6,               # convergence tolerance (ℓ∞ relative residual MRE)
    max_iter=5000,
    max_time=0,             # wall-clock timeout in seconds (0 = no limit)
    anderson_depth=10,
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    hub_sk_threshold=0.0,   # >0: use 1D bisection for nodes with s/k > threshold (see §2.3)
    hub_bisect_max_sweeps=3,# raise (e.g. 10) only for a relaxed hub next to a very tight partner (see §2.8)
    z_clamp=1e-6,           # floor on the pair sum eta_out+eta_in; tied to the residual's own clamp, don't change lightly
    block_newton_gate=0.0,  # >0: exact joint 2x2 (theta,eta) step on ill-conditioned (s~k) rows; opt-in, see §2.9
    block_newton_min_rel=None,  # None = 0.1*tol whenever the gate is active (recommended); see §2.9
    streak_fix_threshold=0, # >0: directly solve a (node,side) pair that monopolizes the argmax this many iters (see §2.8)
    streak_fix_n_sweeps=5,  # phi-fixed bisection sweeps per streak-fix call (see §2.8)
    streak_fix_n_bisect=60, # bisection halvings per sweep (see §2.8)
    blowup_factor=None,     # None = scale-adaptive default; lower (e.g. 20-50) to catch slow drift sooner (see §2.6)
    patience=250,           # stall ladder after this many iters without a significant record (see §2.6); was 750
    patience_rel_tol=0.1,   # only a record improvement by more than this fraction resets patience (0 = any; see §2.8)
    stall_block_newton_gate=None,   # None = 0.05 (reduced path): ladder tier 2 switches block-Newton on after the soft reset fails
    stall_block_newton_off_mre=2e-4, # block-Newton switched off again once the record is below this (only if it was on above it)
    stall_block_newton_revert=True,  # block-Newton switched off if its patience window gives no significant record
    noise_base=1e-4,        # scale of the first perturbed restart's (multiplicative) noise (see §2.6)
    noise_cap_mult=16.0,    # noise scale saturates at noise_base * noise_cap_mult (see §2.6)
    noise_growth=2.0,       # noise growth rate per consecutive failed restart (see §2.6)
    max_stalls=5,           # give up after this many restarts at max noise with no improvement (see §2.6)
    seed=None,              # seed for the restart RNG; irrelevant if no restart ever fires (see §2.6)
    device="cpu",           # "cpu" (default, float64) or "mps"/"cuda" (float32; see §2.7)
                             # covers both the dense and chunked reduced-path steps; stagnation recovery (§2.6) stays CPU-only internally and interoperates transparently
)
# solve_tool returns True if converged and stores the full result:
theta = model.sol.theta     # full 4N parameters [θ_out|θ_in|η_out|η_in]
# topology multipliers (θ_out, θ_in) are model.sol.theta[:2*N]
# weight multipliers  (η_out, η_in) are model.sol.theta[2*N:]
```

Additional model methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `model.pij_matrix(theta)` | `(N, N)` tensor | DECM link-probability matrix (coupled to η) |
| `model.wij_matrix(theta)` | `(N, N)` tensor | Expected weight matrix `W_ij = p_ij · G_ij` |
| `model.residual(theta)` | `(4N,)` tensor | Constraint violation `[F_k_out\|F_k_in\|F_s_out\|F_s_in]` |
| `model.neg_log_likelihood(theta)` | float | Negative log-likelihood `−L(θ,η)` |
| `model.bic(theta)` | float | Bayesian Information Criterion, `4N·ln(N(N−1)) − 2·ln L` |
| `model.hessian_diag(theta)` | `(4N,)` tensor | Diagonal Jacobian elements (all ≤ 0) |
| `model.constraint_error(theta)` | float | `max‖F(θ,η)‖` |
| `model.max_relative_error(theta)` | float | Max relative error over all 4N non-zero constraints |
| `model.initial_theta(method)` | `(4N,)` tensor | Initial guess (see below) |
| `model.sample(seed, chunk_size, method)` | `list[[i,j,w]]` | Sample a weighted network from the fitted DECM; `method="grouped"` (default) is an exact, much faster sampler; `"exact"` is the O(N²) one (see §3.5) |
| `model.p_value_calculator(edgelist)` | structured array `[(source_id, target_id, p_value)]` | Edge-weight p-values under the fitted DECM (see §3.6) |

`initial_theta` methods for DECM:

| Method | Description |
|--------|-------------|
| `"degrees"` (default) | θ from `k/(N-1)` heuristic; η from `β = sqrt(1 − k/s)` mean-field |
| `"random"` | Uniform random `θ ∈ [0.1, 2.0]`, `η ∈ [0.1, 2.0]` |

### 3.5 Sampling synthetic networks — `model.sample()` / `model.sample_many()`

After calling `solve_tool()`, every model exposes two sampling methods:

```python
# Single sample — returns a NumPy array (no Python list overhead)
edges = model.sample(
    seed=42,          # integer or None — random seed for reproducibility
    chunk_size=512,   # rows processed per iteration (exact fallback only)
)
# DCM:   np.ndarray shape (L, 2), columns [source, target]
# other: np.ndarray shape (L, 3), columns [source, target, weight]
# Iteration works identically to lists: for i, j, w in edges: ...

# Parallel batch sampling — returns a list of NumPy arrays
samples = model.sample_many(
    n=2000,           # number of independent samples
    seed=42,          # master seed (per-sample seeds derived from it)
    n_jobs=-1,        # worker threads; -1 = all logical CPUs (default)
)
# each element: np.ndarray of shape (L_k, 2) for DCM, (L_k, 3) for weighted
```

Both methods return `np.ndarray` — iterating `for i, j in arr` or `for i, j, w in arr` works the same as with lists.

The output format and the underlying sampling distribution differ by model:

| Model | Output shape | Sampling distribution |
|-------|--------------|-----------------------|
| `DCMModel` | `(L, 2)` | `A_ij ~ Bernoulli(p_ij)` independently for each `i ≠ j` |
| `DWCMModel` | `(L, 3)` | `w_ij ~ Geom(1 − β_ij) − 1` (starts at 0); pairs with `w=0` omitted |
| `qDECMModel` | `(L, 3)` | Step 1: `A_ij ~ Bernoulli(p_ij)`; step 2 if link: `w_ij ~ Geom(1 − β_ij)` (starts at 1) |
| `DECMModel` | `(L, 3)` | Same two steps, but `p_ij` uses the full coupled DECM formula |

where `β_ij = β_out_i β_in_j = exp(−η_out_i − η_in_j)` and `p_ij` is the relevant model's link probability.

The geometric distributions follow Mastrandrea et al. (2014) / Vallarano et al. (2021):

- **DWCM**: integer weights `w ≥ 0`, `P(w=k) = (1−β_ij) β_ij^k`.  The expected weight is `β_ij / (1−β_ij)`, matching the constraint `s_out_i = Σ_j ⟨w_ij⟩`.
- **qDECM / DECM**: integer weights `w ≥ 1` conditional on the link existing, `P(w=k|A=1) = (1−β_ij) β_ij^{k−1}`.  The unconditional expected weight is `p_ij / (1−β_ij)`.

#### Fast sparse sampler (default for sparse networks)

For sparse networks (mean p < 5 %) DCM, DWCM and qDECM use an O(L) **Poisson-intensity sampler** instead of the O(N²) chunk-based sampler.  The algorithm is:

1. Draw `n_cand ~ Poisson(S_x · S_y)` edge candidates, where `S_x = Σ x_i`, `S_y = Σ y_j`.
2. Sample `src ~ x/S_x` and `dst ~ y/S_y` via the alias method — O(N + L) total.
3. Accept candidate `(i,j)` with probability `1/(1 + x_i y_j) ∈ (0,1]`.
4. Remove self-loops and deduplicate (1D integer encoding, 13× faster than 2D sort).

After deduplication the edge probability is `P(A_ij = 1) = 1 − exp(−p_ij)`, which differs from the exact `p_ij` by `p_ij²/2`.  For typical sparse networks with `p_ij ~ 10⁻³` this error is `~5×10⁻⁷` per edge — negligible for any ensemble observable.

> **Note**: this is *not* the Chung–Lu approximation.  Chung–Lu would skip step 3, giving `P ≈ x_i y_j` rather than `x_i y_j/(1+x_i y_j)`.  The acceptance step in step 3 ensures the exact DCM probabilities are respected; only the Poisson-to-Bernoulli conversion introduces a second-order error.

Dense networks (mean p ≥ 5 %) fall back to the exact chunk-based sampler; `chunk_size` controls peak RAM in that path.

#### Grouped exact sampler for DECM (`method="grouped"`, the default)

`DECMModel.sample(method="grouped")` (the default; also `sample_many(..., method=...)`) draws from the **same** distribution as the default O(N²) sampler, exploiting that nodes with identical constraints (k_out, k_in, s_out, s_in) have identical fitted parameters. For two groups a, b all ordered pairs between them are i.i.d. Bernoulli(p_ab), so the number of links of a group pair is drawn from a Binomial(n_ab, p_ab), the links are placed on distinct uniformly chosen pairs, and the weights are geometric as usual. Cost O(M² + L), with M the number of groups. On the 19 empirical networks N²/M² ranges from 18 to 438 (e.g. N=107 056, M=5 117: 3 s instead of ~530 s per sample). No gain when all nodes are distinct. Pass `method="exact"` for the per-pair O(N²) sampler. Realizations differ between the two for the same seed (same law, different random stream).

#### Parallel batch sampling with `sample_many()`

`sample_many()` uses `ThreadPoolExecutor` to generate samples in parallel.  NumPy releases the GIL during random-number generation and array operations, so threading provides near-linear speedup up to the number of physical cores.  Both `sample()` and `sample_many()` return NumPy arrays directly (no `list` conversion), eliminating the main GIL-holding bottleneck.

**Benchmark** (N = 5 000, power-law degree sequence, ρ = 10⁻³):

| Model | `sample()` single | `sample_many(n, n_jobs=8)` effective per-sample | Speedup |
|-------|-------------------|------------------------------------------------|---------|
| DCM | ~32 ms | ~4 ms | ~8× |
| qDECM | ~37 ms | ~5 ms | ~7× |
| DWCM | ~83 ms | ~11 ms | ~7× |

DECM (8 logical cores): `crisi_dico2` (N = 15 168), grouped sampler, 0.6 s per sample and 3.2× with 8 threads; exact O(N²) sampler on `ita_elections_dico4` (N = 9 010), 2.4 s per sample and 3.6×.

#### Memory estimates for large networks

Each sample is a contiguous `int64` array — RAM per sample scales as `L × cols × 8 bytes`:

| N | ρ | mean degree | L (directed) | Per sample (weighted) | 2 000 samples |
|---|---|-------------|--------------|----------------------|---------------|
| 5 000 | 10⁻³ | 5 | ~25 000 | ~0.6 MB | ~1.2 GB |
| 50 000 | 10⁻³ | 50 | ~2 500 000 | ~60 MB | ~120 GB ❌ |
| 50 000 | 10⁻⁴ | 5 | ~250 000 | ~6 MB | ~12 GB |

> **For N ≳ 10k**, storing all 2 000 samples at once is impractical.  Use a streaming pattern instead:
> ```python
> rng = np.random.default_rng(42)
> measures = []
> for _ in range(2000):
>     arr = model.sample(seed=int(rng.integers(2**31)))
>     measures.append(some_measurement(arr))
>     # arr is released after each iteration — only ~1 sample in memory at a time
> ```
> Or batch `sample_many()` in chunks of `n_jobs` to parallelise without accumulating all results.

Calls to `sample()` or `sample_many()` before `solve_tool()` raise `RuntimeError`.

### 3.6 Statistical validation — `model.p_value_calculator()`

Available on every **weighted** model (`DWCMModel`, `qDECMModel`, `DECMModel` — not `DCMModel`, which is binary-only). Given an observed weighted edge list, computes the p-value of each edge's weight under the fitted null model: the probability of observing a weight at least as large as the one seen, `p_value(w) = P(W_ij >= w)`. This is the standard ingredient for extracting a *statistically validated backbone* from a weighted network.

**Multiple-testing correction is required, not optional.** A backbone typically tests *every* edge (potentially thousands to millions) simultaneously, so thresholding the raw `p_value` directly (e.g. `p_value < 0.01`) is statistically wrong — at that significance level, ~1% of edges would be flagged as "significant" by chance alone even under the null model, and that false-positive count grows with the number of edges tested. Control the **false discovery rate (FDR)** instead, e.g. with the Benjamini-Hochberg procedure via `scipy.stats.false_discovery_control` (SciPy ≥ 1.11, already a `dcms` dependency — see the worked example below).

Each model's conditional weight distribution is geometric (see §3.5's sampling formulas), so the survival probability has a closed form:

| Model | `p_value(w)` | `z_ij` |
|-------|--------------|--------|
| DWCM  | `z_ij ** -w` | `exp(theta_out_i + theta_in_j)` |
| DECM  | `p_ij * z_ij ** -(w - 1)` | `exp(eta_out_i + eta_in_j)` |
| qDECM | `p_ij * z_ij ** -(w - 1)` | `exp(theta_b_out_i + theta_b_in_j)` |

DWCM's weight distribution starts at 0 (no separate link-existence step), so its p-value has no `p_ij` factor; DECM/qDECM gate the geometric branch (starting at 1) behind the link-existence probability `p_ij`, so their survival probability carries it. Computed directly on the given `(source_id, target_id)` pairs — no dense N×N matrix is built — so this scales to large networks.

```python
import numpy as np
from scipy.stats import false_discovery_control

# edgelist: (L, 3) array, each row [source_id, target_id, weight]
# (0-based node indices into k_out/k_in/s_out/s_in; weight >= 1 for
# DECM/qDECM, weight >= 0 for DWCM)
edgelist = np.array([[0, 1, 5], [2, 3, 12], [7, 0, 1]])

pvals = model.p_value_calculator(edgelist)
# structured NumPy array, fields: source_id (int64), target_id (int64), p_value (float64)
# also stored as model.p_value
print(pvals["p_value"])

# FDR control (Benjamini-Hochberg) across all L tested edges at once --
# NOT pvals["p_value"] < 0.01, which ignores the multiple-testing problem.
q_values = false_discovery_control(pvals["p_value"], method="bh")
validated = pvals[q_values < 0.01]
```

Raises `RuntimeError` if called before `solve_tool()`.

### 3.7 SolverResult

`solve_tool()` stores results on the model as `model.sol` for all models.  The `SolverResult` dataclass fields are:

```python
result.theta           # np.ndarray — parameters in log-space; shape (2N,) for DCM/DWCM, (4N,) for qDECM/DECM
result.best_theta      # np.ndarray — iterate with lowest MRE (same shape as theta)
result.converged       # bool
result.iterations      # int — total iterations (qDECM: topo + weight steps summed)
result.residuals       # list[float] — ℓ∞ MRE per step (empty for qDECM; use residuals_topo / residuals_weights)
result.residuals_topo    # list[float] — topology MRE history (qDECM only)
result.residuals_weights # list[float] — weight MRE history (qDECM only)
result.elapsed_time    # float — wall-clock seconds
result.peak_ram_bytes  # int
result.message         # str — warnings or error description
result.mre             # float — MRE at best_theta (min of residuals, or max(min_topo, min_weights) for qDECM)
result.last_mre        # float — MRE at last iterate theta
```

### 3.8 Standalone solvers (advanced)

The underlying solvers can be called directly without the model wrapper, e.g. to interleave topology and weight steps manually:

```python
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm
from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
from dcms.solvers.fixed_point_qdecm import solve_fixed_point_qdecm, solve_fixed_point_qdecm_weight
from dcms.solvers.fixed_point_decm import solve_fixed_point_decm

result = solve_fixed_point_dcm(
    theta0,                  # initial guess (2N,)
    k_out, k_in,             # observed degree sequences
    tol=1e-6,
    max_iter=2000,
    anderson_depth=10,
    max_time=0,
)
```

`solve_fixed_point_dwcm` has the same signature with `s_out, s_in`; `solve_fixed_point_decm` takes `theta0` (4N, `[θ_out|θ_in|η_out|η_in]`) and `k_out, k_in, s_out, s_in`; `solve_fixed_point_qdecm_weight(theta_topo, theta_weight0, k_out, k_in, s_out, s_in, ...)` solves the weight step for a given topology and `solve_fixed_point_qdecm` does topology + weights. All of them group the nodes internally (§2.5), return a `SolverResult` expanded back to the original per-node shape, and accept `device: str = "cpu"` (§2.7: float32 off the CPU, `SolverResult` always CPU/float64).

### 3.9 Parallelism and monitoring

The solvers are single-process PyTorch code (torch's own intra-op threads apply). Long runs can be watched with `verbose`/`monitor`.

**Custom initial conditions (warm restart).**  See §3.10 for details and examples.

```python
# Full log — a new line is printed at every iteration (default behaviour):
model.solve_tool(verbose=True, monitor=False)
# [14:32:07] iteration=    1, elapsed time=   0:00:00, MRE_topo=4.52e-02
# [14:32:08] iteration=    2, elapsed time=   0:00:01, MRE_topo=8.13e-03
# ...

# Live monitor — the line is overwritten in place (end='\r'), ideal for terminals:
model.solve_tool(verbose=True, monitor=True)
# [14:32:12] iteration=  128, elapsed time=   0:00:05, MRE_topo=1.07e-06   ← updates in place

# DECM shows both MRE components on every line:
model_de.solve_tool(verbose=True, monitor=True)
# [14:32:12] iteration=   64, elapsed time=   0:00:10, MRE_topo=4.52e-04, MRE_weights=3.21e-05
```

| Parameter | Behaviour |
|-----------|-----------|
| `verbose=False` (default) | Silent — only prints the final convergence message. |
| `verbose=True, monitor=False` | Prints one new line per iteration; useful for debugging or file logging. |
| `verbose=True, monitor=True` | Overwrites the same terminal line (`end='\r'`); ideal for interactive monitoring of long runs. |

Each line shows: wall-clock timestamp, iteration count, total elapsed time, and the **Maximum Relative Error** (MRE = `max_i |F_i(θ)| / constraint_i`) split by type: `MRE_topo` for degree constraints, `MRE_weights` for strength constraints.

### 3.10 Custom initial conditions and warm restart

Each model internally works with a *parameter vector* in log-space.  After `solve_tool()` finishes, the result is stored in `model.sol.theta` for all models.

**Shape of the parameter vectors**

| Model | Attribute | Shape | Content |
|-------|-----------|-------|---------|
| DCM | `model.sol.theta` | (2N,) | `[θ_out₀ … θ_out_{N-1} | θ_in₀ … θ_in_{N-1}]` |
| DWCM | `model.sol.theta` | (2N,) | `[η_out₀ … | η_in₀ …]` where `β_out_i = exp(−η_out_i)` |
| qDECM | `model.sol.theta` | (4N,) | `[θ_out_topo | θ_in_topo | η_out | η_in]` |
| DECM | `model.sol.theta` | (4N,) | `[θ_out | θ_in | η_out | η_in]` |

The entries are related to the model probabilities by `x_i = exp(−θ_i)` (topology) and `β_i = exp(−η_i)` (weights).  The feasibility constraint is `η_out_i + η_in_j > 0` for every pair (i, j) with a potential link (individual η can be negative as long as the pairwise sum stays positive).

**Why custom ICs?**

By default `solve_tool()` computes the starting point from the observed degree/strength sequences (the `ic` / `ic_topo` / `ic_wei` string choices).  On hard instances — networks with very high-weight hubs or extreme s/k ratios — the automatic starting point may be far from the solution and the solver may not converge within the iteration budget.  All models accept an **array** directly as the `ic` parameter (or `ic_topo` / `ic_wei` for qDECM) to warm-start from any custom vector, enabling two practical strategies:

1. **Warm restart** — if a first call did not converge, the best iterate found so far is always stored in `model.sol.best_theta` (the solver internally tracks the iterate with the smallest residual).  Pass that array back as the starting point for a second call, possibly with a smaller `anderson_depth` to reduce Anderson contamination:

```python
model = qDECMModel(k_out, k_in, s_out, s_in)
model.solve_tool(max_iter=5000, verbose=True, monitor=True)

if not model.sol.converged:
    N = len(k_out)
    # Second attempt from best iterate, less aggressive Anderson mixing
    model.solve_tool(
        ic_topo=model.sol.best_theta[:2*N],   # topology already solved, reuse it
        ic_wei=model.sol.best_theta[2*N:],    # start from best weight iterate
        anderson_depth=3,                      # reduce Anderson depth
        max_iter=10000,
    )
```

2. **Transfer warm start** — use the DWCM solution as the starting point for the qDECM weight step (the weight equations are similar, and DWCM is easier to solve):

```python
dwcm = DWCMModel(s_out, s_in)
dwcm.solve_tool()

qdecm = qDECMModel(k_out, k_in, s_out, s_in)
qdecm.solve_tool(
    ic_wei=dwcm.sol.best_theta,   # DWCM solution as weight IC
)
```

For DCM, DWCM and DECM pass an array of shape (2N,), (2N,) and (4N,) respectively to `ic`.  For DECM, `multi_start` is automatically disabled when `ic` is an array.

### 3.10.1 Checkpointed multi-chunk runs surviving a real crash/preemption

Stagnation and repeated-blowup recovery are now **built into the DECM solver itself** (§2.6) — a single unattended `solve_tool()`/`solve_fixed_point_decm` call handles those on its own via the stall ladder (`patience`/`patience_rel_tol`, block-Newton tier, then `noise_base`/`noise_cap_mult`/`max_stalls`), no external orchestration needed.

What a single in-process call *cannot* survive is an actual **crash, timeout, or preemption** of the process itself — a real concern for instances that need tens of thousands of iterations (hours) unattended. For that, run in **chunks** — `max_iter` iterations at a time, checkpointing state to disk after each chunk, so a killed process never loses more than one chunk of progress:

```python
global_best_theta, best_mre = None, float("inf")
theta = None  # None => solver picks the default ic on the first chunk

for chunk in range(n_chunks):
    res = solve_fixed_point_decm(
        ..., theta0=theta,
        init_best_theta=global_best_theta, init_best_res=best_mre,
        max_iter=chunk_iters,
        # patience/patience_rel_tol/noise_base/noise_cap_mult/max_stalls: leave at their
        # defaults, or pass explicitly -- they apply within EACH chunk,
        # independently of the cross-chunk record tracked here.
    )
    if res.mre < best_mre:
        best_mre = res.mre
        global_best_theta = res.best_theta.copy()
    theta = res.theta          # continue from where this chunk left off
    save_checkpoint(theta, global_best_theta, best_mre)  # survive a restart
    if res.converged:
        break
```

`init_best_theta` / `init_best_res` exist specifically for this: each chunked call is otherwise a *fresh* invocation with no memory of the record set by earlier chunks — its own in-call best-tracking (and now, stagnation/blowup recovery) can only fall back to a chunk-local best, which may already be far worse than the true historical record. Passing the previous chunk's record back in keeps everything anchored to the true global best across chunk boundaries.

**Validated on a real hard instance:** DECM on an empirical online social network (N=15 168, M=3 003 after degeneracy reduction, §2.5, several hub nodes), run unattended from scratch with `anderson_depth=10`, `hub_sk_threshold=5.0`, `patience=750` (the then-default; see the *Stall ladder* paragraph in §2.6 for the current `solve_tool` defaults), converged after 9 460 iterations to MRE=9.45×10⁻⁶ with no manual intervention — first as a chunked+checkpointed prototype (an earlier iteration of this same recovery idea, before it moved into the solver itself), then reproduced with the built-in mechanism (§2.6).

### 3.11 Network generator (`dcms/utils/wng.py`)

```python
from dcms.utils.wng import k_s_generator_pl

k, s = k_s_generator_pl(
    N,                  # number of nodes
    rho=1e-3,           # target edge density
    seed=None,          # reproducibility
    alpha_pareto=2.5,   # Pareto shape (degree heterogeneity)
)
# k: int tensor (2N,) = [k_out | k_in]
# s: int tensor (2N,) = [s_out | s_in]
```

---

## 4. Performance

Measured on the 19 real networks of the project (N from 232 to 107 056, M from 20 to 7 305 groups), 24 threads, `tol = 1e-5`, seed 0, benchmark v2 (package at commit `4a1f492`, `benchmark_19/results_v2/`). All runs converged:

| Model | Networks converged | Iterations | Wall time | Remarks |
|---|---|---|---|---|
| DWCM | 19 / 19 | 13 – 59 | < 10 s | |
| qDECM | 19 / 19 | 26 – 175 | < 2 min | **to be re-measured**: those runs predate the 2026-10-27 fix of the within-group pairs (§2.5), which changes the iteration counts |
| DECM | 19 / 19 | 164 – 3 064 | 1 s – 10 h 19 | recipe `general` at the first attempt on every network (`anderson_depth=10, hub_sk_threshold=5, z_clamp=1e-6, noise_base=2e-3`); the three slowest networks (M ≈ 5 000–7 300) take 1–10 h |

DCM was not part of that benchmark; on the same inputs it converges in 14–24 iterations (tol 1e-8).

---

## 5. Complexity

| Model | Convergence | Memory per iteration | Compute per iteration |
|--------|-------------|-------------------|--------------------|
| DCM, DWCM | superlinear (θ-Newton + Anderson) | O(M²) | O(M²) |
| qDECM | superlinear (topology, then weights) | O(M²) | O(M²) per stage |
| DECM | superlinear (alternating GS-Newton on 4N) | O(M²), or O(chunk × M) when `M > 2000` | O(M²), two passes |

`M` is the number of distinct sufficient-statistics groups (§2.5), `M ≤ N`; on real networks with heavy-tailed degree/strength distributions `M` is far smaller than `N` (N²/M² from 18 to 438 on the 19 networks of the project).

---

## Running Tests

```bash
pytest tests/
```

`tests/test_device.py` runs the non-CPU device tests only if the device exists (MPS on Apple Silicon yes; CUDA never run, see §2.7).

## References

1. Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, **13**, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

2. Park, J. & Newman, M.E.J. (2004). Statistical mechanics of networks. *Physical Review E*, **70**, 066117. https://doi.org/10.1103/PhysRevE.70.066117

3. Vallarano, N., Bruno, M., Marchese, E., Trapani, G., Saracco, F., Cimini, G., Zanon, M. & Squartini, T. (2021). Fast and scalable likelihood maximization for exponential random graph models with local constraints. *Scientific Reports*, **11**, 15227. https://doi.org/10.1038/s41598-021-93830-4 *(NEMtropy)*

4. Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, **49**(4), 1715–1735. https://doi.org/10.1137/10078356X

5. Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM.  Chapter 5.
