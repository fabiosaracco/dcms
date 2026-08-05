# DCMS — Maximum-Entropy Solvers for Directed Networks

This project implements numerical solvers for **maximum-entropy** (MaxEnt) models of directed networks.  Given an observed graph, the models find the probability distribution over all directed graphs that maximises entropy subject to reproducing a chosen set of topological constraints (degree and/or strength sequences).

Two solvers are provided for every model, both proven to scale reliably to N = 5 000 and beyond:

| Solver | Algorithm | When to prefer |
|--------|-----------|----------------|
| **FP-GS Anderson(10)** | Gauss-Seidel fixed-point + Anderson(10) acceleration | DWCM/qDECM where the contraction condition holds (mild heterogeneity) |
| **θ-Newton Anderson(10)** | Coordinate-wise Newton in log-space + Anderson(10) acceleration | Default choice — most robust, fastest at large N |

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

To include optional [Numba](https://numba.pydata.org/) support (only beneficial for very large networks, N ≳ 100 000 — see §3.9 for benchmarked RAM/speed trade-offs):

```bash
pip install "dcms[numba] @ git+https://github.com/fabiosaracco/dcms.git"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy ≥ 1.24, SciPy ≥ 1.10.

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

### 2.1 FP-GS Anderson(10) — Gauss-Seidel Fixed-Point with Anderson Acceleration

#### Rationale

The MaxEnt self-consistency equations can be written as a **fixed-point problem**:

```
θ_new = g(θ)
```

where, for a single out-multiplier of the DCM, `g` isolates the variable on one side:

```
x_i^new = k_out_i / Σ_{j≠i} y_j / (1 + x_i · y_j)   →   θ_out_i^new = -log(x_i^new)
```

The **Gauss-Seidel** ordering updates `θ_out` first and immediately uses the fresh values when computing `θ_in`.  This makes the effective Jacobian of the map (the **spectral radius** ρ of ∂g/∂θ) smaller than the Jacobi (simultaneous) variant, yielding faster convergence.

Convergence is guaranteed when ρ < 1.  For sparse, homogeneous networks this holds comfortably.  For power-law networks with high-degree hubs, some nodes have ρ ≥ 1 and plain FP-GS stagnates; a node-level Newton fallback and the blowup-reset logic handle those cases (see Implementation details below).

For the DWCM and qDECM weight step, the fixed-point map in β-space is a multiplicative scaling rule.  For the **DWCM** (where `p_ij = 1`):

```
β_out_i^new = β_out_i · s_out_i / s_out_hat_i,   s_out_hat_i = Σ_{j≠i} β_in_j / (1 - β_out_i · β_in_j)
```

For the **qDECM weight step** (conditioned DWCM with `p_ij` from the DCM):

```
β_out_i^new = β_out_i · s_out_i / s_out_hat_i,   s_out_hat_i = Σ_{j≠i} p_ij / (1 - β_out_i · β_in_j)
```

Here the spectral radius depends on the inverse of `(1 - β·β)`, which grows rapidly as `β → 1` (hub nodes with `s/k → ∞`).  This is the main failure mode of FP-GS at large N and high heterogeneity.

#### Anderson Acceleration

Anderson mixing (depth m = 10) transforms the plain fixed-point sequence into a quasi-Newton method by finding the linear combination of the last `m` residuals `r_k = g(θ_k) - θ_k` that minimises the mixed residual norm:

```
min_{c, Σc=1}  ‖Σ_k c_k · r_k‖²
```

The coefficients `c` are found by a small `m×m` least-squares system (O(m²) per step).  The acceleration can reduce iteration counts by 5–50× on well-conditioned problems.

**Blowup protection:** if the Anderson iterate produces a residual jump > `_ANDERSON_BLOWUP_FACTOR × best_residual`, the history is cleared and the plain Newton step is used instead.  This prevents one bad linear combination from ruining the run.

**Weighted mixing:** residuals are row-normalised by their component-wise maximum before solving the least-squares problem.  This prevents hub nodes (which have large absolute residuals) from dominating the mixing coefficients.

#### Implementation

- `dcms/solvers/fixed_point_dcm.py` — `solve_fixed_point_dcm(..., variant="gauss-seidel", anderson_depth=10)`
- `dcms/solvers/fixed_point_dwcm.py` — `solve_fixed_point_dwcm(..., variant="gauss-seidel", anderson_depth=10)`
- `dcms/solvers/fixed_point_qdecm.py` — `solve_fixed_point_qdecm(..., variant="gauss-seidel", anderson_depth=10)`
- `dcms/solvers/fixed_point_decm.py` — `solve_fixed_point_decm(..., variant="theta-newton", anderson_depth=10)` (DECM only uses θ-Newton; see §2.2)

All four files share the same algorithmic skeleton:

1. **Dense path** (N ≤ threshold): materialise the N×N probability/weight matrix once per iteration.
2. **Chunked path** (N > threshold): process rows in blocks of 512 to keep peak RAM at O(chunk × N) rather than O(N²).  The dense/chunked crossover threshold is **5 000** for DCM/DWCM and **2 000** for qDECM/DECM (`DCM_LARGE_N_THRESHOLD`, `DWCM_LARGE_N_THRESHOLD`, `qDECM_LARGE_N_THRESHOLD` in `dcms/models/parameters.py`).
3. **Node-level Newton fallback**: when `|Δθ_FP| > _FP_NEWTON_FALLBACK_DELTA` for a node, replace the FP step with an exact diagonal Newton step `Δθ = -F_i / (∂F_i/∂θ_i)`.
4. **Best-θ tracking**: the result always returns the lowest-residual iterate seen, not the final one.

**Literature:**
Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735. https://doi.org/10.1137/10078356X

---

### 2.2 θ-Newton Anderson(10) — Coordinate Newton with Anderson Acceleration

#### Rationale

Instead of solving `θ = g(θ)`, the θ-Newton method treats the constraints directly as a **nonlinear system** `F(θ) = 0` and applies a **coordinate-wise Newton step**:

```
Δθ_out_i = -F_out_i(θ) / (∂F_out_i / ∂θ_out_i)
```

where `F_out_i = k_hat_out_i - k_out_i` is the residual of node `i`'s out-degree constraint, and the denominator is the diagonal element of the Jacobian:

```
∂F_out_i / ∂θ_out_i = -Σ_{j≠i} p_ij · (1 - p_ij)
```

This is equivalent to a **Gauss-Seidel Newton** step: update `θ_out_i` node by node using fresh values immediately.  The step is clipped to `[-max_step, +max_step]` in log-space to prevent large excursions near hubs.

**Key advantage over FP-GS:** the step size is `O(|F_i| / Σ p(1-p))`, which naturally adapts to the curvature of the likelihood surface.  Hub nodes — where FP-GS oscillates or diverges because ρ ≥ 1 — are handled gracefully: their large residual produces a large numerator, but the large denominator (many connections) stabilises the step.

For the DWCM/qDECM weight step, the coordinate Newton formula becomes:

```
Δη_out_i = (s_hat_out_i - s_out_i) / Σ_{j≠i} p_ij · G_ij · (G_ij - 1)
```

where `G_ij = 1/(1 - β_i · β_j)` is the geometric-distribution correction factor and `p_ij` is the topology probability from the DCM step.  The denominator is always negative (the Jacobian diagonal is negative-definite), so the step is in the correct descent direction.

For the DECM, the coupling between degree and strength equations modifies the strength Jacobian diagonal:

```
∂F_s_out_i / ∂η_out_i = −Σ_{j≠i} p_ij · G_ij² · (1 − p_ij + z_ij)
```

which equals the qDECM diagonal plus a correction `Σ p_ij · (1 − p_ij) · G_ij²` reflecting the dependence of `p_ij` on `η`.  The DECM solver therefore uses alternating out-group / in-group GS-Newton passes that update both topology (θ) and weight (η) multipliers simultaneously within each group: pass 1 updates (θ_out, η_out) from row sums; pass 2 updates (θ_in, η_in) from col sums.

**The z-floor mechanism:** define `z_ij = θ_out_i + θ_in_j`.  When `z_ij → 0`, `G_ij → ∞` and the residual blows up.  The solver maintains per-node floors `z_min_out[i]` and `z_min_in[j]` (computed from significant pairs with `p_ij > 0.5/N`) and applies a global floor from `min(θ_in)` over non-zero-strength nodes.  This guarantees `z_ij > _Z_G_CLAMP = 1e-8` for all pairs after every Newton step.

**Anderson acceleration** is applied identically to the FP-GS case, with the same blowup protection and history clearing.  When the Anderson mix violates the z-floor (i.e. `min(θ_out) + min(θ_in) < 0`), the mix is rejected and the plain Newton step is used instead, and the Anderson history is cleared.

#### Implementation

- `dcms/solvers/fixed_point_dcm.py` — `solve_fixed_point_dcm(..., variant="theta-newton", anderson_depth=10)`
- `dcms/solvers/fixed_point_dwcm.py` — `solve_fixed_point_dwcm(..., variant="theta-newton", anderson_depth=10)`
- `dcms/solvers/fixed_point_qdecm.py` — `solve_fixed_point_qdecm(..., variant="theta-newton", anderson_depth=10)`
- `dcms/solvers/fixed_point_decm.py` — `solve_fixed_point_decm(..., variant="theta-newton", anderson_depth=10)` (alternating out/in GS-Newton on 4N vector)

Internally, each file has a `_theta_newton_step_chunked` (and optionally `_theta_newton_step_dense`) function that computes the diagonal Jacobian and applies the clipped step without materialising the full Jacobian matrix (O(N) RAM).

**Numerical constants (tuneable):**

| Constant | Default | Role |
|----------|---------|------|
| `_Z_G_CLAMP` | `1e-8` | Minimum `z = θ_out + θ_in` before clamping |
| `_Z_NEWTON_FLOOR` | `1e-8` | Hard floor on `z` after each Newton step |
| `_Z_NEWTON_FRAC` | `0.5` | Max fractional decrease of `z` per step (prevents period-2 oscillation) |
| `max_step` | `1.0` (DCM/DWCM/qDECM), `0.5` (DECM) | Max `|Δθ|` per coordinate per step (reduces for heterogeneous hubs) |
| `_ANDERSON_BLOWUP_FACTOR` | `5000` | Residual-jump ratio that triggers history clear |

**Literature:**
Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM. Chapter 5.

Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735.

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

### 2.4 Backtracking line search — preventing Newton divergence

On very difficult networks the θ-Newton step can cause the residual to *increase* dramatically at a single iteration (even by a factor of 4–10×), after which Anderson mixing accumulates bad history and the solver diverges.  The **backtracking line search** prevents this by checking the residual at the proposed iterate and dampening the step when necessary.

**Mechanism:** after computing the Newton proposal `θ_fp`, the solver evaluates `F(θ_fp)`.  If `MRE(F(θ_fp)) > backtracking_gamma × MRE(F(θ))`, the step is halved (`α = 0.5 → 0.25 → … → 1/32`) until the condition is met or the minimum step size is reached.  The best dampened iterate is accepted and Anderson history is cleared to prevent contamination.

**Effect on an empirical online social network** (N = 22 754, one hub with s/k = 152):  
- Without backtracking: residual jumps from 0.976 to 3.73 at iteration 2, then diverges.  
- With `backtracking_gamma=1.2, anderson_depth=3, hub_sk_threshold=100`: residuals decrease **monotonically** (0.976 → 0.905 → …) with no divergence.

**Parameters:**
- `backtracking_gamma` (default 0.0 = disabled): threshold ratio.  Typical values 1.2–2.0.  Lower values enforce stricter descent but cost more evaluations.
- **Combine with** `hub_sk_threshold` on networks with extreme-hub nodes.

**Cost:** each iteration triggers at most 1 extra O(N²) residual evaluation (plus up to 4 more if halvings are needed).  For small/medium networks this overhead is negligible; for large chunked networks (N > 5 000) expect 2–4× longer iterations when backtracking is active.

**Example:**

```python
model = qDECMModel(k_out, k_in, s_out, s_in)
converged = model.solve_tool(
    hub_sk_threshold=100,     # bisection for the extreme hub
    backtracking_gamma=1.2,   # strict: allow at most 20% residual increase per step
    anderson_depth=3,         # reduced history to limit contamination
    max_time=3600,
    verbose=True,
)
print(model.sol.mre)
```

---

### 2.5 Degeneracy reduction — collapsing symmetric nodes into groups

Two nodes that share the exact same **sufficient statistics** — the observed sequences that pin down the model's Lagrange multipliers — provably have **identical multipliers at the true solution**. This follows from a label-swap symmetry argument: swapping the labels of two such nodes leaves the log-likelihood unchanged, and since the log-likelihood is strictly concave, its unique maximiser must be symmetric under that swap (Vallarano, N. et al. (2021), *Scientific Reports*, 11, 15227).

| Model | Sufficient statistics (degeneracy key) |
|---|---|
| DCM | `(k_out, k_in)` |
| DWCM | `(s_out, s_in)` |
| qDECM (conditioned-weight step only — the topology step is a plain DCM) | `(k_out, k_in, s_out, s_in)` |
| DECM | `(k_out, k_in, s_out, s_in)` |

On real-world networks with heavy-tailed degree/strength distributions, most of the tail shares a small number of distinct low-degree/low-strength value combinations, so the number of **groups** `M` (unique keys) is often far smaller than the number of nodes `N`. Solving for `M` group-level unknowns instead of `N` per-node unknowns turns the O(N²) pairwise compute into O(M²) — measured **10–67× faster** on real networks (see table below), with the group and per-node solutions agreeing to machine precision (or the model's own convergence tolerance) once expanded back to per-node shape.

> **All four models have a genuine gauge freedom**, not just DCM. `p_ij`/`W_ij` always enter through the additive combination `θ_out_i + θ_in_j` (degree part) and/or `η_out_i + η_in_j` (strength part, DWCM/qDECM-weight/DECM), so `θ_out += c, θ_in -= c` — and, separately, `η_out += c, η_in -= c` where a strength part exists — leaves every `p_ij`/`W_ij` unchanged (verified numerically to machine precision for all four models). **DECM has two independent gauge directions** (θ-shift and η-shift, since its degree and strength equations decouple through this same additive structure). The full and reduced solvers are therefore not expected to land on identical raw θ unless both are very tightly converged along every gauge direction — always compare via a gauge-invariant quantity (`p_ij` and/or `W_ij`), never raw θ. This is also the real explanation for qDECM's weight step showing a looser agreement (~2e-3) than DCM/DWCM/DECM in the table below: it isn't a vague "conditioning artifact", it's this same gauge direction, just less sharply constrained by the model's positivity box than in the other three cases.

**Measured on empirical data from online social networks** (`tol≈1e-9/1e-10`, `theta-newton`, `anderson_depth=10`; network labels below are anonymized, distinct letters = distinct networks):

| Model | Network | N → M | Speedup |
|---|---|---|---|
| DCM | online social network A | 1304 → 61 | 24.2× |
| DCM | online social network B | 20914 → 2310 | 43.5× |
| DWCM | online social network C | 15168 → 1860 | 47.1× |
| DWCM | online social network D | 31874 → 3083 | 66.8× |
| qDECM (weight step) | online social network B | 20914 → 4070 | 23.5× |
| DECM | online social network C | 15168 → 3003 | ~10–25× |

**Status:** on by default. Every model's `solve_tool()` accepts `reduce_degeneracy: bool = True` (§3.1-3.4) — set it to `False` to force the full (unreduced) solver. It's automatically bypassed (with a printed note) when the requested options aren't supported by the reduced path: `variant != "theta-newton"`, `backend == "numba"`, or (qDECM/DECM only) `backtracking_gamma > 0`.

```python
model = DCMModel(k_out, k_in)
converged = model.solve_tool(tol=1e-9)                       # reduced by default
converged = model.solve_tool(tol=1e-9, reduce_degeneracy=False)  # force the full solver
```

The standalone `solve_fixed_point_*_degenerate` functions (§3.8) remain available directly, e.g. to pass a custom residual function or when not using the model wrapper:

**Example:**

```python
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm_degenerate

model = DCMModel(k_out, k_in)
theta0 = model.initial_theta("degrees")

result = solve_fixed_point_dcm_degenerate(
    theta0, k_out, k_in,
    tol=1e-9, max_iter=2000, anderson_depth=10,
)
# result.theta / result.best_theta are already expanded back to shape (2N,) —
# a drop-in replacement for solve_fixed_point_dcm's return value.
print(result.converged, result.mre, result.message)  # message reports "N=... -> M=..."
```

`solve_fixed_point_dwcm_degenerate` and `solve_fixed_point_qdecm_degenerate` follow the same pattern (`s_out, s_in` in place of `k_out, k_in`; qDECM additionally needs `theta_topo0`/`k_out`/`k_in` since it also runs the topology step internally). `solve_fixed_point_decm_degenerate` additionally supports `hub_sk_threshold` (§2.3) and `weight_anderson` (multiplicity-aware Anderson mixing, on by default — needed because a degeneracy class of `mult` identical nodes must count `mult` times in the Anderson least-squares fit to match the unreduced system exactly).

### 2.6 Perturbed restart on stagnation / repeated blowup (DECM)

Even with Anderson acceleration and the blowup guard (§2.1), some hub-heavy instances get stuck at a **quasi-fixed point**: progress halts for hundreds of iterations, or the Anderson blowup guard keeps tripping and rolling back to the same point without ever improving on it. Plain rollback-and-retry alone cannot escape a genuine trap — the deterministic Newton/Anderson dynamics just reproduce the same trajectory. The fix is to occasionally add noise:

- **Stagnation:** if `best_theta_res` hasn't improved for `patience` consecutive iterations, restart from `best_theta` plus Gaussian noise instead of continuing from the stuck iterate.
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

**Status:** on by default (`DECMModel.solve_tool()` and the standalone `solve_fixed_point_decm[_degenerate]`). Inert for well-behaved instances that never stagnate or blow up — no default-value tuning is needed to get the old behaviour on easy networks. Set `patience<=0` to fully disable and restore the old fail-fast-on-stagnation behaviour.

```python
model = DECMModel(k_out, k_in, s_out, s_in)
model.solve_tool(
    max_iter=20000,
    blowup_factor=None,   # None = scale-adaptive default; lower (e.g. 20-50) to catch slow drift sooner
    patience=750,        # restart after 750 iterations with no record improvement
    noise_base=1e-2,      # first restart's noise scale (multiplicative, see above)
    noise_cap_mult=16.0,  # noise saturates at noise_base * 16
    noise_growth=2.0,     # noise doubles per consecutive failed restart; lower (e.g. 1.2-1.5) if escalation overshoots
    max_stalls=5,         # give up after 5 restarts at max noise with no improvement
    seed=None,             # set an int for reproducible restarts (irrelevant if none fire)
)
```

**Validated on a real hard instance:** DECM on the same empirical online social network as above (N=15 168, M=3 003 after degeneracy reduction, §2.5, several hub nodes), run unattended from scratch with `anderson_depth=10`, `hub_sk_threshold=5.0`, `patience=750` (750 iterations translated from the value validated in the external checkpointed-runner prototype, §3.10.1) — one repeated-blowup restart and four stagnation restarts fired, and the run converged after 9 460 iterations to MRE=9.45×10⁻⁶, with no manual intervention.

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
    variant="theta-newton", # "theta-newton" (default) or "gauss-seidel"
    anderson_depth=10,
    backend="auto",         # "auto" (default), "pytorch", or "numba"
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    reduce_degeneracy=True, # collapse nodes sharing (k_out,k_in) into groups (see §2.5); default True
)
theta = model.sol.theta     # converged parameters, shape (2N,)
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
    variant="theta-newton",
    anderson_depth=10,
    backend="auto",         # "auto" (default), "pytorch", or "numba"
    num_threads=0,          # Numba threads: 0 = auto (all available CPUs)
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    reduce_degeneracy=True, # collapse nodes sharing (s_out,s_in) into groups (see §2.5); default True
)
theta = model.sol.theta     # converged parameters, shape (2N,)
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
    variant="theta-newton",
    anderson_depth=10,
    backend="auto",         # "auto" (default), "pytorch", or "numba"
    num_threads=0,          # Numba threads: 0 = auto (all available CPUs)
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    hub_sk_threshold=0.0,   # >0: use 1D bisection for nodes with s/k > threshold (see §2.3)
    backtracking_gamma=0.0, # >0: line search — halve step if MRE increases by > gamma× (see §2.4)
    reduce_degeneracy=True, # collapse degenerate node groups in both steps (see §2.5); default True
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
    backend="auto",         # "auto" (default), "pytorch", or "numba"
    num_threads=0,          # Numba threads: 0 = auto (all available CPUs)
    verbose=False,          # print iteration progress (timestamp, MRE, …)
    monitor=False,          # if True (with verbose), overwrite line in place (end="\r")
    hub_sk_threshold=0.0,   # >0: use 1D bisection for nodes with s/k > threshold (see §2.3)
    backtracking_gamma=0.0, # >0: line search — halve step if MRE increases by > gamma× (see §2.4)
    reduce_degeneracy=True, # collapse nodes sharing (k_out,k_in,s_out,s_in) into groups (see §2.5); default True
    blowup_factor=None,     # None = scale-adaptive default; lower (e.g. 20-50) to catch slow drift sooner (see §2.6)
    patience=750,           # restart from best_theta+noise after this many iters with no improvement (see §2.6)
    noise_base=1e-2,        # scale of the first perturbed restart's (multiplicative) noise (see §2.6)
    noise_cap_mult=16.0,    # noise scale saturates at noise_base * noise_cap_mult (see §2.6)
    noise_growth=2.0,       # noise growth rate per consecutive failed restart (see §2.6)
    max_stalls=5,           # give up after this many restarts at max noise with no improvement (see §2.6)
    seed=None,              # seed for the restart RNG; irrelevant if no restart ever fires (see §2.6)
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
| `model.sample(seed, chunk_size)` | `list[[i,j,w]]` | Sample a weighted network from the fitted DECM (see §3.5) |
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

#### Parallel batch sampling with `sample_many()`

`sample_many()` uses `ThreadPoolExecutor` to generate samples in parallel.  NumPy releases the GIL during random-number generation and array operations, so threading provides near-linear speedup up to the number of physical cores.  Both `sample()` and `sample_many()` return NumPy arrays directly (no `list` conversion), eliminating the main GIL-holding bottleneck.

**Benchmark** (N = 5 000, power-law degree sequence, ρ = 10⁻³):

| Model | `sample()` single | `sample_many(n, n_jobs=8)` effective per-sample | Speedup |
|-------|-------------------|------------------------------------------------|---------|
| DCM | ~32 ms | ~4 ms | ~8× |
| qDECM | ~37 ms | ~5 ms | ~7× |
| DWCM | ~83 ms | ~11 ms | ~7× |

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

Available on every **weighted** model (`DWCMModel`, `qDECMModel`, `DECMModel` — not `DCMModel`, which is binary-only). Given an observed weighted edge list, computes the p-value of each edge's weight under the fitted null model: the probability of observing a weight at least as large as the one seen, `p_value(w) = P(W_ij >= w)`. This is the standard ingredient for extracting a *statistically validated backbone* from a weighted network — keep only edges whose weight is unlikely under the null (e.g. `p_value < 0.01`, typically after a multiple-testing correction such as Benjamini–Hochberg).

Each model's conditional weight distribution is geometric (see §3.5's sampling formulas), so the survival probability has a closed form:

| Model | `p_value(w)` | `z_ij` |
|-------|--------------|--------|
| DWCM  | `z_ij ** -w` | `exp(theta_out_i + theta_in_j)` |
| DECM  | `p_ij * z_ij ** -(w - 1)` | `exp(eta_out_i + eta_in_j)` |
| qDECM | `p_ij * z_ij ** -(w - 1)` | `exp(theta_b_out_i + theta_b_in_j)` |

DWCM's weight distribution starts at 0 (no separate link-existence step), so its p-value has no `p_ij` factor; DECM/qDECM gate the geometric branch (starting at 1) behind the link-existence probability `p_ij`, so their survival probability carries it. Computed directly on the given `(source_id, target_id)` pairs — no dense N×N matrix is built — so this scales to large networks.

```python
import numpy as np

# edgelist: (L, 3) array, each row [source_id, target_id, weight]
# (0-based node indices into k_out/k_in/s_out/s_in; weight >= 1 for
# DECM/qDECM, weight >= 0 for DWCM)
edgelist = np.array([[0, 1, 5], [2, 3, 12], [7, 0, 1]])

pvals = model.p_value_calculator(edgelist)
# structured NumPy array, fields: source_id (int64), target_id (int64), p_value (float64)
# also stored as model.p_value
print(pvals["p_value"])

validated = pvals[pvals["p_value"] < 0.01]
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

The underlying solvers can be called directly without the model wrapper, e.g. to pass a custom residual function or to interleave topology and weight steps manually:

```python
from dcms.solvers.fixed_point_dcm import solve_fixed_point_dcm
from dcms.solvers.fixed_point_dwcm import solve_fixed_point_dwcm
from dcms.solvers.fixed_point_qdecm import solve_fixed_point_qdecm
from dcms.solvers.fixed_point_decm import solve_fixed_point_decm

result = solve_fixed_point_dcm(
    residual_fn,             # callable F(θ) → (2N,) tensor
    theta0,                  # initial guess (2N,)
    k_out, k_in,             # observed degree sequences
    tol=1e-6,
    max_iter=2000,
    variant="theta-newton",  # "theta-newton" (default) or "gauss-seidel"
    anderson_depth=10,
    max_time=0,
    backend="auto",          # "auto" (default), "pytorch", or "numba"
)
```

`solve_fixed_point_dwcm` and `solve_fixed_point_qdecm` share the same signature (replacing `k_out, k_in` with `s_out, s_in`; qDECM additionally requires `theta_topo`).

Each of the four models also has a **degeneracy-reduced** counterpart —
`solve_fixed_point_dcm_degenerate`, `solve_fixed_point_dwcm_degenerate`,
`solve_fixed_point_qdecm_degenerate`, `solve_fixed_point_decm_degenerate` —
that automatically groups nodes with identical sufficient statistics and
solves the resulting smaller system, expanding the result back to the
original per-node shape. See §2.5 for the rationale, measured speedups, and
a usage example.

`solve_fixed_point_decm` requires `k_out, k_in, s_out, s_in` and an initial 4N guess `theta0 = [θ_out|θ_in|η_out|η_in]`.

### 3.9 Compute backend and parallelism

All solvers accept a `backend` parameter that controls which compute engine executes the N×N inner loops:

| Value | Behaviour |
|-------|-----------|
| `"auto"` (default) | PyTorch chunked for N ≤ 100 000; Numba parallel scalar loops for N > 100 000. |
| `"pytorch"` | Always use PyTorch (dense or chunked depending on N). |
| `"numba"` | Always use Numba JIT-compiled scalar loops. |

**Automatic fallback.** If the requested backend is not installed, the solver falls back to whichever is available and emits a `warnings.warn()` plus a `logging.warning()` message so the switch is never silent.

**Why two backends?**

Benchmarked on DECM at N = 50 000 / 100 000 / 200 000 (2026-07-13):

* **Peak RAM is essentially identical between the two backends at every scale tested** (≈4.5 GB at N=50k, ≈7.2 GB at N=100k, ≈16.9 GB at N=200k for both PyTorch-chunked and Numba). Numba does **not** offer a RAM advantage in practice — the widely-assumed "Numba saves RAM" rationale is not borne out by measurement.
* **Speed** is the only real differentiator, and it is scale-dependent: **PyTorch is faster up to N ≈ 50 000** (e.g. ≈307 s/iter vs ≈373 s/iter at N=50k on a representative server), while **Numba becomes ≈13–14% faster only at N ≥ 100 000** (e.g. 818 s/iter vs 955 s/iter at N=100k, 3305 s/iter vs 3820 s/iter at N=200k). The `"auto"` threshold (100 000) is set at this measured crossover point.
* **Numba is not required for old Python versions either.** PyTorch alone runs correctly on Python 3.8 (tested); `pyproject.toml`'s `requires-python` floor is a packaging choice, not a technical necessity tied to the backend.
* **Numba** (optional: `pip install numba`) compiles the update loop to native code and is parallelised with `prange` (OpenMP/TBB) so it can use multiple CPU cores; this is its main advantage over PyTorch at very large N, independent of RAM.

**Controlling the number of threads (Numba only).**  Each `solve_tool()` accepts a `num_threads` parameter:

```python
model.solve_tool(backend="numba", num_threads=4)   # use 4 threads
model.solve_tool(backend="numba", num_threads=0)   # auto: all CPUs available to the process
```

`num_threads=0` (default) automatically uses all CPUs visible to the current process via `os.sched_getaffinity()` on Linux (respects `taskset`/cgroup quotas) or `os.cpu_count()` elsewhere.  Positive values are **clamped** to the available CPU count so requesting more threads than the OS allows never raises a `libgomp: Thread creation failed` error on shared or resource-limited servers.

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

To install with Numba support:

```bash
pip install dcms[numba]          # installs numba as an optional extra
# or
pip install dcms numba           # equivalent
```

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

Stagnation and repeated-blowup recovery are now **built into the DECM solver itself** (§2.6) — a single unattended `solve_tool()`/`solve_fixed_point_decm[_degenerate]` call handles those on its own via `patience`/`noise_base`/`noise_cap_mult`/`max_stalls`, no external orchestration needed.

What a single in-process call *cannot* survive is an actual **crash, timeout, or preemption** of the process itself — a real concern for instances that need tens of thousands of iterations (hours) unattended. For that, run in **chunks** — `max_iter` iterations at a time, checkpointing state to disk after each chunk, so a killed process never loses more than one chunk of progress:

```python
global_best_theta, best_mre = None, float("inf")
theta = None  # None => solver picks the default ic on the first chunk

for chunk in range(n_chunks):
    res = solve_fixed_point_decm_degenerate(
        ..., theta0=theta,
        init_best_theta=global_best_theta, init_best_res=best_mre,
        max_iter=chunk_iters,
        # patience/noise_base/noise_cap_mult/max_stalls: leave at their
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

**Validated on a real hard instance:** DECM on an empirical online social network (N=15 168, M=3 003 after degeneracy reduction, §2.5, several hub nodes), run unattended from scratch with `anderson_depth=10`, `hub_sk_threshold=5.0`, `patience=750`, converged after 9 460 iterations to MRE=9.45×10⁻⁶ with no manual intervention — first as a chunked+checkpointed prototype (an earlier iteration of this same recovery idea, before it moved into the solver itself), then reproduced with the built-in mechanism (§2.6).

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

All benchmarks use `k_s_generator_pl(N, rho=1e-3)` (power-law degree/strength sequences), `tol = 1e-5`, and the `--fast` flag.  Statistics are mean ± 2σ over converged runs.

### DCM — N = 5 000

Benchmark: 5 seeds (0–4), `k_s_generator_pl(N=5000, rho=1e-3)`, `tol=1e-5`.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| FP-GS Anderson(10)    | **100%** |  8 ± 3 | 3.52 ± 1.26 | 1.38e-06 ± 9.28e-07 |
| θ-Newton Anderson(10) | **100%** | 13 ± 1 | 3.67 ± 0.78 | 8.94e-07 ± 1.30e-06 |

### DWCM — N = 5 000

Benchmark over 5 seeds (0–4), `k_s_generator_pl(N=5000, rho=1e-3)`.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| FP-GS Anderson(10) | **100%** | 24 ± 37 | 15.5 ± 33.8 | 9.5e-09 ± 3.2e-08 |
| θ-Newton Anderson(10) | **100%** | 14 ± 7 | 11.2 ± 5.3 | 2.5e-08 ± 3.0e-08 |

> The z-floor and Anderson blowup-reset mechanisms make both methods reliable even on hard seeds (high s/k hubs) that previously caused divergence.  θ-Newton is more consistent (lower variance in time and iterations).

### qDECM — N = 5 000

Benchmark over 5 seeds (0–4), `k_s_generator_pl(N=5000, rho=1e-3)`, 150 s per solver.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| FP-GS Anderson(10) | 0% | — | — | — |
| **θ-Newton Anderson(10)** | **100%** | **44 ± 10** | **36.1 ± 17.3** | **7.6e-08 ± 1.4e-07** |

> FP-GS Anderson(10) fails for qDECM at N = 5 000 because the conditioned weight equations have spectral radius > 1 for power-law hubs: each `p_ij < 1` factor forces `β_i β_j` closer to 1 to satisfy the strength constraint, amplifying the fixed-point Jacobian.  The θ-Newton approach bypasses this limitation by working in log-space where the diagonal Hessian always stabilises the step.

**Hard instances (s/k ≫ 1):** for real-world networks with a handful of nodes whose strength-to-degree ratio greatly exceeds 5, even θ-Newton can stagnate.  Use `hub_sk_threshold=5` (see §2.3); tested on an empirical online social network (N=22 754, directed, with a hub reaching s/k = 152) where the standard solver stalled at MRE≈0.47 and the bisection option achieved **best_MRE=9.5×10⁻⁵** in 46 iterations.

**Extreme hubs that cause Newton divergence:** on networks where the Newton step itself produces a large residual spike (MRE jumps from ~1 to 3–4), use the backtracking line search (see §2.4).  Combine `hub_sk_threshold` with `backtracking_gamma=1.2–2.0` and `anderson_depth=3–5` for the most challenging cases.  Convergence is guaranteed to be monotone but may be slow for very large networks (N > 10 000) where each residual evaluation is O(N²).

### DECM — N = 1 000 and N = 5 000

Benchmarks over 5 seeds each (`k_s_generator_pl(N, rho=1e-3)`, `tol=1e-5`).

The DECM uses the alternating GS-Newton solver (`solve_fixed_point_decm`), which applies θ-Newton steps on both the degree (θ) and strength (η) multipliers within each iteration.  Anderson(10) is applied on the full 4N vector.  `solve_tool()` uses `multi_start=True` by default: if the primary IC ("degrees") does not converge, it automatically retries with the "qdecm" warm-start (run qDECM first and use its 4N solution as starting point) and then "random".

**N = 1 000**

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| **θ-Newton Anderson(10)** | **100%** | **45 ± 8** | **2.3 ± 1.8** | **8.05e-07 ± 6.37e-07** |

**N = 5 000**

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| **θ-Newton Anderson(10)** | **100%** | **67 ± 20** | **77.9 ± 22.8** | **1.50e-07 ± 2.58e-07** |

> The coupling between degree and strength equations makes the DECM more expensive per iteration than the qDECM (two passes over the N×N grid instead of one), but the alternating GS-Newton strategy with multi-start achieves 100% convergence across all tested seeds.  Hard seeds (high s/k hubs) that the "degrees" IC cannot handle are resolved by the "qdecm" warm-start fallback.

---

## 5. Complexity

| Method | Model | Convergence | RAM per iteration | Scales to large N? |
|--------|-------|-------------|-------------------|--------------------|
| FP-GS Anderson(10) | DCM, DWCM, qDECM | linear + acceleration | O(chunk × N) | ✓ (chunked path for N > 5 000, or N > 2 000 for qDECM) |
| θ-Newton Anderson(10) | DCM, DWCM, qDECM | superlinear | O(chunk × N) | ✓ (same chunked path) |
| Alternating GS-Newton Anderson(10) | DECM | superlinear | O(chunk × N) | ✓ (2 passes per iteration, chunked path for N > 2 000) |

All methods are **O(N)** in RAM (with the default chunked path) and **O(N²)** in compute per iteration.  The dense path materialises the full N×N matrix once per step (threshold: N ≤ 5 000 for DCM/DWCM, N ≤ 2 000 for qDECM/DECM); above the threshold rows are processed in chunks of 512, keeping peak RAM under ~1 GB at N = 50 000.

The DECM solver performs 2 passes per iteration (out-group and in-group), compared to 1 pass for DCM/DWCM and 2 passes for qDECM.  This makes the per-iteration cost approximately equal to qDECM.

**Degeneracy-reduced variants** (§2.5) replace the O(N²) pairwise compute with O(M²), where `M` is the number of distinct sufficient-statistics groups — on real-world networks with heavy-tailed degree/strength distributions, typically `M ≪ N`, giving 10–67× measured wall-clock speedups.

---

## Running Tests

```bash
pytest tests/
```

## Running Benchmarks

```bash
# DCM comparison (two methods, N=1000, 10 seeds)
python -m dcms.benchmarks.dcm_comparison --sizes 1000 --n_seeds 10 --fast

# DCM scaling across sizes
python -m dcms.benchmarks.dcm_scaling --sizes 1000 5000 10000

# DWCM comparison
python -m dcms.benchmarks.dwcm_comparison --sizes 1000 --n_seeds 10 --fast

# DWCM at N=5000
python -m dcms.benchmarks.dwcm_comparison --sizes 5000 --n_seeds 5 --fast

# qDECM comparison (N=1000)
python -m dcms.benchmarks.qdecm_comparison --sizes 1000 --n_seeds 10 --fast

# qDECM at N=5000 (θ-Newton only reliable method)
python -m dcms.benchmarks.qdecm_comparison --sizes 5000 --n_seeds 5 --timeout 0 --fast

# DECM comparison (N=1000, 10 seeds)
python -m dcms.benchmarks.decm_comparison --n 1000 --n_seeds 10

# DECM at custom size/seeds
python -m dcms.benchmarks.decm_comparison --n 500 --n_seeds 5
```

## References

1. Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, **13**, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

2. Park, J. & Newman, M.E.J. (2004). Statistical mechanics of networks. *Physical Review E*, **70**, 066117. https://doi.org/10.1103/PhysRevE.70.066117

3. Vallarano, N., Bruno, M., Marchese, E., Trapani, G., Saracco, F., Cimini, G., Zanon, M. & Squartini, T. (2021). Fast and scalable likelihood maximization for exponential random graph models with local constraints. *Scientific Reports*, **11**, 15227. https://doi.org/10.1038/s41598-021-93830-4 *(NEMtropy)*

4. Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, **49**(4), 1715–1735. https://doi.org/10.1137/10078356X

5. Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM.  Chapter 5.
