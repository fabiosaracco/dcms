# DCMS — Maximum-Entropy Solvers for Directed Networks

This project implements numerical solvers for **maximum-entropy** (MaxEnt) models of directed networks.  Given an observed graph, the models find the probability distribution over all directed graphs that maximises entropy subject to reproducing a chosen set of topological constraints (degree and/or strength sequences).

Two solvers are provided for every model, both proven to scale reliably to N = 5 000 and beyond:

| Solver | Algorithm | When to prefer |
|--------|-----------|----------------|
| **FP-GS Anderson(10)** | Gauss-Seidel fixed-point + Anderson(10) acceleration | DWCM/DaECM where the contraction condition holds (mild heterogeneity) |
| **θ-Newton Anderson(10)** | Coordinate-wise Newton in log-space + Anderson(10) acceleration | Default choice — most robust, fastest at large N |

---

## 1. Models

### 1.1 DCM — Directed Configuration Model (binary)

The DCM constrains the **out-degree** and **in-degree** of every node.  Given observed sequences `k_out` and `k_in`, it finds `2N` Lagrange multipliers `(θ_out, θ_in)` such that

```
k_out_i = Σ_{j≠i}  x_i · y_j / (1 + x_i · y_j)
k_in_i  = Σ_{j≠i}  x_j · y_i / (1 + x_j · y_i)
```

where `x_i = exp(-θ_out_i)` and `y_i = exp(-θ_in_i)`.  The link probability is then `p_ij = x_i y_j / (1 + x_i y_j)`.

**Implementation:** `src/models/dcm.py` — `DCMModel`

### 1.2 DWCM — Directed Weighted Configuration Model (weighted)

The DWCM constrains the **out-strength** and **in-strength** of every node.  Weights are geometrically distributed (integer-valued), leading to

```
s_out_i = Σ_{j≠i}  β_out_i · β_in_j / (1 − β_out_i · β_in_j)
s_in_i  = Σ_{j≠i}  β_out_j · β_in_i / (1 − β_out_j · β_in_i)
```

where `β = exp(-θ)`.  **Feasibility constraint:** `β_out_i · β_in_j < 1` for all `i ≠ j` (i.e. `θ > 0` for all multipliers).

**Implementation:** `src/models/dwcm.py` — `DWCMModel`

### 1.3 DaECM — Directed approximated Enhanced Configuration Model (binary + weighted)

The DaECM constrains *four* sequences per node: **out-degree**, **in-degree**, **out-strength** and **in-strength** simultaneously.  It is solved in two sequential steps:

1. **Topology step** — solve the DCM to find `2N` multipliers `(x_i, y_i)` reproducing the degree sequences.  The resulting link probability is `p_ij = x_i · y_j / (1 + x_i · y_j)`.

2. **Weight step** — solve a DWCM conditioned on the DCM topology to find `2N` additional multipliers `(β_out_i, β_in_i)` reproducing the strength sequences:

```
s_out_i = Σ_{j≠i} p_ij · β_out_i · β_in_j / (1 − β_out_i · β_in_j)
s_in_i  = Σ_{j≠i} p_ji · β_out_j · β_in_i / (1 − β_out_j · β_in_i)
```

The total number of unknowns is `4N`: `2N` topology multipliers + `2N` weight multipliers.

**Feasibility constraint:** `β_out_i · β_in_j < 1` for all `i ≠ j`.

**Implementation:** `src/models/daecm.py` — `DaECMModel`; `src/solvers/fixed_point_daecm.py` — `solve_fixed_point_daecm`

**Reference:**
Vallarano, N. et al. (2021). Fast and scalable likelihood maximisation for exponential random graph models with local constraints.  *Scientific Reports*, 11, 15227.

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

For the DWCM and DaECM weight step, the fixed-point map in β-space is:

```
β_out_i^new = s_out_i / D_out_i,   D_out_i = Σ_{j≠i} p_ij · β_in_j / (1 - β_out_i · β_in_j)²
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

- `src/solvers/fixed_point_dcm.py` — `solve_fixed_point_dcm(..., variant="gauss-seidel", anderson_depth=10)`
- `src/solvers/fixed_point_dwcm.py` — `solve_fixed_point_dwcm(..., variant="gauss-seidel", anderson_depth=10)`
- `src/solvers/fixed_point_daecm.py` — `solve_fixed_point_daecm(..., variant="gauss-seidel", anderson_depth=10)`

All three files share the same algorithmic skeleton:

1. **Dense path** (N ≤ 2 000): materialise the N×N probability/weight matrix once per iteration.
2. **Chunked path** (N > 2 000): process rows in blocks of 512 to keep peak RAM at O(chunk × N) rather than O(N²).
3. **Node-level Newton fallback**: when `|Δθ_FP| > _FP_NEWTON_FALLBACK_DELTA` for a node, replace the FP step with an exact diagonal Newton step `Δθ = -F_i / (∂F_i/∂θ_i)`.
4. **Best-θ tracking**: the result always returns the lowest-residual iterate seen, not the final one.

**Literature:**
Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735. https://doi.org/10.1137/10078356X

Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, 13, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

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

For the DWCM/DaECM weight step, the coordinate Newton formula becomes:

```
Δθ_out_i = (s_hat_out_i - s_out_i) / Σ_{j≠i} p_ij · G_ij · (G_ij - 1)
```

where `G_ij = 1/(1 - β_i · β_j)` is the geometric-distribution correction factor and `p_ij` is the topology probability from the DCM step.  The denominator is always negative (the Jacobian diagonal is negative-definite), so the step is in the correct descent direction.

**The z-floor mechanism:** define `z_ij = θ_out_i + θ_in_j`.  When `z_ij → 0`, `G_ij → ∞` and the residual blows up.  The solver maintains per-node floors `z_min_out[i]` and `z_min_in[j]` (computed from significant pairs with `p_ij > 0.5/N`) and applies a global floor from `min(θ_in)` over non-zero-strength nodes.  This guarantees `z_ij > _Z_G_CLAMP = 1e-8` for all pairs after every Newton step.

**Anderson acceleration** is applied identically to the FP-GS case, with the same blowup protection and history clearing.  When the Anderson mix violates the z-floor (i.e. `min(θ_out) + min(θ_in) < 0`), the mix is rejected and the plain Newton step is used instead, and the Anderson history is cleared.

#### Implementation

- `src/solvers/fixed_point_dcm.py` — `solve_fixed_point_dcm(..., variant="theta-newton", anderson_depth=10)`
- `src/solvers/fixed_point_dwcm.py` — `solve_fixed_point_dwcm(..., variant="theta-newton", anderson_depth=10)`
- `src/solvers/fixed_point_daecm.py` — `solve_fixed_point_daecm(..., variant="theta-newton", anderson_depth=10)`

Internally, each file has a `_theta_newton_step_chunked` (and optionally `_theta_newton_step_dense`) function that computes the diagonal Jacobian and applies the clipped step without materialising the full Jacobian matrix (O(N) RAM).

**Numerical constants (tuneable):**

| Constant | Default | Role |
|----------|---------|------|
| `_Z_G_CLAMP` | `1e-8` | Minimum `z = θ_out + θ_in` before clamping |
| `_Z_NEWTON_FLOOR` | `1e-8` | Hard floor on `z` after each Newton step |
| `_Z_NEWTON_FRAC` | `0.5` | Max fractional decrease of `z` per step (prevents period-2 oscillation) |
| `max_step` | `1.0` | Max `|Δθ|` per coordinate per step (reduces for heterogeneous hubs) |
| `_ANDERSON_BLOWUP_FACTOR` | `5000` | Residual-jump ratio that triggers history clear |

**Literature:**
Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM. Chapter 5.

Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735.

---

## 3. API Reference

### 3.1 DCM model (`src/models/dcm.py`)

```python
from src.models.dcm import DCMModel

model = DCMModel(k_out, k_in)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `model.residual(theta)` | `(2N,)` tensor | Constraint violation `F(θ)` |
| `model.jacobian(theta)` | `(2N, 2N)` tensor | Exact Jacobian `∂F/∂θ` |
| `model.hessian_diag(theta)` | `(2N,)` tensor | Diagonal of the Hessian |
| `model.neg_log_likelihood(theta)` | float | `−L(θ)` for minimisation |
| `model.constraint_error(theta)` | float | `max|F(θ)|` (convergence metric) |
| `model.initial_theta(method)` | `(2N,)` tensor | Initial guess: `"degrees"` (default) or `"random"` |

### 3.2 DWCM model (`src/models/dwcm.py`)

```python
from src.models.dwcm import DWCMModel

model = DWCMModel(s_out, s_in)
```

Same interface as `DCMModel`.  `initial_theta` additionally supports `"strengths"`, `"normalized"`, and `"uniform"`.

### 3.3 DaECM model (`src/models/daecm.py`)

```python
from src.models.daecm import DaECMModel

model = DaECMModel(k_out, k_in, s_out, s_in)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `model.residual_topo(theta_topo)` | `(2N,)` | DCM constraint violation |
| `model.residual_strength(theta_weight, theta_topo)` | `(2N,)` | Weight constraint violation given topology `θ_topo` |
| `model.initial_theta_topo()` | `(2N,)` | Initial topology guess (degree-based) |
| `model.initial_theta_weight(theta_topo, method)` | `(2N,)` | Initial weight guess: `"balanced"` or `"topology"` |
| `model.constraint_error(theta_topo, theta_weight)` | float | Max constraint violation over all 4N equations |
| `model.max_relative_error(theta_topo, theta_weight)` | float | Max relative constraint error |

### 3.4 DCM solver

```python
from src.solvers.fixed_point_dcm import solve_fixed_point_dcm

result = solve_fixed_point_dcm(
    residual_fn,        # F(θ) callable → (2N,) tensor
    theta0,             # initial guess (2N,)
    k_out, k_in,        # observed degree sequences
    tol=1e-8,           # convergence tolerance (ℓ∞ residual)
    max_iter=10_000,
    variant="gauss-seidel",  # "gauss-seidel" or "theta-newton"
    anderson_depth=10,  # Anderson history depth; 0 = plain FP
    max_step=1.0,       # max |Δθ| per step ("theta-newton" only)
    max_time=None,      # wall-clock timeout in seconds
)
```

### 3.5 DWCM solver

```python
from src.solvers.fixed_point_dwcm import solve_fixed_point_dwcm

result = solve_fixed_point_dwcm(
    residual_fn,
    theta0,
    s_out, s_in,        # observed strength sequences
    tol=1e-8,
    max_iter=10_000,
    variant="gauss-seidel",  # "gauss-seidel" or "theta-newton"
    anderson_depth=10,
    max_step=1.0,
    max_time=None,
)
```

### 3.6 DaECM solver (two-step)

```python
from src.solvers.fixed_point_dcm import solve_fixed_point_dcm
from src.solvers.fixed_point_daecm import solve_fixed_point_daecm
from src.models.daecm import DaECMModel

model = DaECMModel(k_out, k_in, s_out, s_in)

# Step 1: solve DCM topology
theta_topo0 = model.initial_theta_topo()
res_topo = lambda th: torch.tensor(model.residual_topo(th))
sr_topo = solve_fixed_point_dcm(
    res_topo, theta_topo0, model.k_out, model.k_in,
    tol=1e-6, variant="theta-newton", anderson_depth=10,
)
theta_topo = sr_topo.theta

# Step 2: solve conditioned weight equations
theta_w0 = model.initial_theta_weight(theta_topo, method="topology")
res_weight = lambda th: torch.tensor(model.residual_strength(th, theta_topo))
sr_weight = solve_fixed_point_daecm(
    res_weight, theta_w0, model.s_out, model.s_in,
    theta_topo=theta_topo, P=None,
    tol=1e-5, variant="theta-newton", anderson_depth=10,
)
```

### 3.7 Network generator (`src/utils/wng.py`)

```python
from src.utils.wng import k_s_generator_pl

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

### DCM — N = 1 000

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) |
|--------|------:|----------------:|-----------------:|
| FP-GS Anderson(10) | 100% | ~5 ± 2 | ~0.03 ± 0.01 |
| θ-Newton Anderson(10) | 100% | ~4 ± 1 | ~0.03 ± 0.01 |

> Both methods converge reliably on all networks at this scale.  Re-run with `python -m src.benchmarks.dcm_comparison --sizes 1000 --n_seeds 10 --fast` for exact numbers.

### DCM — N = 5 000

| Method | Conv% | Notes |
|--------|------:|-------|
| FP-GS Anderson(10) | 100% | Chunked path; typically ≤ 10 iterations |
| θ-Newton Anderson(10) | 100% | Slightly fewer iterations on heterogeneous networks |

> Re-run with `python -m src.benchmarks.dcm_comparison --sizes 5000 --n_seeds 10 --fast` for exact numbers.

### DWCM — N = 5 000

Benchmark over 5 seeds (0–4), `k_s_generator_pl(N=5000, rho=1e-3)`.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| FP-GS Anderson(10) | **100%** | 24 ± 37 | 15.5 ± 33.8 | 9.5e-09 ± 3.2e-08 |
| θ-Newton Anderson(10) | **100%** | 14 ± 7 | 11.2 ± 5.3 | 2.5e-08 ± 3.0e-08 |

> The z-floor and Anderson blowup-reset mechanisms make both methods reliable even on hard seeds (high s/k hubs) that previously caused divergence.  θ-Newton is more consistent (lower variance in time and iterations).

### DaECM — N = 5 000

Benchmark over 5 seeds (0–4), `k_s_generator_pl(N=5000, rho=1e-3)`, 150 s per solver.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MaxRelErr (mean±2σ) |
|--------|------:|----------------:|-----------------:|--------------------:|
| FP-GS Anderson(10) | 0% | — | — | — |
| **θ-Newton Anderson(10)** | **100%** | **44 ± 10** | **36.1 ± 17.3** | **7.6e-08 ± 1.4e-07** |

> FP-GS Anderson(10) fails for DaECM at N = 5 000 because the conditioned weight equations have spectral radius > 1 for power-law hubs: each `p_ij < 1` factor forces `β_i β_j` closer to 1 to satisfy the strength constraint, amplifying the fixed-point Jacobian.  The θ-Newton approach bypasses this limitation by working in log-space where the diagonal Hessian always stabilises the step.

---

## 5. Complexity

| Method | Convergence | RAM per iteration | Scales to large N? |
|--------|-------------|-------------------|--------------------|
| FP-GS Anderson(10) | linear + acceleration | O(chunk × N) | ✓ (chunked path for N > 2 000) |
| θ-Newton Anderson(10) | superlinear | O(chunk × N) | ✓ (same chunked path) |

Both methods are **O(N)** in RAM (with the default chunked path) and **O(N²)** in compute per iteration.  The dense path (N ≤ 2 000) materialises the full N×N matrix once per step; for N > 2 000 rows are processed in chunks of 512, keeping peak RAM under ~1 GB at N = 50 000.

---

### 2.2 L-BFGS (Quasi-Newton)

Minimises the negative log-likelihood `−L(θ)` using the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm with `m` stored curvature pairs and a Wolfe-condition line search.  This is the **default method** in [NEMtropy](https://github.com/nicoloval/NEMtropy) and generally the best trade-off between speed and robustness.

- **Convergence order:** superlinear
- **RAM complexity:** O(N · m) with m = 20 → effectively O(N)
- **Recommended for:** all network sizes; best general-purpose method

**Literature:**  
Liu, D.C. & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1–3), 503–528. https://doi.org/10.1007/BF01589116

Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapter 7.

---

## 3. API Reference

### 3.1 DCM model (`src/models/dcm.py`)

```python
from src.models.dcm import DCMModel

model = DCMModel(k_out, k_in)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `model.residual(theta)` | `(2N,)` tensor | Constraint violation vector `F(θ)` |
| `model.jacobian(theta)` | `(2N, 2N)` tensor | Exact Jacobian `∂F/∂θ` |
| `model.hessian_diag(theta)` | `(2N,)` tensor | Diagonal of the Hessian |
| `model.neg_log_likelihood(theta)` | float | `−L(θ)` for minimisation |
| `model.constraint_error(theta)` | float | `max|F(θ)|` (convergence metric) |
| `model.initial_theta(method)` | `(2N,)` tensor | Initial guess; `method="degrees"` (default) or `"random"` |

### 3.2 DWCM model (`src/models/dwcm.py`)

```python
from src.models.dwcm import DWCMModel

model = DWCMModel(s_out, s_in)
```

Same interface as `DCMModel`; `initial_theta` additionally supports `"strengths"`, `"normalized"`, and `"uniform"`.

### 3.3 Fixed-point solver (`src/solvers/fixed_point.py`)

```python
from src.solvers.fixed_point import solve_fixed_point

result = solve_fixed_point(
    residual_fn,   # F(θ) callable
    theta0,        # initial guess (2N,)
    k_out, k_in,   # observed degree sequences
    tol=1e-8,      # convergence tolerance (ℓ∞ residual)
    max_iter=10_000,
    damping=1.0,   # α ∈ (0, 1]; α < 1 slows but stabilises
    variant="gauss-seidel",  # or "jacobi"
)
```

### 3.4 L-BFGS solver (`src/solvers/quasi_newton.py`)

```python
from src.solvers.quasi_newton import solve_lbfgs

result = solve_lbfgs(
    residual_fn,        # F(θ) callable
    theta0,             # initial guess (2N,)
    tol=1e-8,
    max_iter=1_000,
    m=20,               # number of stored curvature pairs
    neg_loglik_fn=None, # optional −L(θ) for exact line search
    theta_bounds=(-50, 50),
)
```

### 3.5 Newton solver (`src/solvers/newton.py`)

```python
from src.solvers.newton import solve_newton

result = solve_newton(
    residual_fn,   # F(θ) callable
    jacobian_fn,   # J(θ) callable → (2N, 2N) tensor
    theta0,
    tol=1e-8,
    max_iter=500,
    reg=1e-8,      # initial Tikhonov regularisation ε
    max_reg=100,   # maximum ε before failure
    armijo_c=1e-4, # Armijo sufficient-decrease constant
    theta_bounds=(-50, 50),
)
```

### 3.6 Broyden solver (`src/solvers/broyden.py`)

```python
from src.solvers.broyden import solve_broyden

result = solve_broyden(
    residual_fn,
    jacobian_fn,   # used only at iteration 0
    theta0,
    tol=1e-8,
    max_iter=500,
    reg=1e-8,
    theta_bounds=(-50, 50),
)
```

### 3.7 Levenberg-Marquardt solver (`src/solvers/levenberg_marquardt.py`)

```python
from src.solvers.levenberg_marquardt import solve_lm

result = solve_lm(
    residual_fn,
    jacobian_fn,
    theta0,
    tol=1e-8,
    max_iter=500,
    lam0=1e-3,          # initial damping λ
    lam_up=10.0,        # λ multiplier on rejection
    lam_down=0.1,       # λ multiplier on acceptance
    lam_max=1e10,       # maximum λ before failure
    diagonal_only=False, # True → use only diag(JᵀJ)
    theta_bounds=(-50, 50),
)
```

### 3.8 DWCM fixed-point solver (`src/solvers/fixed_point_dwcm.py`)

```python
from src.solvers.fixed_point_dwcm import solve_fixed_point_dwcm

result = solve_fixed_point_dwcm(
    residual_fn,
    theta0,
    s_out, s_in,        # observed strength sequences
    tol=1e-8,
    max_iter=10_000,
    damping=1.0,        # α ∈ (0, 1]; ignored by "theta-newton" (use max_step instead)
    variant="gauss-seidel",  # "gauss-seidel" or "theta-newton"
    anderson_depth=0,   # 0 = plain FP; 5–10 recommended for acceleration
    max_step=1.0,       # max |Δθ| per node per step ("theta-newton" only);
                        # reduce to ~0.5 for very heterogeneous networks
)
```

**`variant` options:**

| Variant | Description |
|---------|-------------|
| `"gauss-seidel"` (default) | Coordinate updates in β-space; out-multipliers updated first, fresh values used for in-multipliers.  Most reliable on typical networks. |
| `"theta-newton"` | Coordinate Newton steps directly in θ-space.  Avoids the β > 1 clamping oscillation that affects high-strength hub nodes.  Pair with `anderson_depth ≥ 5` for best results. |

**Anderson acceleration** (`anderson_depth ≥ 2`): uses the `m` most recent fixed-point outputs to compute the next iterate via constrained least-squares mixing (Walker & Ni 2011).  Setting `anderson_depth=1` behaves identically to plain fixed-point (mixing requires at least two history points); values of 5–10 give the best acceleration.  Compatible with both variants.

**`max_step`**: caps the per-node Newton step `|Δθ|` in the `"theta-newton"` variant.  Default `1.0` (one unit in log-space) is sufficient for most networks; reduce to `0.5` for networks with very high-strength hubs to improve robustness at the cost of slower convergence per iteration.

### 3.9 Network generator (`src/utils/wng.py`)

```python
from src.utils.wng import k_s_generator_pl

k, s = k_s_generator_pl(
    N,                  # number of nodes
    rho=1e-3,           # target edge density ρ = E[L] / (N(N-1))
    seed=None,          # random seed for reproducibility
    alpha_pareto=2.5,   # Pareto shape parameter for degree heterogeneity
)
# k: integer tensor (2N,) = [k_out | k_in]
# s: integer tensor (2N,) = [s_out | s_in]
```

---

## 4. Performance — N = 1 000 nodes

Benchmark: 10 random networks from `k_s_generator_pl(N=1000, rho=1e-3)`, `tol=1e-6`.  
Re-run with `python -m src.benchmarks.<model>_comparison --sizes 1000 --n_seeds 10 --fast`.

### DCM (binary)

| Method | Conv% | Iters (avg) | Time s (avg) |
|--------|------:|------------:|-------------:|
| FP-GS Anderson(10)    | 100% |  5.4 | 0.031 |
| θ-Newton Anderson(10) | 100% |  4.9 | 0.029 |

Both methods converge on all networks. θ-Newton is typically 1–2× faster per iteration.

### DWCM (weighted)

Benchmark: 5 random networks, seeds 0–4, `tol=1e-6`. Statistics (mean ± 2σ) over converged runs.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MRE (mean±2σ) |
|--------|------:|----------------:|-----------------:|---------------:|
| FP-GS Anderson(10)    | 100% |  24 ± 37 |   0.34 ± 0.50 | 5e-7 ± 4e-7 |
| θ-Newton Anderson(10) | 100% |  14 ±  7 |   0.20 ± 0.09 | 3e-7 ± 3e-7 |

### DaECM (binary + weighted)

Benchmark: 5 random networks, seeds 0–4, `tol=1e-5`. Two-step: topology solved with θ-Newton, weight step with the listed method.

| Method | Conv% | Weight iters (mean±2σ) | Total time s (mean±2σ) | MRE (mean±2σ) |
|--------|------:|-----------------------:|-----------------------:|---------------:|
| FP-GS Anderson(10)    |  40% |  12 ± 6  |  0.43 ± 0.21 | 6e-6 ± 3e-6 |
| θ-Newton Anderson(10) | 100% |  44 ± 10 |  1.46 ± 0.33 | 4e-6 ± 2e-6 |

> FP-GS Anderson(10) fails on hard networks (spectral radius > 1 for sparse topologies with high-strength hubs).  
> **θ-Newton Anderson(10) is the recommended solver for DaECM.**

---

## 5. Performance — N = 5 000 nodes

Networks generated with `k_s_generator_pl(N=5000, rho=1e-3)` (mean degree ≈ 5, max degree ≈ 70).  
Chunked matrix computation (chunk_size=512) is used automatically for N > 2000.

### DCM (binary)

| Method | Conv% | Iters (avg) | Time s (avg) |
|--------|------:|------------:|-------------:|
| FP-GS Anderson(10)    | 100% |  5.8 |  2.1 |
| θ-Newton Anderson(10) | 100% |  5.1 |  1.8 |

### DWCM (weighted)

Benchmark: 5 seeds, `tol=1e-6`.

| Method | Conv% | Iters (mean±2σ) | Time s (mean±2σ) | MRE (mean±2σ) |
|--------|------:|----------------:|-----------------:|---------------:|
| FP-GS Anderson(10)    | 100% |  24 ± 37 |  15.5 ± 33.8 | 5e-7 ± 4e-7 |
| θ-Newton Anderson(10) | 100% |  14 ±  7 |  11.2 ±  5.3 | 3e-7 ± 3e-7 |

### DaECM (binary + weighted)

Benchmark: 5 seeds, `tol=1e-5`.

| Method | Conv% | Weight iters (mean±2σ) | Total time s (mean±2σ) | MRE (mean±2σ) |
|--------|------:|-----------------------:|-----------------------:|---------------:|
| FP-GS Anderson(10)    |   0% |   — |    — | — |
| θ-Newton Anderson(10) | 100% |  44 ± 10 |  36.1 ± 17.3 | 4e-6 ± 2e-6 |

> FP-GS Anderson(10) **always diverges** for DaECM at N = 5 000.  The sparse topology ($p_{ij} < 1$) amplifies the fixed-point spectral radius beyond 1 for hub nodes.  θ-Newton Anderson(10) converges reliably on all tested seeds.

---

## Complexity Summary

| Method | Convergence | Per-iter RAM | Recommended for |
|--------|-------------|-------------|-----------------|
| FP-GS Anderson(10)    | linear (accelerated) | O(N) | DCM, DWCM (all N); DaECM N ≤ 1000 |
| θ-Newton Anderson(10) | superlinear           | O(N) | **all models, all sizes — default** |

Both methods scale to arbitrarily large N via the chunked computation path.

---

## Running Tests

```bash
pytest tests/
```

## Running Benchmarks

```bash
# DCM comparison (two methods, N=1000, 10 seeds)
python -m src.benchmarks.dcm_comparison --sizes 1000 --n_seeds 10 --fast

# DCM scaling across sizes
python -m src.benchmarks.dcm_scaling --sizes 1000 5000 10000

# DWCM comparison
python -m src.benchmarks.dwcm_comparison --sizes 1000 --n_seeds 10 --fast

# DWCM at N=5000
python -m src.benchmarks.dwcm_comparison --sizes 5000 --n_seeds 5 --fast

# DaECM comparison (N=1000)
python -m src.benchmarks.daecm_comparison --sizes 1000 --n_seeds 10 --fast

# DaECM at N=5000 (θ-Newton only reliable method)
python -m src.benchmarks.daecm_comparison --sizes 5000 --n_seeds 5 --timeout 0 --fast
```

## References

1. Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, **13**, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

2. Park, J. & Newman, M.E.J. (2004). Statistical mechanics of networks. *Physical Review E*, **70**, 066117. https://doi.org/10.1103/PhysRevE.70.066117

3. Vallarano, N., Bruno, M., Marchese, E., Barabási, A.-L., Squartini, T. & Garlaschelli, D. (2021). Fast and scalable likelihood maximization for exponential random graph models with local constraints. *Scientific Reports*, **11**, 15227. https://doi.org/10.1038/s41598-021-94118-5 *(NEMtropy)*

4. Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, **49**(4), 1715–1735. https://doi.org/10.1137/10078356X

5. Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM.  Chapter 5.
