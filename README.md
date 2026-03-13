# DCMS — Maximum-Entropy Solvers for Directed Networks

This project implements numerical solvers for **maximum-entropy** (MaxEnt) models of directed networks.  Given an observed graph, the models find the probability distribution over all directed graphs that maximises entropy subject to reproducing a chosen set of topological constraints (degree and/or strength sequences).

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

---

## 2. Solver Methods

All solvers return a `SolverResult` dataclass with fields `theta`, `converged`, `iterations`, `residuals`, `elapsed_time`, `peak_ram_bytes`, and `message`.

### 2.1 Fixed-Point Iteration

The simplest solver.  In the **Gauss-Seidel** variant, out-multipliers are updated first, and the fresh values are immediately used when computing in-multipliers.  In the **Jacobi** variant, all multipliers are updated simultaneously from the previous step.  An optional damping factor `α ∈ (0, 1]` slows the step to improve stability.

- **Convergence order:** linear
- **RAM complexity:** O(N) — never materialises the N×N matrix for N > 5 000
- **Recommended for:** all network sizes; Gauss-Seidel with `α = 1` is the fastest variant on typical networks

**Literature:**  
Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, 13, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

### 2.2 L-BFGS (Quasi-Newton)

Minimises the negative log-likelihood `−L(θ)` using the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm with `m` stored curvature pairs and a Wolfe-condition line search.  This is the **default method** in [NEMtropy](https://github.com/nicoloval/NEMtropy) and generally the best trade-off between speed and robustness.

- **Convergence order:** superlinear
- **RAM complexity:** O(N · m) with m = 20 → effectively O(N)
- **Recommended for:** all network sizes; best general-purpose method

**Literature:**  
Liu, D.C. & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1–3), 503–528. https://doi.org/10.1007/BF01589116

Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapter 7.

### 2.3 Newton (full Jacobian)

Solves the linearised system `(−J + εI) δθ = F(θ)` at each step via `torch.linalg.solve`, where `J = ∂F/∂θ` is the exact Jacobian and `ε` is a Tikhonov regularisation parameter.  A backtracking Armijo line search is applied.

- **Convergence order:** quadratic
- **RAM complexity:** O(N²) — impractical for N > ~2 000
- **Recommended for:** small networks (N ≤ 2 000) where quadratic convergence matters

**Literature:**  
Kelley, C.T. (1995). *Iterative Methods for Linear and Nonlinear Equations*. SIAM. Chapter 5.

Squartini & Garlaschelli (2011), op. cit.

### 2.4 Broyden (rank-1 Jacobian updates)

Computes the exact Jacobian once at the first step, then uses the Sherman-Morrison formula to update the inverse Jacobian with rank-1 corrections.

- **Convergence order:** superlinear
- **RAM complexity:** O(N²)
- **Recommended for:** medium networks (500 ≤ N ≤ 2 000)

**Literature:**  
Broyden, C.G. (1965). A class of methods for solving nonlinear simultaneous equations. *Mathematics of Computation*, 19(92), 577–593. https://doi.org/10.2307/2003941

### 2.5 Levenberg-Marquardt (LM)

Solves the regularised normal equations `(JᵀJ + λI) δθ = −Jᵀ F(θ)` with an adaptive damping parameter `λ` that increases on rejection and decreases on acceptance.  With `diagonal_only=True` only the diagonal of `JᵀJ` is used, reducing the per-step cost.

- **Convergence order:** quadratic (locally)
- **RAM complexity:** O(N²) for full mode; O(N) when using only the Hessian diagonal
- **Recommended for:** situations where robustness is paramount

**Literature:**  
Moré, J.J. (1978). The Levenberg-Marquardt algorithm: Implementation and theory. In Watson, G.A. (ed.), *Numerical Analysis*, Lecture Notes in Mathematics 630, 105–116. Springer. https://doi.org/10.1007/BFb0067700

---

## 3. Main Functions and Parameters

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
    s_out, s_in,   # observed strength sequences
    tol=1e-8,
    max_iter=10_000,
    damping=1.0,
)
```

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

Benchmark over **10 random networks** generated with `k_s_generator_pl(N=1000, rho=1e-3)`.  
Convergence tolerance: `tol = 1e-6`.  All methods converged on all 10 networks.

### DCM (binary)

| Method | Conv. | Iters (avg) | Time (s, avg) |
|--------|------:|------------:|--------------:|
| Fixed-point GS α=1.0 | 100% | 5.2 | 0.028 |
| Fixed-point GS α=0.5 | 100% | 26.0 | 0.117 |
| Fixed-point Jacobi | 100% | 1 305.5 | 5.742 |
| L-BFGS (m=20) | 100% | 22.7 | 0.548 |
| Newton (exact J) | 100% | 4.8 | 0.455 |
| Broyden (rank-1 J) | 100% | 24.1 | 0.637 |
| LM (full Jacobian) | 100% | 5.3 | 2.543 |

> **Note:** Newton, Broyden and full-Jacobian LM require O(N²) RAM
> (~32 MB at N=1 000) and are not recommended for N > 2 000.

---

## 5. Performance — N = 5 000 nodes

At N = 5 000, methods that require an explicit N × N Jacobian (~800 MB) are not applicable.  The table below covers only the O(N) and O(N·m) methods.  Networks generated with `k_s_generator_pl(N=5000, rho=1e-3)` (mean degree ≈ 5, max degree ≈ 70).

### DCM (binary)

| Method | Conv. | Notes |
|--------|------:|-------|
| Fixed-point GS α=1.0 | 100% | Fastest; typically < 10 iterations |
| Fixed-point GS α=0.5 | 100% | More stable on heterogeneous networks |
| Fixed-point Jacobi | 100% | Slow — thousands of iterations; not recommended |
| L-BFGS (m=20) | 100% | Best convergence rate among scalable methods |
| Newton (exact J) | — | Skipped — O(N²) RAM |
| Broyden (rank-1 J) | — | Skipped — O(N²) RAM |
| LM (full Jacobian) | — | Skipped — O(N²) RAM |

> At N = 5 000 each fixed-point GS iteration materialises a 5 000 × 5 000
> matrix (~200 MB), so wall-clock time per seed is dominated by the number
> of iterations. L-BFGS converges in ~20–30 iterations and is the
> recommended method at this scale.

---

## Complexity Summary

| Method | Convergence | RAM | Recommended N |
|--------|-------------|-----|---------------|
| Fixed-point GS | linear | O(N) | all sizes |
| L-BFGS | superlinear | O(N·m) | all sizes — **default** |
| Newton | quadratic | O(N²) | ≤ 2 000 |
| Broyden | superlinear | O(N²) | ≤ 2 000 |
| LM | quadratic | O(N²) / O(N) | all sizes (diagonal mode) |

---

## Running Tests

```bash
pytest tests/
```

## Running Benchmarks

```bash
# DCM single network
python -m src.benchmarks.dcm_comparison --n 100 --seed 42

# DCM multi-seed comparison
python -m src.benchmarks.dcm_comparison --n 1000 --n_seeds 10

# DCM scaling (N = 1k … 50k)
python -m src.benchmarks.dcm_scaling --sizes 1000 5000 10000

# DWCM comparison
python -m src.benchmarks.dwcm_comparison --n 1000 --n_seeds 10
```

## References

1. Squartini, T. & Garlaschelli, D. (2011). Analytical maximum-likelihood method to detect patterns in real networks. *New Journal of Physics*, **13**, 083001. https://doi.org/10.1088/1367-2630/13/8/083001

2. Park, J. & Newman, M.E.J. (2004). Statistical mechanics of networks. *Physical Review E*, **70**, 066117. https://doi.org/10.1103/PhysRevE.70.066117

3. Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer.

4. Broyden, C.G. (1965). A class of methods for solving nonlinear simultaneous equations. *Mathematics of Computation*, **19**(92), 577–593. https://doi.org/10.2307/2003941

5. Moré, J.J. (1978). The Levenberg-Marquardt algorithm: Implementation and theory. In *Numerical Analysis*, Lecture Notes in Mathematics 630. Springer. https://doi.org/10.1007/BFb0067700

6. Vallarano, N., Bruno, M., Marchese, E., Barabási, A.-L., Squartini, T. & Garlaschelli, D. (2021). Fast and scalable likelihood maximization for exponential random graph models with local constraints. *Scientific Reports*, **11**, 15227. https://doi.org/10.1038/s41598-021-94118-5 *(NEMtropy)*
