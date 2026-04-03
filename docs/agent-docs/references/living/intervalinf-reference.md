# intervalinf — Package Reference

## Overview

`intervalinf` provides concrete implementations of Hilbert spaces for functions defined on 1D intervals, built directly on top of `pygeoinf`. The central design principle is **continuous-first**: functions are represented as callables over an `IntervalDomain` and inner products are computed by numerical integration, not matrix multiplication. Discretisation (choosing a finite basis) is opt-in. The package extends `pygeoinf`'s abstract algebra (Hilbert spaces, linear operators, linear forms, Gaussian measures) for the specific case where the model space is $L^2([a,b])$ or a Sobolev space $H^s([a,b])$.

## Demo Notebooks

The `intervalinf/demos/convex_analysis/` directory contains notebook-scale demonstrations of convex-analysis inversion workflows built on `intervalinf` and `pygeoinf`.

- `dli.ipynb`: deterministic linear inference workflow with dual master-cost optimization.
- `bg_with_errors_minkowski.ipynb`: Backus-Gilbert admissible-region construction with data errors using Minkowski-sum support-function algebra.
- `dli_vs_bg_polyhedral_comparison.ipynb`: side-by-side comparison notebook for DLI and BG in the 2D property-space setting with editable `N_d`, `N_p`, and common polyhedral resolution `N_theta`.
- `realistic_dli.ipynb`: **realistic DLI mirror** of `old_demos/paper_demos/example_1.ipynb`. Uses the same direct-sum model structure (`M_vp ⊕ M_vs ⊕ M_rho ⊕ M_σ₀ ⊕ M_σ₁`), the same `SensitivityKernelCatalog`/`SensitivityKernelProvider` operator construction pattern, and a chi-squared deterministic data error set — reduced to N_d=5, N_p=2. The functional model blocks are now intentionally **basis-free** (`dim=0`, `basis=None`) and the prior is built **operator-theoretically**, not by assembling dense covariance matrices: `BesselSobolevInverse` supplies covariance blocks, `BesselSobolev` supplies the shape/operator-inverse blocks, and nested `BlockDiagonalLinearOperator` objects assemble the full model prior. The model prior is a strict 0.95 ellipsoid $\mathcal{E} = \{m : \langle A m,m\rangle_\mathcal{M} \leq \chi^2_{d_\mathrm{eff}}(0.95)\}$ represented through `EllipsoidSupportFunction`, with effective dof computed by KL truncation (tol=1e-4, helper count N=10, final dof=42). No `.matrix(...)` discretization is used in the notebook’s Phase 3 prior construction. Requires the external kernel catalog `kernels_modeplotaat_Adrian` (set `INTERVALINF_KERNEL_CATALOG_DIR` env var); raises `FileNotFoundError` if the catalog is missing rather than silently substituting synthetic proxies. A clearly-labeled `USE_SYNTHETIC_FALLBACK` flag (default `False`) can be set for offline exploration.

The `intervalinf/demos/old_demos/paper_demos/` directory contains paper-oriented comparison notebooks and legacy prototypes.

- `example_3.ipynb`: older posterior-uncertainty demo retained as a legacy reference point.
- `premature_discretization_posterior_uncertainty.ipynb`: continuous-vs-discretized Bayesian comparison notebook showing that posterior property uncertainty is preserved by covariance-eigenfunction truncation but distorted by naive discretize-first identity priors.



---

## Package Metadata

| Key | Value |
|-----|-------|
| Version | 0.1.0 |
| Status | Alpha (Development Status 3) |
| Python | ≥ 3.11 |
| License | BSD-3-Clause |
| Core dependencies | `numpy≥1.26.0`, `scipy≥1.11.0`, `pygeoinf≥1.4.2` |
| Optional | `dev` (`pytest>=8.0.0`, `pytest-cov>=4.0.0`, `mypy>=1.0.0`, `ruff>=0.1.0`), `docs` (`sphinx>=7.0.0`, `sphinx-rtd-theme>=2.0.0`, `myst-parser>=2.0.0`), `plotting` (`matplotlib>=3.8.0`, `seaborn>=0.13.0`), `all` (`intervalinf[dev,docs,plotting]`) |
| Build backend | hatchling |

**Install:**
```bash
pip install intervalinf           # minimal
pip install "intervalinf[all]"    # includes dev, docs, plotting
pip install -e ".[dev]"           # editable development install
```

**Last Updated:** 2026-04-03 (Lowering fast path for SOLA data-space operators: `SOLAOperator` now exposes `_build_kernel_matrix()`, `_build_quadrature_weights()`, `compute_gram_matrix_fast()`, and `compute_cross_gram_matrix(other)`; new `operators/reduced.py` adds `ReducedGramOperator.from_sola()` and `ReducedCrossGramOperator.from_sola_pair()` returning dense matrix-backed pygeoinf operators. On `benchmarks.baseline_benchmark.build_problem(N_d=10, N_p=5, seed=42)`, slow Gram median = 116.079 ms, fast cold = 0.168 ms, fast hot median = 0.075 ms, hot speedup = 1548.28x, max abs diff = 7.105e-15. Living reference updated for the new reduced-operator module and test coverage.)

---

## Architecture

```
intervalinf/
├── core/             LEVEL 1 – no pygeoinf dependency
│   ├── domain.py         IntervalDomain
│   ├── boundary.py       BoundaryConditions
│   ├── functions.py      Function
│   ├── materialization.py Hidden fixed-grid representation cache helpers
│   └── config.py         IntegrationConfig, ParallelConfig, +hierarchical wrappers
│
├── spaces/           LEVEL 2 – depends on pygeoinf base classes
│   ├── lebesgue.py       Lebesgue  (→ pygeoinf.HilbertSpace)
│   ├── sobolev.py        Sobolev   (→ pygeoinf.MassWeightedHilbertSpace)
│   └── forms.py          LinearFormKernel (→ pygeoinf.LinearForm)
│
├── providers/        LEVEL 2–4 – basis/eigenvalue/spectrum factories
│   ├── base.py           Abstract bases (FunctionProvider, EigenvalueProvider, …)
│   ├── eigenvalues.py    Simple eigenvalue providers (sine, cosine, Fourier, mixed, …)
│   ├── functions/        Concrete function families (trig, FEM, smooth, wavelet, data)
│   ├── laplacian.py      Composite Laplacian providers
│   └── radial.py         Radial Laplacian providers
│
├── operators/        LEVEL 3 – depends on spaces and providers
│   ├── base.py           SpectralOperator (→ pygeoinf.LinearOperator)
│   ├── laplacian.py      Laplacian, InverseLaplacian
│   ├── gradient.py       Gradient
│   ├── bessel.py         BesselSobolev, BesselSobolevInverse
│   ├── sola.py           SOLAOperator
│   ├── reduced.py        ReducedGramOperator, ReducedCrossGramOperator
│   ├── radial.py         RadialLaplacian, InverseRadialLaplacian
│   ├── spectral_helpers.py   Shared spectral algorithms
│   └── _impl/
│       ├── fast_spectral.py  DST/DCT/DFT transforms
│       └── fem_solvers.py    FEM stiffness matrix assembly
│
├── sampling/
│   └── kl_sampler.py    KLSampler (KL / spectral sampling)
│
└── utils/
    └── robin_utils.py   RobinRootFinder
```

**Test layout:**

```
tests/
├── __init__.py          Package marker for test discovery/imports
├── conftest.py          Shared fixtures (`unit_domain`, `pi_domain`, `simple_space`)
├── core/                Unit tests for domain, boundary conditions, config, Function, and hidden materialization caches
├── spaces/              Lebesgue, Sobolev, forms, and Sobolev-operator integration tests
├── operators/           Spectral operator coverage, SOLAOperator regression/optimization suite, and reduced Gram/cross-Gram coverage
└── providers/           Standalone provider tests for domain-only provider usage
```

The current suite is organized by package layer rather than by mathematical workflow. When tracing a behavior change, start with the matching package directory under `tests/`, then check `tests/conftest.py` for shared fixtures reused across modules.

**pygeoinf relationship:**

```
pygeoinf (abstract)              intervalinf (concrete)
─────────────────────────────────────────────────────────
HilbertSpace                 →   Lebesgue
MassWeightedHilbertSpace     →   Sobolev
HilbertSpaceDirectSum        →   LebesgueSpaceDirectSum, SobolevSpaceDirectSum
LinearOperator               →   SpectralOperator, Gradient, BesselSobolev,
                                 BesselSobolevInverse, SOLAOperator
SpectralOperator (interval)  →   Laplacian, InverseLaplacian, RadialLaplacian,
                                 InverseRadialLaplacian
LinearForm                   →   LinearFormKernel
EuclideanSpace               →   codomain of SOLAOperator (used from pygeoinf)
MassWeightedHilbertSpace     →   Sobolev (mass operators are BesselSobolev instances)
```

---

## Connection to pygeoinf

### Abstract classes subclassed

| pygeoinf class | intervalinf subclass | Notes |
|---|---|---|
| `HilbertSpace` | `Lebesgue` | Implements `dim`, `to_dual`, `from_dual`, `to_components`, `from_components`, `__eq__` |
| `MassWeightedHilbertSpace` | `Sobolev` | Wraps a `Lebesgue` as underlying space; mass/inverse-mass operators are `BesselSobolev`/`BesselSobolevInverse` |
| `HilbertSpaceDirectSum` | `LebesgueSpaceDirectSum`, `SobolevSpaceDirectSum` | Direct sums of subspaces |
| `LinearOperator` | `SpectralOperator`, `Gradient`, `BesselSobolev`, `BesselSobolevInverse`, `SOLAOperator` | All inherit `@` composition, `.adjoint`, `.dual`, `.matrix()` from pygeoinf |
| `LinearForm` | `LinearFormKernel` | Kernel-based dual representation; lazily computes components |

### How intervalinf objects flow into pygeoinf algorithms

1. **Spaces as domains/codomains**: `Lebesgue` and `Sobolev` instances are passed directly wherever pygeoinf expects a `HilbertSpace`. pygeoinf inversion routines call `to_dual`/`from_dual` to compute preconditioned residuals.

2. **Functions as vectors**: `Function` objects are the "vectors" of `Lebesgue`/`Sobolev`. pygeoinf algorithms that call `space.zero`, `space.inner_product(u, v)`, `space.norm(u)` work transparently because `Lebesgue` overrides all these methods with integration-based implementations.

3. **Linear forms via kernel**: `LinearFormKernel` wraps a kernel function so that `form(f) = ∫ k(x) f(x) dx`. `SOLAOperator`'s dual mapping returns `LinearFormKernel` objects. When a pygeoinf algorithm needs components it calls `form.components`, which lazily projects onto the basis.

4. **Gaussian measures**: `KLSampler` uses pygeoinf's `MassWeightedHilbertSpace` interface to detect Sobolev spaces and adjust eigenvalues: $\lambda'_i = \lambda_i / \mu_i^2$ where $\mu_i$ are mass-operator eigenvalues.

---

## Module Reference

### `core/` — Domain and Function Primitives

No pygeoinf dependency. Safe to import anywhere.

---

#### `core/domain.py` — `IntervalDomain`

**Purpose:** Represents a 1D interval $[a,b]$ (or open/semi-open variants) with meshing, integration, and subdomain operations.

**Class:** `IntervalDomain`
- **Base:** none
- **Constructor:** `IntervalDomain(a, b, *, boundary_type='closed', name=None, open_epsilon=None)`
  - `boundary_type`: `'closed'` | `'open'` | `'left_open'` | `'right_open'`
  - `open_epsilon`: shift for mesh points at open endpoints (default = 0.001 × length)

| Property / Method | Signature | Description |
|---|---|---|
| `length` | `→ float` | $b - a$ |
| `center` | `→ float` | $(a+b)/2$ |
| `radius` | `→ float` | $(b-a)/2$ |
| `contains(x)` | `(float|ndarray) → bool|ndarray` | Membership test respecting boundary type |
| `uniform_mesh(n)` | `(int) → ndarray` | Uniformly spaced mesh of $n$ points |
| `interior()` | `→ IntervalDomain` | Returns open version |
| `closure()` | `→ IntervalDomain` | Returns closed version |
| `boundary_points()` | `→ (float, float)` | Returns $(a, b)$ |
| `integrate(f, ...)` | `(callable, method, support, n_points, vectorized) → float` | Integrates $f$ via `'simpson'`, `'trapz'`, `'adaptive'`, or `'quad'` (alias for `'adaptive'`); supports subinterval `support` |
| `restriction_to_subinterval(a, b)` | `(float, float) → IntervalDomain` | Creates child domain $[a,b] \subseteq$ self |
| `split_at_discontinuities(pts)` | `(list) → list[IntervalDomain]` | Splits at interior discontinuity points |

---

#### `core/boundary.py` — `BoundaryConditions`

**Purpose:** Unified specification of boundary conditions used by operators and FEM solvers.

**Class:** `BoundaryConditions`
- **Base:** none
- **Constructor:** `BoundaryConditions(bc_type, **kwargs)`

Supported `bc_type` values:
- `'dirichlet'`: $u(a) = $ `left`, $u(b) = $ `right` (default 0)
- `'neumann'`: $u'(a) = $ `left`, $u'(b) = $ `right` (default 0)
- `'robin'`: $\alpha u + \beta u' = $ value at each boundary (requires `left_alpha`, `left_beta`, `left_value`, `right_alpha`, `right_beta`, `right_value`)
- `'periodic'`: $u(a)=u(b)$, $u'(a)=u'(b)$
- `'mixed_dirichlet_neumann'`: Dirichlet at left, Neumann at right
- `'mixed_neumann_dirichlet'`: Neumann at left, Dirichlet at right

| Property / Method | Description |
|---|---|
| `type` | BC type string |
| `is_homogeneous` | True if all values/derivatives are zero |
| `get_parameter(name)` | Access a specific BC parameter |
| `BoundaryConditions.dirichlet(left=0, right=0)` | Factory: homogeneous/non-hom. Dirichlet |
| `BoundaryConditions.neumann(left=0, right=0)` | Factory: Neumann |
| `BoundaryConditions.robin(...)` | Factory: Robin |
| `BoundaryConditions.periodic()` | Factory: periodic |
| `BoundaryConditions.mixed_dirichlet_neumann(left=0, right=0)` | Factory: D at left, N at right |
| `BoundaryConditions.mixed_neumann_dirichlet(left=0, right=0)` | Factory: N at left, D at right |

---

#### `core/functions.py` — `Function`

**Purpose:** Represents a function on an interval, either attached to a `Lebesgue`/`Sobolev` space (traditional mode) or standalone on a domain (new bootstrapping mode).

**Class:** `Function`
- **Base:** none (pure Python)
- **Constructor:** `Function(space_or_domain, *, coefficients=None, evaluate_callable=None, name=None, support=None)`
  - Pass `IntervalDomain` for standalone mode; pass a `HilbertSpace` for attached mode.
  - Exactly one of `coefficients` or `evaluate_callable` must be provided.
  - `support`: `(a, b)` or list of tuples specifying compact support.

| Property | Description |
|---|---|
| `space` | The attached space (None if standalone) |
| `function_domain` | Always available `IntervalDomain` |
| `is_attached` | True if created with a space |
| `has_compact_support` | True if support is specified |
| `coefficients` | Coefficient array if given |
| `evaluate_callable` | Callable rule f(x) if given |

| Method | Signature | Description |
|---|---|---|
| `evaluate(x, check_domain)` | `(float|ndarray, bool) → float|ndarray` | Evaluate at point(s); enforces domain membership if `check_domain` |
| `__call__(x)` | delegates to `evaluate` | Callable interface |
| `attach_to_space(space, copy=True)` | `(HilbertSpace, bool) → Function` | Create space-attached copy |
| `detach(copy=True)` | `(bool) → Function` | Create domain-only copy (requires callable, not coefficients) |
| `restrict(restricted_space)` | `→ Function` | Restrict to a subdomain; if compact support is set, it is intersected with the restricted domain |
| `integrate(weight, method, n_points, vectorized)` | `→ float` | $\int f(x) w(x)\,dx$ |
| `materialize(spec)` | `(RepresentationSpec) → Materialization` | Hidden helper: evaluates once on a fixed grid, caches the resulting `grid` and `values`, and reuses them on repeated requests |
| `get_materialized(spec)` | `(RepresentationSpec) → Materialization \| None` | Returns an already cached materialization, if present |
| `clear_materializations()` | `() → None` | Invalidates all cached materializations for this function |
| Arithmetic: `+`, `-`, `*`, `/`, `__neg__`, `__abs__` | pointwise, returns new `Function` | Standard function arithmetic |

**Note:** For `*`, if both operands have compact support and their supports are disjoint, the product is an identically-zero `Function` with `support=[]`.

**Phase 2 materialization note:** `Function` now keeps a private `_materializations` dict keyed by `RepresentationSpec`. This does not change ordinary `f(x)` semantics; it only supports repeated fixed-grid evaluations such as `Lebesgue.inner_product` on Simpson/trapezoid meshes.

**Mathematical meaning:** Represents an element of $L^2([a,b])$ or $H^s([a,b])$; evaluation at a point is the function value $f(x)$.

---

#### `core/materialization.py` — `RepresentationSpec`, `Materialization`

**Purpose:** Defines the hidden cache key/value objects used to store concrete function values on reusable fixed grids.

| Class | Constructor | Description |
|---|---|---|
| `RepresentationSpec` | `(kind, n_points, interval, method)` | Hashable representation key. Phase 2 supports `kind='fixed_grid'` with `method='uniform'`; `kind='spectral'` is reserved for later phases. |
| `Materialization` | `(spec, grid, values)` | Stores the materialized grid coordinates and the function values evaluated on that grid. |

**Phase 2 behavior:** `RepresentationSpec(interval=(a,b), n_points=N, method='uniform')` materializes on the domain's uniform mesh for `[a,b]`. Repeated calls reuse the same cached `Materialization` object.

---

#### `core/config.py` — Configuration Dataclasses

**Purpose:** Typed configuration objects for numerical integration and parallelisation.

**Module-level constants (Phase 3):**

| Constant | Value | Purpose |
|---|---|---|
| `FIXED_GRID_METHODS` | `frozenset({'simpson', 'trapz'})` | Fixed-point-mesh methods; candidates for Phase 4 batching |
| `ADAPTIVE_METHODS` | `frozenset({'adaptive', 'quad'})` | Methods delegating to `scipy.integrate.quad`; `'quad'` is an alias for `'adaptive'` |

| Class | Constructor | Fields | Presets |
|---|---|---|---|
| `IntegrationConfig` | `(method='simpson', n_points=1000)` | `method`, `n_points`, `is_fixed_grid` (prop), `is_adaptive` (prop) | `.high_accuracy()`, `.fast()`, `.adaptive_quad()`, `.adaptive(dim)` |
| `ParallelConfig` | `(enabled=False, n_jobs=-1)` | `enabled`, `n_jobs` | `.all_cores()`, `.cores(n)`, `.serial()` |
| `LebesgueIntegrationConfig` | `(inner_product, dual, general)` | Three `IntegrationConfig` sub-configs | `.from_single(cfg)`, `.high_accuracy_galerkin()`, `.adaptive_spectral(dim)` |
| `LebesgueParallelConfig` | `(inner_product, dual, general)` | Three `ParallelConfig` sub-configs | `.from_single(cfg)`, `.parallel_dual(n_jobs)`, `.full_parallel(n_jobs)` |

All dataclasses support `.copy(**overrides)`.  `IntegrationConfig.method` accepts `'simpson'`, `'trapz'`, `'adaptive'`, and `'quad'` (alias for `'adaptive'`).  `is_fixed_grid` is `True` for `'simpson'` / `'trapz'` and determines Phase 4 batchability.

---

### `spaces/` — Hilbert Space Implementations

---

#### `spaces/lebesgue.py` — `Lebesgue`, `LebesgueSpaceDirectSum`, `KnownRegion`, `PartitionedLebesgueSpace`

**Purpose:** Concrete $L^2([a,b])$ Hilbert space implementing the full pygeoinf `HilbertSpace` interface.

---

**Class: `Lebesgue`**
- **Base:** `pygeoinf.HilbertSpace`
- **Mathematical space:** $L^2([a,b]; w)$ with inner product $\langle u, v \rangle = \int_a^b u(x)v(x)w(x)\,dx$
- **Constructor:** `Lebesgue(dim, function_domain, /, *, basis=None, weight=None, integration_config=None, parallel_config=None)`
  - `dim`: Number of basis functions used only in basis-backed workflows; may be `0` in basis-free mode
  - `function_domain`: `IntervalDomain`
  - `basis`: `'sine'` | `'cosine'` | `'fourier'` | `'DN'` | `'ND'` | `'hat'` | `'none'` | list of callables | `None` (defaults to `'none'`, i.e. basis-free functional mode)
  - `weight`: Optional weight function $w(x)$
  - `integration_config`: `IntegrationConfig` or `LebesgueIntegrationConfig`
  - `parallel_config`: `ParallelConfig` or `LebesgueParallelConfig`

| Property | Description |
|---|---|
| `dim` | Basis count for basis-backed operations; does not control direct callable/integration-only workflows |
| `function_domain` | The `IntervalDomain` |
| `metric` | Gram matrix $G_{ij} = \langle \phi_i, \phi_j \rangle$ (cached) |
| `integration` | `LebesgueIntegrationConfig` |
| `parallel` | `LebesgueParallelConfig` |

| Method | Description |
|---|---|
| `inner_product(u, v)` | $\int u(x)v(x)w(x)\,dx$ via numerical integration; for `method in {'simpson', 'trapz'}` it first materializes `u` and `v` on a shared uniform grid and integrates the cached arrays directly |
| `distance(u, v)` | $\|u - v\|_{L^2}$ |
| `to_dual(x)` | Returns `LinearFormKernel` with kernel $= x$ (Riesz map is identity in $L^2$) |
| `from_dual(xp)` | Extracts kernel from `LinearFormKernel` |
| `to_components(f)` | Projects $f$ onto the basis: $c_i = \langle f, \phi_i \rangle$ |
| `from_components(c)` | Reconstructs $f = \sum c_i \phi_i$; **infers `support`** from active basis functions (Phase 2): `support=[]` if all-zero, `support=None` if any active basis function is globally supported, else union of active basis-function supports (tol = 1e-14) |
| `zero` | Returns zero function with `support=[]` |
| `multiply(a, x)` | Scalar × coefficient function; propagates `support=[]` when `a==0`, else preserves `x.support` |
| `add(x, y)` | Coefficient addition; returns `Function` with `support = union(x.support, y.support)` |
| `ax(a, x)` | In-place `x *= a`; sets `x.support = []` when `a == 0` and clears any cached materializations on `x` |
| `axpy(a, x, y)` | In-place `y += a*x`; updates `y.support = union(y.support, x.support)` only when `a != 0` (preserves `y.support` for the zero-update case) and clears any cached materializations on `y` |
| `restrict(subspace, ...)` | Creates a `Lebesgue` on a subdomain |
| `gram_matrix()` | Assembles and returns the full Gram matrix |
| `inverse_gram_matrix()` | Returns $G^{-1}$ |

**Basis types and corresponding eigenfunctions:**

**Phase 2 inner-product fast path:** `_continuous_l2_inner_product` now builds a `RepresentationSpec(kind='fixed_grid', method='uniform', ...)` from the space domain and `integration_npoints`. For Simpson/trapezoid rules it obtains `u.materialize(spec)` and `v.materialize(spec)`, multiplies the cached arrays (plus optional weight samples), and integrates them with the same scipy quadrature rule. Adaptive/quad methods keep the original `(u * v).integrate(...)` path.

| `basis` string | Eigenfunctions | BC type |
|---|---|---|
| `'sine'` | $\sqrt{2/L}\sin(k\pi x/L)$ | Dirichlet |
| `'cosine'` | $1/\sqrt{L},\, \sqrt{2/L}\cos(k\pi x/L)$ | Neumann |
| `'fourier'` | $e^{2\pi i k x/L}/\sqrt{L}$ (cos/sin form) | Periodic |
| `'DN'` | $\sqrt{2/L}\sin\!\left((k+\tfrac{1}{2})\pi x/L\right)$ | Dirichlet-Neumann |
| `'ND'` | $\sqrt{2/L}\cos\!\left((k+\tfrac{1}{2})\pi x/L\right)$ | Neumann-Dirichlet |
| `'hat'` | Piecewise-linear hat functions on uniform grid | — |
| `'none'` | No basis (integration-only mode) | — |
| list | User-supplied callables | — |

---

**Class: `LebesgueSpaceDirectSum`**
- **Base:** `pygeoinf.HilbertSpaceDirectSum`
- **Mathematical space:** $L^2([a_1,b_1]) \oplus \cdots \oplus L^2([a_n,b_n])$
- Inner product is sum of component inner products.

---

**Class: `KnownRegion`**
- **Base:** none (dataclass-like)
- Specifies a sub-interval $[a', b'] \subset [a, b]$ where the function value is known (fixed).
- Constructor: `KnownRegion(domain, value_function)` where `value_function(x)` returns the fixed value.

---

**Class: `PartitionedLebesgueSpace`**
- **Base:** `Lebesgue`
- A `Lebesgue` space split into known-value regions and free regions.
- Constructor: `PartitionedLebesgueSpace(dim, function_domain, known_regions, /, *, basis, ...)`
- Restricts optimisation/inference to the free regions.

---

#### `spaces/sobolev.py` — `Sobolev`, `SobolevSpaceDirectSum`

**Purpose:** Sobolev space $H^s([a,b])$ as a mass-weighted $L^2$ space.

---

**Class: `Sobolev`**
- **Base:** `pygeoinf.MassWeightedHilbertSpace`
- **Mathematical space:** $H^s([a,b])$ with inner product $\langle u, v \rangle_{H^s} = \langle (k^2 I + \Delta)^s u, v \rangle_{L^2}$
- **Constructor:** `Sobolev(dim, function_domain, s, k, L, /, *, basis=None, integration_config=None, parallel_config=None)`
  - `s`: Sobolev regularity order (float)
  - `k`: Bessel potential scaling parameter (float); appears in $(k^2 I + \Delta)^s$
  - `L`: Spectral operator (typically a `Laplacian` instance); defines $\Delta$

| Property | Description |
|---|---|
| `function_domain` | `IntervalDomain` |
| `s` | Regularity order |
| `k` | Bessel parameter |
| `mass_operator_factor` | $M^{1/2} = (k^2 I + \Delta)^s$ (`BesselSobolev`) |
| `inverse_mass_operator_factor` | $M^{-1/2} = (k^2 I + \Delta)^{-s}$ (`BesselSobolevInverse`) |

| Method | Description |
|---|---|
| `to_dual(x)` | Returns `LinearFormKernel` with kernel $= Mx$ |
| `from_dual(xp)` | Returns $M^{-1}$kernel |
| `restrict(space, ...)` | Restricts to a subdomain |

**Note:** Full functionality requires the `operators` module (Phase 3). Passing `L=None` creates a deferred placeholder; most operations will raise `NotImplementedError` until a `Laplacian` is provided.

---

**Class: `SobolevSpaceDirectSum`**
- **Base:** `pygeoinf.HilbertSpaceDirectSum` (indirectly via `Sobolev` direct sum wrapper)
- Wraps multiple `Sobolev` spaces.

---

#### `spaces/forms.py` — `LinearFormKernel`

**Purpose:** Implements linear forms via a kernel function so that $\phi(f) = \int k(x) f(x) w(x)\,dx$.

**Class: `LinearFormKernel`**
- **Base:** `pygeoinf.LinearForm`
- **Constructor:** `LinearFormKernel(domain, /, *, mapping=None, kernel=None, components=None, integration_config=IntegrationConfig(), parallel_config=ParallelConfig())`
  - Exactly one of `mapping`, `kernel`, or `components` required.
  - `kernel`: A `Function` (or list of `Function`s for direct sum spaces)

| Property | Description |
|---|---|
| `kernel` | The kernel function (or `None` if built from mapping/components) |
| `components` | Basis-coefficient array; lazily computed on first access |
| `integration` | `IntegrationConfig` |
| `parallel` | `ParallelConfig` |

The `_mapping_impl` evaluates the form as `(kernel * v).integrate(...)`, making it basis-independent.

---

### `operators/` — Linear Operators

All operators inherit from `pygeoinf.LinearOperator` and therefore support:
- `op @ f` — apply operator
- `op.adjoint` — adjoint operator
- `op.dual` — dual operator
- `op.matrix()` — dense matrix representation
- Composition `op1 @ op2`

---

#### `operators/base.py` — `SpectralOperator`

**Purpose:** Abstract base for operators diagonalised by eigenfunctions.

**Class: `SpectralOperator`**
- **Base:** `pygeoinf.LinearOperator`, ABC
- **Abstract methods:**
  - `get_eigenvalue(index: int) → float`
  - `get_eigenfunction(index: int) → Function`
  - `_apply(f: Function) → Function`

All spectral operators share the pattern: project $f$ onto eigenfunctions $\{\phi_i\}$, scale coefficients by eigenvalues $\{\lambda_i\}$, reconstruct.

---

#### `operators/laplacian.py` — `Laplacian`, `InverseLaplacian`

**Class: `Laplacian`**
- **Base:** `SpectralOperator`
- **Mathematical operator:** $-d^2/dx^2$ with specified boundary conditions
- **Constructor:** `Laplacian(domain, boundary_conditions, alpha=1.0, /, *, method='spectral', dofs=None, fd_order=2, n_samples=512, integration_config=None)`
  - `domain`: `Lebesgue` or `Sobolev`
  - `alpha`: scaling factor (default 1.0); operator is $\alpha(-\Delta)$
  - `method`: `'spectral'` (DST/DCT/DFT, fast) or `'fd'` (finite difference)
  - `dofs`: degrees of freedom (default = `domain.dim`)
  - `fd_order`: FD stencil order (2 or 4)
  - `n_samples`: samples for fast transform path

| Method | Description |
|---|---|
| `get_eigenvalue(i)` | $\alpha \lambda_i$ from `LaplacianSpectrumProvider` |
| `get_eigenfunction(i)` | $i$-th eigenfunction from spectrum provider |
| `restrict(space, new_bcs)` | Creates restricted `Laplacian` on subdomain |

**Spectral path** (fast, for Dirichlet / Neumann / Periodic / Mixed BCs):
1. Sample $f$ on $n_\text{samples}$ points
2. Apply DST-I / DCT-I / DFT to get spectral coefficients
3. Scale each coefficient by eigenvalue $\lambda_i$
4. Reconstruct via `build_eigenfunction_expansion`

**FD path** (fallback for Robin BCs or when requested):
- Assembles tridiagonal matrix; applies to function values on grid; interpolates result.

---

**Class: `InverseLaplacian`**
- **Base:** `SpectralOperator`
- **Mathematical operator:** $(-\Delta)^{-1}$ (Green's function / covariance operator)
- Same constructor signature as `Laplacian` but inverts eigenvalues: $\lambda_i \mapsto 1/\lambda_i$
- Uses `GeneralFEMSolver` internally for FD/FEM path
- Self-adjoint and positive-definite (suitable as covariance operator for Gaussian measures)

---

#### `operators/gradient.py` — `Gradient`

**Class: `Gradient`**
- **Base:** `pygeoinf.LinearOperator`
- **Mathematical operator:** $d/dx$ (first derivative)
- **Constructor:** `Gradient(domain, /, *, fd_order=2, fd_step=None, boundary_treatment='one_sided')`
  - Uses finite differences exclusively (no spectral path)
  - `fd_step`: defaults to $(b-a)/1000$
- **Apply:** Uses central differences in interior, forward/backward at boundaries

---

#### `operators/bessel.py` — `BesselSobolev`, `BesselSobolevInverse`

**Class: `BesselSobolev`**
- **Base:** `pygeoinf.LinearOperator`
- **Mathematical operator:** $(k^2 I - \Delta)^{s/2}$ — the Bessel potential operator
- **Constructor:** `BesselSobolev(domain, codomain, k, s, L, dofs=None, n_samples=1024, use_fast_transforms=True, integration_config=...)`
  - `k`: Bessel parameter; `s`: power; `L`: spectral operator (Laplacian)
  - Self-adjoint: adjoint mapping = forward mapping
- **Fast path** (when L is Laplacian with standard BCs): Uses DST/DCT/DFT; scales coefficients by $(k^2 + \lambda_i)^{s/2}$
- **Slow path** (Robin BCs): Uses numerical integration

**Class: `BesselSobolevInverse`**
- **Base:** `pygeoinf.LinearOperator`
- **Mathematical operator:** $(k^2 I - \Delta)^{-s/2}$
- Same structure as `BesselSobolev` but scales by $(k^2 + \lambda_i)^{-s/2}$

**Usage in Sobolev spaces:** `Sobolev._create_mass_operators()` instantiates `BesselSobolev(s=2s)` as mass operator $M = (k^2 I + \Delta)^s$ and `BesselSobolevInverse(s=2s)` as $M^{-1}$.

---

#### `operators/radial.py` — `RadialLaplacian`, `InverseRadialLaplacian`

**Purpose:** Spherical-coordinate radial Laplacian for geophysical applications.

**Class: `RadialLaplacianEigenvalueProvider`**
- **Base:** `EigenvalueProvider`
- Computes eigenvalues of $-\partial_r^2 - (2/r)\partial_r$ (radial $\ell=0$ case) for various BCs on $(0,R)$ or $(a,b)$
- Supports: Dirichlet-Dirichlet, Dirichlet-Neumann, Neumann-Dirichlet, Neumann-Neumann at domain endpoints
- Analytical eigenvalues for DD and regularity-Dirichlet; numerical root-finding (via `RobinRootFinder`) for NN and mixed

**Class: `RadialLaplacianSpectrumProvider`**
- **Base:** `SpectrumProvider`
- Combines `RadialLaplacianEigenvalueProvider` with the corresponding radial function providers from `providers/radial.py`

**Class: `RadialLaplacian`**
- **Base:** `SpectralOperator`
- Self-adjoint with respect to weighted inner product $\langle f, g \rangle = \int f(r)g(r)r^2\,dr$

**Class: `InverseRadialLaplacian`**
- **Base:** `SpectralOperator`
- Inverts `RadialLaplacian` eigenvalues

---

#### `operators/sola.py` — `SOLAOperator`

**Purpose:** Implements the SOLA (Subtractive Optimally Localized Averages) forward operator.

**Class: `SOLAOperator`**
- **Base:** `pygeoinf.LinearOperator`
- **Mathematical operator:** $G: L^2([a,b]) \to \mathbb{R}^N$, $(Gf)_i = \int f(x)k_i(x)\,dx$
- **Constructor:** `SOLAOperator(domain, codomain, kernels=None, cache_kernels=False, integration_config=...)`
  - `domain`: `Lebesgue` or `Sobolev`
  - `codomain`: `pygeoinf.EuclideanSpace` (dimension $N$)
  - `kernels`: `IndexedFunctionProvider` | list of `Function` | list of callables
  - `dual_mapping`: reconstructs a `LinearFormKernel` from data coefficients $\sum y_i k_i(x)$

| Method | Description |
|---|---|
| `get_kernel(i)` | Lazily retrieves $i$-th kernel with optional caching |
| `get_kernels()` | Materialises and returns all kernels as a list |
| `_build_kernel_matrix(xs=None)` | Builds the dense kernel table $K[i, :] = k_i(x_s)$ on the shared mesh; reuses `_kernel_eval_cache` only when `xs` matches the shared fixed-grid mesh |
| `_build_quadrature_weights(xs=None, method=None)` | Builds fixed-grid quadrature weights for Simpson/trapz on the shared mesh; for even-sample Simpson it matches SciPy's Cartwright end correction exactly |
| `_mapping(f)` | Applies $G$: returns `ndarray` of shape `(N,)` |
| `_dual_mapping(yp)` | Returns `LinearFormKernel` reconstructed from data; adjoint is $G^*(y) = \sum_i y_i k_i(x)$ |
| `compute_gram_matrix()` | Returns $N \times N$ matrix $G_{ij} = \int k_i(x)k_j(x)\,dx$ |
| `compute_gram_matrix_fast()` | Builds $K$ and quadrature weights $w$ on a shared mesh and returns the reduced Gram matrix $(K \odot w)K^T$; falls back to `compute_gram_matrix()` when caching or fixed-grid integration is unavailable |
| `compute_cross_gram_matrix(other)` | Builds a reduced cross-Gram matrix $C_{TG} = (L \odot w)K^T$ on a common mesh for a second `SOLAOperator`; falls back to pairwise quadrature for adaptive methods |
| `clear_cache()` | Clears `_kernels_cache` AND `_kernel_eval_cache`; `_shared_mesh` is preserved |
| `clear_mesh_cache()` | Clears only `_kernel_eval_cache`; `_shared_mesh` is preserved (no-op if `cache_kernels=False`) |
| `get_cache_info()` | Returns dict with `caching_enabled`, `shared_mesh_built`; when enabled also `cached_functions`, `total_functions`, `cache_coverage`, `kernel_eval_cache_entries` |
| `stats` | **Phase 2 instrumentation.** Property returning a shallow-copy `dict` with performance counters accumulated since construction or last `reset_stats()`. Keys: `forward_calls`, `disjoint_skips`, `compact_support_fallbacks`, `batched_fixed_grid_kernels`, `forward_time_total_s`, `compact_support_fallback_time_total_s`. See table below. |
| `reset_stats()` | Zeros all `_stats` counters. Call before a timed section to get per-experiment numbers. |
| `for_direct_sum(domain, codomain, kernels, ...)` | **Static.** Creates a `RowLinearOperator` with one `SOLAOperator` per subspace; kernels restricted via `provider.restrict(subspace)` or `Function.restrict(subspace)` |
| `_eval_on_mesh(func, xs)` | **Static.** Evaluates a `Function` on a numpy mesh array with vectorisation fallback for non-vectorised callables; preserves complex dtype |
| `_build_support_mesh(support, n_points)` | **Static. Phase 4.** Builds a concatenated quadrature mesh over a list of support subintervals, reproducing the proportional-allocation + remainder-distribution logic of `IntervalDomain.integrate`. Returns an empty array for empty support; otherwise concatenates `np.linspace(a_i, b_i, alloc_i)` per subinterval (shared boundary endpoints **not** deduplicated). Used by the Phase 5 grouped support-restricted batched integration path. |
| `_apply_kernels(func)` | Dispatch method: routes to `_apply_kernels_fixed_grid` for fixed-grid methods, `_apply_kernels_generic` for adaptive |
| `_apply_kernels_fixed_grid(func)` | Phase 5 forward path — reuses shared mesh, evaluates f once, batches full-domain kernels, and groups compact-support kernels by their exact intersected-support key for batched integration on restricted meshes per subinterval |
| `_apply_kernels_generic(func)` | Original per-kernel loop — calls `domain.integrate()` individually for each kernel; used for adaptive methods |

**Phase 2 instrumentation counters (2026-03-10):**

All counters **accumulate** across calls; use `reset_stats()` before a timed section.

| Counter key | Type | Incremented by |
|---|---|---|
| `forward_calls` | `int` | +1 for every `G(f)` call (fixed-grid **and** adaptive) |
| `batched_fixed_grid_kernels` | `int` | +N per call for kernels handled by the full-domain batched matrix path (no compact-support restriction) |
| `compact_support_fallbacks` | `int` | +1 per kernel handled by the support-restricted grouped batched path in `_apply_kernels_fixed_grid` (per-kernel count: N kernels in the same support group contribute N to this counter) |
| `disjoint_skips` | `int` | +1 per kernel whose support was disjoint from the input function's support (both fixed-grid **and** adaptive paths) |
| `forward_time_total_s` | `float` | cumulative wall time of all `G(f)` calls |
| `compact_support_fallback_time_total_s` | `float` | cumulative wall time of support-restricted grouped batched integrations inside the fixed-grid path |

Example usage:
```python
G.reset_stats()
for f_i in training_set:
    G(f_i)
s = G.stats
print(s["compact_support_fallbacks"], "fallbacks in", s["forward_calls"], "calls")
```

**Phase 3 changes (2026-03-08):**
- `IntegrationConfig(method='quad')` now works correctly — `'quad'` routes to `scipy.integrate.quad` via the `'adaptive'` alias in `IntervalDomain.integrate`.
- `_apply_kernels` no longer allocates a `Function` wrapper per kernel; it calls `domain.integrate()` directly.
- `_apply_kernels` and `compute_gram_matrix` now propagate compact-support metadata: when both the input function and the kernel carry compact-support information, the integration range is narrowed to the support intersection.  Disjoint supports return 0 without evaluating the integrand.
- Reconstructed adjoint functions still loop over kernels on each evaluation (Phase 5/6 scope).

**Phase 5 changes (2026-03-08 + 2026-03-10):** Kernel-eval caching and grouped support-restricted batching.

*Phase 5a — shared mesh reuse and kernel-eval cache:*
- **`_shared_mesh: Optional[np.ndarray]`** — lazy-built on first `_apply_kernels_fixed_grid` call via `np.linspace(a, b, n_points)`. Never cleared; depends only on immutable constructor parameters (domain bounds + n_points).
- **`_kernel_eval_cache: Optional[dict]`** — `None` when `cache_kernels=False`; otherwise a dict mapping kernel index → `ndarray` of shape `(n_points,)` (kernel values on the shared mesh). Populated **only** for full-domain (non-compact-support) kernels. Cleared by `clear_cache()` and `clear_mesh_cache()`; `_shared_mesh` is never cleared.
- **`_get_or_build_mesh()`** — private helper; builds and stores `_shared_mesh` on first call, returns it on subsequent calls.
- **Batched-path loop** now checks eval cache before calling `_eval_on_mesh(kernel, xs)`; stores result if not present.
- **Cache semantics:** A kernel is cached iff it goes through the full-domain batched path. Compact-support kernels (grouped path) and disjoint-support kernels are never cached.
- **Memory:** each entry ≈ 8 KB at n_points=1000; N_d entries ≈ N_d × 8 KB (e.g. 200 × 8 KB = 1.6 MB).
- **Measured speedup** for repeated workloads (N_REPS=50 distinct input functions, n_points=1000): about 1.6x–3.6x across N_d 5–200 on the current benchmark.

*Phase 5b — grouped support-restricted batched integration (2026-03-10):*
- **Replaces per-kernel `domain.integrate` fallback** for compact-support kernels with a grouped batched numpy/scipy pass.
- **Grouping logic:** during the classification loop, kernels with a non-None, non-empty `intersected_support` are inserted into a `support_groups` dict keyed by `tuple(tuple(iv) for iv in intersected_support)`. Only exact tuple equality is used — no floating-point canonicalization.
- **Per-group integration:** for each unique support key, the proportional allocation sizes are computed via `_compute_subinterval_alloc(...)`, then for each subinterval `(a_i, b_i)` with allocation `n_i` a per-subinterval `np.linspace(a_i, b_i, n_i)` mesh is built, `f` and all kernels in the group are evaluated on it, the batched product matrix is integrated with `_simpson`/`_trapz`, and partial results are accumulated. Summing over subintervals gives the complete integral for non-contiguous multi-interval supports.
- **Single-subinterval groups (common case):** exactly one `_build_support_mesh` call per group, one batched integration pass. N kernels sharing the same single-interval support → 1 mesh build instead of N.
- **Multi-interval supports:** `_build_support_mesh` is called once per subinterval within the group. Correctness is guaranteed because each subinterval is integrated separately (avoiding the gap-integration error that would result from applying `_simpson` directly to the concatenated non-contiguous mesh).
- **Stats:** `compact_support_fallbacks` still counts per-kernel (N kernels in a group → N increments), preserving backward compatibility. `compact_support_fallback_time_total_s` now accumulates the per-group timing (from mesh build through integration) rather than per-kernel timing.
- **Complex dtype:** complex kernel/function values are preserved through the batched product and integration, matching the behaviour of the full-domain batched path.

**Phase 4 changes (2026-03-08):**
- `_apply_kernels` now dispatches based on `self.integration.is_fixed_grid`:
  - **Fixed-grid methods** (`'simpson'`, `'trapz'`): automatic batched path via `_apply_kernels_fixed_grid`.
    - Mesh built **once** per `G(f)` call.
    - Input function `f` evaluated **once** on the shared mesh.
    - Full-domain kernels assembled into an `(N_d, n_points)` matrix.
    - Product matrix integrated in a single `scipy.integrate.simpson` or `trapezoid` call.
    - Non-vectorised callables handled via `_eval_on_mesh` fallback (per-point loop).
    - Disjoint-support kernels still skipped without evaluation.
    - Support-restricted kernels are grouped by their exact intersected-support key; each group is integrated in a single batched pass on a restricted mesh (Phase 5b grouped batching).
    - Complex-valued fixed-grid evaluations are preserved end to end; no silent real downcast in the batched or support-restricted fixed-grid paths.
  - **Adaptive methods** (`'adaptive'`, `'quad'`): unchanged generic per-kernel loop via `_apply_kernels_generic`.
- Static helper `_eval_on_mesh(func, xs)` added: tries vectorised `Function.evaluate(xs)` first; on shape mismatch or exception, falls back to per-point evaluation while preserving scalar dtype.
- **Phase 4b (2026-03-10): `_build_support_mesh(support, n_points)` static helper added.** Reproduces `IntervalDomain.integrate` proportional-allocation meshing for support lists inside `SOLAOperator` without running the integrand. Used as a mesh-building primitive in preparation for Phase 6 batched support-restricted kernel integration. The helper is private and does not change existing forward behaviour. Algorithm: `effective_total=max(n_points, 3·n_sub)`, `alloc_i=max(3, floor(raw_i))`, remainder distributed by descending fractional part with stable original-order tie-break; per-interval mesh is `np.linspace(a_i, b_i, alloc_i)` with shared boundary endpoints kept (not deduplicated).
- `IntervalDomain.integrate` fixed-grid methods (`'simpson'`, `'trapz'`) now preserve complex dtype in both vectorised and scalar-fallback evaluation paths.
- Measured speedup over `_apply_kernels_generic` (same method, n_points=1000): about 2–5x for N_d 5–200 on the current benchmark.
- Public API and semantics unchanged: operator remains "continuous operator evaluated numerically".

**Reduced Gram / cross-Gram changes (2026-04-03):**
- **`_build_kernel_matrix(xs=None)`** stacks all kernel evaluations into a dense array of shape `(N_d, n_points)`. When `xs` matches the operator's shared fixed-grid mesh it reuses `_kernel_eval_cache`; otherwise it evaluates kernels directly on the supplied common mesh without polluting the shared-mesh cache.
- **`_build_quadrature_weights(xs=None, method=None)`** returns dense Simpson or trapezoid weights. For odd sample counts Simpson uses the classical $[1,4,2,\ldots,4,1]h/3$ pattern; for even sample counts it reproduces the same Cartwright correction that `scipy.integrate.simpson` applies, so reduced assembly matches the legacy pairwise quadrature path to machine precision.
- **`compute_gram_matrix_fast()`** computes the dense reduced Gram matrix directly as `(K * w[np.newaxis, :]) @ K.T`, avoiding creation of intermediate `Function` objects and repeated `domain.integrate()` calls. When `cache_kernels=False` or the integration method is adaptive it falls back to `compute_gram_matrix()`.
- **`compute_cross_gram_matrix(other)`** computes a dense cross-Gram matrix `self @ other.adjoint` on a common mesh. When both operators share the same fixed-grid configuration it reuses their shared meshes/caches; when mesh sizes differ it evaluates both kernel stacks on `np.linspace(a, b, max(n_points))`; adaptive-method pairs fall back to pairwise quadrature.
- **`operators/reduced.py`** adds `ReducedGramOperator.from_sola(G)` and `ReducedCrossGramOperator.from_sola_pair(T, G)`, both returning dense matrix-backed `pygeoinf` operators on the data spaces.
- **Measured benchmark** on `benchmarks.baseline_benchmark.build_problem(N_d=10, N_p=5, seed=42)`: slow Gram median `116.079 ms`; fast cold `0.168 ms`; fast hot median `0.075 ms`; hot speedup `1548.28x`; max absolute difference `7.105e-15`.

#### `operators/reduced.py` — `ReducedGramOperator`, `ReducedCrossGramOperator`

**Purpose:** Wraps reduced SOLA Gram and cross-Gram matrices as dense pygeoinf matrix operators.

| Factory | Description |
|---|---|
| `ReducedGramOperator.from_sola(G)` | Returns a dense self-adjoint matrix-backed operator on `G.codomain` using `G.compute_gram_matrix_fast()` |
| `ReducedCrossGramOperator.from_sola_pair(T, G)` | Returns a dense matrix-backed operator `G.codomain -> T.codomain` using `T.compute_cross_gram_matrix(G)` |

**Integration method support:**

| `IntegrationConfig.method` | Behaviour | `is_fixed_grid` | Forward path |
|---|---|---|---|
| `'simpson'` | Composite Simpson on uniform mesh | True | `_apply_kernels_fixed_grid` (batched) |
| `'trapz'` | Composite trapezoidal on uniform mesh | True | `_apply_kernels_fixed_grid` (batched) |
| `'adaptive'` | `scipy.integrate.quad` error-controlled | False | `_apply_kernels_generic` (per-kernel) |
| `'quad'` | Alias for `'adaptive'` (legacy name) | False | `_apply_kernels_generic` (per-kernel) |

**Phase 6 end-to-end performance summary (2026-03-08):**

Validated against the forced generic path in the current codebase
(`_apply_kernels_generic`), which reproduces the Phase 2 forward-path behavior
without checking out an earlier revision. The comparison matrix covers N_d
5–200, n_points 200–2000, simpson/trapz, and sine/callable/bump kernels.

| Workload | Phase 4 single-call speedup | Phase 5 single-call speedup (warm) | Phase 5 batch speedup (N=30) |
|---|---|---|---|
| sine_provider, N_d 5–20 | 2.8x–4.9x | 4.7x–13x | 1.6x–2.8x |
| sine_provider, N_d 50–200 | 3.7x–5.5x | 8.4x–19.3x | 2.1x–5.4x |
| callable kernels, N_d=20 | 5.0x | 11.3x | 2.3x |
| bump_provider, N_d=20 | 2.5x | 19x | 7.6x |
| bump_provider, N_d=50 | 2.6x | 28.9x | 10.2x |

Key findings:
- **Global/smooth kernels (full-domain batched path):** Phase 4 achieves clear
  multi-x single-call speedup via batched numpy/scipy integration; Phase 5
  multiplies this further when the kernel-eval cache is warm.
- **Localized bump kernels (bump_provider):** In the Phase 6 scenarios these
  kernels are still eligible for the batched/cached fixed-grid path because the
  test input function carries no compact-support metadata, so no explicit
  support intersection is available. Large warm-cache gains therefore reflect
  forced-generic versus batched/cached comparison on sharply localized kernels,
  not a support-narrowed fallback path. Always use `method='adaptive'` when
  high accuracy is required for peaky bump kernels: fixed-grid Simpson on a
  global mesh can differ from adaptive quad by O(1) for sharp bumps.
- **Accuracy (fast path vs adaptive reference):** machine-precision (< 1e-8) for global/smooth kernels with n_points ≥ 500; n_points=200 shows ~1e-5 error (acceptable for iterative methods).
- **Adjoint residual:** heuristic check stays < 1e-5 for all global/smooth
  scenarios; up to 2e-3 for large-N_d bump providers with fixed-grid
  integration (expected; not a regression — use 'adaptive' for high-accuracy
  bump scenarios).
- **No correctness issues found; no Phase-6 implementation fixes required.**

---

#### `operators/spectral_helpers.py` — Shared Spectral Utilities

**Purpose:** Reusable patterns for eigenfunction expansion across all spectral operators.

| Function | Signature | Description |
|---|---|---|
| `build_eigenfunction_expansion(terms, domain, codomain, tolerance=1e-14)` | `(list[(float, Function)], ...) → Function` | Builds lazy-evaluated eigenfunction expansion, filtering negligible terms |
| `compute_spectral_coefficients_fast(operator, f, coefficients, scale_func)` | `→ list` | Fast path: uses pre-computed DST/DCT/DFT coefficients |
| `compute_spectral_coefficients_slow(operator, f, n_dofs, method, n_pts, scale_func, skip_zero)` | `→ list` | Slow path: numerical integration $\langle f, \phi_i \rangle$ |
| `validate_eigenvalue(eigval, index, allow_negative)` | `→ None` | Raises descriptive error for invalid eigenvalues |

---

#### `operators/_impl/fast_spectral.py` — Fast Transform Backend

**Purpose:** Low-level DST/DCT/DFT coefficient computation.

| Function | Description |
|---|---|
| `fast_spectral_coefficients(f_samples, bc, domain_length, n_coeffs)` | Dispatches to DST-I, DCT-I, DFT, DST-IV, or DCT-IV based on BC |
| `create_uniform_samples(f, domain_tuple, n_samples, bc_type)` | Samples `Function` $f$ on a uniform grid, respecting BC symmetry |

Supported BCs: `'dirichlet'` → DST-I, `'neumann'` → DCT-I, `'periodic'` → DFT, `'mixed_dirichlet_neumann'` → DST-IV, `'mixed_neumann_dirichlet'` → DCT-IV.

---

#### `operators/_impl/fem_solvers.py` — FEM Stiffness Matrix Assembly

**Purpose:** Pure-numpy FEM solver for `InverseLaplacian`.

**Class: `GeneralFEMSolver`**
- **Base:** none
- **Constructor:** `GeneralFEMSolver(function_domain, dofs, operator_domain, boundary_conditions)`
- **Design:** If BCs are homogeneous Dirichlet + hat functions, assembles analytical tridiagonal $K$ with entries $K_{ii} = 2/h$, $K_{i,i\pm1} = -1/h$. Otherwise uses numerical integration of $K_{ij} = \int \phi'_i \phi'_j\,dx$.
- Supports: Dirichlet, Neumann, Periodic, mixed D-N, and mixed N-D BCs (with appropriate Lagrange multiplier or condensation).

| Method | Description |
|---|---|
| `_assemble_stiffness_matrix()` | Builds $K$; called once at init |
| `solve(f)` | Solves $K\mathbf{u} = \mathbf{F}$ where $F_i = \int f \phi_i\,dx$; returns `Function` |

---

### `providers/` — Basis and Quadrature Providers

Providers follow a four-level hierarchy:

```
Level 0: base.py          — Abstract ABCs
Level 1: eigenvalues.py   — Eigenvalue providers (closed-form formulas)
Level 2: functions/       — Concrete function families
Level 3: laplacian.py     — Composite Laplacian eigenbasis
         radial.py        — Composite radial eigenbasis
```

Providers support **dual-mode initialisation**: pass either a `HilbertSpace` (attached mode — functions returned are attached to that space) or an `IntervalDomain` (standalone mode — functions returned are not attached to any space). This solves the bootstrapping problem.

---

#### `providers/base.py` — Abstract Bases

| Class | Base | Description |
|---|---|---|
| `FunctionProvider` | ABC | Base for all function providers; provides `domain`, `space`, `is_standalone`, `function_context` |
| `IndexedFunctionProvider` | `FunctionProvider`, ABC | `get_function_by_index(i)`, `get_functions(indices)`, `restrict(space)` |
| `RestrictedFunctionProvider` | `IndexedFunctionProvider` | Restricts another provider's functions to a subdomain |
| `ParametricFunctionProvider` | `FunctionProvider`, ABC | `get_function_by_parameters(params)`, `get_default_parameters()` |
| `RandomFunctionProvider` | `FunctionProvider`, ABC | `get_random_function()` with `rng` seed |
| `NullFunctionProvider` | `IndexedFunctionProvider` | Returns zero functions |
| `EigenvalueProvider` | ABC | `get_eigenvalue(i) → float` |
| `CustomEigenvalueProvider` | `EigenvalueProvider` | Wraps a callable $i \mapsto \lambda_i$ |
| `BasisProvider` | `IndexedFunctionProvider` | Adds `orthonormal`, `basis_type` metadata |
| `CustomBasisProvider` | `BasisProvider` | Wraps a list of callables |
| `SpectrumProvider` | `BasisProvider` | Combines eigenvalues + eigenfunctions; `get_eigenvalue(i)`, `get_eigenfunction(i)` |
| `CustomSpectrumProvider` | `SpectrumProvider` | Wraps parallel eigenvalue/eigenfunction lists |

---

#### `providers/eigenvalues.py` — Simple Eigenvalue Providers

| Class | Formula | BC |
|---|---|---|
| `SineEigenvalueProvider(L)` | $\lambda_k = (k\pi/L)^2$, $k = i+1$ | Dirichlet |
| `CosineEigenvalueProvider(L)` | $\lambda_0 = 0$; $\lambda_k = (k\pi/L)^2$ | Neumann |
| `FourierEigenvalueProvider(L)` | $\lambda_0 = 0$; pairs at $(2k\pi/L)^2$ | Periodic |
| `MixedDNEigenvalueProvider(L)` | $\lambda_k = ((k+\tfrac{1}{2})\pi/L)^2$ | Dirichlet-Neumann |
| `MixedNDEigenvalueProvider(L)` | $\lambda_k = ((k+\tfrac{1}{2})\pi/L)^2$ | Neumann-Dirichlet |
| `ZeroEigenvalueProvider` | $\lambda_k = 0$ always | — |

---

#### `providers/functions/` — Concrete Function Families

| Provider | Basis type | Formula / notes |
|---|---|---|
| `SineFunctionProvider` | Trigonometric | $\phi_k(x) = \sqrt{2/L}\sin(k\pi x/L)$, $k = i+1$ |
| `CosineFunctionProvider` | Trigonometric | $\phi_0 = 1/\sqrt{L}$; $\phi_k = \sqrt{2/L}\cos(k\pi x/L)$ |
| `FourierFunctionProvider` | Trigonometric | Real-valued Fourier (cos/sin pairs) |
| `MixedDNFunctionProvider` | Trigonometric | Shifted sine: $\sqrt{2/L}\sin((k+\tfrac{1}{2})\pi x/L)$ |
| `MixedNDFunctionProvider` | Trigonometric | Shifted cosine: $\sqrt{2/L}\cos((k+\tfrac{1}{2})\pi x/L)$ |
| `RobinFunctionProvider` | Trigonometric | $\mu_k \cos(\mu_k x) + (\alpha_0/\beta_0)\sin(\mu_k x)$; numerical roots |
| `HatFunctionProvider` | FEM | Piecewise-linear hat functions on uniform grid; supports homogeneous/non-homogeneous; **sets `support=(nodes[i-1], nodes[i+1])`** on each returned `Function` (clamped at domain boundaries) |
| `SplineFunctionProvider` | FEM | B-spline basis; **sets `support=(knots[i], knots[i+degree+1])`** on each returned `Function`; `support=None` for degenerate zero-width spans; `get_function_by_parameters` infers support when `knots`/`degree`/`index` keys are present |
| `BumpFunctionProvider` | Smooth | Smooth compactly-supported bump functions $C^\infty_0$ |
| `BumpFunctionGradientProvider` | Smooth | Gradients of bump functions |
| `WaveletFunctionProvider` | Wavelet | Haar (or other) wavelet basis |
| `BoxCarFunctionProvider` | Step | Indicator functions on sub-intervals; **sets `support=(a, b)`** on each returned `Function` |
| `DiscontinuousFunctionProvider` | Step | Piecewise-constant discontinuous functions |
| `KernelProvider` | Data | Sensitivity kernels loaded from file |
| `NormalModesProvider` | Data | Normal-mode kernels for seismic applications |

---

#### `providers/laplacian.py` — Composite Laplacian Providers

**Class: `LaplacianEigenvalueProvider`**
- **Base:** `EigenvalueProvider`
- Delegates to `SineEigenvalueProvider`, `CosineEigenvalueProvider`, `FourierEigenvalueProvider`, or numerical Robin root-finding via `RobinRootFinder`.
- Supports `inverse=True` for $(-\Delta)^{-1}$ eigenvalues; caches all values.

**Class: `LaplacianSpectrumProvider`**
- **Base:** `SpectrumProvider`
- Combines `LaplacianEigenvalueProvider` with the appropriate function provider (selected by BC type).
- This is the primary entry point for spectral methods; used internally by `Laplacian` and `InverseLaplacian`.

---

#### `providers/radial.py` — Radial Laplacian Providers

All are `IndexedFunctionProvider` subclasses; each wraps a domain and returns radial eigenfunctions $y_n(r)$.

| Class | Domain | BC | Eigenfunction |
|---|---|---|---|
| `RadialLaplacianDirichletProvider` | $(0, R)$ | D at $r=R$ | $y_n = \sqrt{2/R}\sin(n\pi r/R)/r$ |
| `RadialLaplacianNeumannProvider` | $(0, R)$ | N at $r=R$ | Zero mode + $y_n = c_n\sin(k_n r)/r$ where $k_n$ solves $\tan(k_n R) = k_n R$ |
| `RadialLaplacianDDProvider` | $(a, b)$, $a>0$ | DD | $\sqrt{2/L}\sin(n\pi(r-a)/L)/r$ |
| `RadialLaplacianDNProvider` | $(a, b)$ | DN | Numerically computed |
| `RadialLaplacianNDProvider` | $(a, b)$ | ND | Numerically computed |
| `RadialLaplacianNNProvider` | $(a, b)$ | NN | Zero mode + numerical |

---

### `sampling/` — Sampling Utilities

#### `sampling/kl_sampler.py` — `KLSampler`

**Purpose:** Generates samples from a Gaussian measure $\mathcal{N}(m, C)$ via truncated KL expansion.

**Class: `KLSampler`**
- **Base:** none
- **Constructor:** `KLSampler(operator, *, mean=None, n_modes=None, energy_tol=None, max_modes=None, rng=None, cache=True)`
  - `operator`: `SpectralOperator` acting as covariance (typically `InverseLaplacian`)
  - `mean`: `Function` (default = `domain.zero`)
  - `n_modes`: explicit truncation; or `energy_tol`: retain modes until fraction of variance is captured
  - Supports both `Lebesgue` (unweighted) and `Sobolev` (mass-weighted) spaces
  - In Sobolev spaces, adjusts eigenvalues: $\lambda'_i = \lambda_i / \mu_i^2$ where $\mu_i$ are `BesselSobolev` mass eigenvalues

**Dataclass: `TruncationInfo`**
- Fields: `n_modes`, `reason` (`'explicit'` | `'energy_tol-heuristic'` | `'default'`), `energy_fraction`

| Method | Description |
|---|---|
| `sample()` | Draws $u = m + \sum_{i<k} \sqrt{\lambda'_i}\,\xi_i\,\phi_i$, $\xi_i \sim \mathcal{N}(0,1)$ |
| `sample_batch(n)` | Draws $n$ independent samples |
| `covariance_factor()` | Returns the operator $L$ such that $C \approx LL^*$ |
| `truncation_info` | Returns `TruncationInfo` after first `sample()` call |

**Gaussian measure workflow:**
```python
domain = IntervalDomain(0, 1)
space  = Lebesgue(50, domain, basis='sine')
cov    = InverseLaplacian(space, BoundaryConditions.dirichlet())
sampler = KLSampler(cov, n_modes=20)
f_sample = sampler.sample()   # Function object
```

---

### `utils/` — Utilities

#### `utils/robin_utils.py` — `RobinRootFinder`

**Purpose:** Unified root-finding for Robin BC eigenvalues. Used by `LaplacianEigenvalueProvider`, `RadialLaplacian` providers, and `RadialLaplacianEigenvalueProvider`.

**Class: `RobinRootFinder`**
- All methods are `@staticmethod`

| Method | Description |
|---|---|
| `bisect(F, a, b, tol=1e-12, maxit=100)` | Standard bisection on $[a,b]$ |
| `find_bracket_with_expansion(F, left, right)` | Expands bracket until sign change found |
| `find_bracket_by_scanning(F, left, right, n_samples=129)` | Scans for first sign change |
| `compute_robin_eigenvalue(index, α₀, β₀, α_L, β_L, L, tol, maxit)` | Computes $\mu_k$ satisfying the Robin characteristic equation $D(\mu)=0$ |
| `solve_tan_equation(F, R, index)` | Finds $k$ solving $\tan(kR) = F(k)$ (for regularity-Neumann and DN/ND eigenvalues) |

The Robin characteristic equation is:
$$D(\mu) = (\alpha_0 \alpha_L + \beta_0 \beta_L \mu^2)\sin(\mu L) + \mu(\alpha_0 \beta_L - \beta_0 \alpha_L)\cos(\mu L) = 0$$

---

## Key Mathematical Concepts

### Lebesgue Space Discretisation Strategy

`Lebesgue` represents the **infinite-dimensional** $L^2([a,b])$ but allows a **finite-dimensional projection** when a basis is chosen. The discretisation is:

$$f \approx \sum_{i=1}^N c_i \phi_i(x)$$

where $\phi_i$ are chosen from `providers/functions/`. Coefficients are computed as $c_i = \langle f, \phi_i \rangle = \int_a^b f(x)\phi_i(x)\,dx$ via `to_components`, using the integration config. Without a basis (`basis='none'`), the space is purely continuous and operations use numerical integration directly.

### Sobolev Space Construction

$H^s([a,b])$ is constructed as a **mass-weighted** $L^2$ space:

$$\langle u, v \rangle_{H^s} = \langle (k^2 I + \Delta)^s u, v \rangle_{L^2}$$

The mass operator $M = (k^2 I + \Delta)^s$ and its inverse are represented by `BesselSobolev` and `BesselSobolevInverse`. This is the Bessel potential approach (also called the Matérn-type regularisation in geostatistics).

### Inner Product Computation

1. **Callable functions**: $\langle u, v \rangle = \int_a^b u(x)v(x)w(x)\,dx$ via `scipy.integrate.simpson` (default) or `trapezoid` or `quad`.
2. **Coefficient vector** (when basis is set): $\langle u, v \rangle = \mathbf{c}^T G \mathbf{d}$ where $G_{ij} = \langle \phi_i, \phi_j \rangle$ is the cached Gram matrix.
3. **Dual pairing**: `LinearFormKernel` evaluates $\phi(f) = \int k(x)f(x)\,dx$ directly.

### Quadrature Rules

| Method | scipy function | When used |
|---|---|---|
| Simpson's rule | `scipy.integrate.simpson` | Default, integrates on uniform grid |
| Trapezoidal | `scipy.integrate.trapezoid` | Fast preset; slightly less accurate |
| Adaptive quad | `scipy.integrate.quad` | When `method='adaptive'`; arbitrary precision |

Number of points scales with `IntegrationConfig.n_points` (default 1000); `LebesgueIntegrationConfig.adaptive_spectral(dim)` sets `n_points = max(1000, 10*dim)` for inner products.

### Spectral vs FEM Approaches

| Approach | Where | When to use |
|---|---|---|
| **Spectral (DST/DCT/DFT)** | `Laplacian`, `BesselSobolev` | Dirichlet / Neumann / Periodic BCs; fast $O(N \log N)$ |
| **Spectral (slow integration)** | Same classes, fallback | Robin BCs; $O(N^2)$ or $O(N \cdot n_\text{pts})$ |
| **Finite Differences** | `Gradient`, `Laplacian(method='fd')` | When a point-wise derivative is needed |
| **FEM** | `InverseLaplacian`, `GeneralFEMSolver` | Solving $-\Delta u = f$; hat-function basis has analytical tridiagonal stiffness |

**Fast spectral path details:** Given $f$ sampled at $n$ uniform points, the DST-I (for Dirichlet) computes $\hat{f}_k = \int f(x)\sin(k\pi x/L)\,dx \approx \frac{\sqrt{2L}}{2(n+1)}\text{DST-I}(f)[k]$. Scaling by $\lambda_k = (k\pi/L)^2$ and back-transforming gives $-\Delta f$. The `fast_spectral.py` module handles the full normalisation chain for each BC type precisely.

---

## Public API Summary

### Top-level exports (`intervalinf/__init__.py`)

| Class/Symbol | Module | Purpose | pygeoinf connection |
|---|---|---|---|
| `IntervalDomain` | `core/domain.py` | 1D interval with meshing/integration | — |
| `BoundaryConditions` | `core/boundary.py` | BC specification (D/N/R/P/mixed) | — |
| `Function` | `core/functions.py` | Callable function on interval | Vector type for `HilbertSpace` |
| `RepresentationSpec` | `core/materialization.py` | Hashable fixed-grid/spectral cache key | Hidden representation helper |
| `Materialization` | `core/materialization.py` | Cached grid + function values | Hidden representation helper |
| `IntegrationConfig` | `core/config.py` | Quadrature settings | — |
| `ParallelConfig` | `core/config.py` | Parallelisation settings | — |
| `Lebesgue` | `spaces/lebesgue.py` | $L^2([a,b])$ Hilbert space | Implements `HilbertSpace` |
| `LebesgueSpaceDirectSum` | `spaces/lebesgue.py` | Direct sum of $L^2$ spaces | Implements `HilbertSpaceDirectSum` |
| `LebesgueIntegrationConfig` | `spaces/lebesgue.py` | Hierarchical integration config for $L^2$ | — |
| `LebesgueParallelConfig` | `spaces/lebesgue.py` | Hierarchical parallel config for $L^2$ | — |
| `Sobolev` | `spaces/sobolev.py` | $H^s([a,b])$ Hilbert space | Implements `MassWeightedHilbertSpace` |
| `SobolevSpaceDirectSum` | `spaces/sobolev.py` | Direct sum of $H^s$ spaces | Implements `HilbertSpaceDirectSum` |
| `LinearFormKernel` | `spaces/forms.py` | Kernel-based linear form | Implements `LinearForm` |
| `KnownRegion` | `spaces/lebesgue.py` | Fixed-value sub-interval spec | — |
| `PartitionedLebesgueSpace` | `spaces/lebesgue.py` | $L^2$ with known regions | Extends `Lebesgue` |

### Operators (from `intervalinf.operators`)

| Class | Mathematical operator | pygeoinf base |
|---|---|---|
| `SpectralOperator` | Abstract spectral operator | `LinearOperator` |
| `Laplacian` | $\alpha(-\Delta)$ | `SpectralOperator` |
| `InverseLaplacian` | $(-\Delta)^{-1}$ | `SpectralOperator` |
| `Gradient` | $d/dx$ | `LinearOperator` |
| `BesselSobolev` | $(k^2 I + \Delta)^{s/2}$ | `LinearOperator` |
| `BesselSobolevInverse` | $(k^2 I + \Delta)^{-s/2}$ | `LinearOperator` |
| `SOLAOperator` | $G: f \mapsto (\int fk_i)_i$ | `LinearOperator` |
| `RadialLaplacian` | Radial $-\partial_r^2 - (2/r)\partial_r$ | `SpectralOperator` |
| `InverseRadialLaplacian` | Inverse of above | `SpectralOperator` |

### Providers (from `intervalinf.providers`)

| Class | Family | Level |
|---|---|---|
| `FunctionProvider`, `IndexedFunctionProvider`, `EigenvalueProvider`, `BasisProvider`, `SpectrumProvider` | Abstract | 0 |
| `SineEigenvalueProvider`, `CosineEigenvalueProvider`, `FourierEigenvalueProvider`, `MixedDNEigenvalueProvider`, `MixedNDEigenvalueProvider` | Eigenvalues | 1 |
| `SineFunctionProvider`, `CosineFunctionProvider`, `FourierFunctionProvider`, `MixedDNFunctionProvider`, `MixedNDFunctionProvider`, `RobinFunctionProvider` | Trigonometric | 2 |
| `HatFunctionProvider`, `SplineFunctionProvider` | FEM | 2 |
| `BumpFunctionProvider`, `BumpFunctionGradientProvider` | Smooth | 2 |
| `WaveletFunctionProvider` | Wavelet | 2 |
| `BoxCarFunctionProvider`, `DiscontinuousFunctionProvider` | Step | 2 |
| `KernelProvider`, `NormalModesProvider` | Data | 2 |
| `LaplacianEigenvalueProvider`, `LaplacianSpectrumProvider` | Laplacian composite | 3–4 |
| `RadialLaplacianDirichletProvider`, `RadialLaplacianNeumannProvider`, `RadialLaplacianDDProvider`, `RadialLaplacianDNProvider`, `RadialLaplacianNDProvider`, `RadialLaplacianNNProvider` | Radial | 3 |

### Sampling

| Class | Module | Purpose |
|---|---|---|
| `KLSampler` | `sampling/kl_sampler.py` | KL expansion sampler for Gaussian measures |
| `TruncationInfo` | `sampling/kl_sampler.py` | Dataclass with truncation metadata |

### Utils

| Class | Module | Purpose |
|---|---|---|
| `RobinRootFinder` | `utils/robin_utils.py` | Root-finding for Robin BC eigenvalues |
| `GeneralFEMSolver` | `operators/_impl/fem_solvers.py` | Pure-numpy FEM stiffness assembly and solve |

---

## Test Coverage Summary

| Test file | What is tested |
|---|---|
| `tests/conftest.py` | Shared fixtures: `unit_domain`, `pi_domain`, `simple_space` |
| `tests/core/test_domain.py` | `IntervalDomain`: init validation, `length/center/radius`, `contains` (all boundary types, array input), `uniform_mesh`, `integrate`, `restriction_to_subinterval`, `split_at_discontinuities`, `interior`/`closure`, `__repr__` |
| `tests/core/test_boundary.py` | `BoundaryConditions`: all BC types and factory methods; Robin parameter validation; homogeneous detection; equality |
| `tests/core/test_config.py` | `IntegrationConfig`, `ParallelConfig`, `LebesgueIntegrationConfig`, `LebesgueParallelConfig`: defaults, presets, `copy()`, hierarchical override |
| `tests/core/test_functions.py` | `Function`: callable/coefficient creation, standalone vs attached modes, `evaluate`, arithmetic (+/-/*/÷), compact support masking, `integrate`, `attach_to_space`, `detach` |
| `tests/spaces/test_lebesgue.py` | `Lebesgue`: init (basis types, callable basis, dimension mismatch, invalid type), properties, `zero`, `inner_product`, Riesz maps (`to_dual`/`from_dual`), hierarchical config, `gram_matrix`, `restrict`, `KnownRegion`, `PartitionedLebesgueSpace`, `LebesgueSpaceDirectSum` |
| `tests/spaces/test_sobolev.py` | `Sobolev`: init with `None` Laplacian (deferred placeholder), import guards, `SobolevSpaceDirectSum`, docstring existence |
| `tests/spaces/test_forms.py` | `LinearFormKernel`: init with kernel/components/mapping, exactly-one constraint, parallel config, lazy `components`, direct sum evaluation |
| `tests/operators/test_operators.py` | Import tests for all operator classes; `Laplacian` creation (spectral and FD methods); basic application tests; eigenvalue/eigenfunction retrieval |
| `tests/operators/test_sola.py` | **SOLAOperator test suite (Phases 2–5).** 103 tests. Phase 2 coverage: analytic forward-integral checks (constant/polynomial/trig kernels with analytic reference values); linearity; adjoint-consistency $\langle G(f), y\rangle_D = \langle f, G^*(y)\rangle_M$; provider-backed kernels (`SineFunctionProvider`, `BumpFunctionProvider`, `CosineFunctionProvider`); direct `Function`-list and callable-list kernels; `cache_kernels` behavior and `get_cache_info`/`clear_cache` accessors; integration-method coverage; compact-support locality; Gram-matrix symmetry; `for_direct_sum`. Phase 3–4: fixed-grid batched path; complex-valued kernels; adaptive vs fixed dispatch. Phase 5 (`TestPhase5ReuseAndCaching`): shared mesh unbuilt→built→same-object; `shared_mesh_built` in `get_cache_info`; eval cache `None` when disabled; cache populated/correct shape after call; cached == uncached bitwise; provider-backed caching; support-overlap kernel excluded from eval cache; `clear_cache` empties entries; `clear_mesh_cache` clears eval cache but preserves shared mesh; repeated N_REPS=50 workload smoke test. |
| `tests/operators/test_reduced.py` | Reduced SOLA operator coverage (10 tests, full suite now 476 passed): fast Gram matches slow quadrature to `rtol=1e-8`; Gram symmetry and PSD; cross-Gram shape/value checks; reduced Gram and cross-Gram matrix-backed operator apply paths; Simpson/trapz weight normalization. |
| `tests/providers/test_standalone_providers.py` | Trigonometric, FEM, smooth, wavelet, step, and data providers in standalone (domain-only) mode: `is_standalone`, `function_domain`, evaluation correctness for sine functions |

**Testing patterns:**
- Uses `np.testing.assert_allclose` for floating-point comparisons
- Fixtures provide `unit_domain` ($[0,1]$) and `pi_domain` ($[0,\pi]$)
- Operator tests use `@pytest.mark.slow` for expensive spectral applications
- Mock spaces (`MockSpace`, `MockHilbertSpace`) used in `core` and `spaces` tests to avoid circular dependencies

---

### DLI Performance Analysis Report

`docs/agent-docs/references/living/dli-performance-analysis-and-speedup-targets.md` — Phase 4 deliverable. Ranked optimization opportunities derived from Phase 1–3 benchmark evidence. Key findings: oracle dominates at ~95% of total solve time; within oracle, `support_value_model` is ~46% and `support_point_model` is ~31% (both model-prior operations on Lebesgue space); data-side operations are negligible (<0.1%). Top target: eliminate duplicated support evaluations (T1-A, est. 30–44% total speedup). See the report for full tier ranking (T1-A/B, T2-A/B/C, T3-A/B).

### DLI Optimization Roadmap

`docs/agent-docs/references/living/dli-optimization-roadmap.md` — Phase 5 deliverable. Concrete implementation roadmap grounded in Phase 4 evidence. Selects three targets (A: eliminate duplicated support evaluations, B: support-aware batched mesh for compact-support kernels, C: cache adjoint operator object) with acceptance metrics, workstream breakdowns, rollback criteria, and explicit deferral rationale for T2-B/C, T3-A/B, and warm-start enhancements. Benchmark artifacts and rerun commands are included.

---

## Demo Notebooks (`demos/convex_analysis/`)

Interactive Jupyter notebooks demonstrating end-to-end workflows using intervalinf and pygeoinf.

| Notebook | Purpose |
|---|---|
| `demos/convex_analysis/dli.ipynb` | Distributed Linear Inference (DLI) demo: builds `SOLAOperator`-based forward and property operators, runs convex optimisation via proximal/bundle/Chambolle–Pock solvers, visualises the admissible set. |
| `demos/convex_analysis/bg_with_errors_minkowski.ipynb` | **Single-run** Backus–Gilbert admissible-region notebook with data errors. Constructs the Minkowski-sum formulation for one fixed $N_d$, uses chi-square confidence calibration, computes the optimal BG estimator $X^\star = \mathcal{T} G^* (GG^* + \alpha \mathbf{C}_\mathcal{D})^{-1}$ via pygeoinf operator algebra, and plots the 2D admissible property region together with width decomposition diagnostics. |
| `demos/convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb` | **Multi-$N_d$ comparison companion** to the above. Defines reusable helper functions (`build_spaces_and_operators`, `build_true_model_and_noise`, `build_prior_and_confidence`, `build_optimal_estimator`, `build_admissible_support`, `run_single_nd_case`) and executes the full BG pipeline for `N_d_values = [5, 10, 20, 50, 100]`, storing each case in `nd_results`. Key design points: (1) fixed `random_seed` for `NormalModesProvider` gives nested kernels (N_d=10 is a strict superset of N_d=5); (2) per-datum noise calibrated from RMS signal amplitude keeps `sigma_d` consistent across N_d; (3) chi-square confidence level is fixed so ellipsoid scaling is model-driven, not sample-fitted; (4) Phase 2 validation checks exact N_d coverage, required geometry keys, `PolyhedralSet` construction, true-property containment, and positive widths; (5) **overlay figure** plots all 5 absolute admissible property sets on a single plasma-coloured figure via `region_boundary_xy` (support-function polygon reconstruction), saved as PNG and PDF; (6) **diagnostic summary table** reports per-case `sigma_d`, `r_V`, `alpha`, `x_width`, `y_width` with a width-comparison summary. End-to-end validation confirmed (Phase 4): all 14 code cells pass, the overlay figure renders with all 5 distinct regions, the region tightens substantially overall from `N_d=5` to `N_d=100`, and `p_bar` lies inside every admissible region. The notebook also states the interpretation caveat that directional widths need not decrease monotonically for every intermediate `N_d`, because the region depends on operator geometry and sampled support directions. |

---
