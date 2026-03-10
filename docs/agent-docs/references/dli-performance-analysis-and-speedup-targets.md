# DLI Performance Analysis and Speedup Targets

Evidence-based analysis of DLI runtime distribution and ranked optimization
opportunities, produced from Phases 1–3 of the DLI benchmarking plan.

**Plan reference:** `intervalinf/docs/agent-docs/active-plans/dli-benchmarking-and-analysis-plan.md`, Phase 4.

---

## 1. Measured Performance Profile

### 1.1 End-to-End Cost Distribution (Phase 3)

Full proximal-bundle solves on intervalinf-backed DLI problems show:

| Component | % of total solve time | Source |
|-----------|----------------------|--------|
| Dual oracle (`value_and_subgradient`) | ~95% | Phase 3 bundle mode |
| Master QP (`_solve_master`) | ~4–5% | Phase 3 bundle mode |
| Overhead (step logic, bookkeeping) | ~1% | Residual |

*Scope note:* These fractions are from the Phase 3 bundle CSV
(`benchmark_dli_oracle_bundle_bundle.csv`) at $N_d \in \{5, 10\}$ with a
single seed pair (forward=2, data=42).  The 95%/4–5% split may shift
slightly with larger $N_d$ or different solver settings, but the
"oracle-dominated" conclusion is expected to be robust.

**Conclusion:** Optimizing anything outside the oracle has at most ~5% payoff.
All high-leverage targets must reduce per-call oracle cost or reduce the
number of oracle calls.

### 1.2 Oracle Internal Breakdown (Phase 3)

Within each `value_and_subgradient` call, the timed sub-operations are
(measured from Phase 3 oracle CSV, averaged over $N_d = 5$ and $N_d = 10$):

| Sub-operation | % of oracle time | What it computes |
|---------------|-----------------|------------------|
| `support_value_model` | **~46%** | $\sigma_B(T^*q - G^*\lambda)$: model-prior support value (separate call, repeats norm/inner-product computation) |
| `support_point_model` | **~31%** | $v \in \partial\sigma_B(T^*q - G^*\lambda)$: model-prior support point via `BallSupportFunction.support_point` on Lebesgue space |
| `support_point_data` | <0.1% | $w \in \partial\sigma_V(-\lambda)$: data-error support point via `EllipsoidSupportFunction.support_point` on Euclidean space |
| `support_value_data` | <0.1% | $\sigma_V(-\lambda)$: data-error support value on Euclidean space (cheap) |
| `gstar_apply` | <0.3% | $G^*\lambda$: adjoint operator application (SOLA `_dual_mapping`) |
| Unaccounted (arithmetic, $\tilde d$ inner product) | ~23% | Hilbert space arithmetic, Riesz representatives, component extraction |

*Data source:* `benchmark_dli_oracle_bundle_oracle.csv`, $N_d=5$:
`support_pt_model_s=0.056` (30.5%), `support_val_model_s=0.084` (45.8%),
`gstar_s=0.001` (0.3%), data-side totals < 0.2%.

The key structural finding is that **`support_value_model` is the single
largest cost component**, and it repeats computation (norms, inner products)
already performed by `support_point_model`.  Data-side operations are
negligible because they operate on finite-dimensional `EuclideanSpace`
rather than the infinite-dimensional Lebesgue space.

### 1.3 Compact-Support Fallback Penalty (Phase 2)

When kernel compact-support metadata forces per-kernel integration:

| Scenario | Median time (N_d=40, n_pts=500) | Slowdown vs batched |
|----------|-------------------------------|---------------------|
| `full_domain` (batched fixed-grid) | 0.88 ms | 1.0× (baseline) |
| `overlapping_fallback` (per-kernel `domain.integrate`) | 11.30 ms | ~12.8× |
| `mixed_paths` (half batched, half fallback) | 6.07 ms | ~6.9× |
| `disjoint_support` (all skipped) | ~0 ms | (free) |

The 12.8× penalty comes from: (a) Python-level per-kernel loop with closure
allocation and recursive `IntervalDomain.integrate` calls, vs. (b) a single
vectorized batch evaluation + Simpson/trapezoid integration over a shared mesh.

### 1.4 Warm-Start Effectiveness (Phase 3)

Nested $N_d$ warm-start (pad small-problem $\lambda^*$ with zeros):

| Pair | Cold iters | Warm iters | Reduction | Time speedup |
|------|-----------|-----------|-----------|--------------|
| 5 → 10 | 12 | 11 | −1 (−8%) | 1.04× |

The modest improvement reflects: (a) the padded start is geometrically
close to the larger-problem optimum (nested-kernel prefix property), but
(b) only the starting point transfers — the bundle (cutting planes) must
be rebuilt from scratch.

---

## 2. Ranked Optimization Targets

### Tier 1: High Leverage, Low Risk

#### T1-A. Eliminate Duplicated Support-Function Evaluations

**Current state:** Each oracle call computes `support_point(q)` returning a
maximizer $v \in \partial\sigma_C(q)$, then *separately* calls the support
function $\sigma_C(q)$ to get the scalar value. Both calls independently
compute norms, inner products, or operator applications.

**Optimization:** When `support_point` returns a valid maximizer $v$, the
support value is recoverable via the identity:
$$\sigma_C(q) = \langle q, v \rangle$$
No second call to the support function is needed.

**Implementation options:**
- **Option A (local, in `DualMasterCostFunction`):** After obtaining `v` and
  `w`, compute `term_model = M.inner_product(hilbert_residual, v)` and
  `term_data = D.inner_product(neg_lam, w)` instead of calling the support
  functions again. Use the current scalar call only when `v` or `w` is `None`
  (finite-difference fallback path).
- **Option B (API-level):** Add `SupportFunction.value_and_support_point(q)`
  that returns both in one pass with shared intermediates (e.g., reuse
  $\|q\|$ in `BallSupportFunction`, reuse $A^{-1/2}q$ in
  `EllipsoidSupportFunction`).

**Estimated payoff:** Eliminates the `support_value_model` cost (~46% of
oracle time) and `support_value_data` (<0.1%, negligible).  Since oracle
is ~95% of total: **~44% reduction in total solve time.**  Even accounting
for the inner-product computation that replaces the support-value call
(which is cheap on Lebesgue via cached basis coefficients), a conservative
estimate is **30–44% reduction in total solve time.**

**Risk:** Low. The identity $\sigma_C(q) = \langle q, x^*(q)\rangle$ is
exact for any proper convex support function when $x^*$ is a maximizer.
The fallback path (finite-difference) is already implemented for cases
where `support_point` returns `None`.

**Files to modify:** `pygeoinf/pygeoinf/backus_gilbert.py` (primary);
optionally `pygeoinf/pygeoinf/convex_analysis.py` (for Option B).

---

#### T1-B. Cache the Adjoint Operator Object

**Current state:** `self._G.adjoint` is a Python property that allocates a
new `LinearOperator` wrapper each time it is accessed (when `_adjoint_base`
is `None`). In the SOLA case this wrapper delegates to `_dual_mapping`, but
the wrapper itself is freshly allocated every oracle call.

**Optimization:** Store `self._G_adj = self._G.adjoint` once in
`DualMasterCostFunction.__init__` and use `self._G_adj(lam)` in each oracle
call.

**Estimated payoff:** Eliminates one object allocation per oracle call.
Likely very small (< 1% of oracle time) since G* apply is already < 1%.
Still worth doing as a zero-risk cleanup.

**Risk:** Negligible—purely mechanical caching.

**Files to modify:** `pygeoinf/pygeoinf/backus_gilbert.py`.

---

### Tier 2: Moderate Leverage, Moderate Risk

#### T2-A. Support-Aware Batched Mesh for Compact-Support Kernels

**Current state:** When kernels carry overlapping compact-support metadata,
`SOLAOperator._apply_kernels_fixed_grid` falls back to per-kernel
`domain.integrate(product_callable, support=intersection)` calls. This is
~12.8× slower than the batched path.

**Optimization:** Group compact-support kernels by their
`intersected_support` (list of interval endpoints). For each group:
1. Build the exact support-narrowed mesh using the same allocation rules
   as `IntervalDomain.integrate` (proportional point distribution across
   sub-intervals).
2. Evaluate `func` once on that mesh.
3. Stack-evaluate all kernels in the group on the same mesh.
4. Perform a single batched Simpson/trapezoid integration.

**Estimated payoff:** Recovers near-batched performance for compact-support
cases. The 12.8× penalty drops to ~1–2× (mesh build overhead plus the
grouping logic). In the DLI context, the forward operator $G$ is called
once per oracle call (for the subgradient), so the per-call saving is the
full forward-time reduction; this compounds across all bundle iterations.

**Risk:** Moderate. The narrowed-mesh semantics must exactly reproduce
`IntervalDomain.integrate` behavior to satisfy strict tests (e.g.,
`test_fast_path_narrow_support_matches_generic` asserts 1e-12 agreement).
Multi-interval support allocation rules must be faithfully replicated.
Additionally, compact-support kernels are currently excluded from
`_kernel_eval_cache` by design and tests enforce this — any caching
strategy for these kernels needs careful scoping.

**Files to modify:** `intervalinf/intervalinf/operators/sola.py`,
potentially `intervalinf/intervalinf/core/domain.py`.

---

#### T2-B. Persistent OSQP Instance for Master QP

**Current state:** `OSQPQPSolver` creates a fresh OSQP model per
`_solve_master` call. The factorization of the KKT system is discarded
between iterations, even though the proximal QP structure is nearly
identical across consecutive master solves (only one new cutting plane
is appended).

**Optimization:** Maintain a persistent OSQP instance across iterations.
Use OSQP's warm-start API and `update()` method to add new constraints
incrementally rather than rebuilding the entire problem.

**Estimated payoff:** Master QP is ~4–5% of total time. With incremental
updates, the per-iteration QP cost could drop by 50–80%, saving ~2–4% of
total time. The benefit grows for problems requiring many iterations
(large $N_d$, tight tolerances).

**Risk:** Moderate. Requires careful handling of the OSQP model lifecycle,
bundle trimming (when old cuts are discarded), and the changing proximal
weight $\rho$ (which modifies the quadratic term).

**Files to modify:** `pygeoinf/pygeoinf/convex_optimisation.py`
(`OSQPQPSolver`, `ProximalBundleMethod._solve_master`, `Bundle`).

---

#### T2-C. Incremental Bundle Linearization Matrix

**Current state:** `Bundle.linearization_matrix` allocates a fresh dense
matrix $A \in \mathbb{R}^{n_\text{cuts} \times (d+1)}$ and vector
$b \in \mathbb{R}^{n_\text{cuts}}$ each time it is called (once per master
solve). Each cut's components are extracted via `domain.to_components`.

**Optimization:** Maintain a pre-allocated matrix that grows by one row
per new cut (or shrinks when old cuts are discarded). Avoid re-extracting
components for existing cuts.

**Estimated payoff:** Proportional to T2-B (matrix build is part of master
QP overhead). Small absolute benefit (~1–2% of total), but removes
O($n_\text{cuts} \times d$) per-iteration allocation.

**Risk:** Low-moderate. Requires bookkeeping for cut insertion/deletion
indices and handling the bundle-size cap.

**Files to modify:** `pygeoinf/pygeoinf/convex_optimisation.py` (`Bundle`).

---

### Tier 3: Lower Leverage or Higher Risk

#### T3-A. Vectorized Adjoint Reconstruction

**Current state:** `SOLAOperator._reconstruct_function` returns a `Function`
whose `evaluate_callable` loops over nonzero coefficients at each evaluation
point. When the reconstructed function is subsequently integrated (e.g., for
support-point computations), this loop executes at every quadrature point.

**Optimization:** Pre-evaluate the adjoint function on the integration mesh
and cache the result as a numpy array, avoiding repeated per-point loops.

**Estimated payoff:** G* apply is < 0.3% of oracle time currently. The
adjoint is called once per oracle call, and its returned `Function` is
evaluated during subsequent support-point/value computations. Vectorizing
the evaluation closure would help only if evaluation dominates those
computations, which the data does not suggest.

**Risk:** Low implementation risk, but negligible payoff.

**Files to modify:** `intervalinf/intervalinf/operators/sola.py`.

---

#### T3-B. Enhanced Warm-Start with Bundle Persistence

**Current state:** `solve_support_values` warm-starts only the starting
$\lambda$; the bundle (cutting planes) is discarded between solves.

**Optimization:** For nested $N_d$ problems sharing a common kernel prefix,
the first $N_d^{\text{small}}$ rows of each cutting plane could be padded
with zeros to form valid (though suboptimal) cuts for the larger problem.

**Estimated payoff:** Current warm-start gives ~8% iteration reduction.
Bundle persistence might push this to ~20–30% for sequential $N_d$ scaling,
but the benefit depends heavily on the problem structure and the quality of
the padded cuts.

**Risk:** High. Padded cuts are no longer tight bounds on the true
objective — they may mislead the bundle algorithm or violate convergence
assumptions. Requires theoretical analysis to confirm validity.

**Files to modify:** `pygeoinf/pygeoinf/convex_optimisation.py`
(`Bundle`, `solve_support_values`).

---

## 3. Summary: Ranked Speedup Candidates

| Rank | Target | Est. Total Speedup | Risk | Complexity |
|------|--------|--------------------|------|------------|
| 1 | **T1-A: Eliminate duplicated support evals** | 30–44% | Low | Small (local change in oracle) |
| 2 | **T2-A: Support-aware batched mesh** | Variable (up to 12× on forward calls with compact support) | Moderate | Medium (mesh grouping + test compliance) |
| 3 | **T2-B: Persistent OSQP instance** | 2–4% | Moderate | Medium (OSQP lifecycle management) |
| 4 | **T1-B: Cache adjoint operator object** | <1% | Negligible | Trivial |
| 5 | **T2-C: Incremental bundle matrix** | 1–2% | Low-moderate | Medium |
| 6 | **T3-A: Vectorized adjoint** | <1% | Low | Small |
| 7 | **T3-B: Bundle persistence warm-start** | 10–20% (speculative) | High | High (theoretical + implementation) |

---

## 4. Recommended Next Steps

The highest-payoff, lowest-risk optimization is **T1-A (eliminate duplicated
support evaluations)**. It addresses the dominant cost component (~95% oracle
time) with a mathematically exact identity and a localized code change.

**Suggested implementation order:**
1. T1-A + T1-B together in a single phase (oracle cleanup).
2. T2-A if compact-support problems are frequent in practice.
3. T2-B + T2-C if master QP becomes a bottleneck at large $N_d$.
4. T3-B only after theoretical validation of padded-cut convergence.
