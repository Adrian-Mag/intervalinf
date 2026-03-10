# DLI Optimization Roadmap

Evidence-based implementation roadmap for intervalinf-backed proximal-bundle
DLI, derived from the Phase 4 performance analysis.

**Prerequisites:**
- Phase 4 analysis: `intervalinf/docs/agent-docs/references/living/dli-performance-analysis-and-speedup-targets.md`
- Benchmark harnesses: `intervalinf/rough_work/benchmark_dli_oracle_bundle.py`, `intervalinf/rough_work/benchmark_phase2_compact_support.py`
- Benchmark artifacts (CSV): `intervalinf/rough_work/benchmark_dli_oracle_bundle_oracle.csv`, `benchmark_dli_oracle_bundle_bundle.csv`, `benchmark_dli_oracle_bundle_warmstart.csv`, `benchmark_phase2_compact_support_results.csv`

**Plan reference:** `intervalinf/docs/agent-docs/completed-plans/dli-benchmarking-and-analysis-plan.md`, Phase 5.

---

## 1. Measured Cost Summary (from Phase 4)

| Component | Share of total solve time |
|-----------|--------------------------|
| Dual oracle (`value_and_subgradient`) | ~95% |
| Master QP (`_solve_master`) | ~4–5% |
| Overhead | ~1% |

Within the oracle:

| Sub-operation | Share of oracle time |
|---------------|---------------------|
| `support_value_model` | ~46% (dominant, duplicates work) |
| `support_point_model` | ~31% |
| Unaccounted arithmetic / Riesz maps | ~23% |
| `gstar_apply` | <0.3% |
| Data-side operations | <0.2% (negligible) |

Compact-support fallback path: **~12.8× slower** than batched fixed-grid
path.  Warm-start (padded lambda) saves ~8% iterations / ~1.04× wall time in
the measured $N_d = 5 \to 10$ case.

---

## 2. Selected Optimization Targets

Three targets were selected, ordered by expected payoff-to-risk ratio.  All
others are deferred; see §5 for explicit reasoning.

### Target A — Eliminate Duplicated Support Evaluations (T1-A)

**What to change:** In `DualMasterCostFunction.value_and_subgradient`
(`pygeoinf/pygeoinf/backus_gilbert.py`), replace the separate
`support_value_model` call with the inner-product identity.  When
`support_point` returns a maximizer $v$, the support value is recoverable
exactly as:
$$\sigma_C(q) = \langle q, v \rangle$$
No second pass through the support-function code is needed.  The finite-
difference fallback path (when `support_point` returns `None`) is unchanged.

**Why it is the highest priority:** `support_value_model` is ~46% of oracle
time. Oracle is ~95% of total. Eliminating this sub-call alone reduces total
solve time by an estimated **30–44%**, depending on how much of the removed
work is replaced by the inner-product evaluation (which is cheap on `Lebesgue`
via cached basis coefficients).

**Files:** `pygeoinf/pygeoinf/backus_gilbert.py` (primary).  Optionally
`pygeoinf/pygeoinf/convex_analysis.py` if a combined
`value_and_support_point` API is preferred.

**Estimated payoff:** 30–44% reduction in total solve time (conservative lower
bound excludes inner-product replacement cost; upper bound assumes negligible
inner-product cost).

**Risk:** Low.  The identity is exact for any proper convex support function
when $v$ is a true maximizer.  A purely local change in the oracle; no
algorithmic impact on the bundle solver.

---

### Target B — Support-Aware Batched Mesh for Compact-Support Kernels (T2-A)

**What to change:** In `SOLAOperator._apply_kernels_fixed_grid`
(`intervalinf/intervalinf/operators/sola.py`), extend the batched path to
handle compact-support kernels by constructing a support-narrowed mesh per
group of co-support kernels, then evaluating the integrand once on that mesh
and integrating all kernels in the group together.

The current fallback issues one `domain.integrate(product_callable, support=intersection)` call per kernel, incurring a Python loop with closure allocation per kernel. The replacement groups kernels by their intersection support interval(s), builds the exact narrowed mesh once (following the same proportional-point-distribution rules as `IntervalDomain.integrate`), and stacks the group into one batched integration.

**Files:** `intervalinf/intervalinf/operators/sola.py` (primary);
`intervalinf/intervalinf/core/domain.py` may need an internal mesh-building
helper exposed.

**Estimated payoff:** Recovers near-batched performance for compact-support
problems. The ~12.8× penalty drops to ~1–2×. In the DLI context the forward
operator is called once per oracle call, so the saving compounds across all
bundle iterations (multiplied by oracle-call count).

**Risk:** Moderate.  The narrowed-mesh semantics must exactly reproduce
`IntervalDomain.integrate` behavior; `test_fast_path_narrow_support_matches_generic`
asserts ≤ 1e-12 agreement against the per-kernel reference.  Multi-interval
support allocation rules and the `_kernel_eval_cache` exclusion for
compact-support kernels (enforced by existing tests) must both be preserved.

---

### Target C — Cache Adjoint Operator Object (T1-B)

**What to change:** In `DualMasterCostFunction.__init__`
(`pygeoinf/pygeoinf/backus_gilbert.py`), store `self._G_adj = self._G.adjoint`
once and call `self._G_adj(lam)` in each oracle invocation instead of
accessing `self._G.adjoint` on every call (which allocates a new wrapper).

**Files:** `pygeoinf/pygeoinf/backus_gilbert.py`.

**Estimated payoff:** <1% of total solve time. Included as a zero-risk
mechanical cleanup; not a primary speedup driver.

**Risk:** Negligible.

---

## 3. Acceptance Metrics

All three targets must pass the following criteria before being considered
complete.

### 3.1 Numerical Agreement

| Metric | Threshold | How to check |
|--------|-----------|--------------|
| Support values $\sigma_C(q)$ | Relative difference ≤ 1e-10 vs pre-change baseline | Re-run bundle benchmark; compare `support_value` in `benchmark_dli_oracle_bundle_bundle.csv` at $N_d \in \{5,10\}$ |
| Bundle convergence quality | Relative change in `gap` ≤ 1e-6 vs pre-change baseline | Compare `gap` column in `benchmark_dli_oracle_bundle_bundle.csv` |
| Iteration count stability | No increase larger than 1 iteration on measured cases unless accompanied by a larger time win | Compare `num_iterations` column in `benchmark_dli_oracle_bundle_bundle.csv` |
| Forward-operator values `G(f)` (Target B) | Relative difference ≤ 1e-12 vs per-kernel reference | `test_fast_path_narrow_support_matches_generic` in `tests/operators/test_sola.py` must pass |

### 3.2 Runtime Improvement

| Target | Oracle time reduction | Total solve time reduction |
|--------|----------------------|--------------------------|
| A (duplicate evals) | ≥ 25% of oracle wall time at $N_d \in \{5,10\}$ | ≥ 20% of total solve time |
| B (batched mesh) | N/A (affects forward operator, not oracle directly) | `overlapping_fallback` scenario ≤ 2× `full_domain` time at $N_d = 40$, $n\_pts = 500$ |
| C (adjoint cache) | Immeasurable at this scale | Not a performance gate; pass on code correctness alone |

Runtime acceptance uses the existing benchmark harnesses as the measurement
protocol (see §4).

### 3.3 Test Suite Health

- Full `pytest` run on `intervalinf/tests/` and `pygeoinf/tests/` must remain
  green after each target.
- No existing assertions in benchmark harnesses may be weakened to achieve
  the improvement.

---

## 4. Implementation Plan

### Workstream 1: Oracle Cleanup (Targets A + C)

**Scope:** `pygeoinf/pygeoinf/backus_gilbert.py` only.

| Step | Action |
|------|--------|
| W1-1 | Read existing oracle code for `value_and_subgradient`; map support-value call sites |
| W1-2 | Implement Target C (adjoint cache) — one-line change, commit independently for clean diff |
| W1-3 | Implement Target A (inner-product identity), but keep `support_value` call active behind a flag or in the fallback branch |
| W1-4 | Run oracle benchmark at $N_d \in \{5,10\}$ with `n_reps=50`; verify runtime reduction ≥ 25% of oracle time |
| W1-5 | Verify numerical agreement: compare bundle CSV `support_value`, `gap`, and `num_iterations` against baseline; in oracle CSV confirm `support_val_model_s` drops sharply toward zero |
| W1-6 | Run full test suite; fix any regressions |
| W1-7 | Commit: `perf(oracle): eliminate duplicated support_value calls (T1-A, T1-B)` |

**Rollback criterion:** If test suite fails *and* the failure cannot be
traced to a clear correctness bug, revert to pre-W1 state.  If runtime
reduction is < 10% of oracle time, investigate before proceeding to W1-7.

---

### Workstream 2: Batched Mesh for Compact-Support Kernels (Target B)

**Scope:** `intervalinf/intervalinf/operators/sola.py`; possibly
`intervalinf/intervalinf/core/domain.py`.

This workstream is independent of Workstream 1 and can be developed in
parallel or sequentially.

| Step | Action |
|------|--------|
| W2-1 | Re-read `_apply_kernels_fixed_grid` and `IntervalDomain.integrate` to understand the fallback dispatch condition and mesh-allocation rules |
| W2-2 | Write a failing test asserting that `overlapping_fallback` scenario runs ≤ 2× `full_domain` time (add to `tests/operators/test_sola.py` or a scratch test module) |
| W2-3 | Implement a `_group_by_support(kernels)` helper that clusters compact-support kernels by intersected support interval |
| W2-4 | Implement the batched-mesh integration path; ensure the resulting values are bitwise-identical to the per-kernel fallback at 1e-12 tolerance |
| W2-5 | Run `benchmark_phase2_compact_support.py`; verify `overlapping_fallback` ≤ 2× `full_domain` at $N_d = 40$, $n\_pts = 500$ |
| W2-6 | Confirm `_kernel_eval_cache` exclusion behavior unchanged for compact-support kernels (Phase 5 tests must still pass) |
| W2-7 | Run full test suite; fix any regressions |
| W2-8 | Commit: `perf(sola): support-aware batched mesh for compact-support kernels (T2-A)` |

**Rollback criterion:** If the batched path cannot reproduce per-kernel
results at 1e-12 relative tolerance on representative multi-interval support
cases, the intermediate grouping strategy should be revised.  Do not loosen
the existing test tolerance to satisfy this.

---

## 5. What Is NOT Worth Prioritizing Now

The following candidates were explicitly deprioritized based on the benchmark
evidence.  Any future agent should read this section before reopening these
items.

| Target | Reason for deferral |
|--------|---------------------|
| **T2-B: Persistent OSQP instance** | Master QP is only ~4–5% of total time. Even a 70–80% reduction in QP cost saves only ~3% overall. Implementing live OSQP warm-start with incremental constraint addition is non-trivial (bundle trimming, changing $\rho$); payoff does not justify the risk at current problem sizes. Revisit only if profiling at larger $N_d$ (≥ 50) shows QP fraction growing. |
| **T2-C: Incremental bundle linearization matrix** | Also proportional to master QP overhead (~1–2% of total). Marginal payoff. Defer alongside T2-B. |
| **T3-A: Vectorized adjoint reconstruction** | `gstar_apply` is <0.3% of oracle time. No measurable payoff. |
| **T3-B: Bundle persistence warm-start** | Padded cuts are not valid tight bounds on the larger-dimension objective. Current iteration-count savings (~8%) are modest, and extending to bundle persistence requires theoretical validation (padded-cut convergence conditions are not established). High implementation risk relative to unconfirmed payoff. |
| **Warm-start improvements in general** | The measured 1.04× speedup from lambda padding is too small to warrant further investment given the structural change required (bundle rebuilding from scratch dominates). |

---

## 6. Benchmark Artifacts to Rerun After Each Workstream

Run these after each workstream implementation to confirm acceptance criteria:

**After Workstream 1 (Targets A + C):**
```bash
conda run -n inferences3 python intervalinf/rough_work/benchmark_dli_oracle_bundle.py
```
Artifacts produced: `benchmark_dli_oracle_bundle_oracle.csv`,
`benchmark_dli_oracle_bundle_bundle.csv`, `benchmark_dli_oracle_bundle_warmstart.csv`.

Columns to check:
- `oracle.csv`: `support_val_model_s` should be near-zero vs baseline, while `support_pt_model_s` remains of the same order.
- `bundle.csv`: `support_value` and `gap` should match baseline within the numerical thresholds from §3.1.
- `bundle.csv`: `elapsed_s` should be ≥ 20% lower than the pre-change baseline on the measured cases.

**After Workstream 2 (Target B):**
```bash
conda run -n inferences3 python intervalinf/rough_work/benchmark_phase2_compact_support.py
```
Artifact: `benchmark_phase2_compact_support_results.csv`.

Key rows: `scenario=overlapping_fallback`, `N_d=40`, `n_points=500`.
Acceptance: `wall_ms_median` ≤ 2× the `full_domain` row with the same $N_d$/$n\_pts$.

**After both workstreams (regression check):**
```bash
conda run -n inferences3 python -m pytest intervalinf/tests/ pygeoinf/tests/ -x -q
```

---

## 7. Open Assumptions and Caveats

1. **Benchmark scope is narrow.** All timing evidence comes from $N_d \in \{5, 10\}$ with `forward_seed=2, data_seed=42`. The oracle-dominated cost split (~95%) is expected to persist at larger $N_d$, but the absolute and relative times may shift. The acceptance metric thresholds (§3.2) should be re-evaluated at $N_d = 40$ once the workstreams are complete.

2. **Inner-product cost on Lebesgue.** Target A replaces a `BallSupportFunction` call with `Lebesgue.inner_product(hilbert_residual, v)`. The benchmark reports `support_value_model` as ~46% of oracle time; the inner-product call is not separately timed. If the inner-product turns out to be expensive (e.g., due to numerical integration over the full domain), the achieved speedup could be below the 30% lower bound. The post-W1 benchmark will resolve this.

3. **Compact-support grouping edge cases.** Target B groups kernels by intersected support. Kernels with multi-interval support (produced by `split_at_discontinuities`) require grouping by the full set of sub-intervals, not just the outer bounding box. The implementation must correctly identify when two kernels share the same exact support intervals (not merely overlapping outer bounds) before they can be batched onto a shared mesh.

4. **pygeoinf/intervalinf joint test suite.** Workstream 1 modifies `pygeoinf`; its changes must be validated against both `pygeoinf/tests/` and `intervalinf/tests/` (the latter exercises the full DLI stack including `backus_gilbert.py`).

5. **No claim on non-DLI workflows.** These optimizations are profiled in the context of proximal-bundle DLI only. Other pygeoinf workflows using `BallSupportFunction.support_value` may or may not benefit depending on their calling patterns.
