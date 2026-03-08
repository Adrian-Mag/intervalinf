# Plan: SOLAOperator speedup on SOLAOperator_speedup branch

Speed up `intervalinf.operators.SOLAOperator` while preserving its continuous-first
API semantics. The project will begin with a detailed implementation audit and a
thorough benchmarking baseline, then proceed through progressively more aggressive
optimizations with correctness, robustness, and post-change performance comparisons
baked into every phase.

**Context:**
- Work is taking place on local branch `SOLAOperator_speedup`, created from the local
  `convex_analysis` branch in the nested `intervalinf` git repository.
- `SOLAOperator` is currently a continuous operator in user-facing semantics: it maps
  functions on an interval to Euclidean data by numerically integrating against kernel
  functions, rather than by exposing a fixed matrix discretization.
- In practice, repeated optimization workloads such as PLI and DLI call `SOLAOperator`
  many times with unchanged kernels and unchanged integration settings, so the current
  per-call integration overhead becomes the dominant cost.
- The current implementation performs one integration per kernel, rebuilding product
  callables and `Function` wrappers repeatedly inside Python loops.
- Dedicated functional tests for `SOLAOperator` are currently sparse, so a correctness
  and robustness baseline must be established before optimization work starts.
- Benchmark artifacts for this project should remain in `rough_work/`.
- Any accelerated fixed-grid path should be automatic for reusable fixed-grid methods,
  not opt-in.
- The mismatch between `IntegrationConfig(method="quad")` and the domain-level
  integration naming/behavior is in scope and should be cleaned up as part of this
  work.

---

## Phase 1: Detailed implementation and dependency audit

**Status:** ✅ Complete (2026-03-08)

**Objective:** Build an exact understanding of the current `SOLAOperator`
implementation, call graph, dependencies, invariants, and numerical semantics before
any optimization is attempted.

**Files to inspect:**
- `intervalinf/intervalinf/operators/sola.py`
- `intervalinf/intervalinf/core/functions.py`
- `intervalinf/intervalinf/core/domain.py`
- `intervalinf/intervalinf/spaces/lebesgue.py`
- `intervalinf/intervalinf/spaces/forms.py`
- `intervalinf/intervalinf/providers/base.py`
- relevant provider implementations used in practice:
  - `intervalinf/intervalinf/providers/functions/data.py`
  - `intervalinf/intervalinf/providers/functions/smooth.py`
- heavy call sites in downstream workflows:
  - `pygeoinf/pygeoinf/backus_gilbert.py`
  - current PLI/DLI demos using SOLA-based forward and property operators

**Questions to answer in this phase:**
- What exact work is done in forward application, dual mapping, and adjoint recovery?
- Which parts of the pipeline are generic and which parts are effectively fixed-grid already?
- Where do compact support and domain restrictions exist today, and where are they not propagated?
- Which invariants must be preserved to keep the operator continuous-first in semantics?
- What kernel/provider cases must remain supported without behavior changes?
- What does the current integration-config API promise versus what the implementation actually supports?

**Deliverables:**
- A written architectural summary inside the plan change log and follow-up notes.
- A list of invariants to preserve during optimization.
- A list of benchmark surfaces and risk areas discovered during inspection.

**Outcome:**
- Completed a code-path audit spanning `SOLAOperator`, `Function`, `IntervalDomain`,
  `LinearFormKernel`, relevant providers, direct-sum composition, and downstream
  `DualMasterCostFunction` usage.
- Confirmed the dominant repeated costs are per-kernel forward integrations and repeated
  evaluation of reconstructed adjoint functions inside downstream norms/integrals.
- Identified dedicated SOLA test coverage as a major gap to close before optimization.
- Identified support-propagation and integration-method inconsistencies that should be
  addressed before or alongside acceleration work.
- Wrote the audit note:
  `intervalinf/docs/agent-docs/references/sola-operator-phase-1-audit.md`.

---

## Phase 2: Baseline correctness, robustness, and benchmark suite

**Status:** ✅ Complete (2026-03-08)

**Objective:** Establish a rigorous pre-change baseline so that later speedups can be
compared against known behavior, numerical accuracy, and runtime characteristics.

**Files to create/modify:**
- New: `intervalinf/tests/operators/test_sola.py`
- New or extended baseline benchmark(s) in `intervalinf/rough_work/`
- Possibly extend: `intervalinf/rough_work/benchmark_dli_solvers.py`

**Tests to write:**
- Analytic forward-integral tests on simple domains and simple kernels.
- Random regression tests comparing multiple operator construction modes:
  direct function list, callable list, and `IndexedFunctionProvider`.
- Adjoint consistency tests checking
  `⟨G(f), y⟩_D ≈ ⟨f, G* y⟩_M` under explicit tolerances.
- Compact-support tests, including disjoint support, partial overlap, and empty support.
- Integration-method tests covering the current usable methods and the planned cleanup of
  the `quad` naming/behavior.
- Robustness tests for vectorized and non-vectorized callables.

**Benchmarking requirements:**
- Measure isolated operator costs:
  - `G(f)`
  - `G.adjoint(y)`
  - evaluation/integration of the function returned by the adjoint path
  - `DualMasterCostFunction.value_and_subgradient(...)` as a realistic downstream hotspot
- Measure scaling across workload axes:
  - `N_d` and `N_p`
  - model-space basis dimension
  - integration method
  - `n_points`
  - kernel family (`NormalModesProvider`, `BumpFunctionProvider`, data-driven kernels if applicable)
  - function representation (coefficient-based versus callable-based)
- Record accuracy against a higher-accuracy reference configuration.
- Record robustness behavior for edge cases and fallback paths.

**Benchmark outputs to preserve:**
- wall-clock runtime tables
- relative and absolute output errors
- adjoint consistency residuals
- notes on failure modes or cases where the current implementation is unexpectedly slow or fragile

**Outcome:** A trustworthy baseline that later phases can compare against quantitatively.

**Outcome:**
- Added a dedicated SOLA baseline test module:
  `intervalinf/tests/operators/test_sola.py`.
- Added a rough-work benchmark harness:
  `intervalinf/rough_work/benchmark_sola_baseline.py`.
- Updated the living reference file to document the new SOLA tests, benchmark
  harness, and currently known SOLA behavioral quirks.
- Established baseline coverage for analytic forward correctness, linearity,
  adjoint consistency, provider-backed and direct kernels, caching accessors,
  integration-method behavior, compact-support scenarios, Gram-matrix basics,
  and direct-sum smoke behavior.
- Recorded the current `quad` naming inconsistency as baseline behavior rather
  than changing production code in this phase.
- Verified the new SOLA test module passes after minor review cleanup.

---

## Phase 3: Integration API cleanup and low-risk semantic fixes

**Status:** ✅ Complete (2026-03-08)

**Objective:** Clean up inconsistencies and remove low-risk inefficiencies before adding
more ambitious acceleration paths.

**Primary scope:**
- Resolve the `IntegrationConfig(method="quad")` versus domain-level integration naming mismatch.
- Decide and implement the correct behavior for fixed-grid versus adaptive methods.
- Clarify and codify which methods support automatic batched acceleration.
- Improve obvious avoidable overhead where semantics are unambiguous.

**Likely modifications:**
- `intervalinf/intervalinf/core/config.py`
- `intervalinf/intervalinf/core/domain.py`
- `intervalinf/intervalinf/operators/sola.py`
- relevant tests added in Phase 2

**Validation:**
- No behavior regressions in baseline tests.
- Updated benchmarks still reproduce the Phase 2 baseline within tolerance, except where
  the cleanup intentionally fixes broken or inconsistent behavior.

**Outcome:**
- Resolved the `IntegrationConfig(method='quad')` / `IntervalDomain.integrate` naming
  mismatch: both `'quad'` and `'adaptive'` are now accepted everywhere.  `'quad'` is the
  legacy alias; `'adaptive'` is the canonical name.
- Added `FIXED_GRID_METHODS` and `ADAPTIVE_METHODS` module-level frozensets to
  `config.py`; added `is_fixed_grid` and `is_adaptive` properties on `IntegrationConfig`;
  added `IntegrationConfig.adaptive_quad()` classmethod preset.
- Eliminated the unnecessary `Function` wrapper allocation per kernel in
  `SOLAOperator._apply_kernels`; now calls `domain.integrate()` directly.
- Added support-propagation in `SOLAOperator._apply_kernels` and `compute_gram_matrix`:
  when both the input function and the kernel carry compact-support metadata the
  integration range is narrowed to the support intersection, and disjoint supports
  return exactly 0 without evaluating the integrand.
- Added 12 new tests covering `'quad'`/`'adaptive'` aliasing, `is_fixed_grid` /
  `is_adaptive` properties, `FIXED_GRID_METHODS` / `ADAPTIVE_METHODS` constants,
  `adaptive_quad()` preset, disjoint-support early return, and support-propagation
  correctness.  Total: 60 SOLA tests / 362 across the test suite, all passing.
- Updated the living reference to document module-level constants, new config
  properties, the `'quad'` alias, and Phase 3 SOLAOperator behavioral changes.

---

## Phase 4: Automatic batched fixed-grid forward path

**Status:** ✅ Complete (2026-03-08)

**Objective:** Introduce an automatic accelerated path for reusable fixed-grid integration
methods such as `simpson` and `trapz`, while preserving a generic fallback path for cases
that do not fit the accelerated assumptions.

**Key design requirement:**
- The accelerated path must be automatic whenever the operator is using a fixed-grid
  integration mode whose mesh/settings are stable and reusable.
- The public semantics must still be “continuous operator evaluated numerically”, not
  “user-visible matrix discretization”.

**Implementation ideas to evaluate and, if justified, implement:**
- Build the quadrature mesh once per operator/configuration instead of per kernel.
- Evaluate the input function once on the mesh instead of once per kernel.
- Evaluate kernels on the shared mesh and integrate all products in a batched/vectorized way.
- Keep exact fallback behavior for non-vectorized or unusual callables.

**Files likely affected:**
- `intervalinf/intervalinf/operators/sola.py`
- possibly small helpers in `intervalinf/core/functions.py` if needed for safe vectorized evaluation

**Validation:**
- Forward results match the Phase 2 baseline under the same quadrature rule.
- No regression in provider-backed cases or in non-vectorized fallback cases.
- Benchmarks show isolated operator-level improvements before moving on to deeper caching.

**Outcome:**
- Added `_eval_on_mesh` static helper: tries vectorised `Function.evaluate(xs)`, falls
  back to per-point evaluation on shape-mismatch or exception, and preserves dtype.
- Added `_apply_kernels_fixed_grid`: mesh built once, f evaluated once, full-domain
  kernels batched into a shared matrix, single `scipy.integrate.simpson`/`trapezoid`
  call for the batched subset.
- Added `_apply_kernels_generic`: renamed original per-kernel loop for adaptive methods.
- `_apply_kernels` dispatches automatically via `is_fixed_grid`.
- Disjoint-support early exit preserved; support-restricted kernels fall back to the
  Phase 3 generic path so narrowed-support quadrature semantics are preserved.
- Fixed-grid `IntervalDomain.integrate` now preserves complex dtype for `simpson` and
  `trapz`, including scalar fallback evaluation paths used by support-restricted kernels.
- 20 new tests added in `TestFastPathFixedGrid`; focused SOLA suite now has 80 passing tests.
- New benchmark: `intervalinf/rough_work/benchmark_phase4.py`.
- Measured speedup over `_apply_kernels_generic`: about 2-5x for N_d 5-200.
- Updated living reference with new method table, dispatch logic, and speedup figures.

---

## Phase 5: Reuse and caching for repeated workloads

**Status:** ✅ Complete (2026-03-08)

**Objective:** Exploit the fact that iterative inverse problems reuse the same operator,
kernels, and integration settings many times.

**Optimization targets:**
- Reuse fixed-grid mesh and weights across repeated applications.
- Cache kernel evaluations on the shared mesh when safe.
- Use support intersection to reduce the effective integration domain when compact support
  information is available and mathematically safe to use.
- Investigate whether a coefficient-space fast path is worthwhile for `Lebesgue` functions
  represented directly in the active basis.

**Important constraints:**
- Cache design must not change semantics when providers are the kernel source of truth.
- Cache invalidation/refresh rules must be explicit.
- Memory growth must be measured and justified.
- Any coefficient-based fast path must preserve fallback behavior for callable-based functions.

**Benchmark focus:**
- repeated `G(f)` calls with stable settings
- repeated downstream calls inside PLI/DLI workflows
- compact-support versus global-support kernel families

**Outcome:**
- `_shared_mesh: Optional[np.ndarray]` — persistent mesh, built once via `np.linspace(a, b, n_points)` on
  first `_apply_kernels_fixed_grid` call. Never cleared; depends only on immutable constructor params.
- `_kernel_eval_cache: Optional[dict]` — maps kernel index → `ndarray(n_points,)`. Active only when
  `cache_kernels=True`. Populated for kernels taking the batched path; skipped for the generic fallback.
  Cache semantics: a kernel is NOT cached iff BOTH the input function and that kernel have overlapping
  compact-support metadata (the `_intersect_supports` generic path). Disjoint: skipped entirely.
- `_get_or_build_mesh()` — private lazy mesh builder.
- `clear_mesh_cache()` — new public method; clears `_kernel_eval_cache` only (`_shared_mesh` preserved).
- `clear_cache()` — now also calls `clear_mesh_cache()` so all caches are invalidated together.
- `get_cache_info()` — new keys: `shared_mesh_built`, `kernel_eval_cache_entries`.
- Memory footprint: N_d × 8 KB at n_points=1000 (e.g. 200 × 8 KB = 1.6 MB). Justified for workloads
  with hundreds of repeated forward calls.
- 23 new tests added in `TestPhase5ReuseAndCaching`; SOLA suite is now 103 tests.
- New benchmark: `intervalinf/rough_work/benchmark_phase5.py`.
- Measured speedup for repeated workloads (N_REPS=50, n_points=1000): 1.6–3.4x across N_d 5–200.
- Deferred: coefficient-space fast path for Lebesgue functions (scope for Phase 6 or later).

---

## Phase 6: End-to-end validation, documentation, and comparison report

**Status:** ⬜ Not started

**Objective:** Verify the final optimized implementation against the original baseline,
document what improved and where, and update project references so future work has an
accurate map of the operator.

**Required work:**
- Re-run the full benchmark matrix created in Phase 2.
- Compare speed, accuracy, adjoint consistency, and robustness against the baseline.
- Summarize where speedups are substantial, where they are modest, and where the fallback
  path is still required.
- Update living references to reflect the new implementation structure and semantics.

**Files to update:**
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
- any benchmark summary artifacts retained under `rough_work/`
- plan change log and phase completion notes

---

## Success criteria

- `SOLAOperator` remains continuous-first in public semantics.
- A strong dedicated SOLA test suite exists where there was previously little direct coverage.
- Fixed-grid repeated workloads are measurably faster than baseline.
- Accuracy and adjoint consistency remain within agreed tolerances.
- The integration-method API is internally consistent and documented.
- Benchmarking is detailed enough to justify design choices and quantify gains.

---

## Change log

| Date | Phase | Action |
|------|-------|--------|
| 2026-03-08 | Planning | Created `SOLAOperator_speedup` from local `convex_analysis` in the nested `intervalinf` repo |
| 2026-03-08 | Planning | Completed initial research on `SOLAOperator`, function/integration internals, current test coverage, and DLI/PLI benchmark surfaces |
| 2026-03-08 | Planning | Recorded project decisions: keep benchmark artifacts in `rough_work/`, make fixed-grid acceleration automatic, and include integration-method cleanup in scope |
| 2026-03-08 | Phase 1 | Completed implementation and dependency audit; documented exact forward/adjoint call graph, preserved invariants, current inefficiencies, constraints, and recommended Phase 2 benchmark surfaces |
| 2026-03-08 | Phase 2 | Added dedicated SOLA baseline tests, a rough-work benchmark harness, and living-reference updates; review approved after minor cleanup |
| 2026-03-08 | Phase 3 | Resolved `quad`/`adaptive` naming mismatch; added `is_fixed_grid`/`is_adaptive` properties and module-level method-set constants; eliminated `Function` wrapper allocation in `_apply_kernels`; added support-propagation in `_apply_kernels` and `compute_gram_matrix`; 12 new tests added (60 SOLA / 362 total) |
| 2026-03-08 | Phase 4 | Added `_eval_on_mesh`, `_apply_kernels_fixed_grid`, `_apply_kernels_generic`; automatic dispatch in `_apply_kernels`; support-aware generic fallback for narrowed compact supports; fixed-grid complex dtype preservation in `SOLAOperator` and `IntervalDomain.integrate`; non-vectorised callable fallback; 20 new tests (80 SOLA passing); new benchmark `benchmark_phase4.py`; about 2–5x speedup over per-kernel loop |
| 2026-03-08 | Phase 5 | Added `_shared_mesh`, `_kernel_eval_cache`, `_get_or_build_mesh()`; updated `_apply_kernels_fixed_grid` batched loop to cache/reuse kernel evals; added `clear_mesh_cache()`; updated `clear_cache()` and `get_cache_info()`; 23 new tests in `TestPhase5ReuseAndCaching` (103 SOLA passing); new benchmark `benchmark_phase5.py`; 1.6–3.4x speedup for repeated workloads |