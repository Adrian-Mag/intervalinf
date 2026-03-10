## Phase 5 Complete: Reuse and caching for repeated workloads

Added repeated-workload reuse to `SOLAOperator` by persisting the fixed-grid
mesh and caching full-domain kernel evaluations on that mesh when caching is
enabled. This phase improves repeated `G(f)` workloads without changing the
continuous-first public API, and preserves the generic fallback for
support-restricted kernels where narrowed-support semantics matter.

**Files created/changed:**
- `intervalinf/operators/sola.py`
- `tests/operators/test_sola.py`
- `rough_work/benchmark_phase5.py`
- `docs/agent-docs/references/living/intervalinf-reference.md`
- `docs/agent-docs/completed-plans/sola-operator-speedup-plan.md`
- `docs/agent-docs/completed-plans/sola-operator-speedup-phase-5-complete.md`

**Functions created/changed:**
- `SOLAOperator._get_or_build_mesh`
- `SOLAOperator._apply_kernels_fixed_grid`
- `SOLAOperator.clear_mesh_cache`
- `SOLAOperator.clear_cache`
- `SOLAOperator.get_cache_info`

**Tests created/changed:**
- Updated: `tests/operators/test_sola.py`
  - added shared-mesh lifecycle tests
  - added kernel-evaluation cache population and invalidation tests
  - added provider-backed caching correctness tests
  - added support-restricted cache bypass tests
  - added repeated-workload correctness checks
- Validation run: `tests/operators/test_sola.py` -> `103 passed`
- Benchmark validation: `rough_work/benchmark_phase5.py` shows about `1.6x`
  to `3.6x` speedup over the uncached repeated-workload path on the current
  run, depending on `N_d`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(sola): reuse fixed-grid work across repeated calls

- Reuse the shared fixed-grid mesh and cache full-domain kernel evaluations when caching is enabled
- Preserve support-restricted fallback semantics and expose explicit cache invalidation helpers
- Add Phase 5 regression tests and a repeated-workload benchmark for warm versus cold paths

Plan: intervalinf/docs/agent-docs/completed-plans/sola-operator-speedup-plan.md
Phase: 5 of 6
Related: intervalinf/docs/agent-docs/completed-plans/sola-operator-speedup-phase-5-complete.md