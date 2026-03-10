## Phase 4 Complete: Automatic batched fixed-grid forward path

Added the first real operator-level speedup for `SOLAOperator` by batching the
fixed-grid forward path for `simpson` and `trapz` while preserving the generic
path for adaptive integration and support-restricted kernels. This phase keeps
the continuous-first public semantics intact, preserves Phase 3 compact-support
behavior, and delivers a measured operator-level improvement on the baseline
benchmark.

**Files created/changed:**
- `intervalinf/core/domain.py`
- `intervalinf/operators/sola.py`
- `tests/operators/test_sola.py`
- `rough_work/benchmark_phase4.py`
- `docs/agent-docs/references/living/intervalinf-reference.md`
- `docs/agent-docs/completed-plans/sola-operator-speedup-plan.md`
- `docs/agent-docs/completed-plans/sola-operator-speedup-phase-4-complete.md`

**Functions created/changed:**
- `IntervalDomain.integrate`
- `SOLAOperator._apply_kernels`
- `SOLAOperator._eval_on_mesh`
- `SOLAOperator._apply_kernels_fixed_grid`
- `SOLAOperator._apply_kernels_generic`

**Tests created/changed:**
- Updated: `tests/operators/test_sola.py`
  - added fixed-grid fast-path analytic correctness tests
  - added non-vectorized callable fallback tests
  - added narrow-support regression coverage against the generic Phase 3 path
  - added complex-valued end-to-end forward tests for batched and
    support-restricted fixed-grid paths
  - added support-restricted non-vectorized complex-kernel regression coverage
- Validation run: `tests/operators/test_sola.py` -> `80 passed`
- Benchmark validation: `rough_work/benchmark_phase4.py` shows about `2x` to
  `5x` speedup over the generic per-kernel fixed-grid path depending on `N_d`

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(sola): batch fixed-grid forward evaluation

- Add an automatic batched forward path for simpson and trapz SOLA evaluations
- Preserve support-restricted semantics with generic fallback and keep complex fixed-grid outputs intact
- Add Phase 4 regression tests and a benchmark comparing batched and generic forward paths

Plan: intervalinf/docs/agent-docs/completed-plans/sola-operator-speedup-plan.md
Phase: 4 of 6
Related: intervalinf/docs/agent-docs/completed-plans/sola-operator-speedup-phase-4-complete.md