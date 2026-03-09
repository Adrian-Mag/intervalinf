## Phase 2 Complete: Baseline correctness, robustness, and benchmark suite

Established the pre-optimization baseline for `SOLAOperator` by adding a dedicated
test module and a rough-work benchmark harness. This phase now gives the project a
repeatable correctness baseline and a quantitative runtime baseline for forward,
adjoint, and `DualMasterCostFunction.value_and_subgradient(...)` workloads.

**Files created/changed:**
- `tests/operators/test_sola.py`
- `rough_work/benchmark_sola_baseline.py`
- `docs/agent-docs/references/living/intervalinf-reference.md`
- `docs/agent-docs/active-plans/sola-operator-speedup-plan.md`
- `docs/agent-docs/completed-plans/sola-operator-speedup-phase-2-complete.md`

**Functions created/changed:**
- No production functions changed in this phase.

**Tests created/changed:**
- New: `tests/operators/test_sola.py`
  - baseline forward-integral correctness
  - linearity checks
  - adjoint-consistency checks
  - provider-backed kernel coverage
  - callable-list and Function-list kernel coverage
  - caching behavior and cache accessors
  - integration-method baseline behavior, including current `quad` mismatch
  - compact-support baseline behavior
  - Gram-matrix basics
  - direct-sum smoke coverage

**Review Status:** APPROVED

**Git Commit Message:**
test(sola): add SOLAOperator baseline tests and benchmarks

- Add dedicated SOLAOperator correctness and robustness test coverage
- Add rough-work baseline benchmark for forward, adjoint, and dual-master hotspot timings
- Update living reference with SOLA test coverage, benchmark harness, and known quirks

Plan: intervalinf/docs/agent-docs/active-plans/sola-operator-speedup-plan.md
Phase: 2 of 6
Related: intervalinf/docs/agent-docs/completed-plans/sola-operator-speedup-phase-2-complete.md