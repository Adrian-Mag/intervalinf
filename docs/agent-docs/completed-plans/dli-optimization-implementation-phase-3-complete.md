## Phase 3 Complete: Optimize DualMasterCostFunction Oracle Path

Optimized the dual-master oracle in `pygeoinf` by caching the adjoint operator once and switching the fast path to the fused `value_and_support_point` API. The redundant scalar support-function evaluations are gone from `value_and_subgradient`, the Phase 1 guardrails now pass, and the fallback path remains covered and instrumented.

**Files created/changed:**
- `pygeoinf/pygeoinf/backus_gilbert.py`
- `pygeoinf/tests/test_dual_master_cost.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
- `intervalinf/docs/agent-docs/references/living/dli-optimization-roadmap.md`

**Functions created/changed:**
- `DualMasterCostFunction.__init__`
- `DualMasterCostFunction._mapping`
- `DualMasterCostFunction._subgradient`
- `DualMasterCostFunction.value_and_subgradient`
- `test_no_redundant_scalar_eval_when_support_point_exists`
- `test_no_repeated_adjoint_fetch_across_oracle_calls`
- `test_fallback_branch_correctness_and_instrumentation`

**Tests created/changed:**
- `test_no_redundant_scalar_eval_when_support_point_exists`
- `test_no_repeated_adjoint_fetch_across_oracle_calls`
- `test_fallback_branch_correctness_and_instrumentation`
- `test_value_and_subgradient_consistency`

**Review Status:** APPROVED

**Git Commit Message:**
feat(oracle): fuse dual-master support evaluation path

- Cache the adjoint operator in DualMasterCostFunction
- Use value_and_support_point in the oracle fast path
- Restore coherent fallback timing and add fallback regression coverage
- Make the Phase 1 oracle guardrail tests pass
- Update pygeoinf and roadmap references for fused timing semantics

Plan: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-plan.md
Phase: 3 of 5
Related: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-phase-3-complete.md