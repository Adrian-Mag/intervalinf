## Phase 1 Complete: Add Oracle Optimization Guardrails

Added two focused guardrail tests that lock in the intended oracle-path efficiency goals before any production code changes. The phase is intentionally test-only, and the new tests currently fail against the existing implementation for the expected reasons: redundant scalar support evaluation and repeated `G.adjoint` property access inside `value_and_subgradient`.

**Files created/changed:**
- `pygeoinf/tests/test_dual_master_cost.py`

**Functions created/changed:**
- `_CountingBallSupportFunction`
- `_AdjointCountingLinearOperator`
- `test_no_redundant_scalar_eval_when_support_point_exists`
- `test_no_repeated_adjoint_fetch_across_oracle_calls`

**Tests created/changed:**
- `test_no_redundant_scalar_eval_when_support_point_exists`
- `test_no_repeated_adjoint_fetch_across_oracle_calls`
- `test_value_and_subgradient_consistency` (unchanged baseline test, re-run during validation)

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
test(oracle): add phase 1 optimization guardrails

- Add failing guardrails for redundant support-value calls
- Add failing guardrail for repeated adjoint access
- Keep Phase 1 scoped to dual-master tests only

Plan: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-plan.md
Phase: 1 of 5
Related: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-phase-1-complete.md