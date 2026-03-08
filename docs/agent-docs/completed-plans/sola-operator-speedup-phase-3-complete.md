## Phase 3 Complete: Integration API cleanup and low-risk semantic fixes

Resolved the integration-method naming mismatch around `quad` versus `adaptive`
and removed the main low-risk inefficiencies identified in the Phase 1 audit.
This phase keeps `SOLAOperator` continuous-first while making adaptive and
fixed-grid behavior explicit for later acceleration phases.

**Files created/changed:**
- `intervalinf/core/config.py`
- `intervalinf/core/domain.py`
- `intervalinf/operators/sola.py`
- `tests/operators/test_sola.py`
- `docs/agent-docs/references/living/intervalinf-reference.md`
- `docs/agent-docs/active-plans/sola-operator-speedup-plan.md`
- `docs/agent-docs/completed-plans/sola-operator-speedup-phase-3-complete.md`

**Functions created/changed:**
- `IntegrationConfig.is_fixed_grid`
- `IntegrationConfig.is_adaptive`
- `IntegrationConfig.adaptive_quad`
- `IntervalDomain.integrate`
- `SOLAOperator._apply_kernels`
- `SOLAOperator.compute_gram_matrix`

**Tests created/changed:**
- Updated: `tests/operators/test_sola.py`
  - replaced the old failing-`quad` baseline with working `quad`/`adaptive`
    coverage
  - added `IntegrationConfig` property tests for fixed-grid versus adaptive
    classification
  - added compact-support propagation checks and disjoint-support exact-zero
    coverage
- Validation run: `tests/operators/test_sola.py` -> `60 passed`
- Benchmark smoke validation: first two baseline scenarios ran successfully in
  the `inferences3` environment

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
fix(sola): clean up integration semantics before batching

- Resolve the `quad` versus `adaptive` naming mismatch across config and domain integration
- Add fixed-grid versus adaptive integration metadata and preserve low-risk SOLA support propagation
- Update SOLA tests and living references for the Phase 3 behavior contract

Plan: intervalinf/docs/agent-docs/active-plans/sola-operator-speedup-plan.md
Phase: 3 of 6
Related: intervalinf/docs/agent-docs/completed-plans/sola-operator-speedup-phase-3-complete.md