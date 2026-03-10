## Phase 1 Complete: Add Lightweight DLI Instrumentation

Phase 1 added passive timing and counter instrumentation to the dual master oracle and proximal bundle solver, plus a small intervalinf-backed validation harness. The instrumentation was verified to populate consistently without changing DLI solve behavior, and the living references were updated to document the new API surface.

**Files created/changed:**
- pygeoinf/pygeoinf/backus_gilbert.py
- pygeoinf/pygeoinf/convex_optimisation.py
- intervalinf/rough_work/benchmark_dli_instrumentation.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- DualMasterStats
- DualMasterCostFunction.instrumentation_stats
- DualMasterCostFunction.reset_instrumentation
- DualMasterCostFunction.value_and_subgradient
- ProximalBundleStats
- ProximalBundleMethod.instrumentation_stats
- ProximalBundleMethod.solve
- _get_value_and_subgradient
- build_small_problem
- run_instrumented_dli
- assert_instrumentation_populated

**Tests created/changed:**
- intervalinf/rough_work/benchmark_dli_instrumentation.py
- assert_instrumentation_populated

**Review Status:** APPROVED

**Git Commit Message:**
feat(dli): add passive proximal-bundle instrumentation

- instrument DualMasterCostFunction and ProximalBundleMethod timing/counters
- add intervalinf rough-work harness validating instrumentation consistency
- update living references for the new instrumentation API and benchmark

Plan: intervalinf/docs/agent-docs/active-plans/dli-benchmarking-and-analysis-plan.md
Phase: 1 of 5
Related: intervalinf/docs/agent-docs/active-plans/dli-benchmarking-and-analysis-phase-1-complete.md