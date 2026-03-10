## Phase 2 Complete: Benchmark SOLA Compact-Support Fallbacks

Phase 2 added passive SOLA forward-path instrumentation and a focused compact-support benchmark harness that isolates the fixed-grid fallback penalty. The benchmark was run successfully and showed the overlapping compact-support fallback is substantially slower than the batched full-domain path while remaining numerically accurate against an adaptive reference.

**Files created/changed:**
- intervalinf/intervalinf/operators/sola.py
- intervalinf/rough_work/benchmark_phase2_compact_support.py
- intervalinf/tests/operators/test_sola.py
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- SOLAOperator.stats
- SOLAOperator.reset_stats
- SOLAOperator._apply_kernels
- SOLAOperator._apply_kernels_fixed_grid
- SOLAOperator._apply_kernels_generic
- build_full_domain
- build_disjoint_support
- build_overlapping_fallback
- build_mixed_paths
- run_benchmark
- test_instrumentation_and_reset_smoke

**Tests created/changed:**
- TestPhase2Instrumentation.test_instrumentation_and_reset_smoke
- intervalinf/rough_work/benchmark_phase2_compact_support.py built-in sanity and accuracy assertions

**Review Status:** APPROVED

**Git Commit Message:**
feat(sola): benchmark compact-support fallback routing

- add passive SOLA forward-path counters for batched, fallback, and disjoint cases
- add a focused compact-support benchmark harness with routing and accuracy checks
- extend SOLA coverage and update the living reference for Phase 2 tooling

Plan: intervalinf/docs/agent-docs/active-plans/dli-benchmarking-and-analysis-plan.md
Phase: 2 of 5
Related: intervalinf/docs/agent-docs/active-plans/dli-benchmarking-and-analysis-phase-2-complete.md