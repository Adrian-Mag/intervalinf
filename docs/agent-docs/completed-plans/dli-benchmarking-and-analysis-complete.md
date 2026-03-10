## Plan Complete: DLI Benchmarking And Analysis

This plan built a full benchmarking workflow for intervalinf-backed DLI, traced the dominant runtime costs through the dual oracle and proximal-bundle solver, quantified the compact-support fallback penalty in `SOLAOperator`, and converted the measurements into a concrete optimization roadmap. The resulting artifacts establish that the oracle dominates solve time, identify duplicated model-support evaluation as the highest-payoff next target, and define measurable acceptance criteria for the follow-on performance work.

**Phases Completed:** 5 of 5
1. ✅ Phase 1: Add Lightweight DLI Instrumentation
2. ✅ Phase 2: Benchmark SOLA Compact-Support Fallbacks
3. ✅ Phase 3: Benchmark DLI Oracle And Proximal-Bundle End-To-End Cost
4. ✅ Phase 4: Analyze Results And Rank Speedup Targets
5. ✅ Phase 5: Produce Optimization Roadmap

**All Files Created/Modified:**
- intervalinf/rough_work/benchmark_dli_instrumentation.py
- intervalinf/intervalinf/operators/sola.py
- intervalinf/rough_work/benchmark_phase2_compact_support.py
- intervalinf/tests/operators/test_sola.py
- intervalinf/rough_work/benchmark_dli_oracle_bundle.py
- intervalinf/docs/agent-docs/references/living/dli-performance-analysis-and-speedup-targets.md
- intervalinf/docs/agent-docs/references/living/dli-optimization-roadmap.md
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md
- pygeoinf/pygeoinf/backus_gilbert.py
- pygeoinf/pygeoinf/convex_optimisation.py
- pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md

**Key Functions/Classes Added:**
- DualMasterStats
- DualMasterCostFunction.instrumentation_stats
- DualMasterCostFunction.reset_instrumentation
- ProximalBundleStats
- ProximalBundleMethod.instrumentation_stats
- SOLAOperator.stats
- SOLAOperator.reset_stats
- bench_oracle
- bench_bundle
- bench_warmstart

**Test Coverage:**
- Total tests written: benchmark harness assertions across 3 benchmark scripts plus SOLA instrumentation coverage
- All tests passing: ✅

**Recommendations for Next Steps:**
- Implement the oracle cleanup from the roadmap first: eliminate duplicated support-value evaluation and cache the adjoint operator object.
- Follow with the support-aware batched mesh path for compact-support SOLA kernels.
- Re-run the Phase 2 and Phase 3 benchmark harnesses after each optimization workstream to confirm the acceptance thresholds from the roadmap.