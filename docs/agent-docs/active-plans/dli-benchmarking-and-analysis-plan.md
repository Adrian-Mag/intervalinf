## Plan: DLI Benchmarking And Analysis

Build a benchmarking and analysis workflow for intervalinf-backed DLI that localizes runtime across the dual oracle, `SOLAOperator`, adjoint reconstruction, and proximal-bundle master solves. The goal is to measure where the remaining cost sits after the recent SOLA improvements, explicitly quantify the compact-support fallback penalty, and produce an evidence-based optimization roadmap before any major refactor.

**Phases 5**
1. **Phase 1: Add Lightweight DLI Instrumentation**
    - **Objective:** Measure where time is spent inside intervalinf-backed proximal-bundle DLI without changing solver behavior.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/backus_gilbert.py`, `pygeoinf/pygeoinf/convex_optimisation.py`, optional rough-work harness under `intervalinf/rough_work/`
    - **Tests to Write:** Small regression checks that instrumentation leaves numerical results unchanged and that timing/counter outputs populate on a minimal DLI case.
    - **Steps:**
        1. Add timers/counters around `DualMasterCostFunction.value_and_subgradient`, support-point/value work, and proximal-bundle master solves.
        2. Record iteration counts, QP backend, cut counts, and support-point availability/fallback information.
        3. Run a small failing/then passing validation that instrumentation collects non-empty measurements.

2. **Phase 2: Benchmark SOLA Compact-Support Fallbacks**
    - **Objective:** Quantify the cost of losing the fixed-grid cached path when overlapping compact-support metadata forces support-aware fallback.
    - **Files/Functions to Modify/Create:** `intervalinf/rough_work/benchmark_phase6_comparison.py` or a new compact-support benchmark script in `intervalinf/rough_work/`, optional helper counters in `intervalinf/intervalinf/operators/sola.py`
    - **Tests to Write:** Scenario checks proving the benchmark covers full-domain, disjoint-support, and overlapping-support cases with the intended support metadata.
    - **Steps:**
        1. Benchmark `G(f)` for kernel-support on/off and input-support on/off cases.
        2. Sweep `N_d`, integration resolution, support widths, and warm/cold cache states.
        3. Record wall time, cache coverage, fallback counts, disjoint-skip counts, and adaptive-reference accuracy.

3. **Phase 3: Benchmark DLI Oracle And Proximal-Bundle End-To-End Cost**
    - **Objective:** Separate operator cost from dual-oracle cost and full proximal-bundle solve cost on realistic intervalinf DLI problems.
    - **Files/Functions to Modify/Create:** `intervalinf/rough_work/benchmark_dli_solvers.py`, `intervalinf/rough_work/bg_dli_1d_sweep.py`, or a new oracle-focused benchmark harness
    - **Tests to Write:** Reproducibility checks across fixed seeds and summary checks that timing breakdowns are internally consistent.
    - **Steps:**
        1. Benchmark oracle-only calls to `DualMasterCostFunction.value_and_subgradient` on nested `N_d` cases.
        2. Benchmark full proximal-bundle solves only; no alternative solver comparison in this plan.
        3. Test warm-start behavior across nested `N_d` by padding smaller dual variables into larger problems.

4. **Phase 4: Analyze Results And Rank Speedup Targets**
    - **Objective:** Convert benchmark evidence into a ranked list of realistic optimization opportunities in `pygeoinf` and `intervalinf`.
    - **Files/Functions to Modify/Create:** Analysis report under `intervalinf/docs/agent-docs/references/` and, if useful, a companion note under `pygeoinf/docs/agent-docs/references/`
    - **Tests to Write:** None beyond benchmark reproducibility and sanity checks on reported tables.
    - **Steps:**
        1. Compare compact-support fallback cost against adjoint reconstruction and master-QP overhead.
        2. Rank low-risk candidates such as removing duplicated support evaluations, caching adjoint operators, reducing dense master-matrix allocations, and improving warm-start usage.
        3. Rank higher-risk candidates such as support-aware batched meshes, vectorized adjoint reconstruction, and persistent QP-solver state reuse.

5. **Phase 5: Produce Optimization Roadmap**
    - **Objective:** Deliver a concrete next-step implementation roadmap grounded in measured performance results.
    - **Files/Functions to Modify/Create:** Final benchmark summary and roadmap document under `intervalinf/docs/agent-docs/references/`
    - **Tests to Write:** None.
    - **Steps:**
        1. Select the most promising 2-3 optimization targets by payoff versus implementation risk.
        2. Define acceptance metrics in terms of oracle time, total solve time, and preserved numerical agreement.
        3. Prepare a follow-on implementation plan for the chosen optimizations.

**Open Questions 0**