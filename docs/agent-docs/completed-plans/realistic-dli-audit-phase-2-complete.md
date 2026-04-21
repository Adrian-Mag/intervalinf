## Phase 2 Complete: Instrument the dual solve behavior

Added a live `run_phase2_instrumentation()` function that reconstructs the realistic DLI problem from scratch and runs the proximal bundle solve per property direction, recording per-direction λ* norms and reduction factors. The primary finding is that the data is data-constraining but highly asymmetric: property p_0 (‖λ*‖₂ ≈ 0.001, reduction ≈ 1.03×) is nearly prior-dominated while p_1 (‖λ*‖₂ ≈ 0.02–0.03, reduction ≈ 1.70–1.96×) is genuinely constrained.

**Files created/changed:**
- intervalinf/demos/convex_analysis/realistic_dli_audit.py
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- run_phase2_instrumentation (new Phase 2 section in realistic_dli_audit.py)

**Tests created/changed:**
- Phase 2 self-check assertion: posterior ≤ prior for all directions (passed)
- Script end-to-end run with real kernel catalog (passed, all 4 directions converge)

**Key findings:**
- Direction +e_0: ‖λ*‖₂ = 0.0013, reduction = 1.031× (nearly prior-dominated)
- Direction +e_1: ‖λ*‖₂ = 0.0218, reduction = 1.700× (genuinely constrained)
- Direction -e_0: ‖λ*‖₂ = 0.0015, reduction = 1.030×
- Direction -e_1: ‖λ*‖₂ = 0.0302, reduction = 1.961×
- Mean reduction: 1.431× (matches notebook baseline)
- Catalog used: True (real kernels, not synthetic fallback)
- All 4 directions converge

**Review Status:** APPROVED (with minor recommendations addressed — catalog robustness try/except added)

**Git Commit Message:**
feat(convex-analysis): add phase 2 instrumented dual solve to dli audit

- Add run_phase2_instrumentation() with live problem reconstruction
- Record per-direction λ* norms, iters, convergence, reduction factors
- Add catalog fallback guard (try/except on real-kernel branch)
- Update living reference for phase 2 diagnostics

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-plan.md
Phase: 2 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-phase-2-complete.md
