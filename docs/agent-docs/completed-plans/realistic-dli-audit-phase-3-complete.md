## Phase 3 Complete: Run controlled experiments

Added a controlled `N_d` sweep to the standalone audit and verified that the suspicious behavior is not a generic solver defect. With the real kernel catalog and all other assumptions fixed, the realistic formulation is data-constraining at `N_d = 5, 10` but becomes fully prior-dominated for `N_d >= 20`, showing that the extra modes being added in this setup do not tighten the two selected bump-function property bounds.

**Files created/changed:**
- intervalinf/demos/convex_analysis/realistic_dli_audit.py
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- run_phase3_nd_sweep

**Tests created/changed:**
- Phase 3 self-check asserting posterior interval bounds stay inside prior bounds for every `N_d` and property
- End-to-end audit script run covering Phases 1–3 with real kernel catalog
- Error check on audit script and living reference (clean)

**Key findings:**
- Sweep values: `N_d ∈ {5, 10, 20, 30, 40, 50}` with `N_p = 2` fixed
- `N_d = 5`: `p0` reduction `1.030×`, `p1` reduction `1.821×`, mean `1.426×`, max `‖λ*‖ = 0.0302`
- `N_d = 10`: `p0` reduction `1.048×`, `p1` reduction `1.899×`, mean `1.474×`, max `‖λ*‖ = 0.0078`
- `N_d >= 20`: both properties reduction `1.000×`, max `‖λ*‖ = 0.0000` (fully prior-dominated)
- Runtime is non-monotone because the prior-dominated cases converge in very few bundle iterations
- Kernel source: real catalog

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(convex-analysis): add controlled N_d sweep to dli audit

- Add Phase 3 sweep over N_d for the realistic DLI audit
- Report true prior/posterior half-widths and reduction factors
- Show transition to prior-dominated behavior for N_d >= 20
- Update living reference with Phase 3 findings

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-plan.md
Phase: 3 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-phase-3-complete.md
