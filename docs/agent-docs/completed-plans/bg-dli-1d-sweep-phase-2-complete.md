## Phase 2 Complete: Implement DLI and BG Interval Solvers

Phase 2 adds one-dimensional DLI and BG admissible-interval solvers to the sweep script and validates that both methods return finite intervals with correct sign conventions and nonnegative widths. The implementation uses `DualMasterCostFunction` plus `solve_support_values` for DLI and the notebook-style BG estimator/support-function algebra for BG.

**Files created/changed:**
- `intervalinf/rough_work/bg_dli_1d_sweep.py`
- `intervalinf/docs/agent-docs/active-plans/bg-dli-1d-sweep-plan.md`

**Functions created/changed:**
- `compute_dli_interval`
- `compute_bg_interval`
- `run_phase2_validation`
- `main` (extended to run Phase 2 validation)

**Tests created/changed:**
- finite-endpoint checks for DLI and BG interval endpoints
- nonnegative-width checks for DLI and BG intervals
- sign-convention validation that lower bound is computed as `-h(-e1)`
- containment sanity check for true property `p_bar` in both intervals
- end-to-end script execution via `conda run -n inferences3 python intervalinf/rough_work/bg_dli_1d_sweep.py`

**Validation output:**
- Problem: `N_d=5`, `forward_seed=2`, `data_seed=42`
- True property: `p_bar = 1.120511`
- DLI interval: `[-0.699726, +1.702132]`, width `2.401859`, `21` iterations, `0.41s`, converged `(True, True)`
- BG interval: `[-0.726593, +1.735874]`, width `2.462467`, `1.162s`, `alpha_bg = 5.0994e-02`
- Width ratio: `BG/DLI = 1.0252`
- Result: all Phase 2 assertions passed

**Review Status:** APPROVED

**Git Commit Message:**
feat(rough_work): add 1d dli and bg interval solvers

- Implement DLI interval computation with DualMasterCostFunction
- Add BG interval construction from estimator support algebra
- Validate finite bounds, sign conventions, and width sanity

Plan: intervalinf/docs/agent-docs/active-plans/bg-dli-1d-sweep-plan.md
Phase: 2 of 4
Related: intervalinf/docs/agent-docs/active-plans/bg-dli-1d-sweep-phase-2-complete.md
