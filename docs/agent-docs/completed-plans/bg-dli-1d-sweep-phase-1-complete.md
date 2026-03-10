## Phase 1 Complete: Create Script Scaffold and Shared Builders

Phase 1 establishes the new 1D BG-vs-DLI sweep script scaffold in `intervalinf/rough_work/`, with reusable builders for shared spaces/operators, synthetic data, and deterministic prior/confidence sets. The script validates the `N_p = 1` design, nested forward-operator construction across `N_d`, and reproducible separation of forward and data seeds.

**Files created/changed:**
- `intervalinf/rough_work/bg_dli_1d_sweep.py`
- `intervalinf/docs/agent-docs/completed-plans/bg-dli-1d-sweep-plan.md`

**Functions created/changed:**
- `build_integration_configs`
- `build_spaces_and_operators`
- `build_true_model_and_data`
- `_scalar_op`
- `build_prior_and_confidence`
- `build_problem`
- `run_phase1_validation`
- `main`

**Tests created/changed:**
- `P.dim == 1` validation for all configured `N_d`
- `D.dim == N_d` validation for all configured `N_d`
- shape validation for `d_bar`, `d_tilde`, and `p_bar`
- positivity/finiteness checks for `sigma_d`, `model_radius`, `r_data_conf`, and `s_confidence`
- nested-kernel prefix check comparing `N_d=5` and `N_d=10` under the same `forward_seed`
- independent data-seed check asserting identical `d_bar` and different `d_tilde`
- end-to-end script execution via `conda run -n inferences3 python intervalinf/rough_work/bg_dli_1d_sweep.py`

**Review Status:** APPROVED

**Git Commit Message:**
feat(rough_work): scaffold 1d bg dli sweep script

- Add Phase 1 script structure for the 1D BG vs DLI sweep
- Build shared operators, synthetic data, and confidence-set helpers
- Validate nested forward kernels and independent noise seeds

Plan: intervalinf/docs/agent-docs/completed-plans/bg-dli-1d-sweep-plan.md
Phase: 1 of 4
Related: intervalinf/docs/agent-docs/completed-plans/bg-dli-1d-sweep-phase-1-complete.md
