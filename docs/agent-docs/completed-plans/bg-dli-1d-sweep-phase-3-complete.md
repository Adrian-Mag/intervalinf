## Phase 3 Complete: Run Multi-Seed Sweep and Compute Gap Metrics

Phase 3 adds the full multi-seed BG-vs-DLI sweep across all configured `N_d`, forward-operator seeds, and data seeds, computes the per-case gap metrics, and writes deterministic raw/summary CSV outputs. The validation now enforces near-zero containment violation as a hard assertion, and all 36 cases pass.

**Files created/changed:**
- `intervalinf/rough_work/bg_dli_1d_sweep.py`
- `intervalinf/rough_work/bg_dli_1d_sweep_results/bg_dli_raw.csv`
- `intervalinf/rough_work/bg_dli_1d_sweep_results/bg_dli_summary_by_nd.csv`
- `intervalinf/docs/agent-docs/completed-plans/bg-dli-1d-sweep-plan.md`

**Functions created/changed:**
- `_results_dir`
- `compute_gap_metrics`
- `evaluate_case`
- `run_sweep`
- `_raw_csv_columns`
- `_summary_csv_columns`
- `write_raw_csv`
- `summarize_results_by_nd`
- `write_summary_csv`
- `run_phase3_validation`
- `main` (extended to run Phase 3)

**Tests created/changed:**
- row-count validation for `len(N_D_VALUES) * len(FORWARD_SEEDS) * len(DATA_SEEDS) = 36`
- finiteness checks for all key interval and gap metrics on every row
- hard assertion that `max(containment_violation) <= CONTAINMENT_TOL`
- file-existence checks for raw and summary CSV outputs
- end-to-end script execution via `conda run -n inferences3 python intervalinf/rough_work/bg_dli_1d_sweep.py`

**Validation output:**
- Total cases: `36`
- Sweep runtime: `108.5s`
- `containment_violation_max = 0.0` for all `N_d`
- Summary by `N_d`:
  - `N_d=5`: `ratio_mean=1.0199`, `ratio_median=1.0193`, `dli_w_mean=2.39961`, `bg_w_mean=2.44722`
  - `N_d=10`: `ratio_mean=1.0377`, `ratio_median=1.0360`, `dli_w_mean=2.31716`, `bg_w_mean=2.40412`
  - `N_d=20`: `ratio_mean=1.1709`, `ratio_median=1.1366`, `dli_w_mean=2.09726`, `bg_w_mean=2.44778`
  - `N_d=40`: `ratio_mean=1.3788`, `ratio_median=1.4003`, `dli_w_mean=1.32210`, `bg_w_mean=1.78390`
- Interpretation: BG remains wider than DLI in all tested cases, and the gap grows substantially with `N_d` in this setup.

**Review Status:** APPROVED

**Git Commit Message:**
feat(rough_work): add bg dli sweep metrics and csv outputs

- Run the full 36-case nested-seed sweep across N_d values
- Compute BG vs DLI gap metrics and aggregate them by N_d
- Write raw and summary CSV outputs with strict containment validation

Plan: intervalinf/docs/agent-docs/completed-plans/bg-dli-1d-sweep-plan.md
Phase: 3 of 4
Related: intervalinf/docs/agent-docs/completed-plans/bg-dli-1d-sweep-phase-3-complete.md
