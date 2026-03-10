## Plan Complete: BG vs DLI 1D Sweep Script

A standalone Python script comparing one-dimensional BG and DLI admissible intervals was built and validated across 36 cases (N_d ∈ {5,10,20,40} × 3 forward seeds × 3 data seeds). The script confirms that DLI intervals are strictly tighter than BG (width ratio grows from ~1.02 at N_d=5 to ~1.38 at N_d=40), with zero containment violations across all cases. CSV outputs, PNG/PDF plots, and a textual summary are generated automatically.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Script Scaffold and Shared Builders
2. ✅ Phase 2: DLI and BG Interval Solvers
3. ✅ Phase 3: Multi-Seed Sweep, Gap Metrics, and CSV Outputs
4. ✅ Phase 4: Plots, CLI Polish, and Reference Update

**All Files Created/Modified:**
- `intervalinf/rough_work/bg_dli_1d_sweep.py` (main script, all phases)
- `intervalinf/rough_work/bg_dli_1d_sweep_results/bg_dli_raw.csv` (36 rows × 20 columns)
- `intervalinf/rough_work/bg_dli_1d_sweep_results/bg_dli_summary_by_nd.csv` (4 rows × 12 columns)
- `intervalinf/rough_work/bg_dli_1d_sweep_figures/width_ratio_vs_nd.png`
- `intervalinf/rough_work/bg_dli_1d_sweep_figures/width_ratio_vs_nd.pdf`
- `intervalinf/rough_work/bg_dli_1d_sweep_figures/widths_vs_nd.png`
- `intervalinf/rough_work/bg_dli_1d_sweep_figures/widths_vs_nd.pdf`
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` (updated with script entry)

**Key Functions/Classes Added:**
- `build_integration_configs()` — shared Lebesgue/SOLA/parallel configs
- `build_spaces_and_operators(N_d, forward_seed, ...)` — M, D, P, G, T builders (nested kernels)
- `build_true_model_and_data(M, G, T, N_d, ...)` — synthetic data + noise draw
- `build_prior_and_confidence(M, D, m_bar, sigma_d, N_d, ...)` — chi-square calibrated confidence set
- `build_problem(N_d, forward_seed, data_seed, ...)` — convenience wrapper
- `compute_dli_interval(...)` — DLI via `DualMasterCostFunction` + `ProximalBundleMethod`
- `compute_bg_interval(...)` — BG via Minkowski-sum support algebra
- `compute_gap_metrics(dli, bg)` — 6 BG-vs-DLI comparison metrics
- `evaluate_case(N_d, forward_seed, data_seed)` — single sweep case
- `run_sweep(verbose)` — triple loop over all seeds, returns 36 row dicts
- `summarize_results_by_nd(rows)` — per-N_d aggregation
- `write_raw_csv / write_summary_csv` — stdlib CSV writers
- `plot_width_ratio_vs_nd / plot_absolute_widths_vs_nd` — matplotlib PNG/PDF outputs
- `print_sweep_summary(summary_rows)` — formatted console table
- `run_phase1/2/3/4_validation` — self-contained validation harnesses per phase

**Test Coverage:**
- Total tests written: 4 validation harnesses (≈ 25 individual assertions)
- All tests passing: ✅
- Containment violation across all 36 cases: 0.0 (exact)

**Key Findings:**
| N_d | ratio_mean | ratio_median | DLI width | BG width |
|-----|-----------|--------------|-----------|----------|
| 5   | 1.0199    | 1.0193       | 2.3996    | 2.4472   |
| 10  | 1.0377    | 1.0360       | 2.3172    | 2.4041   |
| 20  | 1.1709    | 1.1366       | 2.0973    | 2.4478   |
| 40  | 1.3788    | 1.4003       | 1.3221    | 1.7839   |

**Recommendations for Next Steps:**
- Consider a similar sweep with N_p=2 (2D property space) to compare polyhedral hull sizes
- Explore sensitivity of the BG–DLI gap to the confidence level parameter (currently fixed at 0.95)
- Profile DLI iterations vs N_d to understand whether convergence slows with more data
