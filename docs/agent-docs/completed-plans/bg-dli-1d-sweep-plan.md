## Plan: BG vs DLI 1D Sweep Script

Create a standalone script in `intervalinf/rough_work/` that compares one-dimensional BG and DLI admissible intervals across varying `NormalModesProvider` random states and increasing `N_d`. The script will keep `N_p = 1`, use nested forward operators across `N_d`, vary data realizations independently, and report how the BG interval gap compares to the DLI interval gap as the number of data increases.

**Phases 4**
1. **Phase 1: Create Script Scaffold and Shared Builders**
    - **Objective:** Add a new Python script with reusable helpers for spaces, operators, synthetic data, and deterministic confidence/prior sets.
    - **Files/Functions to Modify/Create:** `intervalinf/rough_work/bg_dli_1d_sweep.py`
    - **Tests to Write:** Script-level validation checks for dimensions, positive radii, finite noise scale, nested-kernel construction, and reproducible seeds.
    - **Steps:**
        1. Create top-level configuration for `N_d_values`, forward seeds, data seeds, noise fraction, confidence level, and output paths.
        2. Implement helper builders for `M, D, P, G, T`, with `N_p = 1` and nested forward-operator construction.
        3. Implement synthetic-data generation using a fixed true model and independent data seeds.
        4. Implement prior/confidence-set builders shared by DLI and BG, plus validation helpers.

2. **Phase 2: Implement DLI and BG Interval Solvers**
    - **Objective:** Compute 1D admissible intervals for both methods on the same problem instances.
    - **Files/Functions to Modify/Create:** `intervalinf/rough_work/bg_dli_1d_sweep.py`
    - **Tests to Write:** Checks that DLI and BG widths are finite and nonnegative, lower bounds use `-h(-e1)`, and containment sanity checks pass.
    - **Steps:**
        1. Implement DLI bounds with `DualMasterCostFunction` and `solve_support_values` for `+e1` and `-e1`.
        2. Implement BG bounds from the existing estimator/support-function construction used in the notebook demos.
        3. Record interval widths and solver diagnostics for each case.

3. **Phase 3: Run Multi-Seed Sweep and Compute Gap Metrics**
    - **Objective:** Aggregate repeated runs over different forward operators and data realizations.
    - **Files/Functions to Modify/Create:** `intervalinf/rough_work/bg_dli_1d_sweep.py`
    - **Tests to Write:** Validation checks for row counts, metric finiteness, and zero/near-zero containment-violation rates.
    - **Steps:**
        1. Sweep `N_d` over `5, 10, 20, 40`.
        2. For each `N_d`, loop over forward seeds and data seeds.
        3. Compute primary metric `width_ratio = width_BG / width_DLI`, plus excess width, endpoint slack, and Hausdorff gap.
        4. Save a raw CSV and an aggregated summary table by `N_d`.

4. **Phase 4: Add Plots, CLI Polish, and Reference Update**
    - **Objective:** Make the script directly usable and generate interpretable outputs.
    - **Files/Functions to Modify/Create:** `intervalinf/rough_work/bg_dli_1d_sweep.py`, `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** File-output checks for CSV/PNG/PDF creation and a final summary printout.
    - **Steps:**
        1. Add plots for width ratio vs `N_d` and absolute widths vs `N_d`.
        2. Save outputs under a script-specific figures/results folder.
        3. Print a concise summary of how the BG–DLI gap changes with `N_d`.
        4. Update the intervalinf living reference to mention the new comparison script.

**Open Questions 0**
