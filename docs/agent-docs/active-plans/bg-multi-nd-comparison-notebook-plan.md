## Plan: BG Multi-Nd Comparison Notebook

Create a new convex-analysis demo notebook that reproduces the Backus-Gilbert admissible-region construction for a user-specified list of `N_d` values and overlays the resulting absolute property sets on a single figure. The notebook will preserve the operator-first support-function construction from the existing demo, use a fair comparison calibration across data dimensions, and keep the presentation focused on one final overlay plot plus concise per-case diagnostics.

**Phases 4**
1. **Phase 1: Create Comparison Notebook Skeleton**
    - **Objective:** Create a new companion notebook with the comparison narrative, configuration cell, and reusable helper-function structure needed to run multiple `N_d` cases cleanly.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb`; `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** notebook execution smoke checks for imports, configuration, and helper definitions.
    - **Steps:**
        1. Create a new notebook in the convex-analysis demos folder rather than modifying the existing single-run notebook.
        2. Add a top-level configuration cell exposing `N_d_values = [5, 10, 20, 50, 100]`, `N_theta`, `noise_fraction`, `confidence_level`, and plotting settings.
        3. Refactor the single-case pipeline into notebook helper functions for spaces/operators, noise calibration, estimator construction, admissible support construction, and polyhedral region generation.
        4. Keep the helper API notebook-readable and ready for a simple loop over `N_d` values.

2. **Phase 2: Implement Fair Multi-Nd BG Pipeline**
    - **Objective:** Execute the full BG admissible-region construction for each requested `N_d` under a comparison design that preserves consistent kernel generation and fair uncertainty calibration.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb`
    - **Tests to Write:** notebook execution checks that each `N_d` case produces `X_star_operator`, `H_operator`, `admissible_support`, and a valid polyhedral admissible region.
    - **Steps:**
        1. Implement a single-case runner function that returns all geometry needed for plotting and diagnostics.
        2. Reuse a consistent kernel-generation strategy across `N_d` values so comparisons are not dominated by unrelated random changes.
        3. Use the RMS-based noise rule and chi-square confidence calibration already established in the single-case notebook.
        4. Store each case in a structured dictionary keyed by `N_d`.

3. **Phase 3: Add Absolute Overlay Plot and Diagnostics**
    - **Objective:** Plot all final absolute admissible property sets in one figure and add concise printed diagnostics for each `N_d`.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb`
    - **Tests to Write:** notebook execution checks for the combined overlay figure and per-case diagnostic summary.
    - **Steps:**
        1. Plot the absolute admissible regions on a common property-space axis using a clear color progression over `N_d`.
        2. Label each region explicitly by its `N_d` value.
        3. Print a concise per-case summary including `sigma_d`, `r_V`, `alpha`, and x/y widths.
        4. Keep the visualization focused on the single requested final overlay figure.

4. **Phase 4: Validate Notebook and Sync Reference**
    - **Objective:** Execute the new notebook end to end, verify the overlay plot is valid for the default list, and update the intervalinf living reference.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb`; `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** end-to-end notebook execution checks.
    - **Steps:**
        1. Run the notebook cells in order with the default `N_d` list.
        2. Verify that every case contributes one valid region to the final overlay figure.
        3. Confirm the narrative explains the fair-comparison calibration and the remaining interpretation caveats.
        4. Update the living reference to mention the new comparison notebook.

**Open Questions**
1. None at present; approved decisions are to plot only the absolute admissible sets, use `[5, 10, 20, 50, 100]` as the default list, and keep the output to one final overlay figure.
