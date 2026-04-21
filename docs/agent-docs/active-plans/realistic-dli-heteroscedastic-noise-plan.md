## Plan: Realistic DLI Noise Model

Replace the current signal-scaled synthetic noise in the realistic DLI notebook with an explicit heteroscedastic diagonal measurement model, keeping the notebook at N_d = 50 so changes in posterior width reflect information gain rather than automatic noise inflation. The implementation will keep the existing DLI machinery intact while making the noise assumptions visible, reproducible, and appropriate for real-application style comparisons.

**Phases 3 phases**
1. **Phase 1: Add explicit heteroscedastic noise configuration**
    - **Objective:** Introduce notebook parameters for a fixed diagonal measurement-noise model that does not depend on the forward signal amplitude.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli.ipynb parameter cell, synthetic-data cell
    - **Tests to Write:** Re-execute the parameter and synthetic-data cells to confirm the configured noise vector is constructed and sampled correctly.
    - **Steps:**
        1. Add notebook parameters for the heteroscedastic diagonal noise model at N_d = 50.
        2. Write the data-generation logic so synthetic noise is sampled from the configured standard deviations rather than from max(abs(d_bar)).
        3. Run the updated cells first to confirm the new configuration is active and the printed diagnostics are coherent.

2. **Phase 2: Rebuild the data error set from the diagonal covariance**
    - **Objective:** Construct the deterministic noise ellipsoid from the configured heteroscedastic covariance and update the explanatory markdown accordingly.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli.ipynb data-error markdown cell, data covariance cell
    - **Tests to Write:** Re-execute the data covariance / ellipsoid cell to confirm positive definiteness, correct chi-squared radius reporting, and consistency of the realised noise check.
    - **Steps:**
        1. Update the markdown to describe diagonal heteroscedastic covariance rather than isotropic signal-scaled noise.
        2. Build C_D, C_D^{-1}, and C_D^{1/2} from the configured standard deviation vector.
        3. Run the updated covariance cell and confirm the support-function construction still succeeds.

3. **Phase 3: Validate the controlled DLI comparison at N_d = 50**
    - **Objective:** Re-run the downstream notebook cells and verify the posterior behavior under the fixed noise model.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli.ipynb results cells and any updated narrative text needed for interpretation
    - **Tests to Write:** Re-execute the prior-bounds, DLI solve, and results cells and compare posterior half-widths under the new noise model.
    - **Steps:**
        1. Run the updated notebook from synthetic data generation through the final results summary.
        2. Confirm the reported noise diagnostics now stay tied to the configured measurement model rather than the signal amplitude.
        3. Summarize whether increasing data count under fixed heteroscedastic noise changes the DLI intervals.

**Open Questions 0 questions**
