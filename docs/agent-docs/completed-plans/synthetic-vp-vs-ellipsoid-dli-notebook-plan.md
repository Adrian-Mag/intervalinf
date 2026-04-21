## Plan: Synthetic vp/vs Ellipsoid DLI Notebook

Create a new synthetic DLI notebook that stays lightweight like `synthetic_vp_vs_dli.ipynb` but replaces both norm-ball constraints with ellipsoids. The model prior will be built operator-theoretically from Bessel operators on basis-free function spaces, and the data-confidence set will use a heteroscedastic ellipsoid in data space, with no model-space discretization.

**Phases 3**
1. **Phase 1: Build Bessel Model Ellipsoid**
    - **Objective:** Create the new notebook scaffold and define a non-discretized model prior ellipsoid on the synthetic vp/vs direct-sum model space.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/synthetic_vp_vs_ellipsoid_dli.ipynb`; `BesselSobolev`, `BesselSobolevInverse`, `EllipsoidSupportFunction`, `HilbertSpaceDirectSum`, `RowLinearOperator`
    - **Tests to Write:** Notebook execution checks for imports/configuration/model-prior cells; validation that the model prior is built from operators without calling `.matrix(...)` on model operators.
    - **Steps:**
        1. Write notebook markdown and import/configuration cells for the new synthetic vp/vs ellipsoid example.
        2. Build basis-free `Lebesgue` blocks, synthetic forward/property operators, and a seeded synthetic truth.
        3. Write the model-prior construction using Bessel operators and direct-sum block assembly, then run the relevant notebook cells to confirm they execute.

2. **Phase 2: Add Heteroscedastic Data Ellipsoid and Solve**
    - **Objective:** Add the heteroscedastic data-confidence ellipsoid and run the DLI solve with ellipsoidal constraints.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/synthetic_vp_vs_ellipsoid_dli.ipynb`; `EllipsoidSupportFunction`, `DualMasterCostFunction`, `ProximalBundleMethod`, `solve_support_values`
    - **Tests to Write:** Notebook execution checks for data-ellipsoid cell and solve cell; posterior interval sanity checks ensuring posterior half-widths do not exceed prior half-widths.
    - **Steps:**
        1. Define a heteroscedastic diagonal data covariance and construct the data ellipsoid in `EuclideanSpace`.
        2. Add the support-value solve in the `±e_i` directions using the ellipsoidal model and data constraints.
        3. Run the solve cell and verify the expected interval outputs are produced.

3. **Phase 3: Results, Documentation, and Reference Update**
    - **Objective:** Finalize figures/explanatory text and update the intervalinf living reference to describe the new notebook.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/synthetic_vp_vs_ellipsoid_dli.ipynb`; `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** Notebook execution checks for results/summary cells; reference update verification.
    - **Steps:**
        1. Add result plots and concise markdown explaining how the ellipsoidal constraints differ from the ball-based notebook.
        2. Update the living reference entry for the new notebook and its operator-theoretic ellipsoid construction.
        3. Re-run the full notebook or final result cells to confirm the finished demo works as described.

**Open Questions 1**
1. Resolved: use a heteroscedastic diagonal data ellipsoid in data space as the synthetic analogue of `realistic_dli.ipynb`.
