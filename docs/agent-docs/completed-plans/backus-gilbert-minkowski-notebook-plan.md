## Plan: Backus-Gilbert Minkowski Notebook

Create a new notebook in the DLI demo area that mirrors the setup of the existing DLI notebook while reducing the problem to two properties and five data points. The notebook will construct and visualize the admissible property set through the support identity for the Minkowski sum $\pt + H\Bset + X\Vdata$, with the data-confidence term represented by a Gaussian covariance ellipsoid.

**Phases 3**
1. **Phase 1: Build the Reduced DLI Setup**
    - **Objective:** Create the reduced continuous setup with the same synthetic model and operator patterns as the existing DLI notebook, but with a two-dimensional property space and five data points.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb`; helper logic only if a reusable extraction is needed.
    - **Tests to Write:** Helper tests for reduced-dimension setup if any Python helper module is introduced.
    - **Steps:**
        1. Create the notebook skeleton with imports, narrative, configuration, spaces, operators, and synthetic data setup.
        2. Reduce the problem dimensions to `N_d = 5` and `N_p = 2` while preserving the existing kernel-provider patterns.
        3. Execute the setup cells needed to confirm that the reduced operators, truth, and noisy data are internally consistent.

2. **Phase 2: Construct the Minkowski-Sum Admissible Set**
    - **Objective:** Form the support oracle for $\pt + H\Bset + X\Vdata$ and turn sampled directional bounds into a two-dimensional admissible region.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb`; optional helper code only if needed for clean support-oracle construction.
    - **Tests to Write:** Support-decomposition tests only if helper code is added outside the notebook.
    - **Steps:**
        1. Define the estimator operator $X$ using the Backus--Gilbert-with-errors formula from the theory section.
        2. Build $\pt = X\tilde d$, $H = T - XG$, the model-prior support, and the Gaussian covariance ellipsoid support for the data term.
        3. Evaluate directional support values on the unit circle and construct the outer polyhedral approximation of the admissible set in property space.

3. **Phase 3: Visualize and Explain the Geometry**
    - **Objective:** Add final figures and narrative that show how the admissible set, estimator point, and component uncertainty terms relate to Section 3.4.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb`; update living references only if reusable code is added elsewhere.
    - **Tests to Write:** No additional tests unless reusable plotting helpers are introduced.
    - **Steps:**
        1. Add markdown explaining the meaning of $\pt$, $H\Bset$, and $X\Vdata$ in the notebook notation.
        2. Plot the final two-dimensional admissible region together with the true property and estimator point.
        3. Execute the relevant notebook cells to confirm the notebook works end to end.

**Open Questions 0**