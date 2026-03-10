## Plan: Premature Discretization Posterior Uncertainty

Create a new notebook beside the old paper demo that cleanly demonstrates how posterior property uncertainty depends on model-space geometry. The reference workflow will solve the actual continuous function-space Bayesian problem using intervalinf and pygeoinf, while the correctly discretized finite-dimensional workflow will use covariance-eigenfunction truncation; both will then be compared against naive discretize-first identity-prior baselines across resolution.

**Phases 3**
1. **Phase 1: Build the Shared Continuous Experiment Setup**
    - **Objective:** Create the new notebook scaffold and implement the common continuous problem used by every workflow and every resolution.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/paper_demos/` new notebook; notebook-local helpers for kernels, truth, property functional, and result tabulation.
    - **Tests to Write:** Execute setup cells and verify that the forward operator, scalar property operator, and fixed noise realization are reused unchanged across all workflows.
    - **Steps:**
        1. Create a new notebook in the same folder as `example_3.ipynb` with narrative framing around premature discretization and posterior uncertainty.
        2. Define the continuous problem on `[0,1]`: `m \in L^2(0,1)`, `N_d=20` Gaussian-bump forward kernels, smooth two-bump truth, one scalar Gaussian-bump property functional, and a fixed Gaussian noise realization.
        3. Implement the continuous prior operator $C_0 = (k^2 I + \alpha(-\Delta))^{-s/2}$ with Neumann boundary conditions using `Laplacian`, `BesselSobolevInverse`, and matching precision factors.
        4. Add notebook-local helper routines for resolution sweeps and numerical result collection.

2. **Phase 2: Implement Continuous, Correctly Discretized, and Naive Workflows**
    - **Objective:** Compute posterior property variance for three workflows while isolating model-space geometry as the only changing ingredient.
    - **Files/Functions to Modify/Create:** Same new notebook; no external Python module unless notebook code becomes unreasonably repetitive.
    - **Tests to Write:** Run the workflow cells for `N \in {32, 64, 128, 256, 512}` and verify that only the prior/model-space geometry differs between workflows.
    - **Steps:**
        1. Implement Workflow A as the actual continuous function-space posterior using intervalinf + pygeoinf operators directly, then evaluate the scalar posterior variance through the continuous property operator.
        2. Implement the correctly discretized workflow using the first `N` covariance eigenfunctions as basis functions, so the finite-dimensional prior is the spectral truncation of the continuous prior.
        3. Implement Workflow B with a uniform-grid Euclidean model and naive identity covariance `I_N`, and optional Workflow C with scaled identity covariance `h I_N`.
        4. Store posterior property variances and optional posterior-covariance traces in a compact table indexed by resolution.

3. **Phase 3: Visualize Resolution Dependence and Write Interpretation**
    - **Objective:** Produce the final figures, numerical summary table, and short interpretation paragraph suitable for Section 5.3.
    - **Files/Functions to Modify/Create:** Same new notebook.
    - **Tests to Write:** Execute the full notebook end to end and confirm all figures and tables render from a clean run.
    - **Steps:**
        1. Plot resolution `N` on a log scale against the posterior property variance for the continuous reference, the correctly discretized eigenfunction truncation, the naive identity-prior workflow, and the optional scaled-identity workflow.
        2. Plot the relative error of the naive workflow against the continuous reference.
        3. Add a table of all numerical values and optional trace diagnostics.
        4. Write a concise interpretation paragraph explaining that posterior uncertainty depends on the geometry imposed before discretization, not on the forward map and data alone.

**Open Questions 1**
1. Optional Workflow C will be included unless runtime or clarity suffers materially; if it does, retain the code path but leave the plot toggled off by default.