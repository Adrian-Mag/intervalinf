## Plan: Realistic DLI Notebook Mirror

Build a deterministic-linear-inference counterpart to the realistic Bayesian example so the same spaces, mappings, true model, and synthetic data can be solved using confidence sets rather than Gaussian measures. Start with a simpler sanity-pass notebook configuration to expose missing infrastructure safely, then extend toward the full covariance-matched prior construction.

**Phases 4**
1. **Phase 1: Build Sanity-Pass DLI Mirror**
    - **Objective:** Create a reduced realistic DLI notebook that mirrors the core setup of the realistic Bayesian example while using a covariance-derived data confidence ellipsoid and a simpler deterministic prior set.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/realistic_dli.ipynb`, `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** None initially unless helper logic is extracted; notebook execution will be used to surface missing functionality.
    - **Steps:**
        1. Copy the realistic model-space, forward-operator, property-operator, true-model, and synthetic-data construction from `intervalinf/demos/old_demos/paper_demos/example_1.ipynb` into a new notebook with reduced `N_d=5` and `N_p=2`.
        2. Replace the Gaussian data measure with a chi-squared `0.95` confidence ellipsoid built from the same data covariance.
        3. Use a simpler deterministic model prior set for the first pass so the DLI workflow can be exercised before full prior-ellipsoid support is required.
        4. Attempt to solve the DLI support problems and record the first runtime blockers.

2. **Phase 2: Add Missing Convex-Analysis Infrastructure**
    - **Objective:** Implement the minimal pygeoinf or intervalinf functionality needed to support the realistic DLI notebook once runtime gaps are identified.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/convex_analysis.py`, `pygeoinf/pygeoinf/gaussian_measure.py`, `pygeoinf/tests/*`, and any additional small helpers needed by the notebook
    - **Tests to Write:** Targeted regression tests for the first missing deterministic-set or support-function functionality encountered.
    - **Steps:**
        1. Write a failing test that reproduces the first missing capability exposed by Phase 1.
        2. Implement the smallest possible library change to make that capability available.
        3. Re-run the targeted tests to confirm the fix.
        4. Update living reference files to document any new or changed API.

3. **Phase 3: Upgrade to Covariance-Matched Prior Ellipsoid**
    - **Objective:** Replace the simpler prior set with a deterministic prior confidence ellipsoid that mirrors the covariance structure of the realistic Bayesian prior.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/realistic_dli.ipynb`, any helper code introduced in Phase 2, `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`, `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
    - **Tests to Write:** Tests for model-space ellipsoid support evaluation, especially for direct-sum or operator-defined covariances if new helpers are introduced.
    - **Steps:**
        1. Construct a deterministic model confidence set using the same covariance structure as `example_1.ipynb` and a `0.95` chi-squared threshold.
        2. Use KL truncation as the effective degrees of freedom for the model-space confidence radius.
        3. Run DLI on the mirrored problem with the covariance-matched prior ellipsoid.
        4. Verify the notebook remains executable at the reduced test scale.

4. **Phase 4: Validate and Document Scaling Constraints**
    - **Objective:** Execute the notebook end to end, fix any remaining blockers, and document what still limits scaling to larger `N_d` and `N_p`.
    - **Files/Functions to Modify/Create:** Runtime-dependent; plus reference docs as needed
    - **Tests to Write:** Focused regression tests for any additional runtime failure that requires a library fix.
    - **Steps:**
        1. Execute the notebook from top to bottom in the configured environment.
        2. Fix any residual issues using test-first changes where code extraction or reusable helpers are needed.
        3. Record execution constraints, performance bottlenecks, and any unresolved generalization issues.
        4. Leave the notebook and references in a state that documents the path to a larger realistic DLI run.

**Open Questions 1**
1. If full model-space ellipsoids prove too expensive at the notebook level, should the next step be notebook-only approximation helpers or first-class operator support in `pygeoinf`?