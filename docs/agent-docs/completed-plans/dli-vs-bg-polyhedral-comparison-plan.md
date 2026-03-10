## Plan: DLI vs BG Polyhedral Comparison

Create a new comparison notebook in `intervalinf/demos/convex_analysis/` that runs the DLI and BG admissible-region pipelines on the same problem with editable top-level parameters `N_d`, `N_p`, and `N_theta`. Both methods will build user-controlled approximating polyhedra with the same `N_theta` so the property-space comparison is fair and visually exact.

**Phases 3**
1. **Phase 1: Create Shared Setup and DLI Polyhedral Path**
    - **Objective:** Create the new notebook, define the shared experiment setup, and implement the DLI pathway with a user-facing `N_theta` polyhedral approximation in property space.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb`
    - **Tests to Write:** Notebook validation cells for dimensions, shared object construction, DLI support evaluation, DLI polyhedron assembly, and containment diagnostics.
    - **Steps:**
        1. Create the notebook skeleton and a top-level configuration cell exposing editable `N_d`, `N_p`, and `N_theta` defaults.
        2. Add shared setup cells for spaces, operators, synthetic data, model prior, and data-confidence set so both methods use the same ingredients.
        3. Implement the DLI pathway and build a `PolyhedralSet` approximation in 2D property space using exactly `N_theta` sampled support directions.
        4. Add validation and summary cells confirming the DLI result is well-formed.

2. **Phase 2: Add BG Polyhedral Path and Exact Overlay Plot**
    - **Objective:** Implement the BG Minkowski-sum pathway and compare both admissible polyhedra on the same property-space figure using exact polyhedral plotting.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb`
    - **Tests to Write:** Notebook validation cells for BG estimator construction, BG polyhedron assembly, common `N_theta` usage, and plot-input consistency.
    - **Steps:**
        1. Implement the BG operator-algebra path with `X^*`, `H`, the admissible support function, and a `PolyhedralSet` using the same `N_theta`.
        2. Add exact polyhedral plotting helpers and an overlay figure that plots the DLI and BG polyhedra together without extra angular resampling.
        3. Add validation cells to confirm both regions are constructed and plotted from the stored polyhedra.

3. **Phase 3: Add Comparative Diagnostics and Final Polish**
    - **Objective:** Add concise diagnostics, interpretation, and output polish so the notebook is self-contained and reproducible.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb`, `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** End-to-end notebook validation cells for width summaries, containment checks, and figure export.
    - **Steps:**
        1. Add side-by-side summaries of widths, centers, and containment for DLI and BG.
        2. Save the comparison figure and make the shared-vs-method-specific choices explicit in markdown.
        3. Update the intervalinf living reference document if the new notebook changes the demo inventory.

**Open Questions 0**
