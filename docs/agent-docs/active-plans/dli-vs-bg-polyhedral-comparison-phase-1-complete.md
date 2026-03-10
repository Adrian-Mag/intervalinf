## Phase 1 Complete: Create Shared Setup and DLI Polyhedral Path

Created a new comparison notebook for the DLI vs BG study with editable `N_d`, `N_p`, and `N_theta` parameters, then implemented the shared setup and the DLI admissible-region pathway. The DLI side now builds a 2D `PolyhedralSet` approximation using exactly `N_theta` sampled support directions and validates containment of the true property against the sampled halfspaces.

**Files created/changed:**
- intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb
- intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-plan.md
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- Notebook configuration cell exposing `N_d`, `N_p`, and `N_theta`
- Shared setup cells for spaces, operators, synthetic data, and prior/confidence sets
- DLI support-solve cell using `DualMasterCostFunction`, `ProximalBundleMethod`, and `solve_support_values`
- DLI polyhedron assembly and containment-validation cell producing `dli_admissible_region`

**Tests created/changed:**
- Shared validation cell for dimensions and shared object construction
- Shared containment checks for `m_bar` and the sampled noise vector
- DLI validation checks for support solve completion, polyhedral assembly, and strict `p_bar` halfspace containment

**Review Status:** APPROVED with minor recommendations addressed

**Git Commit Message:**
feat(demos): add phase-1 DLI vs BG comparison notebook

- Create shared comparison notebook with editable N_d, N_p, and N_theta
- Implement the common setup and DLI admissible polyhedron path for 2D property space
- Add validation cells and update the intervalinf living reference for the new demo

Plan: intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-plan.md
Phase: 1 of 3
Related: intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-phase-1-complete.md
