## Phase 2 Complete: Add BG Polyhedral Path and Exact Overlay Plot

Added the Backus-Gilbert pathway to the comparison notebook using the shared Phase 1 setup, then built a BG `PolyhedralSet` with the same editable `N_theta` directions used by DLI. The notebook now renders an exact overlay comparison of the stored DLI and BG polyhedra without any angular resampling at plot time.

**Files created/changed:**
- intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb

**Functions created/changed:**
- BG import/setup cells for `LinearOperator`, `CholeskySolver`, and `SupportFunction`
- BG estimator construction cell for `X_star_bg`
- BG support assembly cell for `p_t_bg`, `H_bg`, and `bg_admissible_support`
- BG polyhedral assembly and validation cells producing `bg_admissible_region`
- Exact overlay plotting cell using the stored `PolyhedralSet` objects directly
- Plot-input consistency validation covering shared `N_theta` and stored halfspace offsets

**Tests created/changed:**
- BG construction validation for shared directions, containment of `p_bar`, and operator-domain consistency
- Plot-input consistency validation asserting that both plotted objects are `PolyhedralSet`s with offsets matching the stored support values
- Re-run validation cells after review-driven fixes to containment logic and Phase 2 scope boundaries

**Review Status:** APPROVED after scope and consistency fixes

**Git Commit Message:**
feat(demos): add BG polyhedral comparison path

- Extend the DLI vs BG comparison notebook with the BG operator-algebra pathway
- Build the BG admissible polyhedron with the same editable N_theta directions as DLI
- Add an exact overlay plot from stored PolyhedralSet objects with validation coverage

Plan: intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-plan.md
Phase: 2 of 3
Related: intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-phase-2-complete.md
