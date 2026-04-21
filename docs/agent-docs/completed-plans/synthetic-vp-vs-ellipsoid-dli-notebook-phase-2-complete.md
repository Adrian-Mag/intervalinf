## Phase 2 Complete: Add Heteroscedastic Data Ellipsoid and Solve

Added the heteroscedastic data-confidence ellipsoid to the synthetic vp/vs ellipsoid DLI notebook and ran the DLI solve with ellipsoidal model and data constraints. The notebook now constructs a diagonal data ellipsoid in `EuclideanSpace`, solves the support-value problems in the `±e_i` directions with the proximal bundle method, and checks that posterior intervals tighten relative to the prior.

**Files created/changed:**
- intervalinf/demos/convex_analysis/synthetic_vp_vs_ellipsoid_dli.ipynb
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- Notebook data-ellipsoid construction using `DiagonalSparseMatrixLinearOperator`
- Notebook data support construction using `EllipsoidSupportFunction`
- Notebook DLI solve using `DualMasterCostFunction`
- Notebook support-value solve using `ProximalBundleMethod`
- Notebook posterior summary and sanity-check cells

**Tests created/changed:**
- Notebook execution checks for data-ellipsoid cells and solve cells
- Mahalanobis check confirming realized noise lies inside the heteroscedastic ellipsoid
- Posterior sanity assertion confirming posterior half-widths do not exceed prior half-widths
- Capture check confirming both true properties lie inside posterior intervals

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(demos): add phase-2 ellipsoidal data solve to synthetic vp/vs notebook

- Add heteroscedastic diagonal data ellipsoid in Euclidean space
- Run DLI support-value solve with ellipsoidal model and data constraints
- Update living reference with phase-2 notebook behavior and outputs

Plan: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-plan.md
Phase: 2 of 3
Related: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-phase-2-complete.md
