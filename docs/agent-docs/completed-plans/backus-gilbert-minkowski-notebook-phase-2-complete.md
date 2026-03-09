## Phase 2 Complete: Construct the Minkowski-Sum Admissible Set

Extended the notebook to compute the estimator point, define the residual operator at the operator level, and build the admissible property region from support-function pullbacks. The resulting two-dimensional admissible region is assembled as a halfspace intersection and plotted together with the true property and estimator point.

**Files created/changed:**
- `intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb`

**Functions created/changed:**
- Notebook computation of `p_t = X_star_operator(d_tilde)`
- Notebook construction of `H_operator = T - X_star_operator @ G`
- Notebook support oracle for `p_t + H B + X_star V_data`
- Notebook construction of the `PolyhedralSet` admissible region in property space

**Tests created/changed:**
- No standalone test files were added in this phase
- Phase 2 notebook cells executed successfully after review cleanup
- Containment checks for `p_t` and `p_bar` passed in the admissible region

**Review Status:** APPROVED

**Git Commit Message:**
feat(demos): add operator-based phase 2 admissible region

- Build p_t, H, and the support oracle without explicit Gram matrices
- Construct the 2D admissible region from directional support halfspaces
- Plot the Minkowski-sum region with true and estimated properties

Plan: intervalinf/docs/agent-docs/active-plans/backus-gilbert-minkowski-notebook-plan.md
Phase: 2 of 3
Related: intervalinf/docs/agent-docs/completed-plans/backus-gilbert-minkowski-notebook-phase-2-complete.md