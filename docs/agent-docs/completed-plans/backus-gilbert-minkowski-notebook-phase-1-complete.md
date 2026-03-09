## Phase 1 Complete: Build the Reduced DLI Setup

Created a new notebook in the DLI demo area that mirrors the continuous setup of the existing DLI notebook while reducing the problem to five data points and two properties. The notebook executes successfully through the reduced operator setup, synthetic data generation, model prior support, Gaussian data ellipsoid support, and estimator-preparation stage.

**Files created/changed:**
- `intervalinf/demos/old_demos/pli_demos/bg_with_errors_minkowski.ipynb`

**Functions created/changed:**
- Notebook setup for the reduced continuous spaces and operators
- Notebook construction of the model prior ball support and Gaussian data ellipsoid support
- Notebook computation of the Backus--Gilbert-with-errors estimator ingredients

**Tests created/changed:**
- No standalone test files were added in this phase
- All 9 code cells in the notebook executed successfully

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(demos): add reduced BG-with-errors notebook setup

- Create a new DLI-area notebook with 5 data points and 2 properties
- Add Gaussian covariance ellipsoid setup and reduced continuous operators
- Execute all setup cells successfully for the phase 1 scaffold

Plan: intervalinf/docs/agent-docs/active-plans/backus-gilbert-minkowski-notebook-plan.md
Phase: 1 of 3
Related: intervalinf/docs/agent-docs/completed-plans/backus-gilbert-minkowski-notebook-phase-1-complete.md