## Plan Complete: Premature Discretization Posterior Uncertainty

This plan delivered a replacement paper-demo notebook that isolates how posterior property uncertainty depends on model-space geometry rather than on the forward map and data alone. The finished notebook compares the true continuous function-space posterior against covariance-eigenfunction truncations and naive discretize-first identity priors, and it now includes saved figures, tabulated diagnostics, interpretation text, and living-reference coverage.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Build the Shared Continuous Experiment Setup
2. ✅ Phase 2: Implement Continuous, Correctly Discretized, and Naive Workflows
3. ✅ Phase 3: Visualize Resolution Dependence and Write Interpretation

**All Files Created/Modified:**
- intervalinf/demos/old_demos/paper_demos/premature_discretization_posterior_uncertainty.ipynb
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Key Functions/Classes Added:**
- Notebook helper `gaussian_kernel`
- Notebook helper `make_results_container`
- Notebook helper `neumann_eigfuncs_matrix`
- Notebook helper `c0_prior_eigenvalues`
- Notebook helper `finite_dim_posterior_stats`
- Notebook cells for the Phase 3 comparison plots, tabulated summary, and closing interpretation

**Test Coverage:**
- Total tests written: notebook execution, assertion, and saved-output validation across all 3 phases
- All tests passing: ✅

**Recommendations for Next Steps:**
- Use this notebook as the primary paper-demo reference when discussing premature discretization and posterior uncertainty in Section 5.3.
- If the experiment parameters are changed later, rerun the full notebook from a clean kernel so the saved sweep tables and figure outputs remain synchronized.