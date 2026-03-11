## Phase 2 Complete: Implement Continuous, Correctly Discretized, and Naive Workflows

Implemented the resolution sweep comparing the continuous reference posterior variance against covariance-eigenfunction truncation and naive discretize-first baselines. The notebook now demonstrates the intended effect clearly: the eigenfunction truncation matches the continuous reference while the naive identity and scaled-identity workflows drift with resolution.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/premature_discretization_posterior_uncertainty.ipynb`

**Functions created/changed:**
- Notebook helper `neumann_eigfuncs_matrix`
- Notebook helper `c0_prior_eigenvalues`
- Notebook helper `finite_dim_posterior_stats`
- Phase 2 sweep logic over `N in [32, 64, 128, 256, 512]`
- Results aggregation into the notebook `results` container and dataframe summary

**Tests created/changed:**
- No standalone test files were added
- Phase 2 sweep cell executed successfully and stored notebook output
- Phase 2 results-table cell executed successfully and stored notebook output
- Review approved the scientific comparison and found no blockers for Phase 3

**Key numerical outcome:**
- Continuous reference posterior variance: approximately `3.21e-4`
- Eigenfunction truncation: matches the reference to near machine precision across all tested `N`
- Naive identity prior: resolution-dependent drift away from the reference
- Scaled identity prior: stronger resolution-dependent collapse of posterior variance

**Review Status:** APPROVED

**Git Commit Message:**

```text
feat(demos): add posterior-uncertainty resolution sweep workflows

- Implement covariance-eigenfunction truncation and naive discretize-first baselines against the continuous reference
- Add finite-dimensional posterior variance and trace diagnostics across resolutions 32 to 512
- Persist the Phase 2 sweep outputs and summary table in the notebook artifact

Plan: intervalinf/docs/agent-docs/active-plans/premature-discretization-posterior-uncertainty-plan.md
Phase: 2 of 3
Related: intervalinf/docs/agent-docs/active-plans/premature-discretization-posterior-uncertainty-phase-2-complete.md
```