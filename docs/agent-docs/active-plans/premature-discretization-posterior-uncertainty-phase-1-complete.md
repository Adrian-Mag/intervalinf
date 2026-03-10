## Phase 1 Complete: Build the Shared Continuous Experiment Setup

Created a new notebook scaffold for the posterior-uncertainty experiment and validated the continuous reference problem end to end. The notebook now fixes the shared domain, forward operator, scalar property functional, true model, fixed noise realization, and continuous Whittle-Matern/Bessel-Sobolev prior that later phases will compare against naive discretize-first workflows.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/premature_discretization_posterior_uncertainty.ipynb`

**Functions created/changed:**
- Notebook helper `gaussian_kernel`
- Notebook helper `make_results_container`
- Continuous reference setup for `G`, `T`, `m_true`, `d_true`, `d_noisy`, `prior_continuous`, and `data_error_measure`
- Continuous posterior validation block producing the scalar reference posterior variance

**Tests created/changed:**
- No standalone test files were added
- All 10 code cells in the new notebook executed successfully end to end
- Review-approved fixes applied: finer continuous reference (`N_CONT=2048`, `SPECTRAL_DOFS=2048`), explicit `alpha` wiring in the Laplacian, `functools.partial` kernel callables, consistent results key, and robust single-process setup

**Review Status:** APPROVED

**Git Commit Message:**
docs(demos): add continuous reference scaffold for posterior-uncertainty experiment

- Create a new paper-demo notebook for premature discretization vs posterior uncertainty
- Build the shared continuous reference problem with fixed forward map, scalar property, noise realization, and Bessel-Sobolev prior
- Validate the continuous posterior variance end to end and prepare Phase 2 results storage

Plan: intervalinf/docs/agent-docs/active-plans/premature-discretization-posterior-uncertainty-plan.md
Phase: 1 of 3
Related: intervalinf/docs/agent-docs/active-plans/premature-discretization-posterior-uncertainty-phase-1-complete.md