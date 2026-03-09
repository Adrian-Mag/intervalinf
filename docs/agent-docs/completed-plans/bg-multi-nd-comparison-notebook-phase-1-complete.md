## Phase 1 Complete: Create Comparison Notebook Skeleton

Phase 1 created a new companion notebook for multi-`N_d` Backus-Gilbert comparisons, with a fair-comparison narrative, centralized configuration, reusable helper functions, and smoke-test execution cells. The intervalinf living reference was also updated to mention the new notebook and its comparison-oriented design.

**Files created/changed:**
- intervalinf/demos/convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- build_integration_and_parallel_configs
- build_spaces_and_operators
- build_true_model_and_noise
- build_prior_and_confidence
- build_optimal_estimator
- build_admissible_support
- run_single_nd_case

**Tests created/changed:**
- Notebook import/config smoke-test cell
- Notebook helper-definition smoke-test cell
- Notebook single-case `N_d=5` end-to-end smoke-test cell

**Review Status:** APPROVED

**Git Commit Message:**
feat(notebook): add multi-nd BG comparison skeleton

- Create companion notebook for multi-N_d BG comparisons
- Add reusable single-case helper functions and config cell
- Update intervalinf reference with the new demo notebook

Plan: intervalinf/docs/agent-docs/active-plans/bg-multi-nd-comparison-notebook-plan.md
Phase: 1 of 4
Related: intervalinf/docs/agent-docs/completed-plans/bg-multi-nd-comparison-notebook-phase-1-complete.md
