## Phase 3 Complete: Solve the DLI problem and document the demo

Phase 3 finished the notebook by adding the ball-prior / ball-data DLI solve, prior-vs-posterior bounds, result plots, and summary output. The package living reference was also updated so the new notebook is documented alongside the existing convex-analysis demos.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb`
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`

**Functions created/changed:**
- Notebook cells defining `model_prior_support`, `data_error_support`, `cost`, `bundle_solver`
- Notebook cells computing prior bounds, posterior bounds, reduction factors, and summary plots

**Tests created/changed:**
- Notebook validation via VS Code notebook summary
- Code review of direct-sum structure, support-function choices, and reference update

**Review Status:** APPROVED

**Git Commit Message:**
feat(convex-analysis): finish synthetic vp vs dli notebook

- Add ball-prior and ball-data DLI solve with proximal bundle optimization
- Plot prior versus posterior property bounds and reduction factors
- Document synthetic_vp_vs_dli.ipynb in the intervalinf living reference

Plan: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-plan.md
Phase: 3 of 3
Related: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-phase-3-complete.md
