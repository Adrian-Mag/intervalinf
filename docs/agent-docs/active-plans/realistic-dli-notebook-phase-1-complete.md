## Phase 1 Complete: Build Sanity-Pass DLI Mirror

Created `realistic_dli.ipynb` as a faithful mirror of `example_1.ipynb` at reduced scale (N_d=5, N_p=2).  The notebook uses the real seismic kernel catalog, identical direct-sum model space and forward-operator construction, and chi-squared 0.95 confidence ellipsoid for the data set.  Execution revealed the nested-list norm bug in `LinearFormKernel._mapping_impl` (addressed in Phase 2) which was the only blocker.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/realistic_dli.ipynb` — new notebook (25 cells)
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — updated to document new notebook

**Functions created/changed:**
- N/A (no library code changed in Phase 1)

**Tests created/changed:**
- N/A (Phase 1 was notebook-execution-driven; test infrastructure added in Phase 2)

**Review Status:** APPROVED (cells 1–17 execute cleanly; blocker surfaced and delegated to Phase 2)

**Git Commit Message:**
```
feat(notebooks): create realistic_dli.ipynb sanity-pass DLI mirror

- Add realistic_dli.ipynb in demos/convex_analysis/ mirroring example_1.ipynb
- Real seismic kernel catalog (kernels_modeplotaat_Adrian) loaded via locator cell
- Direct-sum model space M_model (vp, vs_IC, vs_M, rho, sigma_0, sigma_1), dim=42
- Chi-squared 0.95 data confidence ellipsoid (df=5, chi2_radius=3.327)
- BallSupportFunction prior; DLI solve with DualMasterCostFunction
- N_d=5, N_p=2 test scale; cells 1-17 pass, blocker exposed for Phase 2

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-notebook-plan.md
Phase: 1 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-notebook-phase-1-complete.md
```
