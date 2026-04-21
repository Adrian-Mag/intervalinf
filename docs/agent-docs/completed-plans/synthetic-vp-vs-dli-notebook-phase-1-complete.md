## Phase 1 Complete: Build notebook scaffold and block spaces

Phase 1 created the new synthetic notebook scaffold and established the two-block direct-sum model structure. The notebook now introduces the synthetic vp/vs DLI setup, defines numerical integration settings, and constructs `M_vp`, `M_vs`, `M_model`, `D`, and `P` in a form consistent with `intervalinf` and `pygeoinf` operator workflows.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb`

**Functions created/changed:**
- Notebook setup cells defining `M_vp`, `M_vs`, `M_model`, `D`, `P`

**Tests created/changed:**
- Notebook structure validation via VS Code notebook summary

**Review Status:** APPROVED

**Git Commit Message:**
feat(convex-analysis): scaffold synthetic vp vs dli notebook

- Create synthetic_vp_vs_dli.ipynb as a midpoint between dli and realistic_dli
- Add notebook introduction, numerical configuration, and vp plus vs block spaces
- Define direct-sum model, data, and property spaces for later DLI steps

Plan: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-plan.md
Phase: 1 of 3
Related: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-phase-1-complete.md
