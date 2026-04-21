## Phase 2 Complete: Add synthetic operators and synthetic data

Phase 2 added the synthetic forward and property operators, together with a reproducible synthetic true model and noisy observations. Both `vp` and `vs` now contribute through separate `NormalModesProvider`-driven forward operators, while the property operator stays simple with bump targets on `vp` and null contribution from `vs`.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb`

**Functions created/changed:**
- Notebook cells defining `G_vp`, `G_vs`, `G`, `T_vp`, `T_vs`, `T`
- Notebook cells defining `cosine_series_function`, `m_bar`, `d_bar`, `d_tilde`

**Tests created/changed:**
- Notebook structure validation via VS Code notebook summary
- Review of direct-sum consistency for `m_bar = [m_bar_vp, m_bar_vs]`

**Review Status:** APPROVED

**Git Commit Message:**
feat(convex-analysis): add synthetic block operators to vp vs dli notebook

- Build vp and vs forward blocks from separate NormalModesProvider instances
- Add vp-only bump-function property targets with null vs property contribution
- Generate seeded synthetic vp and vs truth functions and noisy data plots

Plan: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-plan.md
Phase: 2 of 3
Related: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-phase-2-complete.md
