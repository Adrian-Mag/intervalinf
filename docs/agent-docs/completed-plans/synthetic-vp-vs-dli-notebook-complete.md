## Plan Complete: Synthetic vp vs DLI Notebook

A new notebook, `synthetic_vp_vs_dli.ipynb`, has been added to the convex-analysis demos as a lightweight but structured deterministic linear inference example. It uses a two-block direct-sum model space `vp ⊕ vs`, synthetic normal-mode forward kernels, simple ball support functions for the prior and data-confidence sets, and a full proximal-bundle DLI solve with plots and summary output.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Build notebook scaffold and block spaces
2. ✅ Phase 2: Add synthetic operators and synthetic data
3. ✅ Phase 3: Solve the DLI problem and document the demo

**All Files Created/Modified:**
- `intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb`
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
- `intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-plan.md`
- `intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-phase-1-complete.md`
- `intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-phase-2-complete.md`
- `intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-phase-3-complete.md`
- `intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-dli-notebook-complete.md`

**Key Functions/Classes Added:**
- Notebook cells creating `G_vp`, `G_vs`, `G`, `T_vp`, `T_vs`, and `T`
- Notebook cell defining `cosine_series_function`
- Notebook cells building `BallSupportFunction` prior/data sets and solving the DLI support problems

**Test Coverage:**
- Notebook recognized by VS Code notebook tooling: ✅
- Direct-sum model/operator structure review: ✅
- Living reference updated: ✅
- Code review status: APPROVED

**Recommendations for Next Steps:**
- Run the notebook once in your preferred environment to cache plots and inspect the numerical results
- If you want the next step up in complexity, add a second property family on `vs` instead of keeping properties vp-only
