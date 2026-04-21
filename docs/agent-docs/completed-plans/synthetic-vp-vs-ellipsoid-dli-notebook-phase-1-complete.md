## Phase 1 Complete: Build Bessel Model Ellipsoid

Created a new synthetic vp/vs ellipsoid DLI notebook scaffold and implemented the non-discretized Bessel-based model-prior ellipsoid. The notebook now builds a basis-free direct-sum model, synthetic forward/property operators, seeded synthetic truth/data, and an operator-theoretic `EllipsoidSupportFunction` for the model prior without calling `.matrix(...)` on model operators.

**Files created/changed:**
- intervalinf/demos/convex_analysis/synthetic_vp_vs_ellipsoid_dli.ipynb
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- Notebook cell helper `cosine_series_function`
- Notebook Bessel-prior construction using `BesselSobolev`
- Notebook Bessel-prior construction using `BesselSobolevInverse`
- Notebook model-prior construction using `EllipsoidSupportFunction`

**Tests created/changed:**
- Notebook execution through the model-prior setup cells
- Prior sanity checks: `σ_ε(0)=0`, `m_bar ∈ E`, support dominates inner product in a test direction
- Prior interval check: both true properties lie inside the prior bounds

**Review Status:** APPROVED

**Git Commit Message:**
feat(demos): add phase-1 synthetic vp/vs ellipsoid notebook

- Create operator-theoretic vp/vs ellipsoid DLI demo scaffold
- Build basis-free Bessel prior ellipsoid without model discretization
- Update living reference with new ellipsoid notebook entry

Plan: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-plan.md
Phase: 1 of 3
Related: intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-phase-1-complete.md
