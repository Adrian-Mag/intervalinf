## Plan Complete: Synthetic vp/vs Ellipsoid DLI Notebook

Built a complete synthetic vp/vs DLI notebook that replaces the ball-based prior and data-confidence sets with ellipsoids while keeping the model-space construction continuous and operator-theoretic. The finished demo now shows a basis-free Bessel-Sobolev model prior, a heteroscedastic diagonal data ellipsoid, a full DLI solve, and a closing explanation of why ellipsoidal constraints are more informative than norm balls in this setting.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Build Bessel Model Ellipsoid
2. ✅ Phase 2: Add Heteroscedastic Data Ellipsoid and Solve
3. ✅ Phase 3: Results, Documentation, and Reference Update

**All Files Created/Modified:**
- intervalinf/demos/convex_analysis/synthetic_vp_vs_ellipsoid_dli.ipynb
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md
- intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-plan.md
- intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-phase-1-complete.md
- intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-phase-2-complete.md
- intervalinf/docs/agent-docs/active-plans/synthetic-vp-vs-ellipsoid-dli-notebook-phase-3-complete.md

**Key Functions/Classes Added:**
- `EllipsoidSupportFunction` usage for the model prior in a basis-free direct-sum setting
- `EllipsoidSupportFunction` usage for heteroscedastic diagonal data noise in `EuclideanSpace`
- `BesselSobolev` / `BesselSobolevInverse` block assembly for operator-native prior construction
- `DualMasterCostFunction` + `ProximalBundleMethod` solve for ellipsoidal DLI bounds

**Test Coverage:**
- Total notebook validation checkpoints written: 8
- All tests passing: ✅

**Recommendations for Next Steps:**
- Compare this notebook directly against `synthetic_vp_vs_dli.ipynb` in one side-by-side benchmark notebook if you want a compact pedagogical comparison.
- If you want to move closer to `realistic_dli.ipynb`, swap the synthetic normal-mode kernels for a catalog-backed realistic kernel provider while keeping the same ellipsoid machinery.
