## Phase 3 Complete: Per-block prior covariance with shared Bessel-Sobolev operators

Implemented shared Bessel-Sobolev covariance operators built once per radial parameter, scaled by τ_{p,s}² per block, and assembled into a `GaussianMeasure` for each (s, t) block. All Lebesgue model spaces are **basis-free** (`dim=0, basis=None`); `Lebesgue.from_components`/`to_components` were fixed to handle this. Prior domain `dim=2` (Euclidean topography DOFs only) matches M_st from Phase 2.

**Files created/changed:**
- `intervalinf/intervalinf/spaces/lebesgue.py` — added dim=0 early-exit in `from_components` and `to_components`
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_utils.py` — added Phase 3 functions; all Lebesgue model spaces changed to `Lebesgue(0, ..., basis=None)`
- `intervalinf/demos/old_demos/paper_demos/tests/test_full_spectrum_utils.py` — added/rewrote 3 Phase 3 tests to use callable `Function` objects
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` — added Phase 3 markdown + code cells

**Functions created/changed:**
- `prior_power_spectrum_default(p, s) -> float` — flat τ=1.0 default amplitude
- `build_shared_bessel_blocks(specs) -> dict[str, BesselSobolevInverse]` — builds 4 reference covariance operators
- `build_block_prior(s, t, shared_bessel, specs, *, tau_fn, sigma_var) -> GaussianMeasure` — assembles nested block-diagonal prior matching M_st structure
- `_BESSEL_PARAMS` dict — hyperparameters (s_order, length, overall_var, bc) for each parameter from example_2.ipynb
- New imports in `full_spectrum_utils.py`: `BoundaryConditions` (from intervalinf), `Laplacian, BesselSobolevInverse` (from intervalinf.operators), `Function as _IFunction` (from intervalinf.core.functions)

**Tests created/changed:**
- `test_shared_bessel_is_built_once_per_parameter` — asserts keys = {'vp','vs_IC','vs_M','rho'}, each a BesselSobolevInverse
- `test_block_prior_covariance_scales_with_tau` — doubles τ → 4× radial covariance output; sigma components unchanged
- `test_block_prior_is_block_diagonal` — nonzero vp input → zero vs/rho output (off-diagonal blocks = 0)

**Review Status:** APPROVED

**Key design decisions:**
- Prior nested structure matches M_st exactly: `from_direct_sum([prior_functions, prior_euclidean])` with `prior_functions = from_direct_sum([prior_vp, prior_vs=[prior_vs_IC, prior_vs_M], prior_rho])` — required for BlockDiagonalLinearOperator._mapping to zip correctly with the output of G_st.adjoint
- Space reassociation via `_IFunction(shared_op.domain, evaluate_callable=f.__call__)` allows one shared BesselSobolevInverse to serve all 15 blocks
- `tau_sq` is captured at closure creation time (float, immutable) — no stale-closure risk
- `expectation=None` in GaussianMeasure → zero mean throughout

**Git Commit Message:**
```
feat(paper-demo): Phase 3 — shared Bessel-Sobolev block priors (basis-free)

- Add prior_power_spectrum_default, build_shared_bessel_blocks, build_block_prior
- Add _BESSEL_PARAMS hyperparameter dict (from example_2.ipynb)
- All Lebesgue model spaces use basis-free mode (dim=0, basis=None)
- Fix Lebesgue.from_components / to_components for dim=0 early-exit
- 3 Phase 3 tests rewritten to use callable Function objects (no from_components)
- 10/10 tests pass; notebook cells execute clean; prior domain dim=2

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 3 of 8
Related: intervalinf/docs/agent-docs/completed-plans/full-spectrum-splitting-pli-example-phase-3-complete.md
```
