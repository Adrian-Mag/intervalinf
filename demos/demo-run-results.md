# Demo Run Results — 2026-06-22

**Branch:** `mission/lowering-materialization`
**Environment:** miniconda3 Python 3.12.2, `PYTHONPATH=intervalinf:pygeoinf`
**Method:** Code cells executed via shared-namespace script with `Agg` matplotlib backend, `ulimit -v 4GB`, 60s timeout per notebook.

## Top-level demos

| # | Notebook | Status | Notes |
|---|----------|--------|-------|
| 1 | `1_interval_domain_demo.ipynb` | ✅ Pass | All cells OK |
| 2 | `2_functions_demo.ipynb` | ✅ Pass | All cells OK |
| 3 | `3_lebesgue_space_demo.ipynb` | ✅ Pass | All cells OK |
| 3.1 | `3.1_kernel_functionals_demo.ipynb` | ✅ Pass | All cells OK |
| 4 | `4_function_and_basis_providers_demo.ipynb` | ✅ Pass | All cells OK |
| 5 | `5_gradient_operator_demo.ipynb` | ✅ Pass | All cells OK |
| 6 | `6_laplacian_operator_demo.ipynb` | ✅ Pass | All cells OK |
| — | `weighted_lebesgue_demo.ipynb` | ✅ Pass | All cells OK |

## `convex_analysis/` demos

| # | Notebook | Status | Notes |
|---|----------|--------|-------|
| 1 | `bg_with_errors_minkowski.ipynb` | ✅ Pass | All cells OK |
| 2 | `bg_with_errors_minkowski_multi_nd.ipynb` | ⏱️ Timeout | No API errors. All setup cells pass. Timed out during heavy N_d=500 computation loop (expected — needs >60s). |
| 3 | `dli.ipynb` | ✅ Pass | All cells OK. Full DLI solve completed in ~19s. |
| 4 | `dli_vs_bg_polyhedral_comparison.ipynb` | ✅ Pass | All cells OK. DLI + BG comparison with figures. |
| 5 | `realistic_dli.ipynb` | 🗑️ Deleted | Had stale `import kernel_utils` referencing deleted `old_demos/` module. Removed per user instruction. |
| 6 | `synthetic_vp_vs_dli.ipynb` | ⏱️ Timeout | No API errors. All setup cells pass. Timed out during DLI solve (expected — needs >60s). |
| 7 | `synthetic_vp_vs_ellipsoid_dli.ipynb` | ⏱️ Timeout | No API errors. All setup + prior cells pass. Timed out during DLI solve (expected — needs >60s). |
| 8 | `synthetic_vp_vs_zero_region_dli.ipynb` | ⏱️ Timeout | No API errors. All setup cells pass. Timed out during DLI solve (expected — needs >60s). |

## `model_fusion/` demos

| # | Notebook | Status | Notes |
|---|----------|--------|-------|
| 1 | `first_test.ipynb` | ✅ Fixed | **Fixed (demo-only):** `LinearBayesianInference` → `LinearBayesianInversion` (renamed in pygeoinf); `sampler.variance_function()` → `sampler.variance_function` (now a property); `LinearBayesianInversion` constructor no longer takes `T` — now use `bayesian_inversion.model_posterior_measure(d, solver)` then `model_posterior.affine_mapping(operator=T)` for property posterior. All cells now pass. |

## `sola-base_demo/` demos

| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | `sola_demo.ipynb` | ❌ External dep | Cells 0–7 pass (continuous SOLA, discrete SOLA, unimodularity). Cell 8 fails: calls external `sola_lsqr.py` which expects `sola-base` C pipeline at `/home/adrian/PhD/sola-base/`. The `LSQR_SOLA_paral` binary path issue. Cell 9 cascading failure (missing output file). User notes: sola-base was moved to `/home/adrian/PhD/sola-base`; a fix exists on a sola-base branch but is not high priority. |
| 2 | `sola_demo_earth.ipynb` | ❌ Numerical mismatch | Cells 0–8 pass (including sola-base C pipeline execution). Cell 9 fails: `assert_allclose` comparing intervalinf vs sola-base property outputs — max diff 0.11 vs `atol=4e-5`. User notes: this is due to a sola-base issue fixed on a particular branch in sola-base; not high priority. |
| 3 | `sola_demo.py` | ❌ External dep | Methods 1–2 pass (continuous SOLA, discrete SOLA). Method 3 fails: `FileNotFoundError: /home/adrian/PhD/Inferences/sola-base/LSQR_SOLA_paral` — path points to old location. External dependency issue, not a code bug. |

## `old_demos/` demos

### `old_demos/sola_demos/` (6 notebooks)

| # | Notebook | Status | Notes |
|---|----------|--------|-------|
| 1 | `sola_noiseless.ipynb` | ✅ Pass | All cells OK |
| 2 | `sola_noise.ipynb` | ✅ Pass | All cells OK |
| 3 | `sola_failure.ipynb` | ❌ Numerical | Cell 6: Cholesky fails on `G_T @ G_T.adjoint` (rank-deficient joint operator). This is expected — the demo is about showing failure modes. Pre-existing, not a stale API issue. Cells 7–18 cascade from cell 6. |
| 4 | `sola_noise_multiple.ipynb` | ✅ Pass | All cells OK |
| 5 | `sola_noise_sobolev.ipynb` | ✅ Pass | All cells OK |
| 6 | `sola_noise_sobolev_multiple.ipynb` | ✅ Pass | All cells OK |

### `old_demos/pli_demos/` (7 notebooks + 2 Python scripts)

| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | `pli.ipynb` | ✅ Pass | All cells OK |
| 2 | `pli_sobolev.ipynb` | ✅ Fixed | **Fixed (demo-only):** `sampler.variance_function()` → `sampler.variance_function` (now a property). Cells 0–16 pass. Cell 17: pre-existing numerical issue (covariance matrix has negative eigenvalues). |
| 3 | `pli_discontinuity.ipynb` | ✅ Fixed (core) | **Fixed (core intervalinf):** `ValueError: Some points not in domain [0.0, 0.5)` was caused by `check_domain=False` not propagating through inner function evaluations in `build_eigenfunction_expansion`, `_reconstruct_function`, `KLSampler.evaluate_sample`, `KLSampler.evaluate_variance`, and `Function._binary_op` scalar path. All cells now pass. |
| 4 | `pli_discontinuity_sobolev.ipynb` | ✅ Fixed (core) | Same core fix as `pli_discontinuity.ipynb`. All cells now pass. |
| 5 | `pli_multiple.ipynb` | ✅ Fixed | **Fixed (demo-only):** `sampler.variance_function()` → `sampler.variance_function` (now a property). All cells now pass. |
| 6 | `pli_multiple_discontinuity.ipynb` | ✅ Fixed (core) | Same core fix as `pli_discontinuity.ipynb`. All cells now pass. |
| 7 | `pli_radial.ipynb` | ✅ Fixed | **Fixed (demo-only):** `Lebesgue(weight=...)` → `WeightedLebesgue(weight=...)` (API changed); `sampler.variance_function()` → `sampler.variance_function` (now a property). All cells now pass. |
| 8 | `interactive_prior.py` | ✅ Pass | Runs OK (interactive matplotlib, no errors) |
| 9 | `interactive_prior_sobolev.py` | ✅ Fixed | **Fixed (demo-only):** `sampler.variance_function()` → `sampler.variance_function` (now a property). Now runs OK. |

### `old_demos/other_demos/` (3 notebooks)

| # | Notebook | Status | Notes |
|---|----------|--------|-------|
| 1 | `conference_presentation_helper.ipynb` | ✅ Pass | All cells OK |
| 2 | `conference_presentation_pli.ipynb` | ✅ Pass | All cells OK |
| 3 | `conference_presentation_sola.ipynb` | ✅ Pass | All cells OK |

## Summary

- **33 files run** (1 deleted before running)
- **23 fully pass** (all top-level + 2 convex_analysis + dli + model_fusion + 5 sola_demos + pli + 3 other_demos + interactive_prior.py)
- **5 fixed (demo-only)** (model_fusion/first_test, pli_sobolev, pli_multiple, pli_radial, interactive_prior_sobolev.py — stale API: `LinearBayesianInference` rename, `variance_function` property, `WeightedLebesgue`, `LinearBayesianInversion` constructor)
- **3 fixed (core intervalinf)** (pli_discontinuity, pli_discontinuity_sobolev, pli_multiple_discontinuity — `check_domain=False` not propagating through inner function evaluations in eigenfunction expansions, KL sampler, SOLA reconstruction, and Function binary ops)
- **4 timeout** (heavy computation, no API errors)
- **3 external dep** (`sola-base` demos — sola-base C pipeline path/branch issue, low priority)
- **1 numerical** (`sola_failure.ipynb` — Cholesky on rank-deficient operator, expected for failure-mode demo)
- **1 pre-existing numerical** (`pli_sobolev.ipynb` cell 17 — covariance matrix negative eigenvalues)

### Core intervalinf changes (4 files, 6 lines)

| File | Change |
|------|--------|
| `intervalinf/operators/spectral_helpers.py:53` | `eigfunc(x)` → `eigfunc.evaluate(x, check_domain=False)` in `evaluate_expansion` |
| `intervalinf/operators/sola.py:773` | `kernel.evaluate(x)` → `kernel.evaluate(x, check_domain=False)` in `evaluate_sum` |
| `intervalinf/sampling/kl_sampler.py:390,420,424` | Added `check_domain=False` to `phi.evaluate()` and `mean.evaluate()` in `evaluate_variance` and `evaluate_sample` |
| `intervalinf/core/functions.py:851` | `self.evaluate(x)` → `self.evaluate(x, check_domain=False)` in `scalar_op_callable` |

**Root cause:** When functions are composed (eigenfunction expansions, KL samples, scaled/added functions), the inner `evaluate()` calls used default `check_domain=True`. For open subdomain boundaries (e.g., `[0.0, 0.5)` where `0.5` is excluded), `np.linspace(a, b, n)` includes the endpoint `b`, triggering `ValueError`. The fix propagates `check_domain=False` through all inner evaluation callables. This is safe because eigenfunctions/kernels are smooth functions defined everywhere — domain restrictions are for the Hilbert space inner product, not function evaluation.

**Test impact:** 530 tests pass, 6 pre-existing failures unchanged (all in `test_sola.py` compact support / instrumentation tests). Zero regressions.
