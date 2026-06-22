## Phase 6 Complete: Full Property Posterior Assembly

Implemented `assemble_property_posterior` in `full_spectrum_utils.py`, which
pushes per-block model posteriors through per-block property operators to
produce the full $\mathcal{N}(\mu_P, C_P)$ Gaussian measure on
$\mathbb{R}^{N_p}$. The Phase 6 slice also now includes the notebook/runtime
follow-up needed to make block posterior solves use process-based parallelism
reliably, plus TODO notes documenting the remaining mass-weighted fast-path gap.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_utils.py` — added `assemble_property_posterior`; switched `solve_all_blocks` from thread workers to process workers with per-worker `threadpool_limits(limits=1)`
- `intervalinf/demos/old_demos/paper_demos/tests/test_full_spectrum_utils.py` — 3 new Phase 6 tests (np.all assertion strengthened post-review)
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` — Phase 6 markdown + code cells added; block-solve cell updated for stable parallel execution
- `intervalinf/intervalinf/operators/sola.py` — TODO notes on mass-weighted adjoint corrections required by current fast paths
- `intervalinf/intervalinf/operators/reduced.py` — TODO note on mass-weighted reduced-covariance correction
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — updated to Phases 1–6 and documented the process-based block parallel backend

**Functions created/changed:**
- `assemble_property_posterior(property_op_dict, model_posterior_dict) -> GaussianMeasure` — accumulates mean $\mu_P = \sum_{st} T_{st}(\tilde{m}_{st})$ and covariance $C_P = \sum_{st} T_{st} C^{post}_{st} T^*_{st}$ column-by-column; symmetrizes $C_P$; wraps result in `GaussianMeasure` with dense matrix `LinearOperator` covariance
- `solve_all_blocks(forward_dict, prior_dict, split, n_jobs) -> Dict[BlockIndex, GaussianMeasure]` — serial for `n_jobs=1`; otherwise uses process workers and caps native BLAS/FFT/OpenMP thread pools inside each worker to avoid oversubscription

**Tests created/changed:**
- `test_assemble_property_posterior_mean_matches_pushforward` — mean equals direct sum of T_st applied to block posteriors, rtol=1e-10 ✅
- `test_assemble_property_posterior_cov_symmetric_psd` — C_P symmetric (atol=1e-8) and PSD (min eigval ≥ −1e-8) ✅
- `test_assemble_property_posterior_reduces_uncertainty` — ALL posterior std-devs ≤ prior std-devs (np.all) ✅

**Review Status:** APPROVED (warnings addressed: living reference updated, np.any → np.all)

**Git Commit Message:**
```
feat(paper-demo): assemble property posterior and fix block solve parallelism

- Add assemble_property_posterior in full_spectrum_utils.py and notebook
  Phase 6 cells for mean/std reporting and correlation heatmap diagnostics
- Add 3 Phase 6 tests for pushforward mean, symmetric PSD covariance, and
  uncertainty reduction; strengthen the reduction check from np.any to np.all
- Switch solve_all_blocks to process-based joblib workers with per-worker
  threadpool limits so independent block posteriors scale reliably
- Document the remaining mass-weighted fast-path gap in SOLA/reduced
  operators and refresh the intervalinf living reference

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 6 of 8
Related: intervalinf/docs/agent-docs/completed-plans/full-spectrum-splitting-pli-example-phase-6-complete.md
```
