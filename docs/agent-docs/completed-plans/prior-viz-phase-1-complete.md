## Phase 1 Complete: PriorViewer (precompute + get_display_data)

`PriorViewer` implemented with TDD. Precomputes eigenpair-based reference std
and samples (τ=1) once; `get_display_data` is pure arithmetic thereafter.
All 4 tests green; no regressions against existing 3 viz tests.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/prior_viz.py` — created
- `intervalinf/demos/old_demos/paper_demos/tests/test_prior_viz.py` — created
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_viz.py` — bug fix to `_plot_radial_panel`

**Functions created/changed:**
- `_PriorComponentData` (dataclass) — holds r_grid, ref_std, ref_samples, _eigenvalues, _phi_matrix
- `PriorViewer.__init__` — shared_bessel, specs, n_grid, n_samples, seed
- `PriorViewer.precompute(progress_cb)` — builds phi_matrix, eigenvalues, ref_std, ref_samples for all 4 params
- `PriorViewer.resample(seed)` — redraws samples without rebuilding eigenpairs
- `PriorViewer.invalidate_bessel(new_shared_bessel)` — replaces operators, clears cache
- `PriorViewer.get_display_data(block, *, tau_vp, tau_vs_IC, tau_vs_M, tau_rho, sigma_var)` — returns τ-scaled dict
- `PriorViewer.is_precomputed` — property
- `_plot_radial_panel` (bug fix) — std band now interpolated to dense grid before fill_betweenx

**Tests created/changed:**
- `test_prior_viewer_precompute_runs` — checks shapes and std ≥ 0
- `test_get_display_data_scales_with_tau` — tau_vp=2 doubles std and samples
- `test_get_display_data_samples_shape` — (n_samples, n_grid) shapes; sigma_1 float std
- `test_prior_viewer_resample_changes_samples` — different seed changes samples, not ref_std

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(prior-viz): Phase 1 — PriorViewer precompute + get_display_data

- Add _PriorComponentData dataclass and PriorViewer class in prior_viz.py
- Precomputes eigenpair-based ref std and KL samples at tau=1 for all 4 params
- get_display_data scales by tau (arithmetic only, no operator calls)
- resample() redraws samples without rebuilding eigenpairs
- Add 4 TDD tests (all green, 1.4 s with n_basis=5)
- Fix _plot_radial_panel std-band alignment bug (interp to dense grid)

Plan: intervalinf/docs/agent-docs/active-plans/prior-viz-plan.md
Phase: 1 of 2
Related: intervalinf/docs/agent-docs/active-plans/prior-viz-phase-1-complete.md
```
