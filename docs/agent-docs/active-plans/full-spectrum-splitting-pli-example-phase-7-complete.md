## Phase 7 Complete: Visualisation and Reporting

Four headline visualisation functions were implemented in `full_spectrum_viz.py` and four corresponding cells were appended to `example.ipynb`. Three smoke tests cover `plot_block_posterior`, `plot_cmb_map`, and the gated `plot_equatorial_slice`; the block-posterior test is fully self-contained (no external data files, ~4 s runtime). A code-review cycle caught a dead-figure artifact and a duplicate panel; both were fixed before finalising.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_viz.py` (new)
- `intervalinf/demos/old_demos/paper_demos/tests/test_full_spectrum_viz.py` (new)
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` (4 Phase 7 cells appended)
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` (updated)

**Functions created/changed:**
- `plot_block_posterior` — 5-panel figure: {δvp, δvs-IC, δvs-M, δρ, σ₁}; ±1σ band from posterior-covariance probing with `n_probes=20` normalised Gaussian bumps per component
- `plot_cmb_map` — Mollweide CMB topography map via pyshtools synthesis (Cartopy fallback to imshow)
- `plot_equatorial_slice` — gated volumetric equatorial slice; returns `None` when `enabled=False`
- `plot_property_posterior_summary` — 2-panel figure: mean ± 1σ bar chart + N_p × N_p correlation heatmap

**Tests created/changed:**
- `test_plot_block_posterior_returns_figure` (self-contained: builds prior without external data)
- `test_plot_cmb_map_returns_figure` (synthetic `_FakePosterior`)
- `test_plot_equatorial_slice_gated_off_by_default`

**Review Status:** APPROVED (after review-cycle fixes: removed dead first-figure block, corrected panel layout from duplicate vs-IC to clean {vp, vs-IC, vs-M, ρ, σ₁}, made block-posterior test self-contained)

**Git Commit Message:**
```
feat(paper-demo): Phase 7 — headline visualisation figures

- Add full_spectrum_viz.py with four plotting helpers:
  plot_block_posterior (5-panel radial posterior with ±1σ covariance probing),
  plot_cmb_map (pyshtools SH synthesis + Cartopy Mollweide),
  plot_equatorial_slice (gated off by default, returns None when disabled),
  plot_property_posterior_summary (bar chart + correlation heatmap)
- Add tests/test_full_spectrum_viz.py with 3 self-contained smoke tests
  (block-posterior test uses build_block_prior, no external data, ~4 s)
- Append 4 Phase 7 notebook cells to example.ipynb
- Update intervalinf living reference

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 7 of 8
Related: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-phase-7-complete.md
```
