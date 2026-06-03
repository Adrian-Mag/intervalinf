## Plan Complete: Prior Visualisation for Full-Spectrum PLI

`PriorViewer` precomputes eigenpair-based reference std and KL samples at τ=1
for all four radial parameters.  `_render_prior_figure` renders a 5-panel
matplotlib figure (δvp, δvs_IC, δvs_M, δρ, σ₁) with ±1σ/±2σ bands and sample
curves.  `make_prior_widget` wraps everything in an ipywidgets interactive panel
with block dropdown, four τ sliders, σ_var log-slider, Precompute + Resample
buttons.  All 30 tests pass; notebook updated with Phase 3.5 cells.

**Phases Completed:** 2 of 2
1. ✅ Phase 1: PriorViewer (precompute + get_display_data)
2. ✅ Phase 2: Widget + notebook cells

**All Files Created/Modified:**
- `intervalinf/demos/old_demos/paper_demos/prior_viz.py` — created (Phase 1) + extended (Phase 2)
- `intervalinf/demos/old_demos/paper_demos/tests/test_prior_viz.py` — created (6 tests)
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_viz.py` — `_plot_radial_panel` bug fix
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` — 3 cells added (Phase 3.5)
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — updated

**Key Functions/Classes Added:**
- `_PriorComponentData` dataclass
- `PriorViewer.__init__` / `.precompute()` / `.resample()` / `.invalidate_bessel()` / `.get_display_data()`
- `_render_prior_figure(display_data, block, specs)`
- `make_prior_widget(viewer, blocks, specs)`
- `_plot_prior_panel` / `_plot_prior_sigma1_panel` helpers

**Test Coverage:**
- Total tests written: 6 (prior_viz) + existing 3 (viz) + 16 (utils) + 5 (other) = 30 total
- All tests passing: ✅ (30/30, 11 min full suite)

**Recommendations for Next Steps:**
- Consider adding a `plot_prior_panel_combined_vs` helper that shows IC + mantle on one axis with an outer-core gap (mirrors `plot_block_posterior` combined vs panel)
- For large n_basis (>200), the `precompute()` eigenfunction loop could be parallelised with joblib
