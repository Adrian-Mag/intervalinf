## Phase 2 Complete: Widget + notebook cells

`_render_prior_figure` (5-panel matplotlib figure) and `make_prior_widget`
(ipywidgets interactive viewer with instant τ-scaling updates) added to
`prior_viz.py`.  Two tests added (6/6 green).  Three cells inserted into
`example.ipynb` after the Phase 3 shared-Bessel build cell.  Living reference
updated.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/prior_viz.py` — appended `_render_prior_figure`, `make_prior_widget`, helpers `_plot_prior_panel`, `_plot_prior_sigma1_panel`, colour/label dicts
- `intervalinf/demos/old_demos/paper_demos/tests/test_prior_viz.py` — added 2 Phase 2 tests
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` — inserted Phase 3.5 markdown + 2 code cells (PriorViewer create+precompute; make_prior_widget display)
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — updated with Phase 8 entries

**Functions created/changed:**
- `_PARAM_COLORS` / `_PARAM_LABELS` — colour and label look-up dicts
- `_plot_prior_panel(ax, r, std, samples, label, color, n_sigma)` — draws ±nσ bands + sample curves + dashed zero-mean
- `_plot_prior_sigma1_panel(ax, std, samples, color, n_sigma)` — scatter + horizontal bands for scalar σ₁
- `_render_prior_figure(display_data, block, specs, *, n_sigma, figsize)` — 5-panel 1×5 Figure
- `make_prior_widget(viewer, blocks, specs, *, tau_init, sigma_var_init)` — ipywidgets VBox with dropdown, τ sliders, σ_var log-slider, Precompute + Resample buttons, Output widget

**Tests created/changed:**
- `test_render_prior_figure_returns_figure` — checks Figure with 5 axes
- `test_make_prior_widget_returns_widget` — checks ipywidgets Widget return

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(prior-viz): Phase 2 — _render_prior_figure + make_prior_widget + notebook cells

- Add _render_prior_figure: 5-panel 1x5 matplotlib figure (vp/vs_IC/vs_M/rho/sigma_1)
- Add make_prior_widget: ipywidgets VBox with block dropdown, tau sliders,
  sigma_var log-slider, Precompute/Resample buttons; instant redraw from cache
- Add helper functions: _plot_prior_panel, _plot_prior_sigma1_panel
- Add 2 Phase 2 TDD tests (all 6 tests green, 1.6 s)
- Insert Phase 3.5 markdown + 2 code cells in example.ipynb
- Update intervalinf living reference with Phase 8 entries

Plan: intervalinf/docs/agent-docs/active-plans/prior-viz-plan.md
Phase: 2 of 2
Related: intervalinf/docs/agent-docs/active-plans/prior-viz-phase-2-complete.md
```
