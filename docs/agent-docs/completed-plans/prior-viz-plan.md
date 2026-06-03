## Plan: Prior Visualisation for Full-Spectrum PLI

Add a `PriorViewer` class and ipywidgets interactive panel so that each
subproblem's prior (mean, ±2σ band, sample curves) can be inspected live
while adjusting τ (amplitude) and σ_var (topography variance) sliders.
All expensive eigenpair computation is done once; interactive updates are
pure arithmetic.

**Phases (2)**

1. **Phase 1: PriorViewer (precompute + get_display_data)**
    - **Objective:** Implement `_PriorComponentData` dataclass and `PriorViewer`
      class with `precompute()`, `resample()`, `invalidate_bessel()`, and
      `get_display_data()` driven by TDD.
    - **Files/Functions to Modify/Create:**
        - `intervalinf/demos/old_demos/paper_demos/prior_viz.py` — create
        - `intervalinf/demos/old_demos/paper_demos/tests/test_prior_viz.py` — create
    - **Tests to Write:**
        - `test_prior_viewer_precompute_runs`
        - `test_get_display_data_scales_with_tau`
        - `test_get_display_data_samples_shape`
        - `test_prior_viewer_resample_changes_samples`
    - **Steps:**
        1. Write 4 failing tests → confirm ImportError (red)
        2. Implement `prior_viz.py` → all 4 pass (green)
        3. Run full suite; confirm no regressions

2. **Phase 2: ipywidgets panel + notebook cells**
    - **Objective:** Add `_render_prior_figure` (5-panel matplotlib figure)
      and `make_prior_widget` (ipywidgets interactive viewer) to `prior_viz.py`,
      and add two cells in `example.ipynb`.
    - **Files/Functions to Modify/Create:**
        - `intervalinf/demos/old_demos/paper_demos/prior_viz.py` — extend
        - `intervalinf/demos/old_demos/paper_demos/example.ipynb` — add 2 cells
        - `intervalinf/demos/old_demos/paper_demos/tests/test_prior_viz.py` — extend
        - `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — update
    - **Tests to Write:**
        - `test_render_prior_figure_returns_figure`
        - `test_make_prior_widget_returns_widget`
    - **Steps:**
        1. Write 2 failing tests for figure + widget
        2. Implement `_render_prior_figure` (5 axes: vp, vs-combined, rho, σ₁, block title)
        3. Implement `make_prior_widget` (block dropdown, τ sliders, sigma_var slider,
           Precompute button, Output widget)
        4. Add 2 cells to `example.ipynb` after Phase 3 Bessel-build cell
        5. Update living reference

**Open Questions**
1. Should vs_IC and vs_M be shown in separate panels or combined with a gap?
   → **Combined** with a gap at ICB/CMB boundaries (matches `plot_block_posterior` style).
2. Should τ sliders be flat per-parameter or per-parameter-per-s?
   → **Flat** (single slider per parameter); consistent with `prior_power_spectrum_default`.
3. Layout: 1×5 (landscape) or 2×3?
   → **1×5 landscape** matching `plot_block_posterior`.
