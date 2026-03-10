## Phase 3 Complete: Comparative Diagnostics, Final Polish, and Reproducibility Fix

Phase 3 adds figure export, LP-based width diagnostics, closing markdown cells, and restores the missing synthetic-data cell so the notebook is fully reproducible from a fresh kernel. All Phase 3 code cells execute cleanly.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb`

**Functions created/changed:**
- Cell `#VSC-65b6f733` (cell 9): **Restored** — synthetic data cell that defines `m_bar`, `p_bar`, `d_bar`, `noise_vector`, `d_tilde`, `signal_rms`, `noise_fraction`, `sigma_d`; was accidentally overwritten with a duplicate model-prior cell during user edits between phases
- Cell `#VSC-242f2466` (cell 24): **Edited** — added `savefig` PNG/PDF export block (self-contained: defines `figures_folder` inside the cell)
- Cell `#VSC-5fce6e54` (cell 26): **New** — export setup confirmation cell
- Cell `#VSC-13854eea` (cell 27): **New** — LP-based width diagnostics using `scipy.optimize.linprog` (correct support function via LP, replacing incorrect polar-radial formula); uses `hs.normal_vector`/`hs.offset` attributes; includes width finiteness/non-negativity assertions
- Cell `#VSC-b31abe11` (cell 28): **New** — interpretation markdown (DLI vs BG methodology, region comparison, BG regularisation effect)
- Cell `#VSC-1776b350` (cell 29): **New** — final summary markdown (parameters table, methods table, reproduction instructions)

**Tests created/changed:**
- `assert dli_width_x >= 0 and np.isfinite(dli_width_x)` — DLI x-width validity
- `assert dli_width_y >= 0 and np.isfinite(dli_width_y)` — DLI y-width validity
- `assert bg_width_x >= 0 and np.isfinite(bg_width_x)` — BG x-width validity
- `assert bg_width_y >= 0 and np.isfinite(bg_width_y)` — BG y-width validity
- Containment checks `dli_admissible_region.is_element(p_bar)` and `bg_admissible_region.is_element(p_bar)`

**Diagnostics output (cell 27):**
```
Experiment parameters:  N_d=5,  N_p=2,  N_θ=10

True property:     p_bar  = [-0.1363  0.3637]
BG estimator:      p_t_bg = [-0.0501  0.2857]
||p_t_bg - p_bar|| = 0.1163

DLI solve time: 3.01 s  (134 bundle iterations)

Directional width  (h(+d) + h(-d) = diameter along axis):
  Direction       DLI width     BG width   Ratio BG/DLI
  e1  (x-axis)       2.2290       2.2957         1.0299
  e2  (y-axis)       2.0628       2.1309         1.0330

  x (p_1) [-0.9354, +1.2936]   BG: [-0.9719, +1.3238]
  y (p_2) [-0.3106, +1.7522]   BG: [-0.3479, +1.7830]

p_bar in DLI admissible region: True
p_bar in BG  admissible region: True
```

**Review Status:** APPROVED (after two revision cycles — incorrect support formula replaced with LP, missing synthetic data cell restored)

**Git Commit Message:**
```
feat(demos): add comparative diagnostics and final polish

- Restore synthetic data cell (m_bar, p_bar, d_bar, noise_vector, d_tilde)
  that was accidentally overwritten; notebook is now fully reproducible
- Add LP-based width diagnostics (scipy linprog) replacing incorrect
  polar-radial formula; widths for DLI and BG in cardinal directions
- Add savefig export of comparison figure (PNG + PDF)
- Add interpretation and final summary markdown cells

Plan: intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-plan.md
Phase: 3 of 3
Related: intervalinf/docs/agent-docs/active-plans/dli-vs-bg-polyhedral-comparison-phase-3-complete.md
```
