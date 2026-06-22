## Phase 2 Complete: Per-block radial model space and forward operator

`RadialSpecs`, `make_radial_space`, and `build_block_forward` added to `full_spectrum_utils.py`. Each (s,t) block now gets a full model space (vp ⊕ vs_IC ⊕ vs_M ⊕ rho ⊕ σ_ICB ⊕ σ_CMB), SOLA-based forward operator, and diagonal noise covariance, following the exact `example_2.ipynb` pattern.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_utils.py` (Phase 2 additions)
- `intervalinf/demos/old_demos/paper_demos/tests/test_full_spectrum_utils.py` (4 new tests)
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` (Phase 2 notebook cell added)

**Functions created/changed:**
- `RadialSpecs` — mutable dataclass with shared config (n_basis, radii, integration/parallel configs, sensible defaults in `__post_init__`)
- `make_radial_space(n, domain, weight, basis, ...)` — returns `Lebesgue`; raises `NotImplementedError` for `weight != 1.0`
- `build_block_forward(s, t, reg_st, catalog, specs)` — returns `(G_st, C_D_st, M_st, D_st)`

**Tests created/changed:**
- `test_make_radial_space_unit_weight_matches_lebesgue`
- `test_make_radial_space_nonunit_weight_raises`
- `test_block_forward_shapes`
- `test_block_forward_noise_covariance_diagonal`

**Post-review fixes applied:**
- Removed duplicate Phase 1 code (BlockIndex, enumerate_blocks, block_data_split) that Sisyphus had appended; file is now 375 lines with no duplicate definitions.
- `frozen=True` on `RadialSpecs` was considered but reverted — incompatible with `__post_init__` field assignment; documented as known deviation.

**Review Status:** APPROVED (after critical duplicate-removal fix)

**Git Commit Message:**
```
feat(paper-demos): Phase 2 — per-block radial model space and forward operator

- Add RadialSpecs dataclass (shared integration/parallel config)
- Add make_radial_space: Lebesgue wrapper with weight guard
- Add build_block_forward: returns (G_st, C_D_st, M_st, D_st)
- 4 new tests, all 7 passing (2.24s)
- Fix: remove duplicate Phase 1 definitions appended by Sisyphus

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 2 of 8
Related: intervalinf/docs/agent-docs/completed-plans/full-spectrum-splitting-pli-example-phase-2-complete.md
```
