## Phase 5 Complete: Property Operators via SH Projection

Added `property_targets.py` and `build_property_operator` to assemble per-block `LinearOperator` objects
mapping each block's model space $M_{st}$ to $\mathbb{R}^{N_p}$. Targets are specified as `BulkTarget`
or `CMBTarget` dataclasses with angular and radial localization parameters.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/property_targets.py` (NEW)
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_utils.py` (Phase 5 section added)
- `intervalinf/demos/old_demos/paper_demos/tests/test_property_targets.py` (NEW)
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` (Phase 5 cells added)
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` (updated)

**Functions created/changed:**
- `extract_st_coeff(coeffs_array, s, t)` — Dahlen & Tromp t-index → pyshtools coefficient lookup
- `angular_bump_coeffs(sigma_deg, s_max, lmax_grid)` — Gaussian cap at pole via DH grid
- `rotate_bump_coeffs(coeffs_array, lat_deg, lon_deg)` — ZYZ rotation with pyshtools
- `radial_bump(r_km, r0_km, sigma_r_km)` — Gaussian radial weight
- `BulkTarget` (dataclass: param, lat_deg, lon_deg, sigma_ang_deg, r0_km, sigma_r_km)
- `CMBTarget` (dataclass: lat_deg, lon_deg, sigma_ang_deg)
- `build_block_property_coeffs(targets, blocks, s_max)` — pre-computes B_st per (block, target) with pole-template cache
- `build_property_operator(targets, blocks, forward_dict, specs, s_max, n_radial)` — assembles `dict{BlockIndex: LinearOperator}`

**Tests created/changed:**
- `test_angular_bump_coeffs_axisymmetric_at_pole` — m>0 coefficients ≈ 0 for pole bump
- `test_rotation_preserves_power` — Parseval identity under rotation (rtol=1e-6)
- `test_radial_bump_peak_location` — peak at r0, value ≈ 1.0
- `test_property_operator_shape` — forward output shape (N_p,) for all blocks
- `test_property_operator_forward_cmb_target` — T(m)[i] = B_st * sigma_1 for CMB target
- `test_property_operator_adjoint_consistency` — `<T(m),λ> = <m,T*(λ)>` to rtol=1e-3
- `test_property_operator_vs_target` — vs IC/M subdomain adjoint path; non-zero IC component for r0=500 km

**Review Status:** APPROVED (all 3 review issues resolved: living reference updated, redundant import removed, vs-target test added)

**Git Commit Message:**
```
feat(paper-demo): add property operators via SH projection (Phase 5)

- Add property_targets.py: extract_st_coeff, angular_bump_coeffs,
  rotate_bump_coeffs, radial_bump, BulkTarget, CMBTarget,
  build_block_property_coeffs (pole-template cache on sigma_ang_deg)
- Add build_property_operator in full_spectrum_utils.py: per-block
  LinearOperator with forward + adjoint for bulk (vp/vs/rho) and
  CMB targets; vs uses IC and M subdomain grids separately
- 7 new tests (test_property_targets.py): axisymmetry, Parseval,
  radial peak, shape, CMB forward, adjoint consistency, vs subdomains;
  all 7 pass (2.03s)
- Add Phase 5 notebook cells: 8 property targets, build_property_operator
- Update intervalinf living reference with Phase 5 entries

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 5 of 8
Related: intervalinf/docs/agent-docs/completed-plans/full-spectrum-splitting-pli-example-phase-5-complete.md
```
