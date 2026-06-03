# Target Kernels Refactor Plan

Replace the Gaussian angular/radial target functions in `paper_demos` with
compact-support alternatives: spherical-cap indicators on S² and C∞ / boxcar
radial kernels. The Gaussian approach has infinite support and is a poor
approximation to the localised averages that the inversion is meant to produce.

---

## Scope

Four files change. No new packages are needed. No pygeoinf or intervalinf
core code changes.

| File | Change type |
|---|---|
| `utils/property_targets.py` | Full replacement of public API |
| `utils/full_spectrum_utils.py` | Update `build_property_operator` |
| `visualization/target_kernel_viz.py` | Update viewer methods + defaults |
| `tests/test_property_targets.py` | Replace Gaussian tests with new ones |

One additional file needs a minor touch:
`utils/prior_sensitivity_study.py` — direct target construction, update to new types.

---

## What Already Exists (Do Not Reinvent)

### pygeoinf — `symmetric_space/sphere.py`

`spherical_cap_integral(center, angular_radius, *, normalize)` and
`spherical_cap_average(center, angular_radius)` already compute the **exact**
SH expansion of the cap indicator via `pyshtools.SHCoeffs.from_cap`. The
result is a `LinearForm` whose `.components` vector is the representation in
the truncated SH basis.

We do **not** need to construct a `Lebesgue` space to use this — `from_cap` can
be called directly. `property_targets.py` already imports pyshtools directly
and this stays the same.

Key rescaling already established by `spherical_cap_integral`:
```
from_cap(..., normalization='4pi') / (4π)  →  unit-surface-integral coefficients
```
i.e. the zonal coefficient `c₀₀ = 1/(4π)`, matching the test
`test_normalized_angular_bump_has_expected_zonal_mass` which verifies this for
the Gaussian case. The cap must satisfy the same invariant.

**No rotation step needed.** `from_cap` accepts `clat=lat_deg, clon=lon_deg`
directly and places the cap at the right location. The current pole-template
+ `SHCoeffs.rotate` approach is only necessary because the Gaussian template is
built at the north pole.

### intervalinf — providers/functions/

`BumpFunctionProvider` (C∞ compact support) and `BoxCarFunctionProvider`
(piecewise constant) both exist and work correctly. They normalise to
`∫ f dr = 1` (Lebesgue measure). We need `∫ a(r) r² dr = 1` (spherical
volume measure). Rather than wrapping the providers through this measure
mismatch, it is cleaner to implement the two radial kernels directly in
`property_targets.py` (the C∞ formula is five lines). The providers remain
available for other uses; they are not a dependency of the target system.

### Low-level utilities to keep unchanged

`extract_st_coeff(coeffs_array, s, t)` — already correct for any SH coefficient
array in 4π normalization. Keep it verbatim.

---

## What Gets Removed

From `property_targets.py`:

| Symbol | Reason |
|---|---|
| `angular_bump_coeffs` | Gaussian, infinite support, replaced by `angular_cap_coeffs` |
| `normalized_angular_bump_coeffs` | Same |
| `rotate_bump_coeffs` | Unnecessary with `from_cap` |
| `radial_bump` | Gaussian, replaced |
| `normalized_radial_bump` | Same |
| `BulkTarget` | Replaced by `CapBulkTarget` / `BoxcarBulkTarget` |
| `CMBTarget` | Replaced by `CapCMBTarget` |

---

## New API: `property_targets.py`

### Angular kernel factory

```python
def angular_cap_coeffs(
    lat_deg: float,
    lon_deg: float,
    cap_radius_deg: float,
    s_max: int,
) -> np.ndarray:
    """
    SH coefficients of the spherical-cap indicator, normalised to unit surface
    integral (∫ a dΩ = 1), in 4π-normalised real-harmonic convention.

    Uses pyshtools.SHCoeffs.from_cap placed directly at (lat_deg, lon_deg).
    No pole template, no rotation.  Exact in the truncated basis at degree
    s_max.

    Returns
    -------
    np.ndarray, shape (2, s_max+1, s_max+1)
        Coefficient array compatible with extract_st_coeff and SHCoeffs.from_array.
    """
    coeffs = sh.SHCoeffs.from_cap(
        cap_radius_deg, s_max,
        clat=lat_deg, clon=lon_deg,
        normalization='4pi', csphase=1, kind='real', degrees=True,
    )
    return coeffs.coeffs.copy() / (4.0 * np.pi)
```

**Invariant (testable):** For a full-sphere cap (radius 180°), the zonal
coefficient must equal `1/(4π)`, matching the Gaussian normalization test.

**Caching in `build_block_property_coeffs`:** cache key is
`(lat_deg, lon_deg, cap_radius_deg)` — unlike Gaussian where the key was
only `sigma_ang_deg` (the pole template) and rotation was applied per target.
Each cap target calls `from_cap` once; the result is reused across all blocks.

### Radial kernel factories

```python
def normalized_radial_boxcar(
    r_km: np.ndarray,
    r_low_km: float,
    r_high_km: float,
    *,
    normalization_r_km: np.ndarray | None = None,
) -> np.ndarray:
    """
    Boxcar indicator on [r_low, r_high], normalised so ∫ a(r) r² dr = 1.
    True compact support. Normalization grid defaults to the evaluation grid.
    """

def normalized_radial_bump_compact(
    r_km: np.ndarray,
    r0_km: float,
    width_km: float,
    *,
    normalization_r_km: np.ndarray | None = None,
) -> np.ndarray:
    """
    C∞ bump with hard support [r0 - width/2, r0 + width/2], normalised so
    ∫ a(r) r² dr = 1.

    Uses the standard mollifier: exp(t²/(t²-1)) where t = 2(r-r0)/width,
    exactly zero for |t| ≥ 1.  Normalization grid defaults to the evaluation
    grid.
    """
```

The boxcar normalisation is `1 / ∫_{r_low}^{r_high} r² dr` — analytic:
`1 / ((r_high³ - r_low³) / 3)`.  The bump normalisation is numerical via
`np.trapezoid` on the normalization grid (same pattern as the current
`normalized_radial_bump`).

### New target dataclasses

```python
@dataclasses.dataclass(frozen=True)
class CapBulkTarget:
    """Spherical-cap average × C∞ compact radial bump.

    T(m) = [∫ cap(ξ) dΩ] × [∫ bump(r) m_param(r) r² dr]
    """
    param: str           # 'vp', 'vs', or 'rho'
    lat_deg: float
    lon_deg: float
    cap_radius_deg: float
    r0_km: float         # bump centre
    width_km: float      # full support width (bump is zero outside ±width/2)


@dataclasses.dataclass(frozen=True)
class BoxcarBulkTarget:
    """Spherical-cap average × radial boxcar.

    T(m) = [∫ cap(ξ) dΩ] × [∫_{r_low}^{r_high} m_param(r) r² dr / vol]
    """
    param: str
    lat_deg: float
    lon_deg: float
    cap_radius_deg: float
    r_low_km: float
    r_high_km: float


@dataclasses.dataclass(frozen=True)
class CapCMBTarget:
    """Spherical-cap average of CMB topography.

    T(σ_CMB) = ∫ cap(ξ) σ_CMB(ξ) dΩ  (cap normalised to unit surface integral)
    """
    lat_deg: float
    lon_deg: float
    cap_radius_deg: float
```

### Updated `build_block_property_coeffs`

```python
def build_block_property_coeffs(targets, blocks, s_max) -> dict:
    # Cache full coefficient array per unique (lat, lon, cap_radius)
    cap_cache: dict = {}

    def _get_cap_coeffs(target):
        key = (target.lat_deg, target.lon_deg, target.cap_radius_deg)
        if key not in cap_cache:
            cap_cache[key] = angular_cap_coeffs(
                target.lat_deg, target.lon_deg, target.cap_radius_deg, s_max
            )
        return cap_cache[key]

    result = {}
    for block in blocks:
        coeffs_vec = np.zeros(len(targets))
        for i, target in enumerate(targets):
            coeffs = _get_cap_coeffs(target)
            coeffs_vec[i] = extract_st_coeff(coeffs, block.s, block.t)
        result[block] = coeffs_vec
    return result
```

Note: all three new target types have `lat_deg`, `lon_deg`, `cap_radius_deg`,
so dispatch is not needed for the angular part.

---

## Changes: `full_spectrum_utils.py` — `build_property_operator`

Update the import block and the radial-factor precomputation loop:

```python
from property_targets import (
    CapBulkTarget,
    BoxcarBulkTarget,
    CapCMBTarget,
    build_block_property_coeffs,
    normalized_radial_boxcar,
    normalized_radial_bump_compact,
)
```

Replace the `isinstance(target, BulkTarget)` dispatch with:

```python
for target in targets:
    if isinstance(target, CapBulkTarget):
        radial_fn = normalized_radial_bump_compact
        kwargs = dict(r0_km=target.r0_km, width_km=target.width_km)
    elif isinstance(target, BoxcarBulkTarget):
        radial_fn = normalized_radial_boxcar
        kwargs = dict(r_low_km=target.r_low_km, r_high_km=target.r_high_km)
    else:  # CapCMBTarget
        a_full.append(np.zeros_like(r_full))
        a_IC.append(np.zeros_like(r_IC))
        a_M.append(np.zeros_like(r_M))
        continue
    # evaluate on full, IC, and M grids with shared normalization grid
    a_full.append(radial_fn(r_full, **kwargs, normalization_r_km=r_full))
    a_IC.append(radial_fn(r_IC, **kwargs, normalization_r_km=r_full))
    a_M.append(radial_fn(r_M, **kwargs, normalization_r_km=r_full))
```

The `forward_fn` and `adjoint_fn` closures inside `_make_operator` are
unchanged in structure — they already access `_a_full[i]`, `_a_IC[i]`,
`_a_M[i]` via index, not via target type.

Update the isinstance check in forward and adjoint from `BulkTarget`/
`CMBTarget` to `(CapBulkTarget, BoxcarBulkTarget)` / `CapCMBTarget`.

---

## Changes: `visualization/target_kernel_viz.py`

### `TargetKernelViewer.__init__`

Add:
```python
self._full_coeffs: List[np.ndarray] = self._precompute_full_coeffs(targets, s_max)
```

New private method:
```python
def _precompute_full_coeffs(self, targets, s_max):
    from property_targets import angular_cap_coeffs
    result = []
    cache = {}
    for t in targets:
        key = (t.lat_deg, t.lon_deg, t.cap_radius_deg)
        if key not in cache:
            cache[key] = angular_cap_coeffs(t.lat_deg, t.lon_deg, t.cap_radius_deg, s_max)
        result.append(cache[key])
    return result
```

### `geographic_map`

Replace the `normalized_angular_bump_coeffs` + `rotate_bump_coeffs` block with:

```python
coeffs_rot = self._full_coeffs[target_idx].copy()
```

The "blocks" reconstruction path is unchanged (it masks `coeffs_rot` using
`_block_coeffs`).

Remove imports of `normalized_angular_bump_coeffs` and `rotate_bump_coeffs`
from this method.

### `radial_profile`

Replace:
```python
from property_targets import BulkTarget, normalized_radial_bump
target = self._targets[target_idx]
if not isinstance(target, BulkTarget):
    raise TypeError(...)
a_r = normalized_radial_bump(r_km, target.r0_km, target.sigma_r_km, ...)
```

With:
```python
from property_targets import (
    CapBulkTarget, BoxcarBulkTarget,
    normalized_radial_bump_compact, normalized_radial_boxcar,
)
target = self._targets[target_idx]
if isinstance(target, CapBulkTarget):
    a_r = normalized_radial_bump_compact(r_km, target.r0_km, target.width_km,
                                         normalization_r_km=self._radial_norm_grid)
elif isinstance(target, BoxcarBulkTarget):
    a_r = normalized_radial_boxcar(r_km, target.r_low_km, target.r_high_km,
                                   normalization_r_km=self._radial_norm_grid)
else:
    raise TypeError(f"targets[{target_idx}] is a {type(target).__name__}, "
                    "not a bulk target — no radial profile available.")
```

### `_make_default_targets`

Replace with new types. Example:
```python
def _make_default_targets(specs, s_max):
    cmb_r = specs.cmb_radius_km
    icb_r = specs.icb_radius_km
    earth_r = specs.earth_radius_km

    return [
        # Mantle vp: cap 15° radius, bump 400 km wide in mid-mantle
        CapBulkTarget("vp",   0.0,    0.0, 15.0, (cmb_r + earth_r) / 2, 400.0),
        CapBulkTarget("vp",  30.0,   60.0, 15.0, (cmb_r + earth_r) / 2, 400.0),
        CapBulkTarget("vp", -30.0, -120.0, 15.0, (cmb_r + earth_r) / 2, 400.0),
        # Mantle vs: boxcar covering D'' layer
        BoxcarBulkTarget("vs",  0.0,   0.0, 15.0, cmb_r, cmb_r + 300.0),
        BoxcarBulkTarget("vs", 45.0,  90.0, 15.0, cmb_r, cmb_r + 300.0),
        # Inner core vp: bump centred at half-IC radius
        CapBulkTarget("vp",   0.0,    0.0, 20.0, icb_r / 2, 300.0),
        # CMB topography caps
        CapCMBTarget(  0.0,    0.0, 15.0),
        CapCMBTarget( 30.0,   60.0, 15.0),
        CapCMBTarget(-30.0, -120.0, 15.0),
    ]
```

### `_target_label` in `prior_posterior_tuner_app.py`

```python
from property_targets import CapBulkTarget, BoxcarBulkTarget, CapCMBTarget

def _target_label(idx: int, target) -> str:
    loc = f"lat={target.lat_deg:.0f}°, lon={target.lon_deg:.0f}°"
    cap = f"cap={target.cap_radius_deg:.0f}°"
    if isinstance(target, CapBulkTarget):
        return f"{idx}: {target.param} cap-bump ({loc}, r₀={target.r0_km:.0f} km, {cap})"
    elif isinstance(target, BoxcarBulkTarget):
        return (f"{idx}: {target.param} cap-box ({loc}, "
                f"r∈[{target.r_low_km:.0f},{target.r_high_km:.0f}] km, {cap})")
    else:
        return f"{idx}: CMB cap ({loc}, {cap})"
```

---

## Changes: `tests/test_property_targets.py`

Remove all tests for `angular_bump_coeffs`, `normalized_angular_bump_coeffs`,
`rotate_bump_coeffs`, `radial_bump`, `normalized_radial_bump`, `BulkTarget`,
`CMBTarget`.

Add:

| Test | What it checks |
|---|---|
| `test_angular_cap_coeffs_zonal_mass` | `c₀₀ = 1/(4π)` for any cap radius; matches Gaussian invariant |
| `test_angular_cap_coeffs_full_sphere` | 180° cap → same as uniform function |
| `test_angular_cap_coeffs_axisymmetric_at_pole` | Pole cap has zero `m > 0` components |
| `test_angular_cap_coeffs_vs_spherical_cap_integral` | `angular_cap_coeffs` matches coefficients extracted from `sphere.Lebesgue.spherical_cap_average` for a cross-check against pygeoinf |
| `test_normalized_radial_boxcar_unit_volume_integral` | `∫ a(r) r² dr = 1` |
| `test_normalized_radial_boxcar_zero_outside_support` | Values exactly 0 outside `[r_low, r_high]` |
| `test_normalized_radial_bump_compact_unit_volume_integral` | `∫ a(r) r² dr = 1` |
| `test_normalized_radial_bump_compact_zero_outside_support` | Values exactly 0 outside `[r0−w/2, r0+w/2]` |
| `test_property_operator_shape` | Unchanged structure, with new target types |
| `test_property_operator_forward_cmb_target` | `CapCMBTarget` version |
| `test_property_operator_forward_bulk_target_volume_normalization` | `CapBulkTarget` and `BoxcarBulkTarget` versions |
| `test_property_operator_adjoint_consistency` | With new types |
| `test_property_operator_vs_target` | `BoxcarBulkTarget(param='vs')` path |

The cross-check `test_angular_cap_coeffs_vs_spherical_cap_integral` is
particularly important: it verifies that `angular_cap_coeffs` and
`sphere.Lebesgue.spherical_cap_average` agree on the `B_{st}` coefficients,
confirming the normalization convention is the same throughout.

---

## Changes: `utils/prior_sensitivity_study.py`

This file builds `K_sigma_0` and `K_sigma_1` directly (using
`get_topo_layer_value` in a loop) and constructs forward operators manually
without using the target dataclasses. Update the few lines that reference
`BulkTarget` / `CMBTarget` instances.

---

## Execution Order

These phases are sequential (each depends on the previous):

1. **`property_targets.py`** — new functions + dataclasses + updated
   `build_block_property_coeffs`. Tests written simultaneously (TDD).
2. **`full_spectrum_utils.py`** — update `build_property_operator`. Run full
   test suite to confirm no regressions.
3. **`target_kernel_viz.py`** — update viewer + defaults. Manual smoke-test
   with the tuner app.
4. **`prior_sensitivity_study.py`** — minor cleanup.

Estimated test count delta: −7 Gaussian tests, +13 compact-support tests.

---

## Out of Scope

- Smooth C∞ bumps on S² (as opposed to cap indicators): mathematically
  possible via `sphere.project_function` but not needed for the current paper.
- Non-separable 3D target functions.
- Any changes to pygeoinf, pygeoinf3D, or intervalinf core libraries.
- The `build_topo_matrix` refactor (tracked separately; independent of this
  plan).
