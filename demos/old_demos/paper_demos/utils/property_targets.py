"""
property_targets.py
===================

Target-function factories for building property operators in the full-spectrum
splitting-function PLI example.

Three families of compact-support targets:

1. **CapBulkTarget** — spherical-cap average × C∞ compact radial bump.
2. **BoxcarBulkTarget** — spherical-cap average × radial boxcar.
3. **CapCMBTarget** — spherical-cap average of CMB topography.

All angular kernels use the exact SH expansion of the spherical-cap indicator
via ``pyshtools.SHCoeffs.from_cap`` (4π-normalised real harmonics, placed
directly at the target location — no pole template, no rotation step).
All radial kernels have true compact support and are normalised to
$\\int a(r)\\,r^2\\,dr = 1$.
"""

import dataclasses
from typing import Optional

import numpy as np
import pyshtools as sh


# =============================================================================
# SH coefficient helpers
# =============================================================================

def extract_st_coeff(coeffs_array: np.ndarray, s: int, t: int) -> float:
    """Extract the (s, t) real SH coefficient from a pyshtools coeffs array.

    Uses the Dahlen & Tromp (1998) real-harmonic convention:

    - ``t=0``       → ``coeffs[0, s, 0]``  (zonal, m=0)
    - ``t=2k-1``    → ``coeffs[0, s, k]``  (cosine of order m=k)
    - ``t=2k``      → ``coeffs[1, s, k]``  (sine of order m=k)

    Parameters
    ----------
    coeffs_array : np.ndarray
        Shape ``(2, lmax+1, lmax+1)`` — the ``.coeffs`` attribute of a
        pyshtools ``SHCoeffs`` object.
    s : int
        Spherical harmonic degree.
    t : int
        Real-harmonic component index, $0 \\leq t \\leq 2s$.

    Returns
    -------
    float
    """
    if t == 0:
        return float(coeffs_array[0, s, 0])
    elif t % 2 == 1:
        m = (t + 1) // 2
        return float(coeffs_array[0, s, m])
    else:
        m = t // 2
        return float(coeffs_array[1, s, m])


# =============================================================================
# Angular kernel factory
# =============================================================================

def angular_cap_coeffs(
    lat_deg: float,
    lon_deg: float,
    cap_radius_deg: float,
    s_max: int,
) -> np.ndarray:
    r"""SH coefficients of the spherical-cap indicator, normalised to unit
    surface integral ($\int a\,d\Omega = 1$), in $4\pi$-normalised
    real-harmonic convention.

    Uses ``pyshtools.SHCoeffs.from_cap`` placed directly at
    ``(lat_deg, lon_deg)``.  No pole template and no rotation are needed.
    The expansion is exact in the truncated basis at degree *s_max*.

    **Normalisation invariant:** the zonal coefficient satisfies
    $c_{00} = 1/(4\pi)$, matching the convention used throughout this module.

    Parameters
    ----------
    lat_deg : float
        Cap centre latitude in degrees.
    lon_deg : float
        Cap centre longitude in degrees.
    cap_radius_deg : float
        Cap angular radius in degrees, in ``(0, 180]``.
    s_max : int
        Maximum spherical-harmonic degree to compute.

    Returns
    -------
    np.ndarray, shape ``(2, s_max+1, s_max+1)``
        Coefficient array compatible with :func:`extract_st_coeff` and
        ``pyshtools.SHCoeffs.from_array``.
    """
    if cap_radius_deg <= 0.0 or cap_radius_deg > 180.0:
        raise ValueError(
            f"cap_radius_deg must be in (0, 180], got {cap_radius_deg}."
        )
    coeffs = sh.SHCoeffs.from_cap(
        cap_radius_deg,
        s_max,
        clat=lat_deg,
        clon=lon_deg,
        normalization='4pi',
        csphase=1,
        kind='real',
        degrees=True,
    )
    return coeffs.coeffs.copy() / (4.0 * np.pi)


# =============================================================================
# Radial kernel factories
# =============================================================================

def normalized_radial_boxcar(
    r_km: np.ndarray,
    r_low_km: float,
    r_high_km: float,
    *,
    normalization_r_km: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Boxcar indicator on $[r_{\rm low}, r_{\rm high}]$, normalised so
    $\int a(r)\,r^2\,dr = 1$.

    The normalisation constant is analytic:
    $(r_{\rm high}^3 - r_{\rm low}^3)/3$.

    Parameters
    ----------
    r_km : array_like
        Radial evaluation grid in km.
    r_low_km : float
        Lower support boundary in km.
    r_high_km : float
        Upper support boundary in km.
    normalization_r_km : np.ndarray, optional
        Unused; present for API symmetry with
        :func:`normalized_radial_bump_compact`.

    Returns
    -------
    np.ndarray
        Same shape as *r_km*.
    """
    if r_high_km <= r_low_km:
        raise ValueError(
            f"r_high_km ({r_high_km}) must be greater than r_low_km ({r_low_km})."
        )
    r = np.asarray(r_km, dtype=float)
    raw = np.where((r >= r_low_km) & (r <= r_high_km), 1.0, 0.0)
    vol_integral = (r_high_km ** 3 - r_low_km ** 3) / 3.0
    return raw / vol_integral


def normalized_radial_bump_compact(
    r_km: np.ndarray,
    r0_km: float,
    width_km: float,
    *,
    normalization_r_km: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""C$^\infty$ bump with hard support $[r_0 - w/2,\; r_0 + w/2]$,
    normalised so $\int a(r)\,r^2\,dr = 1$.

    Uses the standard mollifier $\exp(t^2/(t^2-1))$ where
    $t = 2(r - r_0)/w$; exactly zero for $|t| \ge 1$.

    Parameters
    ----------
    r_km : array_like
        Radial evaluation grid in km.
    r0_km : float
        Bump centre in km.
    width_km : float
        Full support width in km.
    normalization_r_km : np.ndarray, optional
        Grid used to compute the normalisation constant.  Provide a fine
        global grid (e.g. 0 → R_Earth at 500 points) when *r_km* covers only
        a subdomain; the bump may be partially outside the evaluation grid.

    Returns
    -------
    np.ndarray
        Same shape as *r_km*.
    """
    if width_km <= 0.0:
        raise ValueError(f"width_km must be positive, got {width_km}.")

    def _eval(r_arr: np.ndarray) -> np.ndarray:
        t = 2.0 * (r_arr - r0_km) / width_km
        result = np.zeros_like(r_arr, dtype=float)
        interior = np.abs(t) < 1.0
        if np.any(interior):
            ti = t[interior]
            result[interior] = np.exp(ti ** 2 / (ti ** 2 - 1.0))
        return result

    r = np.asarray(r_km, dtype=float)
    raw = _eval(r)

    r_norm = r if normalization_r_km is None else np.asarray(
        normalization_r_km, dtype=float
    )
    raw_norm = _eval(r_norm)
    vol_integral = float(np.trapezoid(raw_norm * r_norm ** 2, r_norm))
    if vol_integral <= 0.0:
        raise ValueError(
            "C∞ bump has non-positive volume integral — "
            "check that [r0 ± width/2] is within the radial domain."
        )
    return raw / vol_integral


# =============================================================================
# Target dataclasses
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CapBulkTarget:
    """Spherical-cap average × C∞ compact radial bump.

    $T(m) = \\left[\\int_{\\rm cap} a(\\xi)\\,d\\Omega\\right]
             \\times \\left[\\int {\\rm bump}(r)\\,m_{\\rm param}(r)\\,r^2\\,dr\\right]$

    Attributes
    ----------
    param : str
        Radial parameter: ``'vp'``, ``'vs'``, or ``'rho'``.
    lat_deg : float
        Cap centre latitude in degrees.
    lon_deg : float
        Cap centre longitude in degrees.
    cap_radius_deg : float
        Cap angular radius in degrees.
    r0_km : float
        Bump centre in km.
    width_km : float
        Full bump support width in km (zero outside ``[r0 ± width/2]``).
    """
    param: str
    lat_deg: float
    lon_deg: float
    cap_radius_deg: float
    r0_km: float
    width_km: float


@dataclasses.dataclass(frozen=True)
class BoxcarBulkTarget:
    """Spherical-cap average × radial boxcar.

    $T(m) = \\left[\\int_{\\rm cap} a(\\xi)\\,d\\Omega\\right]
             \\times \\frac{1}{V}\\int_{r_{\\rm low}}^{r_{\\rm high}}
             m_{\\rm param}(r)\\,r^2\\,dr$

    Attributes
    ----------
    param : str
        Radial parameter: ``'vp'``, ``'vs'``, or ``'rho'``.
    lat_deg : float
    lon_deg : float
    cap_radius_deg : float
    r_low_km : float
        Lower radial boundary in km.
    r_high_km : float
        Upper radial boundary in km.
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

    $T(\\sigma_{\\rm CMB}) = \\int_{\\rm cap} a(\\xi)\\,\\sigma_{\\rm CMB}(\\xi)\\,d\\Omega$

    Attributes
    ----------
    lat_deg : float
    lon_deg : float
    cap_radius_deg : float
    """
    lat_deg: float
    lon_deg: float
    cap_radius_deg: float


# =============================================================================
# Block angular coefficients
# =============================================================================

def build_block_property_coeffs(
    targets: list,
    blocks: list,
    s_max: int,
) -> dict:
    """Pre-compute per-block angular SH coefficients for all property targets.

    For each block ``(s, t)``, computes the scalar $B_{st,i}$ for each target
    $i$ — the $(s, t)$-th real SH coefficient of target $i$'s unit-surface-integral
    spherical-cap indicator.

    A cache keyed on ``(lat_deg, lon_deg, cap_radius_deg)`` ensures that
    identical caps call ``angular_cap_coeffs`` only once.

    Parameters
    ----------
    targets : list
        List of :class:`CapBulkTarget`, :class:`BoxcarBulkTarget`, or
        :class:`CapCMBTarget` instances.
    blocks : list of BlockIndex
        Blocks to build coefficients for.
    s_max : int
        Maximum splitting degree.

    Returns
    -------
    dict
        ``{BlockIndex: np.ndarray}`` of shape ``(N_p,)``.
    """
    N_p = len(targets)
    cap_cache: dict = {}

    def _get_coeffs(target) -> np.ndarray:
        key = (target.lat_deg, target.lon_deg, target.cap_radius_deg)
        if key not in cap_cache:
            cap_cache[key] = angular_cap_coeffs(
                target.lat_deg, target.lon_deg, target.cap_radius_deg, s_max
            )
        return cap_cache[key]

    result = {}
    for block in blocks:
        coeffs_vec = np.zeros(N_p)
        for i, target in enumerate(targets):
            coeffs_vec[i] = extract_st_coeff(_get_coeffs(target), block.s, block.t)
        result[block] = coeffs_vec
    return result


