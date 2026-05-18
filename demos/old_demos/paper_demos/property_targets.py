"""
property_targets.py
===================

Target-function factories for building property operators in the full-spectrum
splitting-function PLI example.

Property targets define what physical quantities $\\mathcal{T}_i$ are estimated
from the inversion. Two families are supported:

1. **Bulk volumetric averages** — weighted average of a radial parameter
   (vp, vs, rho) over an angular-bump × radial-bump region.
2. **CMB topography angular bumps** — weighted CMB boundary displacement
   at an angular-bump location.

All bumps use $4\\pi$-normalised real spherical harmonics (pyshtools convention).
"""

import dataclasses
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
        The real SH coefficient.
    """
    if t == 0:
        return float(coeffs_array[0, s, 0])
    elif t % 2 == 1:   # odd: cosine term
        m = (t + 1) // 2
        return float(coeffs_array[0, s, m])
    else:              # even > 0: sine term
        m = t // 2
        return float(coeffs_array[1, s, m])


def angular_bump_coeffs(
    sigma_deg: float,
    s_max: int,
    lmax_grid: int = None,
) -> np.ndarray:
    """Compute $4\\pi$-normalised SH expansion of a Gaussian angular bump at the
    north pole.

    The bump is $f(\\theta) = \\exp\\!\\left(-\\theta^2 / (2\\sigma^2)\\right)$
    where $\\theta$ is the colatitude in radians and $\\sigma = \\sigma_\\text{deg}
    \\cdot \\pi/180$.

    Uses a Driscoll–Healy (DH) grid with resolution ``lmax_grid`` for the
    forward SH expansion.

    Parameters
    ----------
    sigma_deg : float
        Angular half-width of the bump in degrees.
    s_max : int
        Maximum spherical harmonic degree to compute.
    lmax_grid : int, optional
        Grid resolution for the DH expansion.  Defaults to
        ``max(s_max * 8, 64)``.

    Returns
    -------
    np.ndarray
        Shape ``(2, s_max+1, s_max+1)`` — the ``coeffs`` array from pyshtools
        (``4pi`` normalisation).
    """
    if lmax_grid is None:
        lmax_grid = max(s_max * 8, 64)

    nlat = 2 * lmax_grid + 2
    nlon = 2 * nlat

    # Latitude and longitude grids (degrees, north-pole first)
    lats = 90.0 - np.arange(nlat) * 180.0 / nlat
    lons = np.arange(nlon) * 360.0 / nlon
    lat_grid, _ = np.meshgrid(lats, lons, indexing='ij')

    # Colatitude (radians)
    theta = np.radians(90.0 - lat_grid)
    sigma_rad = np.radians(sigma_deg)
    vals = np.exp(-theta ** 2 / (2.0 * sigma_rad ** 2))

    grid_obj = sh.SHGrid.from_array(vals, grid='DH')
    coeffs_obj = grid_obj.expand(lmax_calc=s_max, normalization='4pi')
    return coeffs_obj.coeffs.copy()


def rotate_bump_coeffs(
    coeffs_array: np.ndarray,
    lat_deg: float,
    lon_deg: float,
) -> np.ndarray:
    """Rotate an axisymmetric bump template to a target geographic location.

    Parameters
    ----------
    coeffs_array : np.ndarray
        Shape ``(2, lmax+1, lmax+1)`` from :func:`angular_bump_coeffs`.
    lat_deg : float
        Target latitude in degrees.
    lon_deg : float
        Target longitude in degrees.

    Returns
    -------
    np.ndarray
        Rotated ``(2, lmax+1, lmax+1)`` coeffs array ($4\\pi$ normalisation).
    """
    lmax = coeffs_array.shape[1] - 1
    coeffs_obj = sh.SHCoeffs.from_array(coeffs_array, normalization='4pi', lmax=lmax)

    alpha = np.radians(lon_deg)           # rotation about z-axis
    beta = np.radians(90.0 - lat_deg)     # tilt from north pole (= colatitude)
    gamma = 0.0

    rotated = coeffs_obj.rotate(alpha, beta, gamma, degrees=False)
    return rotated.coeffs.copy()


# =============================================================================
# Radial bump
# =============================================================================

def radial_bump(
    r_km: np.ndarray,
    r0_km: float,
    sigma_r_km: float,
) -> np.ndarray:
    """Gaussian radial bump centred at *r0_km*.

    $h(r) = \\exp\\!\\left(-\\frac{(r - r_0)^2}{2\\,\\sigma_r^2}\\right)$

    Parameters
    ----------
    r_km : array_like
        Radial coordinate(s) in km.
    r0_km : float
        Centre of the bump in km.
    sigma_r_km : float
        Half-width of the bump in km.

    Returns
    -------
    np.ndarray
        Same shape as *r_km*.
    """
    r = np.asarray(r_km, dtype=float)
    return np.exp(-((r - r0_km) ** 2) / (2.0 * sigma_r_km ** 2))


# =============================================================================
# Target dataclasses
# =============================================================================

@dataclasses.dataclass
class BulkTarget:
    """A bulk volumetric property target: $h(r) \\cdot b(\\xi - \\xi_0)$ for
    parameter *param*.

    Attributes
    ----------
    param : str
        Radial parameter: ``'vp'``, ``'vs'``, or ``'rho'``.
    lat_deg : float
        Target latitude in degrees.
    lon_deg : float
        Target longitude in degrees.
    sigma_ang_deg : float
        Angular bump half-width in degrees.
    r0_km : float
        Radial bump centre in km.
    sigma_r_km : float
        Radial bump half-width in km.
    """

    param: str
    lat_deg: float
    lon_deg: float
    sigma_ang_deg: float
    r0_km: float
    sigma_r_km: float


@dataclasses.dataclass
class CMBTarget:
    """A CMB topography angular-bump property target.

    Attributes
    ----------
    lat_deg : float
        Target latitude in degrees.
    lon_deg : float
        Target longitude in degrees.
    sigma_ang_deg : float
        Angular bump half-width in degrees.
    """

    lat_deg: float
    lon_deg: float
    sigma_ang_deg: float


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
    $i$ — the $(s, t)$-th real SH coefficient of target $i$'s angular bump
    evaluated at the target's location.

    A cache is maintained keyed on ``sigma_ang_deg`` so that Gaussian templates
    at the north pole are only expanded once per unique half-width.

    Parameters
    ----------
    targets : list
        List of :class:`BulkTarget` or :class:`CMBTarget` instances.
    blocks : list of BlockIndex
        Blocks to build coefficients for.
    s_max : int
        Maximum splitting degree.

    Returns
    -------
    dict
        ``{BlockIndex: np.ndarray}`` of shape ``(N_p,)`` — each entry is the
        vector of angular coefficients for that block.
    """
    N_p = len(targets)

    # Cache pole templates by sigma_ang_deg
    pole_cache: dict = {}

    def _get_pole(sigma_ang: float) -> np.ndarray:
        if sigma_ang not in pole_cache:
            pole_cache[sigma_ang] = angular_bump_coeffs(sigma_ang, s_max)
        return pole_cache[sigma_ang]

    result = {}
    for block in blocks:
        coeffs_vec = np.zeros(N_p)
        for i, target in enumerate(targets):
            coeffs_pole = _get_pole(target.sigma_ang_deg)
            coeffs_rot = rotate_bump_coeffs(coeffs_pole, target.lat_deg, target.lon_deg)
            coeffs_vec[i] = extract_st_coeff(coeffs_rot, block.s, block.t)
        result[block] = coeffs_vec
    return result
