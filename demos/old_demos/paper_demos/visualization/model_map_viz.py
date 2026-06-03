"""
model_map_viz.py
================

Synthesise geographic maps of the **posterior mean** model recovered by the
full-spectrum Bayesian inversion (one solve per ``(s, t)`` block).

Each block's posterior mean supplies a radial coefficient function for every
3-D component (``vp``, ``vs``, ``rho``) and a single scalar CMB-topography
coefficient ``σ₁``.  For a depth slice at radius ``r`` the per-block radial
coefficient $c_{st}(r)$ is read off the cached mean curves and used as the
$(s, t)$ real spherical-harmonic coefficient; the field is then synthesised on
a lat-lon grid with pyshtools (``4π``-normalised, DH2).  The CMB map uses the
``σ₁`` coefficients directly.

Provides:

    ``ModelMapViewer``
        Pure data engine.  Built from a ``{BlockIndex: _PostBlockDisplayData}``
        mapping plus the radial domain radii.  Methods ``depth_slice`` and
        ``cmb_map`` return ``(lats, lons, data)`` triples.

    ``_draw_model_map``
        Draw one map (depth slice or CMB) into a Matplotlib ``Figure`` using a
        Cartopy ``PlateCarree`` projection with coastline outlines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure
    from full_spectrum_utils import BlockIndex
    from posterior_viz import _PostBlockDisplayData


# 3-D components handled by ``depth_slice`` (CMB is handled separately).
_COMPONENTS_3D = ("vp", "vs", "rho")


def _set_st_coeff(coeffs_array: np.ndarray, s: int, t: int, value: float) -> None:
    """Set the ``(s, t)`` real spherical-harmonic coefficient in a pyshtools
    coeffs array (``4π`` normalisation).

    Mirrors ``target_kernel_viz._set_st_coeff`` so the posterior-mean maps use
    exactly the same real-SH ordering as the block reconstruction elsewhere in
    the demo.
    """
    if t == 0:
        coeffs_array[0, s, 0] = value
    elif t % 2 == 1:
        m = (t + 1) // 2
        coeffs_array[0, s, m] = value
    else:
        m = t // 2
        coeffs_array[1, s, m] = value


class ModelMapViewer:
    """Pure data engine for posterior-mean model maps.

    Parameters
    ----------
    display_dict : dict
        ``{BlockIndex: _PostBlockDisplayData}`` for the solved blocks.
    s_max : int
        Maximum splitting degree (controls the SH truncation).
    earth_radius_km, icb_radius_km, cmb_radius_km : float
        Radial domain boundaries (km).
    """

    def __init__(
        self,
        display_dict: "Dict[BlockIndex, _PostBlockDisplayData]",
        *,
        s_max: int,
        earth_radius_km: float = 6371.0,
        icb_radius_km: float = 1221.0,
        cmb_radius_km: float = 3480.0,
    ) -> None:
        if not display_dict:
            raise ValueError("display_dict is empty — no blocks to map.")
        self._dd = dict(display_dict)
        self._blocks = list(display_dict.keys())
        self._s_max = int(s_max)
        self.earth_radius_km = float(earth_radius_km)
        self.icb_radius_km = float(icb_radius_km)
        self.cmb_radius_km = float(cmb_radius_km)

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def blocks(self) -> list:
        return list(self._blocks)

    @property
    def s_max(self) -> int:
        return self._s_max

    # ── Radial helpers ────────────────────────────────────────────────────
    def radius_bounds(self, component: str) -> Tuple[float, float]:
        """Inclusive ``(r_min, r_max)`` radii (km) over which *component* is
        defined.  For ``vs`` this spans the full earth but the outer core is a
        gap (see :meth:`is_in_vs_gap`)."""
        if component in ("vp", "rho", "vs"):
            return (0.0, self.earth_radius_km)
        raise ValueError(f"Unknown 3-D component: {component!r}")

    def is_in_vs_gap(self, radius_km: float) -> bool:
        """True when *radius_km* lies in the outer core, where the shear-wave
        speed perturbation ``vs`` is undefined (no shear support)."""
        return self.icb_radius_km < radius_km < self.cmb_radius_km

    # ── Per-block radial coefficient ──────────────────────────────────────
    def _block_coeff_3d(self, dd, component: str, radius_km: float) -> Optional[float]:
        """Radial coefficient $c_{st}(r)$ of one block for a 3-D component.

        Returns ``None`` when *radius_km* falls in the ``vs`` outer-core gap.
        """
        if component == "vp":
            return float(np.interp(radius_km, dd.r_vp, dd.mean_vp))
        if component == "rho":
            return float(np.interp(radius_km, dd.r_rho, dd.mean_rho))
        if component == "vs":
            if radius_km <= self.icb_radius_km:
                return float(np.interp(radius_km, dd.r_vs_IC, dd.mean_vs_IC))
            if radius_km >= self.cmb_radius_km:
                return float(np.interp(radius_km, dd.r_vs_M, dd.mean_vs_M))
            return None  # outer core: no shear
        raise ValueError(f"Unknown 3-D component: {component!r}")

    # ── Grid synthesis ────────────────────────────────────────────────────
    def _synthesise(
        self, coeff_of_block, *, grid_lmax: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand per-block real-SH coefficients onto a lat-lon grid.

        Parameters
        ----------
        coeff_of_block : callable
            ``block -> float`` giving the ``(s, t)`` coefficient.
        grid_lmax : int, optional
            Expansion degree.  Defaults to ``max(s_max * 8, 72)``.
        """
        import pyshtools as sh

        if grid_lmax is None:
            grid_lmax = max(self._s_max * 8, 72)

        coeffs = np.zeros((2, grid_lmax + 1, grid_lmax + 1))
        for block in self._blocks:
            _set_st_coeff(coeffs, block.s, block.t, float(coeff_of_block(block)))

        sh_coeffs = sh.SHCoeffs.from_array(coeffs, normalization="4pi")
        grid_obj = sh_coeffs.expand(grid="DH2")
        data_raw = grid_obj.to_array()  # lat: 90→-90, lon: 0→360

        nlon = data_raw.shape[1]
        half = nlon // 2
        data = np.roll(data_raw, half, axis=1)  # lon → -180…180
        lats = grid_obj.lats()
        lons = np.linspace(-180.0, 180.0, nlon, endpoint=False)
        return lats, lons, data

    def depth_slice(
        self, component: str, radius_km: float, *, grid_lmax: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Posterior-mean map of a 3-D component on the sphere of radius
        *radius_km*.

        Returns ``(lats, lons, data)`` or ``None`` when the radius lies in the
        ``vs`` outer-core gap.
        """
        if component not in _COMPONENTS_3D:
            raise ValueError(
                f"component must be one of {_COMPONENTS_3D}, got {component!r}")
        if component == "vs" and self.is_in_vs_gap(radius_km):
            return None

        def _coeff(block):
            c = self._block_coeff_3d(self._dd[block], component, radius_km)
            return 0.0 if c is None else c

        return self._synthesise(_coeff, grid_lmax=grid_lmax)

    def cmb_map(
        self, *, grid_lmax: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior-mean CMB-topography map from the per-block ``σ₁``
        coefficients."""
        def _coeff(block):
            return float(self._dd[block].sigma_1_mean)

        return self._synthesise(_coeff, grid_lmax=grid_lmax)


# =============================================================================
# Drawing
# =============================================================================

_COMPONENT_LABEL = {
    "vp": "δvₚ/vₚ",
    "vs": "δvₛ/vₛ",
    "rho": "δρ/ρ",
    "CMB": "CMB topography",
}


def _draw_model_map(
    fig: "matplotlib.figure.Figure",
    viewer: ModelMapViewer,
    component: str,
    radius_km: Optional[float],
    *,
    plt,
) -> None:
    """Draw one posterior-mean map into *fig* using a Cartopy projection with
    coastline outlines.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to clear and draw into (a single GeoAxes is created).
    viewer : ModelMapViewer
    component : {"vp", "vs", "rho", "CMB"}
    radius_km : float or None
        Slice radius for 3-D components; ignored for ``"CMB"``.
    plt : module
        The ``matplotlib.pyplot`` module (used for the colorbar).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig.clf()
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if component == "CMB":
        lats, lons, data = viewer.cmb_map()
        title = "CMB topography — posterior mean"
    else:
        result = viewer.depth_slice(component, float(radius_km))
        if result is None:
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.4")
            ax.set_title(
                f"{_COMPONENT_LABEL[component]} — r={radius_km:.0f} km\n"
                f"(outer core: no shear support for vs)",
                fontsize=10,
            )
            return
        lats, lons, data = result
        depth = viewer.earth_radius_km - float(radius_km)
        title = (f"{_COMPONENT_LABEL[component]} — posterior mean\n"
                 f"r = {radius_km:.0f} km   (depth {depth:.0f} km)")

    vmax = float(np.max(np.abs(data)))
    vmax = vmax if vmax > 0 else 1.0

    im = ax.imshow(
        data,
        extent=[-180, 180, -90, 90],
        origin="upper",
        transform=proj,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.2")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08,
                      fraction=0.045)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(_COMPONENT_LABEL[component], fontsize=9)

    ax.set_title(title, fontsize=10)
