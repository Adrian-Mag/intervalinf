"""
target_kernel_viz.py
====================

Interactive viewer for target kernels used in the full-spectrum PLI example.

Provides:

    ``TargetKernelViewer``
        Pure data engine that synthesises geographic maps (lat-lon cap
        indicator), volume-normalized radial profiles $a(r)$, and per-block
        angular coefficients $B_{st,i}$ for a list of
        :class:`~property_targets.CapBulkTarget`,
        :class:`~property_targets.BoxcarBulkTarget`, or
        :class:`~property_targets.CapCMBTarget` instances.

    ``_draw_geographic_axes``
        Draw geographic view (lat-lon map + optional radial profile) into
        existing Axes objects.

    ``_draw_st_axes``
        Draw s-t spectral view (radial kernels per block for bulk targets, or
        scalar bar chart for CMB targets) into existing Axes objects.

    ``MatplotlibTargetKernelViewer``
        Standalone Matplotlib interactive control panel with target selector
        and mode toggle (geographic ↔ s-t). Geographic view can switch
        between the full `s <= s_max` reconstruction and the reconstruction
        using only coefficients on the available model/data blocks.

    ``build_parser`` / ``main``
        CLI entry point.  Loads catalog, builds default targets, and opens
        the viewer.

Usage (standalone)::

    python visualization/target_kernel_viz.py
    python visualization/target_kernel_viz.py --s-max 4

Keyboard shortcuts in the interactive window:

    ← / →   Previous / next target.
    m       Toggle geographic ↔ s-t mode.
    g       Toggle geographic reconstruction mode.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for interactive GUI
import numpy as np

import _path_setup  # noqa: F401

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    from full_spectrum_utils import BlockIndex


def _set_st_coeff(coeffs_array: np.ndarray, s: int, t: int, value: float) -> None:
    """Set the (s, t) real spherical-harmonic coefficient in a pyshtools array."""
    if t == 0:
        coeffs_array[0, s, 0] = value
    elif t % 2 == 1:
        m = (t + 1) // 2
        coeffs_array[0, s, m] = value
    else:
        m = t // 2
        coeffs_array[1, s, m] = value


# =============================================================================
# TargetKernelViewer — data engine
# =============================================================================


class TargetKernelViewer:
    """Pure data engine for target-kernel visualisation.

    Pre-computes per-block angular SH coefficients $B_{st,i}$ for all targets
    on construction.  All synthesis methods are on-demand.

    Parameters
    ----------
    targets : list
        List of :class:`~property_targets.CapBulkTarget`,
        :class:`~property_targets.BoxcarBulkTarget`, or
        :class:`~property_targets.CapCMBTarget` instances.
    blocks : list of BlockIndex
        (s, t) blocks to pre-compute coefficients for.
    s_max : int
        Maximum splitting degree.
    n_radial : int
        Default number of points for radial-profile synthesis.
    earth_radius_km : float
        Earth outer radius in km.  Default 6371.0.
    icb_radius_km : float
        Inner-core boundary radius in km.  Default 1221.0.
    cmb_radius_km : float
        Core-mantle boundary radius in km.  Default 3480.0.
    """

    def __init__(
        self,
        targets: list,
        blocks: list,
        s_max: int,
        *,
        n_radial: int = 300,
        earth_radius_km: float = 6371.0,
        icb_radius_km: float = 1221.0,
        cmb_radius_km: float = 3480.0,
    ) -> None:
        from property_targets import build_block_property_coeffs, angular_cap_coeffs

        self._targets = list(targets)
        self._blocks = list(blocks)
        self._s_max = s_max
        self._n_radial = n_radial
        self._earth_radius_km = earth_radius_km
        self._icb_radius_km = icb_radius_km
        self._cmb_radius_km = cmb_radius_km
        self._radial_norm_grid = np.linspace(
            0.0, self._earth_radius_km, max(4 * n_radial, 2000)
        )

        self._block_coeffs: Dict = build_block_property_coeffs(
            targets, blocks, s_max
        )

        # Pre-compute full (2, s_max+1, s_max+1) coefficient array per target
        # for geographic map synthesis — avoids recomputing on every call.
        _cap_cache: dict = {}
        self._full_coeffs: List[np.ndarray] = []
        for t in targets:
            key = (t.lat_deg, t.lon_deg, t.cap_radius_deg)
            if key not in _cap_cache:
                _cap_cache[key] = angular_cap_coeffs(
                    t.lat_deg, t.lon_deg, t.cap_radius_deg, s_max
                )
            self._full_coeffs.append(_cap_cache[key])

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def targets(self) -> list:
        return list(self._targets)

    @property
    def n_targets(self) -> int:
        return len(self._targets)

    @property
    def blocks(self) -> list:
        return list(self._blocks)

    @property
    def s_max(self) -> int:
        return self._s_max

    # ── Synthesis methods ─────────────────────────────────────────────────

    def geographic_map(
        self,
        target_idx: int,
        *,
        grid_lmax: Optional[int] = None,
        reconstruction: str = "smax",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synthesise the angular bump of target *target_idx* on a lat-lon grid.

        The SH coefficients are always truncated at ``s_max``. With
        ``reconstruction='smax'`` the full truncated target expansion is used;
        with ``reconstruction='blocks'`` only the coefficients associated with
        the precomputed available blocks are retained. For smooth display the
        chosen coefficient array is zero-padded to ``grid_lmax`` before grid
        synthesis.

        Parameters
        ----------
        target_idx : int
            Index into :attr:`targets`.
        grid_lmax : int, optional
            Expansion degree for the output grid.  Defaults to
            ``max(s_max * 8, 72)``.
        reconstruction : {"smax", "blocks"}
            Geographic reconstruction mode. ``"smax"`` uses the full target
            expansion up to ``s_max``; ``"blocks"`` uses only coefficients on
            the available model/data blocks.

        Returns
        -------
        lats : np.ndarray, shape (nlat,)
            Latitudes in degrees, 90 → -90 (north first).
        lons : np.ndarray, shape (nlon,)
            Longitudes in degrees, -180 → 180 (centred).
        data : np.ndarray, shape (nlat, nlon)
            Amplitude of the angular bump.
        """
        import pyshtools as sh

        if reconstruction not in {"smax", "blocks"}:
            raise ValueError(
                "reconstruction must be 'smax' or 'blocks', got "
                f"{reconstruction!r}"
            )

        if grid_lmax is None:
            grid_lmax = max(self._s_max * 8, 72)

        coeffs_rot = self._full_coeffs[target_idx].copy()

        if reconstruction == "blocks":
            coeffs_masked = np.zeros_like(coeffs_rot)
            for block in self._blocks:
                _set_st_coeff(
                    coeffs_masked,
                    block.s,
                    block.t,
                    float(self._block_coeffs[block][target_idx]),
                )
            coeffs_rot = coeffs_masked

        # Zero-pad to grid_lmax for smooth visualisation
        if grid_lmax > self._s_max:
            padded = np.zeros((2, grid_lmax + 1, grid_lmax + 1))
            n = self._s_max + 1
            padded[:, :n, :n] = coeffs_rot
            coeffs_for_grid = padded
        else:
            coeffs_for_grid = coeffs_rot

        sh_coeffs = sh.SHCoeffs.from_array(coeffs_for_grid, normalization="4pi")
        grid_obj = sh_coeffs.expand(grid="DH2")
        data_raw = grid_obj.to_array()  # lat: 90→-90, lon: 0→360

        nlat, nlon = data_raw.shape
        # Centre longitude: roll half-way so lon goes -180 → 180
        half = nlon // 2
        data = np.roll(data_raw, half, axis=1)
        lats = grid_obj.lats()
        lons = np.linspace(-180.0, 180.0, nlon, endpoint=False)

        return lats, lons, data

    def radial_profile(
        self,
        target_idx: int,
        *,
        n_r: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Volume-normalized radial factor $a(r)$ for a BulkTarget.

        Parameters
        ----------
        target_idx : int
            Index into :attr:`targets`.
        n_r : int, optional
            Number of radial points.  Defaults to :attr:`_n_radial`.

        Returns
        -------
        r_km : np.ndarray, shape (n_r,)
            Radial coordinate in km.
        h_r : np.ndarray, shape (n_r,)
            Radial bump amplitude.

        Raises
        ------
        TypeError
            If ``targets[target_idx]`` is not a
            :class:`~property_targets.BulkTarget`.
        """
        from property_targets import (
            CapBulkTarget, BoxcarBulkTarget,
            normalized_radial_bump_compact, normalized_radial_boxcar,
        )

        target = self._targets[target_idx]
        n = n_r or self._n_radial
        r_km = np.linspace(0.0, self._earth_radius_km, n)

        if isinstance(target, CapBulkTarget):
            a_r = normalized_radial_bump_compact(
                r_km, target.r0_km, target.width_km,
                normalization_r_km=self._radial_norm_grid,
            )
        elif isinstance(target, BoxcarBulkTarget):
            a_r = normalized_radial_boxcar(
                r_km, target.r_low_km, target.r_high_km,
            )
        else:
            raise TypeError(
                f"targets[{target_idx}] is a {type(target).__name__}, "
                "not a bulk target — no radial profile available."
            )
        return r_km, a_r

    def st_coefficients(self, target_idx: int) -> dict:
        """Per-block angular SH coefficient $B_{st,i}$ for target *i*.

        Parameters
        ----------
        target_idx : int
            Index into :attr:`targets`.

        Returns
        -------
        dict
            ``{BlockIndex: float}`` — one entry per block.
        """
        return {
            block: float(self._block_coeffs[block][target_idx])
            for block in self._blocks
        }

    def st_radial_kernels(
        self,
        target_idx: int,
        *,
        n_r: Optional[int] = None,
    ) -> dict:
        """Per-block radial kernels $B_{st,i} \\cdot a(r)$ for a BulkTarget.

        Parameters
        ----------
        target_idx : int
            Index into :attr:`targets`.
        n_r : int, optional
            Number of radial points.  Defaults to :attr:`_n_radial`.

        Returns
        -------
        dict
            ``{BlockIndex: (r_km, kernel)}`` where *r_km* and *kernel* are
            ``np.ndarray`` of shape ``(n_r,)``.

        Raises
        ------
        TypeError
            If ``targets[target_idx]`` is not a BulkTarget.
        """
        from property_targets import (
            CapBulkTarget, BoxcarBulkTarget,
            normalized_radial_bump_compact, normalized_radial_boxcar,
        )

        target = self._targets[target_idx]
        n = n_r or self._n_radial
        r_km = np.linspace(0.0, self._earth_radius_km, n)

        if isinstance(target, CapBulkTarget):
            a_r = normalized_radial_bump_compact(
                r_km, target.r0_km, target.width_km,
                normalization_r_km=self._radial_norm_grid,
            )
        elif isinstance(target, BoxcarBulkTarget):
            a_r = normalized_radial_boxcar(r_km, target.r_low_km, target.r_high_km)
        else:
            raise TypeError(
                f"targets[{target_idx}] is a {type(target).__name__}, "
                "not a bulk target."
            )

        return {
            block: (r_km, float(self._block_coeffs[block][target_idx]) * a_r)
            for block in self._blocks
        }


# =============================================================================
# Drawing helpers
# =============================================================================


def _draw_geographic_axes(
    ax_map: "matplotlib.axes.Axes",
    ax_radial: Optional["matplotlib.axes.Axes"],
    viewer: TargetKernelViewer,
    target_idx: int,
    *,
    reconstruction: str,
    plt,
) -> None:
    """Draw geographic view into *ax_map* (and *ax_radial* for BulkTarget).

    Parameters
    ----------
    ax_map : Axes
        Main panel for the lat-lon map.
    ax_radial : Axes or None
        Side panel for the radial profile.  Only used for BulkTarget; pass
        ``None`` to skip.
    viewer : TargetKernelViewer
    target_idx : int
    plt : module
        The ``matplotlib.pyplot`` module.
    """
    from property_targets import CapBulkTarget, BoxcarBulkTarget, CapCMBTarget

    ax_map.clear()

    target = viewer.targets[target_idx]
    lats, lons, data = viewer.geographic_map(
        target_idx,
        reconstruction=reconstruction,
    )
    recon_label = (
        "$s \\leq s_{\\max}$"
        if reconstruction == "smax"
        else "model blocks only"
    )

    vmax = np.max(np.abs(data))
    vmax = vmax if vmax > 0 else 1.0
    im = ax_map.imshow(
        data,
        extent=[-180, 180, -90, 90],
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax_map.set_xlabel("Longitude (°)", fontsize=9)
    ax_map.set_ylabel("Latitude (°)", fontsize=9)
    ax_map.tick_params(labelsize=8)

    ax_map.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax_map.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.5)

    cb = plt.colorbar(im, ax=ax_map, orientation="horizontal", pad=0.14,
                      fraction=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Amplitude", fontsize=8)

    loc_str = f"lat={target.lat_deg:.1f}°  lon={target.lon_deg:.1f}°"
    cap_str = f"cap={target.cap_radius_deg:.1f}°"

    if isinstance(target, CapBulkTarget):
        ax_map.set_title(
            f"Target {target_idx}  [{target.param}]  {loc_str}  "
            f"r₀={target.r0_km:.0f} km  w={target.width_km:.0f} km  {cap_str}\n"
            f"geographic reconstruction: {recon_label}",
            fontsize=9,
        )
        if ax_radial is not None:
            ax_radial.clear()
            r_km, a_r = viewer.radial_profile(target_idx)
            ax_radial.plot(a_r, r_km / 1e3, "b-", lw=1.5)
            ax_radial.set_xlabel("a(r)", fontsize=9)
            ax_radial.set_ylabel("Radius (10³ km)", fontsize=9)
            ax_radial.set_title("Radial profile (C∞ bump)", fontsize=9)
            ax_radial.axhline(
                target.r0_km / 1e3, color="gray", lw=0.8, ls="--", alpha=0.7,
                label=f"r₀={target.r0_km:.0f} km",
            )
            ax_radial.legend(fontsize=7)
            ax_radial.tick_params(labelsize=8)
    elif isinstance(target, BoxcarBulkTarget):
        ax_map.set_title(
            f"Target {target_idx}  [{target.param}]  {loc_str}  "
            f"r∈[{target.r_low_km:.0f},{target.r_high_km:.0f}] km  {cap_str}\n"
            f"geographic reconstruction: {recon_label}",
            fontsize=9,
        )
        if ax_radial is not None:
            ax_radial.clear()
            r_km, a_r = viewer.radial_profile(target_idx)
            ax_radial.plot(a_r, r_km / 1e3, "b-", lw=1.5)
            ax_radial.set_xlabel("a(r)", fontsize=9)
            ax_radial.set_ylabel("Radius (10³ km)", fontsize=9)
            ax_radial.set_title("Radial profile (boxcar)", fontsize=9)
            ax_radial.axhspan(
                target.r_low_km / 1e3, target.r_high_km / 1e3,
                color="lightblue", alpha=0.3,
                label=f"[{target.r_low_km:.0f},{target.r_high_km:.0f}] km",
            )
            ax_radial.legend(fontsize=7)
            ax_radial.tick_params(labelsize=8)
    else:
        # CapCMBTarget
        ax_map.set_title(
            f"Target {target_idx}  [CMB]  {loc_str}  {cap_str}\n"
            f"geographic reconstruction: {recon_label}",
            fontsize=9,
        )


def _draw_st_axes(
    ax_profiles: "matplotlib.axes.Axes",
    ax_bar: Optional["matplotlib.axes.Axes"],
    viewer: TargetKernelViewer,
    target_idx: int,
    *,
    plt,
) -> None:
    """Draw s-t spectral view into *ax_profiles* and *ax_bar*.

    For BulkTarget:
    - *ax_profiles*: one radial kernel $B_{st,i} \\cdot a(r)$ per block,
      all overlaid and coloured by block index.
    - *ax_bar*: bar chart of $B_{st,i}$ scalar values.

    For CMBTarget:
    - *ax_profiles*: bar chart of $B_{st,i}$ scalar values (full width).
    - *ax_bar*: unused (pass ``None``).

    Parameters
    ----------
    ax_profiles : Axes
        Left panel (radial kernels or bar chart for CMBTarget).
    ax_bar : Axes or None
        Right panel (scalar bar chart for BulkTarget).
    viewer : TargetKernelViewer
    target_idx : int
    plt : module
        The ``matplotlib.pyplot`` module.
    """
    from property_targets import CapBulkTarget, BoxcarBulkTarget, CapCMBTarget

    ax_profiles.clear()
    if ax_bar is not None:
        ax_bar.clear()

    target = viewer.targets[target_idx]
    blocks = viewer.blocks
    block_labels = [f"s{b.s}t{b.t}" for b in blocks]
    st_coeffs = viewer.st_coefficients(target_idx)
    values = [st_coeffs[b] for b in blocks]

    if isinstance(target, (CapBulkTarget, BoxcarBulkTarget)):
        # ── Left panel: radial kernels per block ──────────────────────────
        kernels = viewer.st_radial_kernels(target_idx)
        cmap = plt.colormaps.get_cmap("tab20").resampled(len(blocks))
        for k, block in enumerate(blocks):
            r_km, kernel = kernels[block]
            ax_profiles.plot(
                kernel,
                r_km / 1e3,
                color=cmap(k),
                lw=1.0,
                label=block_labels[k],
                alpha=0.85,
            )
        if isinstance(target, CapBulkTarget):
            ax_profiles.axhline(
                target.r0_km / 1e3, color="gray", lw=0.8, ls="--", alpha=0.7,
                label=f"r₀={target.r0_km:.0f} km",
            )
        else:
            ax_profiles.axhspan(
                target.r_low_km / 1e3, target.r_high_km / 1e3,
                color="lightblue", alpha=0.25,
            )
        ax_profiles.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax_profiles.set_xlabel("$B_{st,i} \\cdot a(r)$", fontsize=9)
        ax_profiles.set_ylabel("Radius (10³ km)", fontsize=9)
        ax_profiles.set_title(
            f"Target {target_idx}  [{target.param}]  radial kernels per block",
            fontsize=9,
        )
        ax_profiles.legend(
            fontsize=6,
            ncol=2,
            loc="upper right",
            framealpha=0.7,
        )
        ax_profiles.tick_params(labelsize=8)

        # ── Right panel: bar chart of B_{st,i} ────────────────────────────
        if ax_bar is not None:
            colors = ["steelblue" if v >= 0 else "coral" for v in values]
            ax_bar.bar(
                range(len(blocks)),
                values,
                color=colors,
                edgecolor="k",
                linewidth=0.5,
            )
            ax_bar.set_xticks(range(len(blocks)))
            ax_bar.set_xticklabels(block_labels, rotation=90, fontsize=7)
            ax_bar.axhline(0, color="k", lw=0.5)
            ax_bar.set_ylabel("$B_{st,i}$", fontsize=9)
            ax_bar.set_title(
                f"Target {target_idx}  [{target.param}]  angular coefficients",
                fontsize=9,
            )
            ax_bar.tick_params(axis="y", labelsize=8)
    else:
        # ── CapCMBTarget: scalar bar chart (full width in ax_profiles) ────
        colors = ["steelblue" if v >= 0 else "coral" for v in values]
        ax_profiles.bar(
            range(len(blocks)),
            values,
            color=colors,
            edgecolor="k",
            linewidth=0.5,
        )
        ax_profiles.set_xticks(range(len(blocks)))
        ax_profiles.set_xticklabels(block_labels, rotation=90, fontsize=7)
        ax_profiles.axhline(0, color="k", lw=0.5)
        ax_profiles.set_ylabel("$B_{st,i}$", fontsize=9)
        ax_profiles.set_title(
            f"Target {target_idx}  [CMB]  "
            f"lat={target.lat_deg:.1f}°  lon={target.lon_deg:.1f}°  "
            f"cap={target.cap_radius_deg:.1f}°  angular coefficients per block",
            fontsize=9,
        )
        ax_profiles.tick_params(labelsize=8)


# =============================================================================
# MatplotlibTargetKernelViewer — interactive GUI
# =============================================================================


class MatplotlibTargetKernelViewer:
    """Standalone Matplotlib interactive viewer for target kernels.

    Parameters
    ----------
    viewer : TargetKernelViewer
    plt : module
        The ``matplotlib.pyplot`` module.
    button_class : type
        ``matplotlib.widgets.Button``.
    slider_class : type
        ``matplotlib.widgets.Slider``.
    initial_mode : str
        ``'geo'`` or ``'st'``.  Default ``'geo'``.

    Keyboard shortcuts
    ------------------
    ← / →   Previous / next target.
    m       Toggle geographic ↔ s-t mode.
    g       Toggle geographic reconstruction mode.
    """

    # Bounding boxes for the two content axes in each configuration.
    # Format: [left, bottom, width, height] in figure-fraction units.
    _POS_SINGLE = [0.07, 0.23, 0.89, 0.68]           # full-width main panel
    _POS_LEFT   = [0.07, 0.23, 0.57, 0.68]            # left panel (70% of plot)
    _POS_RIGHT  = [0.70, 0.23, 0.27, 0.68]            # right panel (27%)

    def __init__(
        self,
        viewer: TargetKernelViewer,
        plt,
        button_class,
        slider_class,
        *,
        initial_mode: str = "geo",
    ) -> None:
        if viewer.n_targets == 0:
            raise ValueError("MatplotlibTargetKernelViewer requires at least one target.")

        self.viewer = viewer
        self.plt = plt
        self.Button = button_class
        self.Slider = slider_class
        self._mode: str = initial_mode  # 'geo' or 'st'
        self._geo_reconstruction: str = "smax"

        self.fig = self.plt.figure(figsize=(14, 7))
        self.fig.subplots_adjust(
            left=0.07, right=0.97, bottom=0.22, top=0.94
        )

        # Two content axes; positions are updated on each redraw
        self.ax_main = self.fig.add_axes(self._POS_SINGLE)
        self.ax_side = self.fig.add_axes(self._POS_RIGHT)
        self.ax_side.set_visible(False)

        self.status_text = self.fig.text(0.07, 0.97, "", fontsize=8, va="top")

        self._build_widgets()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._redraw()

    # ── Public API ─────────────────────────────────────────────────────────

    def show(self) -> None:
        """Open the interactive Matplotlib window."""
        self.plt.show()

    def save(self, path) -> None:
        """Save the current figure to *path*."""
        self.fig.savefig(path, dpi=160, bbox_inches="tight")

    # ── Widget construction ────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        n_targets = self.viewer.n_targets
        valmax = max(n_targets - 1, 1)

        # Target slider
        slider_ax = self.fig.add_axes([0.20, 0.14, 0.50, 0.03])
        self.target_slider = self.Slider(
            slider_ax,
            "target",
            0,
            valmax,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        self.target_slider.on_changed(self._on_target_changed)

        # Prev / Next buttons
        prev_ax = self.fig.add_axes([0.07, 0.13, 0.09, 0.045])
        next_ax = self.fig.add_axes([0.74, 0.13, 0.09, 0.045])
        self.prev_button = self.Button(prev_ax, "◀ prev")
        self.next_button = self.Button(next_ax, "next ▶")
        self.prev_button.on_clicked(lambda _e: self._step_target(-1))
        self.next_button.on_clicked(lambda _e: self._step_target(1))

        # Mode toggle button
        mode_ax = self.fig.add_axes([0.07, 0.05, 0.16, 0.05])
        self.mode_button = self.Button(mode_ax, "Switch to s-t ↕")
        self.mode_button.on_clicked(self._on_mode_toggle)

        # Geographic reconstruction toggle button
        geo_ax = self.fig.add_axes([0.26, 0.05, 0.22, 0.05])
        self.geo_button = self.Button(geo_ax, "Geo: full s<=smax")
        self.geo_button.on_clicked(self._on_geo_reconstruction_toggle)

    # ── Internal helpers ───────────────────────────────────────────────────

    @property
    def _current_idx(self) -> int:
        return min(int(round(self.target_slider.val)), self.viewer.n_targets - 1)

    def _step_target(self, delta: int) -> None:
        n = self.viewer.n_targets
        new_val = (self._current_idx + delta) % n
        self.target_slider.set_val(new_val)
        # on_changed will fire _redraw

    def _on_target_changed(self, _val) -> None:
        self._redraw()

    def _on_mode_toggle(self, _event) -> None:
        if self._mode == "geo":
            self._mode = "st"
            self.mode_button.label.set_text("Switch to geo (map)")
        else:
            self._mode = "geo"
            self.mode_button.label.set_text("Switch to s-t ↕")
        self._redraw()

    def _on_geo_reconstruction_toggle(self, _event) -> None:
        if self._geo_reconstruction == "smax":
            self._geo_reconstruction = "blocks"
            self.geo_button.label.set_text("Geo: model blocks")
        else:
            self._geo_reconstruction = "smax"
            self.geo_button.label.set_text("Geo: full s<=smax")
        if self._mode == "geo":
            self._redraw()

    def _on_key_press(self, event) -> None:
        if event.key in ("left", "right"):
            self._step_target(-1 if event.key == "left" else 1)
        elif event.key == "m":
            self._on_mode_toggle(None)
        elif event.key == "g":
            self._on_geo_reconstruction_toggle(None)

    def _set_axes_layout(self, two_panel: bool) -> None:
        """Resize/hide the two content axes according to layout mode."""
        if two_panel:
            self.ax_main.set_position(self._POS_LEFT)
            self.ax_side.set_position(self._POS_RIGHT)
            self.ax_side.set_visible(True)
        else:
            self.ax_main.set_position(self._POS_SINGLE)
            self.ax_side.set_visible(False)

    def _redraw(self) -> None:
        from property_targets import CapBulkTarget, BoxcarBulkTarget

        idx = self._current_idx
        target = self.viewer.targets[idx]
        is_bulk = isinstance(target, (CapBulkTarget, BoxcarBulkTarget))
        type_name = type(target).__name__

        self.status_text.set_text(
            f"Target {idx}/{self.viewer.n_targets - 1}  "
            f"({type_name})  "
            f"mode: {self._mode}  "
            f"geo: {self._geo_reconstruction}"
        )

        if self._mode == "geo":
            two_panel = is_bulk
            self._set_axes_layout(two_panel)
            _draw_geographic_axes(
                self.ax_main,
                self.ax_side if two_panel else None,
                self.viewer,
                idx,
                reconstruction=self._geo_reconstruction,
                plt=self.plt,
            )
        else:  # 'st'
            two_panel = is_bulk
            self._set_axes_layout(two_panel)
            _draw_st_axes(
                self.ax_main,
                self.ax_side if two_panel else None,
                self.viewer,
                idx,
                plt=self.plt,
            )

        self.fig.canvas.draw_idle()


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the standalone CLI."""
    p = argparse.ArgumentParser(
        description="Interactive target-kernel viewer for the full-spectrum PLI example."
    )
    _here = Path(__file__).resolve().parent
    _demo_dir = _here.parent
    default_data = str(_demo_dir / "data" / "normal-mode-data")
    default_kernels = str(
        _demo_dir
        / "data"
        / "normal-mode-kernels"
        / "kernels-all_PREM-layers_Adrian"
    )
    p.add_argument("--s-max", type=int, default=4,
                   help="Maximum splitting degree (default: 4).")
    p.add_argument("--data-dir", default=default_data,
                   help="Path to normal-mode data directory.")
    p.add_argument("--kernel-dir", default=default_kernels,
                   help="Path to normal-mode kernel directory.")
    p.add_argument("--n-radial", type=int, default=300,
                   help="Number of radial points for synthesis (default: 300).")
    p.add_argument("--mode", choices=["geo", "st"], default="geo",
                   help="Initial display mode (default: geo).")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() (for testing / batch use).")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Save the initial figure to this path and exit.")
    return p


def _make_default_targets(specs, s_max):
    """Build a small default set of targets for demonstration."""
    from property_targets import CapBulkTarget, BoxcarBulkTarget, CapCMBTarget

    earth_r = specs.earth_radius_km
    icb_r = specs.icb_radius_km
    cmb_r = specs.cmb_radius_km

    return [
        # Mid-mantle vp — C∞ bump, 15° cap
        CapBulkTarget("vp",   0.0,    0.0, 15.0, (cmb_r + earth_r) / 2, 400.0),
        CapBulkTarget("vp",  30.0,   60.0, 15.0, (cmb_r + earth_r) / 2, 400.0),
        CapBulkTarget("vp", -30.0, -120.0, 15.0, (cmb_r + earth_r) / 2, 400.0),
        # D'' layer vs — boxcar, 15° cap
        BoxcarBulkTarget("vs",   0.0,   0.0, 15.0, cmb_r, cmb_r + 300.0),
        BoxcarBulkTarget("vs",  45.0,  90.0, 15.0, cmb_r, cmb_r + 300.0),
        # Inner core vp — C∞ bump, 20° cap
        CapBulkTarget("vp",   0.0,    0.0, 20.0, icb_r / 2, 300.0),
        # CMB topography caps
        CapCMBTarget(  0.0,    0.0, 15.0),
        CapCMBTarget( 30.0,   60.0, 15.0),
        CapCMBTarget(-30.0, -120.0, 15.0),
    ]


def main(argv=None) -> None:
    """CLI entry point for the target-kernel viewer."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    args = build_parser().parse_args(argv)

    from full_spectrum_utils import enumerate_blocks, RadialSpecs
    from normal_mode_kernel_utils import (
        NormalModeDataRegistry,
        NormalModeKernelCatalog,
        EARTH_RADIUS_KM,
    )
    from intervalinf import ParallelConfig

    catalog = NormalModeKernelCatalog(args.kernel_dir)
    reg = NormalModeDataRegistry(
        args.data_dir, mode_filter=catalog.list_modes()
    )
    blocks = enumerate_blocks(reg, s_max=args.s_max)

    specs = RadialSpecs(
        n_basis=20,  # only used for default RadialSpecs; not needed for viewer
        parallel_cfg=ParallelConfig(enabled=False, n_jobs=1),
    )

    targets = _make_default_targets(specs, args.s_max)

    print(f"Loaded {len(blocks)} blocks, {len(targets)} targets.")
    print("Building TargetKernelViewer (pre-computing angular coefficients)…")

    viewer = TargetKernelViewer(
        targets,
        blocks,
        s_max=args.s_max,
        n_radial=args.n_radial,
        earth_radius_km=specs.earth_radius_km,
        icb_radius_km=specs.icb_radius_km,
        cmb_radius_km=specs.cmb_radius_km,
    )

    mpl_viewer = MatplotlibTargetKernelViewer(
        viewer,
        plt,
        Button,
        Slider,
        initial_mode=args.mode,
    )

    if args.save:
        mpl_viewer.save(Path(args.save))
        print(f"Saved to {args.save}")

    if not args.no_show:
        mpl_viewer.show()


if __name__ == "__main__":
    main()
