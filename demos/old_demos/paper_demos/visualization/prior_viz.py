"""
prior_viz.py
============

Prior visualisation helpers for the full-spectrum splitting-function PLI example.

Phase 1 — PriorViewer
~~~~~~~~~~~~~~~~~~~~~
:class:`PriorViewer` precomputes reference std arrays and sample curves for
each of the four radial model parameters (``'vp'``, ``'vs_IC'``, ``'vs_M'``,
``'rho'``) using the eigendecomposition already stored inside the shared
:class:`~intervalinf.operators.BesselSobolevInverse` operators.

All quantities are cached at τ=1.  Interactive updates (τ slider moves,
block switches) are then purely arithmetic — no operator calls are made after
:meth:`PriorViewer.precompute` returns.

Phase 2 — make_prior_widget (ipywidgets interactive viewer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
See :func:`make_prior_widget` (added in Phase 2).

Usage example
-------------
::

    from full_spectrum_utils import build_shared_bessel_blocks, RadialSpecs
    from prior_viz import PriorViewer

    specs = RadialSpecs(n_basis=100)
    shared = build_shared_bessel_blocks(specs)   # pre-built from notebook Phase 3

    viewer = PriorViewer(shared, specs, n_grid=300, n_samples=5)
    viewer.precompute()   # ~10-30 s depending on n_basis

    data = viewer.get_display_data(block, tau_vp=1.5, sigma_var=100.0)
    # data['vp']['std'], data['vp']['samples'], etc. — instant from cache

Standalone usage
----------------
::

    python visualization/prior_viz.py --precompute

This opens a Matplotlib control panel with block navigation, τ sliders,
σ_var control, and a resample button, analogous to
``visualization/normal_mode_data_viewer.py``.

Mathematical basis
------------------
The covariance operator for parameter *p* is

    C_p = τ²  C_ref_p

where C_ref_p = :class:`~intervalinf.operators.BesselSobolevInverse` has the
KL decomposition

    C_ref_p  f  =  Σ_k  λ_k  ⟨f, φ_k⟩  φ_k

with (λ_k, φ_k) the eigenpairs of the Laplacian-based spectral operator.
Pointwise variance at τ=1:

    σ²_ref(r)  =  Σ_k  λ_k  φ_k(r)²

A sample from N(0, C_ref_p) is:

    f_ref(r)  =  Σ_k  √λ_k  z_k  φ_k(r),   z_k ~ N(0, 1)

At arbitrary τ: std = τ · std_ref,  sample = τ · sample_ref.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

import _path_setup  # noqa: F401

if TYPE_CHECKING:
    import matplotlib.figure
    from full_spectrum_utils import BlockIndex, RadialSpecs
    from intervalinf.operators import BesselSobolevInverse


# =============================================================================
# Internal data containers
# =============================================================================

@dataclasses.dataclass
class _PriorComponentData:
    """Reference data at τ=1 for one radial parameter.

    All arrays are at τ=1; :meth:`PriorViewer.get_display_data` scales them.

    Attributes
    ----------
    r_grid : np.ndarray, shape (n_grid,)
        Radial evaluation grid in km.
    ref_std : np.ndarray, shape (n_grid,)
        Pointwise standard deviation at τ=1.
    ref_samples : np.ndarray, shape (n_samples, n_grid)
        Random sample curves at τ=1.
    _eigenvalues : np.ndarray, shape (n_basis,)
        KL eigenvalues (stored to allow :meth:`PriorViewer.resample` without
        re-fetching eigenpairs).
    _phi_matrix : np.ndarray, shape (n_basis, n_grid)
        Eigenfunction values on ``r_grid`` (stored for :meth:`PriorViewer.resample`).
    """

    r_grid: np.ndarray
    ref_std: np.ndarray
    ref_samples: np.ndarray
    _eigenvalues: np.ndarray
    _phi_matrix: np.ndarray


# Domain bounds per parameter — expressed as (a, b) lambdas over RadialSpecs.
_PARAM_DOMAINS: Dict[str, Callable] = {
    "vp":    lambda s: (0.0, s.earth_radius_km),
    "vs_IC": lambda s: (0.0, s.icb_radius_km),
    "vs_M":  lambda s: (s.cmb_radius_km, s.earth_radius_km),
    "rho":   lambda s: (0.0, s.earth_radius_km),
}

_RADIAL_PARAMS = ("vp", "vs_IC", "vs_M", "rho")


# =============================================================================
# PriorViewer
# =============================================================================

class PriorViewer:
    """Precompute and serve display data for block prior visualisation.

    All expensive work is done once in :meth:`precompute`.  After that,
    :meth:`get_display_data` performs only arithmetic (τ scaling), so
    interactive updates are instant regardless of how many blocks exist.

    Parameters
    ----------
    shared_bessel : dict
        Output of :func:`~full_spectrum_utils.build_shared_bessel_blocks`.
        Keys: ``'vp'``, ``'vs_IC'``, ``'vs_M'``, ``'rho'``.
    specs : RadialSpecs
        Shared domain / configuration (domain bounds, ``n_basis``).
    n_grid : int
        Number of radial evaluation points per component.  Default 300.
    n_samples : int
        Number of sample curves to precompute per component.  Default 5.
    seed : int
        Random seed for reproducible samples.  Default 42.
    """

    def __init__(
        self,
        shared_bessel: Dict[str, "BesselSobolevInverse"],
        specs: "RadialSpecs",
        *,
        n_grid: int = 300,
        n_samples: int = 5,
        seed: int = 42,
    ) -> None:
        self._shared_bessel = shared_bessel
        self._specs = specs
        self._n_grid = n_grid
        self._n_samples = n_samples
        self._seed = seed
        self._ref_data: Dict[str, _PriorComponentData] = {}
        self._scalar_z: Optional[np.ndarray] = None  # shape (n_samples,)

    # ── Precomputation ───────────────────────────────────────────────────────

    def precompute(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Precompute reference std and sample curves for all four parameters.

        Iterates over ``('vp', 'vs_IC', 'vs_M', 'rho')``.  For each parameter
        it evaluates all ``specs.n_basis`` eigenpairs of the corresponding
        :class:`~intervalinf.operators.BesselSobolevInverse` on a dense grid,
        then constructs the pointwise-variance std and ``n_samples`` KL samples.

        After this call, :attr:`is_precomputed` is ``True``.

        Parameters
        ----------
        progress_cb : callable, optional
            Invoked with each parameter name string after it completes.
            Useful for notebook progress bars.

        Notes
        -----
        Eigenfunction evaluation uses array-valued calls (``phi_k(r_grid)``)
        when the underlying callable is vectorised (the default for spectral
        providers).  If a particular provider is not vectorised, the code
        falls back to a scalar loop automatically.
        """
        rng = np.random.default_rng(self._seed)
        n_basis = self._specs.n_basis

        for p in _RADIAL_PARAMS:
            a, b = _PARAM_DOMAINS[p](self._specs)
            r_grid = np.linspace(a, b, self._n_grid)
            bessel = self._shared_bessel[p]

            eigenvalues = np.empty(n_basis)
            phi_matrix = np.empty((n_basis, self._n_grid))

            for k in range(n_basis):
                eigenvalues[k] = bessel.get_eigenvalue(k)
                phi_k = bessel.get_eigenfunction(k)
                # Try vectorised evaluation; fall back to scalar loop.
                try:
                    vals = phi_k(r_grid)
                    if not isinstance(vals, np.ndarray) or vals.shape != (self._n_grid,):
                        raise ValueError
                    phi_matrix[k] = vals
                except Exception:
                    phi_matrix[k] = np.array([phi_k(r) for r in r_grid])

            # Reference pointwise std: sqrt(Σ_k λ_k φ_k(r)²)
            ref_var = eigenvalues @ (phi_matrix ** 2)  # (n_grid,)
            ref_std = np.sqrt(np.maximum(0.0, ref_var))

            # Reference samples: shape (n_samples, n_grid)
            # f_ref(r) = Σ_k sqrt(λ_k) z_k φ_k(r),  z ~ N(0,1)
            z = rng.standard_normal((self._n_samples, n_basis))  # (n_samples, n_basis)
            sqrt_lam = np.sqrt(np.maximum(0.0, eigenvalues))      # (n_basis,)
            ref_samples = z @ (sqrt_lam[:, None] * phi_matrix)    # (n_samples, n_grid)

            self._ref_data[p] = _PriorComponentData(
                r_grid=r_grid,
                ref_std=ref_std,
                ref_samples=ref_samples,
                _eigenvalues=eigenvalues,
                _phi_matrix=phi_matrix,
            )

            if progress_cb is not None:
                progress_cb(p)

        # Scalar component z-values (scaled by sqrt(sigma_var) in get_display_data)
        self._scalar_z = rng.standard_normal(self._n_samples)

    def resample(self, seed: Optional[int] = None) -> None:
        """Redraw sample curves without rebuilding eigenpairs.

        The pointwise std (:attr:`~_PriorComponentData.ref_std`) is
        unchanged — it depends only on eigenvalues, not on the random draw.

        Parameters
        ----------
        seed : int, optional
            New random seed.  If ``None``, uses a fresh random integer.

        Raises
        ------
        RuntimeError
            If :meth:`precompute` has not been called yet.
        """
        if not self._ref_data:
            raise RuntimeError("Call precompute() before resample().")
        rng = np.random.default_rng(
            seed if seed is not None else int(np.random.randint(0, 2**31))
        )
        for data in self._ref_data.values():
            n_basis = len(data._eigenvalues)
            z = rng.standard_normal((self._n_samples, n_basis))
            sqrt_lam = np.sqrt(np.maximum(0.0, data._eigenvalues))
            data.ref_samples = z @ (sqrt_lam[:, None] * data._phi_matrix)

        self._scalar_z = rng.standard_normal(self._n_samples)

    def invalidate_bessel(
        self,
        new_shared_bessel: Dict[str, "BesselSobolevInverse"],
    ) -> None:
        """Replace the underlying Bessel operators and clear the cache.

        Call this when ``specs.n_basis`` or the Bessel hyperparameters change
        (e.g., after rebuilding
        :func:`~full_spectrum_utils.build_shared_bessel_blocks`), then call
        :meth:`precompute` again.

        Parameters
        ----------
        new_shared_bessel : dict
            Fresh output of
            :func:`~full_spectrum_utils.build_shared_bessel_blocks`.
        """
        self._shared_bessel = new_shared_bessel
        self._ref_data.clear()
        self._scalar_z = None

    # ── Display data ─────────────────────────────────────────────────────────

    def get_display_data(
        self,
        block: "BlockIndex",
        *,
        tau_vp: float = 1.0,
        tau_vs_IC: float = 1.0,
        tau_vs_M: float = 1.0,
        tau_rho: float = 1.0,
        sigma_var: float = 100.0,
    ) -> Dict:
        """Return τ-scaled display data for *block*.

        This method performs only arithmetic (multiplication by τ); all
        operator calls were completed during :meth:`precompute`.

        Parameters
        ----------
        block : BlockIndex
            The (s, t) spectral block to display.  Not used computationally;
            carried through for labelling and future per-s τ support.
        tau_vp, tau_vs_IC, tau_vs_M, tau_rho : float
            Amplitude scaling applied uniformly across all s.  The prior
            covariance is $τ^2 C_\\mathrm{ref}$, so std scales by τ and
            samples scale by τ.
        sigma_var : float
            Variance for the scalar topography components σ₀ and σ₁.

        Returns
        -------
        dict
            Keys ``'vp'``, ``'vs_IC'``, ``'vs_M'``, ``'rho'`` each map to::

                {
                    'r':       np.ndarray of shape (n_grid,),   # km
                    'mean':    np.ndarray of shape (n_grid,),   # zeros
                    'std':     np.ndarray of shape (n_grid,),   # τ · std_ref
                    'samples': np.ndarray of shape (n_samples, n_grid),
                }

            Key ``'sigma_1'`` maps to::

                {
                    'mean':    0.0,
                    'std':     float  (= sqrt(sigma_var)),
                    'samples': np.ndarray of shape (n_samples,),
                }

        Raises
        ------
        RuntimeError
            If :meth:`precompute` has not been called yet.
        """
        if not self._ref_data:
            raise RuntimeError("Call precompute() before get_display_data().")

        taus = {
            "vp":    tau_vp,
            "vs_IC": tau_vs_IC,
            "vs_M":  tau_vs_M,
            "rho":   tau_rho,
        }

        result: Dict = {}
        for p, tau in taus.items():
            data = self._ref_data[p]
            result[p] = {
                "r":       data.r_grid,
                "mean":    np.zeros(self._n_grid),
                "std":     tau * data.ref_std,
                "samples": tau * data.ref_samples,  # (n_samples, n_grid)
            }

        sigma_std = float(np.sqrt(sigma_var))
        scalar_samples = (
            sigma_std * self._scalar_z
            if self._scalar_z is not None
            else np.zeros(self._n_samples)
        )
        result["sigma_1"] = {
            "mean":    0.0,
            "std":     sigma_std,
            "samples": scalar_samples,
        }

        return result

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_precomputed(self) -> bool:
        """``True`` if :meth:`precompute` has been called and data is available."""
        return bool(self._ref_data)

    @property
    def n_grid(self) -> int:
        """Number of radial evaluation points per component."""
        return self._n_grid

    @property
    def n_samples(self) -> int:
        """Number of precomputed sample curves per component."""
        return self._n_samples


# =============================================================================
# Phase 2 — Figure renderer
# =============================================================================

# Colour palette matching plot_block_posterior
_PARAM_COLORS = {
    "vp":    "steelblue",
    "vs_IC": "coral",
    "vs_M":  "coral",
    "rho":   "seagreen",
    "sigma_1": "mediumpurple",
}

_PARAM_LABELS = {
    "vp":    "δvp (km s⁻¹)",
    "vs_IC": "δvs IC (km s⁻¹)",
    "vs_M":  "δvs mantle (km s⁻¹)",
    "rho":   "δρ (g cm⁻³)",
    "sigma_1": "σ₁ (km)",
}


def _plot_prior_panel(
    ax,
    r: np.ndarray,
    std: np.ndarray,
    samples: np.ndarray,
    label: str,
    color: str = "steelblue",
    n_sigma: float = 2.0,
) -> None:
    """Draw zero mean, ±n_sigma band, and sample curves on *ax* (radial component)."""
    # ±n_sigma shaded band
    ax.fill_betweenx(r, -n_sigma * std, n_sigma * std,
                     alpha=0.20, color=color, label=f"±{n_sigma:.0f}σ")
    # ±1σ band (darker)
    ax.fill_betweenx(r, -std, std,
                     alpha=0.20, color=color)
    # sample curves
    for s_row in samples:
        ax.plot(s_row, r, color=color, lw=0.6, alpha=0.45)
    # zero mean (dashed)
    ax.axvline(0.0, color="k", lw=0.8, ls="--", label="mean")
    ax.set_xlabel(label, fontsize=7)
    ax.set_ylabel("radius (km)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)
    # Set x-axis limits based on maximum amplitude in data (with 10% padding)
    max_amp = max(n_sigma * np.max(std), np.max(np.abs(samples)))
    limit = max_amp * 1.1 if max_amp > 0 else 1.0
    ax.set_xlim(-limit, limit)


def _plot_prior_sigma1_panel(
    ax,
    std: float,
    samples: np.ndarray,
    color: str = "mediumpurple",
    n_sigma: float = 2.0,
) -> None:
    """Draw σ₁ prior as a vertical strip at x=0 with sample dots."""
    ax.axhspan(-n_sigma * std, n_sigma * std, alpha=0.20, color=color,
               label=f"±{n_sigma:.0f}σ")
    ax.axhspan(-std, std, alpha=0.20, color=color)
    ax.axhline(0.0, color="k", lw=0.8, ls="--", label="mean")
    # sample dots
    ax.scatter(
        np.zeros(len(samples)), samples,
        color=color, s=20, alpha=0.7, zorder=3,
    )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("σ₁ (km)", fontsize=7)
    ax.set_xlabel("", fontsize=7)
    ax.set_xticks([])
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)


def _plot_prior_vs_panel(
    ax,
    r_IC: np.ndarray,
    std_IC: np.ndarray,
    samples_IC: np.ndarray,
    r_M: np.ndarray,
    std_M: np.ndarray,
    samples_M: np.ndarray,
    icb_km: float,
    cmb_km: float,
    color: str = "coral",
    n_sigma: float = 2.0,
) -> None:
    """Draw vs_IC and vs_M on a single axis with a gray outer-core band.

    The outer core (ICB → CMB) is shaded light gray.  IC and mantle vs
    uncertainty bands and sample curves share the same colour and axes.
    """
    # Outer core — gray shaded region
    ax.axhspan(icb_km, cmb_km, color="lightgray", alpha=0.6, label="outer core")
    # Inner-core vs (r ∈ [0, ICB])
    ax.fill_betweenx(r_IC, -n_sigma * std_IC, n_sigma * std_IC,
                     alpha=0.20, color=color, label=f"±{n_sigma:.0f}σ")
    ax.fill_betweenx(r_IC, -std_IC, std_IC, alpha=0.20, color=color)
    for s_row in samples_IC:
        ax.plot(s_row, r_IC, color=color, lw=0.6, alpha=0.45)
    # Mantle vs (r ∈ [CMB, R_earth])
    ax.fill_betweenx(r_M, -n_sigma * std_M, n_sigma * std_M,
                     alpha=0.20, color=color)
    ax.fill_betweenx(r_M, -std_M, std_M, alpha=0.20, color=color)
    for s_row in samples_M:
        ax.plot(s_row, r_M, color=color, lw=0.6, alpha=0.45)
    # Zero mean
    ax.axvline(0.0, color="k", lw=0.8, ls="--", label="mean")
    ax.set_xlabel("δvs (km s⁻¹)", fontsize=7)
    ax.set_ylabel("radius (km)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)
    # Set x-axis limits based on maximum amplitude across IC and mantle (with 10% padding)
    max_amp = max(
        n_sigma * np.max(std_IC),
        n_sigma * np.max(std_M),
        np.max(np.abs(samples_IC)),
        np.max(np.abs(samples_M)),
    )
    limit = max_amp * 1.1 if max_amp > 0 else 1.0
    ax.set_xlim(-limit, limit)


def _render_prior_figure(
    display_data: Dict,
    block: "BlockIndex",
    specs: "RadialSpecs",
    *,
    n_sigma: float = 2.0,
    figsize: tuple = (15, 4),
) -> "matplotlib.figure.Figure":
    """Render a 4-panel figure showing the prior for one (s, t) block.

    Panels
    ------
    1. δvp — radial prior with ±{n_sigma}σ band and samples
    2. δvs — IC and mantle shear velocity on one axis (gray outer core)
    3. δρ — density prior
    4. σ₁ — CMB topography (scalar Gaussian prior)

    Parameters
    ----------
    display_data : dict
        Output of :meth:`PriorViewer.get_display_data`.
    block : BlockIndex
        Used for the figure suptitle.
    specs : RadialSpecs
        Domain specs (used for ICB / CMB boundary markers).
    n_sigma : float
        Half-width of the shaded uncertainty band in units of σ.  Default 2.
    figsize : tuple
        Figure size in inches.  Default ``(15, 4)``.

    Returns
    -------
    matplotlib.figure.Figure
        Four-panel figure; caller is responsible for closing/saving.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(f"Prior — block (s={block.s}, t={block.t})", fontsize=9)
    _draw_prior_axes(axes, display_data, block, specs, n_sigma=n_sigma)

    fig.tight_layout()
    return fig


def _draw_prior_axes(
    axes,
    display_data: Dict,
    block: "BlockIndex",
    specs: "RadialSpecs",
    *,
    n_sigma: float = 2.0,
) -> None:
    """Draw the four prior panels onto an existing axes array."""
    _ = block
    if len(axes) != 4:
        raise ValueError("Expected exactly 4 axes for prior rendering.")

    for ax in axes:
        ax.clear()
        ax.set_axis_on()

    # Panel 0 — δvp
    d = display_data["vp"]
    _plot_prior_panel(
        axes[0],
        r=d["r"], std=d["std"], samples=d["samples"],
        label=_PARAM_LABELS["vp"], color=_PARAM_COLORS["vp"],
        n_sigma=n_sigma,
    )

    # Panel 1 — δvs (IC + mantle combined, with gray outer core)
    d_IC = display_data["vs_IC"]
    d_M  = display_data["vs_M"]
    _plot_prior_vs_panel(
        axes[1],
        r_IC=d_IC["r"], std_IC=d_IC["std"], samples_IC=d_IC["samples"],
        r_M=d_M["r"],   std_M=d_M["std"],   samples_M=d_M["samples"],
        icb_km=specs.icb_radius_km,
        cmb_km=specs.cmb_radius_km,
        color=_PARAM_COLORS["vs_IC"],
        n_sigma=n_sigma,
    )

    # Panel 2 — δρ
    d = display_data["rho"]
    _plot_prior_panel(
        axes[2],
        r=d["r"], std=d["std"], samples=d["samples"],
        label=_PARAM_LABELS["rho"], color=_PARAM_COLORS["rho"],
        n_sigma=n_sigma,
    )

    # Panel 3 — σ₁
    d1 = display_data["sigma_1"]
    _plot_prior_sigma1_panel(
        axes[3],
        std=d1["std"],
        samples=d1["samples"],
        color=_PARAM_COLORS["sigma_1"],
        n_sigma=n_sigma,
    )


# =============================================================================
# Phase 2 — ipywidgets interactive viewer
# =============================================================================

def make_prior_widget(
    viewer: "PriorViewer",
    blocks: list,
    specs: "RadialSpecs",
    *,
    tau_init: Optional[Dict[str, float]] = None,
    sigma_var_init: float = 100.0,
) -> "ipywidgets.Widget":
    """Create an ipywidgets interactive panel for browsing the prior.

    The widget allows the user to:

    - select a block from a dropdown
    - adjust τ (amplitude scale) independently for each of the four radial
      parameters via float sliders
    - adjust σ_var for the CMB topography component
    - trigger a full precompute (with per-parameter progress feedback)
    - resample sample curves without rebuilding eigenpairs

    All slider interactions after precomputation are instant — they scale
    the cached reference arrays without calling any operators.

    Parameters
    ----------
    viewer : PriorViewer
        A (possibly not yet precomputed) :class:`PriorViewer` instance.
    blocks : list of BlockIndex
        Blocks to expose in the dropdown.
    specs : RadialSpecs
        Domain specs forwarded to :func:`_render_prior_figure`.
    tau_init : dict, optional
        Initial τ values, e.g. ``{'vp': 1.0, 'vs_IC': 1.0, ...}``.
        Defaults to 1.0 for all parameters.
    sigma_var_init : float
        Initial σ_var value.  Default 100.0.

    Returns
    -------
    ipywidgets.VBox
        A self-contained widget ready to be displayed in a Jupyter cell.
    """
    import ipywidgets as widgets
    from IPython.display import display as _display, clear_output

    _tau = tau_init or {}
    tau_defaults = {
        "vp":    _tau.get("vp",    1.0),
        "vs_IC": _tau.get("vs_IC", 1.0),
        "vs_M":  _tau.get("vs_M",  1.0),
        "rho":   _tau.get("rho",   1.0),
    }

    # ── Controls ─────────────────────────────────────────────────────────────

    block_options = [f"s={b.s}, t={b.t}" for b in blocks]
    block_dd = widgets.Dropdown(
        options=list(zip(block_options, blocks)),
        description="Block:",
        layout=widgets.Layout(width="200px"),
    )

    def _tau_slider(name, val):
        return widgets.FloatSlider(
            value=val, min=0.0, max=10.0, step=0.05,
            description=f"τ_{name}:", continuous_update=True,
            style={"description_width": "70px"},
            layout=widgets.Layout(width="300px"),
        )

    sl_vp    = _tau_slider("vp",    tau_defaults["vp"])
    sl_vs_IC = _tau_slider("vs_IC", tau_defaults["vs_IC"])
    sl_vs_M  = _tau_slider("vs_M",  tau_defaults["vs_M"])
    sl_rho   = _tau_slider("rho",   tau_defaults["rho"])
    sl_sigma = widgets.FloatLogSlider(
        value=sigma_var_init, base=10, min=-1, max=4, step=0.05,
        description="σ_var:", continuous_update=True,
        style={"description_width": "70px"},
        layout=widgets.Layout(width="300px"),
    )

    btn_precompute = widgets.Button(
        description="Precompute", button_style="primary",
        layout=widgets.Layout(width="120px"),
    )
    btn_resample = widgets.Button(
        description="Resample", button_style="",
        layout=widgets.Layout(width="100px"),
    )
    progress_label = widgets.Label(value="")

    out = widgets.Output()

    # ── Render helper ─────────────────────────────────────────────────────────

    def _redraw(*_args):
        if not viewer.is_precomputed:
            with out:
                clear_output(wait=True)
                print("Press 'Precompute' to build the prior display data.")
            return
        block = block_dd.value
        data = viewer.get_display_data(
            block,
            tau_vp=sl_vp.value,
            tau_vs_IC=sl_vs_IC.value,
            tau_vs_M=sl_vs_M.value,
            tau_rho=sl_rho.value,
            sigma_var=sl_sigma.value,
        )
        import matplotlib.pyplot as plt
        fig = _render_prior_figure(data, block, specs)
        with out:
            clear_output(wait=True)
            _display(fig)
        plt.close(fig)

    # ── Precompute callback ───────────────────────────────────────────────────

    def _on_precompute(_btn):
        btn_precompute.disabled = True
        progress_label.value = "Precomputing…"
        completed: List[str] = []

        def _cb(p: str) -> None:
            completed.append(p)
            progress_label.value = f"Done: {', '.join(completed)}"

        viewer.precompute(progress_cb=_cb)
        progress_label.value = f"Done: {', '.join(completed)} ✓"
        btn_precompute.disabled = False
        _redraw()

    def _on_resample(_btn):
        if viewer.is_precomputed:
            viewer.resample()
            _redraw()

    btn_precompute.on_click(_on_precompute)
    btn_resample.on_click(_on_resample)

    # Observe all interactive controls
    for ctrl in (block_dd, sl_vp, sl_vs_IC, sl_vs_M, sl_rho, sl_sigma):
        ctrl.observe(_redraw, names="value")

    # ── Layout ────────────────────────────────────────────────────────────────

    controls = widgets.VBox([
        widgets.HBox([block_dd]),
        widgets.HBox([sl_vp, sl_vs_IC]),
        widgets.HBox([sl_vs_M, sl_rho]),
        widgets.HBox([sl_sigma]),
        widgets.HBox([btn_precompute, btn_resample, progress_label]),
    ])

    return widgets.VBox([controls, out])


# =============================================================================
# Phase 3 — Standalone Matplotlib viewer
# =============================================================================

class MatplotlibPriorViewer:
    """Standalone Matplotlib control panel for browsing the cached prior."""

    def __init__(
        self,
        viewer: "PriorViewer",
        blocks: list,
        specs: "RadialSpecs",
        plt,
        button_class,
        slider_class,
        *,
        tau_init: Optional[Dict[str, float]] = None,
        sigma_var_init: float = 100.0,
        n_sigma: float = 2.0,
    ) -> None:
        if not blocks:
            raise ValueError("MatplotlibPriorViewer requires at least one block.")

        self.viewer = viewer
        self.blocks = list(blocks)
        self.specs = specs
        self.plt = plt
        self.Button = button_class
        self.Slider = slider_class
        self.n_sigma = n_sigma

        tau_defaults = tau_init or {}
        self._tau_defaults = {
            "vp": tau_defaults.get("vp", 1.0),
            "vs_IC": tau_defaults.get("vs_IC", 1.0),
            "vs_M": tau_defaults.get("vs_M", 1.0),
            "rho": tau_defaults.get("rho", 1.0),
        }

        self.fig, self.axes = self.plt.subplots(1, 4, figsize=(15, 6))
        self.fig.subplots_adjust(left=0.05, right=0.98, bottom=0.28, top=0.86, wspace=0.35)
        self.status_text = self.fig.text(0.05, 0.91, "", fontsize=9)
        self._build_widgets(sigma_var_init)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._show_placeholder("Press 'Precompute' to build the prior cache.")

    @property
    def current_block(self) -> "BlockIndex":
        """Return the block currently selected by the block slider."""
        index = min(int(round(self.block_slider.val)), len(self.blocks) - 1)
        return self.blocks[index]

    @property
    def current_sigma_var(self) -> float:
        """Return the current scalar prior variance from the log slider."""
        return float(10.0 ** self.sigma_slider.val)

    def show(self) -> None:
        """Open the interactive Matplotlib window."""
        self.plt.show()

    def save(self, path: Path) -> None:
        """Save the current figure state to disk."""
        self.fig.savefig(path, dpi=160, bbox_inches="tight")

    def precompute(self) -> None:
        """Run the full prior precompute and redraw the current block."""
        self._on_precompute(None)

    def _build_widgets(self, sigma_var_init: float) -> None:
        block_valmax = max(len(self.blocks) - 1, 1)
        block_ax = self.fig.add_axes([0.18, 0.18, 0.50, 0.03])
        self.block_slider = self.Slider(
            block_ax,
            "block index",
            0,
            block_valmax,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        self.block_slider.on_changed(self._set_block_index)

        prev_ax = self.fig.add_axes([0.05, 0.17, 0.09, 0.045])
        next_ax = self.fig.add_axes([0.71, 0.17, 0.09, 0.045])
        self.prev_button = self.Button(prev_ax, "previous")
        self.next_button = self.Button(next_ax, "next")
        self.prev_button.on_clicked(lambda _event: self._step_block(-1))
        self.next_button.on_clicked(lambda _event: self._step_block(1))

        self.tau_vp_slider = self._make_tau_slider([0.08, 0.11, 0.26, 0.03], "τ_vp", self._tau_defaults["vp"])
        self.tau_vs_ic_slider = self._make_tau_slider([0.39, 0.11, 0.26, 0.03], "τ_vs_IC", self._tau_defaults["vs_IC"])
        self.tau_vs_m_slider = self._make_tau_slider([0.08, 0.06, 0.26, 0.03], "τ_vs_M", self._tau_defaults["vs_M"])
        self.tau_rho_slider = self._make_tau_slider([0.39, 0.06, 0.26, 0.03], "τ_rho", self._tau_defaults["rho"])

        sigma_ax = self.fig.add_axes([0.72, 0.11, 0.23, 0.03])
        self.sigma_slider = self.Slider(
            sigma_ax,
            "log10 σ_var",
            -1.0,
            4.0,
            valinit=float(np.log10(max(sigma_var_init, 1e-1))),
            valstep=0.05,
        )
        self.sigma_slider.on_changed(self._set_block_index)

        precompute_ax = self.fig.add_axes([0.72, 0.05, 0.11, 0.045])
        resample_ax = self.fig.add_axes([0.84, 0.05, 0.11, 0.045])
        self.precompute_button = self.Button(precompute_ax, "precompute")
        self.resample_button = self.Button(resample_ax, "resample")
        self.precompute_button.on_clicked(self._on_precompute)
        self.resample_button.on_clicked(self._on_resample)

    def _make_tau_slider(self, bounds, label: str, value: float):
        slider_ax = self.fig.add_axes(bounds)
        slider = self.Slider(
            slider_ax,
            label,
            0.0,
            10.0,
            valinit=value,
            valstep=0.05,
        )
        slider.on_changed(self._set_block_index)
        return slider

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _show_placeholder(self, message: str) -> None:
        for ax in self.axes:
            ax.clear()
            ax.set_axis_off()
        self.axes[2].text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.axes[2].transAxes,
            fontsize=11,
        )
        self.fig.suptitle("Prior viewer", fontsize=10)
        self._set_status(message)

    def _redraw(self) -> None:
        if not self.viewer.is_precomputed:
            self._show_placeholder("Press 'Precompute' to build the prior cache.")
            return

        block = self.current_block
        display_data = self.viewer.get_display_data(
            block,
            tau_vp=self.tau_vp_slider.val,
            tau_vs_IC=self.tau_vs_ic_slider.val,
            tau_vs_M=self.tau_vs_m_slider.val,
            tau_rho=self.tau_rho_slider.val,
            sigma_var=self.current_sigma_var,
        )
        _draw_prior_axes(self.axes, display_data, block, self.specs, n_sigma=self.n_sigma)
        self.fig.suptitle(f"Prior — block (s={block.s}, t={block.t})", fontsize=10)
        self._set_status(
            f"Ready — block (s={block.s}, t={block.t}), σ_var={self.current_sigma_var:.3g}"
        )

    def _set_block_index(self, _value: float) -> None:
        self._redraw()

    def _step_block(self, step: int) -> None:
        new_index = (int(round(self.block_slider.val)) + step) % len(self.blocks)
        self.block_slider.set_val(new_index)

    def _on_key_press(self, event) -> None:
        if event.key == "left":
            self._step_block(-1)
        elif event.key == "right":
            self._step_block(1)
        elif event.key == "r":
            self._on_resample(None)
        elif event.key == "p":
            self._on_precompute(None)

    def _on_precompute(self, _event) -> None:
        self._set_status("Precomputing prior cache...")
        completed: List[str] = []

        def _cb(param: str) -> None:
            completed.append(param)
            self._set_status(f"Precomputing prior cache... {', '.join(completed)}")

        self.viewer.precompute(progress_cb=_cb)
        self._redraw()

    def _on_resample(self, _event) -> None:
        if not self.viewer.is_precomputed:
            self._show_placeholder("Press 'Precompute' before resampling.")
            return
        self.viewer.resample()
        self._redraw()


def _default_blocks(s_max: int) -> list:
    """Return a simple synthetic block list for standalone prior browsing."""
    from full_spectrum_utils import BlockIndex

    blocks = [
        BlockIndex(s=s, t=t)
        for s in range(0, max(s_max, 0) + 1, 2)
        for t in range(2 * s + 1)
    ]
    return blocks or [BlockIndex(s=0, t=0)]


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone prior-viewer CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-max", type=int, default=4, help="Maximum even s to include in the block selector.")
    parser.add_argument("--n-basis", type=int, default=100, help="Number of radial basis functions per parameter.")
    parser.add_argument("--n-grid", type=int, default=300, help="Number of radial evaluation points per panel.")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of prior sample curves to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KL samples.")
    parser.add_argument("--sigma-var", type=float, default=100.0, help="Initial scalar topography prior variance.")
    parser.add_argument("--precompute", action="store_true", help="Precompute the prior cache before opening the window.")
    parser.add_argument("--save", type=Path, default=None, help="Save the current viewer figure to this path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive window.")
    return parser


def main() -> None:
    """Run the standalone Matplotlib prior viewer."""
    parser = build_parser()
    args = parser.parse_args()

    import matplotlib

    if args.no_show or args.save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from full_spectrum_utils import RadialSpecs, build_shared_bessel_blocks

    specs = RadialSpecs(n_basis=args.n_basis)
    shared_bessel = build_shared_bessel_blocks(specs)
    viewer = PriorViewer(
        shared_bessel,
        specs,
        n_grid=args.n_grid,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    app = MatplotlibPriorViewer(
        viewer,
        _default_blocks(args.s_max),
        specs,
        plt,
        Button,
        Slider,
        sigma_var_init=args.sigma_var,
    )

    if args.precompute or args.save:
        app.precompute()
    if args.save is not None:
        app.save(args.save)
    if not args.no_show:
        app.show()


if __name__ == "__main__":
    main()
