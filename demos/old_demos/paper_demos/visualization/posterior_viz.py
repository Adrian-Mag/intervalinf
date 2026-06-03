"""
posterior_viz.py
================

Posterior model-space visualisation helpers for the full-spectrum PLI example.

Provides:

    ``_PostBlockDisplayData``
        Frozen dataclass holding all precomputed arrays for one block's posterior
        (mean functions evaluated on a dense grid plus uncertainty std values at
        probe locations from covariance probing).

    ``PosteriorViewer``
        Engine that caches ``_PostBlockDisplayData`` objects per
        :class:`~full_spectrum_utils.BlockIndex`.  Expensive covariance probing
        is done lazily on first request and then cached.

    ``_draw_posterior_axes``
        Draw the 5 posterior panels (δvp, δvs IC, δvs mantle, δρ, σ₁) into an
        existing array of :class:`matplotlib.axes.Axes`.

    ``render_posterior_figure``
        Create a new figure + 5-panel axes and delegate to ``_draw_posterior_axes``.

    ``MatplotlibPosteriorViewer``
        Standalone Matplotlib interactive control panel for browsing
        pre-computed block posteriors.

    ``build_parser`` / ``main``
        CLI entry point.  Runs the full pipeline (load catalog, build forward
        operators, solve per-block posteriors) then opens the viewer.

Usage (standalone)::

    python visualization/posterior_viz.py
    python visualization/posterior_viz.py --s-max 2 --n-basis 50 --n-probes 10
    python visualization/posterior_viz.py --no-show --save /tmp/post.png --precompute

Data directories default to ``data/normal-mode-data`` and
``data/normal-mode-kernels/kernels-all_PREM-layers_Adrian`` relative to the
``paper_demos`` directory (same defaults as the example notebook).  Override with
``--data-dir`` and ``--kernel-dir``.

Keyboard shortcuts in the interactive window:

    ←  / →      previous / next block
    c           compute current block
    a           precompute all blocks
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

import _path_setup  # noqa: F401

if TYPE_CHECKING:
    import matplotlib.figure
    from full_spectrum_utils import BlockIndex, RadialSpecs
    from pygeoinf import GaussianMeasure

# ---------------------------------------------------------------------------
# Colour / label constants
# ---------------------------------------------------------------------------

_PARAM_COLORS: Dict[str, str] = {
    "vp":     "steelblue",
    "vs_IC":  "forestgreen",
    "vs_M":   "seagreen",
    "rho":    "goldenrod",
    "sigma_1": "firebrick",
}

_PARAM_LABELS: Dict[str, str] = {
    "vp":     "δvp (km s⁻¹)",
    "vs_IC":  "δvs IC (km s⁻¹)",
    "vs_M":   "δvs mantle (km s⁻¹)",
    "rho":    "δρ (g cm⁻³)",
    "sigma_1": "σ₁ (km)",
}


# ---------------------------------------------------------------------------
# _PostBlockDisplayData
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _PostBlockDisplayData:
    """All arrays needed to draw one block's posterior in ``_draw_posterior_axes``.

    Attributes
    ----------
    r_vp, r_vs_IC, r_vs_M, r_rho : np.ndarray, shape (n_grid,)
        Dense radial grids for each component.
    mean_vp, mean_vs_IC, mean_vs_M, mean_rho : np.ndarray, shape (n_grid,)
        Posterior mean function values on the corresponding grid.
    probe_r_vp, probe_r_vs_IC, probe_r_vs_M, probe_r_rho : np.ndarray, shape (n_probes,)
        Radial positions at which the covariance was probed.
    std_vp, std_vs_IC, std_vs_M, std_rho : np.ndarray, shape (n_probes,)
        Pointwise std estimates at the probe positions.
    sigma_1_mean : float
        Posterior mean of the scalar CMB topography coefficient.
    sigma_1_std : float
        Posterior std of the scalar CMB topography coefficient.
    """

    r_vp:     np.ndarray
    r_vs_IC:  np.ndarray
    r_vs_M:   np.ndarray
    r_rho:    np.ndarray

    mean_vp:     np.ndarray
    mean_vs_IC:  np.ndarray
    mean_vs_M:   np.ndarray
    mean_rho:    np.ndarray

    probe_r_vp:     np.ndarray
    probe_r_vs_IC:  np.ndarray
    probe_r_vs_M:   np.ndarray
    probe_r_rho:    np.ndarray

    std_vp:     np.ndarray
    std_vs_IC:  np.ndarray
    std_vs_M:   np.ndarray
    std_rho:    np.ndarray

    sigma_1_mean: float
    sigma_1_std:  float

    d_pred_post: np.ndarray
    """Posterior predictive: G @ posterior_mean, shape (N_d,)."""


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_posterior_display_data(
    block: "BlockIndex",
    posterior: "GaussianMeasure",
    forward_dict: dict,
    specs: "RadialSpecs",
    *,
    n_grid: int = 200,
    n_probes: int = 20,
) -> "_PostBlockDisplayData":
    """Compute mean and ±1σ std at probe positions for one block (expensive).

    For the functional components, the uncertainty std is estimated by probing
    the posterior covariance operator with ``n_probes`` narrow normalised
    Gaussian bumps distributed uniformly across each component's domain.  Each
    bump $b_i$ satisfies $\\|b_i\\|_{L^2} = 1$.  The pointwise std at probe
    centre $r_i$ is

    .. math::

        \\hat{\\sigma}(r_i) = \\sqrt{\\max(0,\\, \\langle b_i, C_{\\text{post}}(b_i) \\rangle)}.

    The scalar ``sigma_1_std`` is extracted directly from the posterior
    covariance by applying it to the unit vector in the ``σ₁`` component.

    Parameters
    ----------
    block : BlockIndex
    posterior : GaussianMeasure
    forward_dict : dict
        ``{BlockIndex: (G_st, C_D_st, M_st, D_st)}``.
    specs : RadialSpecs
    n_grid : int
    n_probes : int

    Returns
    -------
    _PostBlockDisplayData
    """
    from intervalinf.core.functions import Function as _IFunction

    R   = specs.earth_radius_km
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km

    # ── Unpack model space ────────────────────────────────────────────────
    G_st, _C_D_st, M_st, _D_st = forward_dict[block]
    exp = posterior.expectation

    f_vp    = exp[0][0]
    f_vs_IC = exp[0][1][0]
    f_vs_M  = exp[0][1][1]
    f_rho   = exp[0][2]
    sigma_1_mean = float(exp[1][1][0])

    M_functions  = M_st.subspaces[0]
    M_vp_space   = M_functions.subspaces[0]
    M_vs_space   = M_functions.subspaces[1]
    M_rho_space  = M_functions.subspaces[2]
    M_vs_IC_space = M_vs_space.subspaces[0]
    M_vs_M_space  = M_vs_space.subspaces[1]

    # ── Dense radial grids ────────────────────────────────────────────────
    r_vp    = np.linspace(0.0, R,   n_grid)
    r_vs_IC = np.linspace(0.0, ICB, n_grid)
    r_vs_M  = np.linspace(CMB, R,   n_grid)
    r_rho   = np.linspace(0.0, R,   n_grid)

    # ── Evaluate posterior mean ────────────────────────────────────────────
    mean_vp    = np.asarray([f_vp(r)    for r in r_vp],    dtype=float)
    mean_vs_IC = np.asarray([f_vs_IC(r) for r in r_vs_IC], dtype=float)
    mean_vs_M  = np.asarray([f_vs_M(r)  for r in r_vs_M],  dtype=float)
    mean_rho   = np.asarray([f_rho(r)   for r in r_rho],   dtype=float)

    # ── Covariance-probing helpers ─────────────────────────────────────────

    def _make_zero_fn(M_space):
        return _IFunction(
            M_space,
            evaluate_callable=lambda r: np.zeros_like(np.asarray(r, dtype=float)),
        )

    zero_vp    = _make_zero_fn(M_vp_space)
    zero_vs_IC = _make_zero_fn(M_vs_IC_space)
    zero_vs_M  = _make_zero_fn(M_vs_M_space)
    zero_rho   = _make_zero_fn(M_rho_space)
    zeros1     = np.zeros(1)

    def _embed_bump(bump_fn, comp_name: str) -> list:
        """Embed bump_fn into the full model-space nesting for comp_name."""
        if comp_name == "vp":
            return [[bump_fn, [zero_vs_IC, zero_vs_M], zero_rho],  [zeros1, zeros1]]
        elif comp_name == "vs_IC":
            return [[zero_vp, [bump_fn,    zero_vs_M], zero_rho],  [zeros1, zeros1]]
        elif comp_name == "vs_M":
            return [[zero_vp, [zero_vs_IC, bump_fn],   zero_rho],  [zeros1, zeros1]]
        elif comp_name == "rho":
            return [[zero_vp, [zero_vs_IC, zero_vs_M], bump_fn],   [zeros1, zeros1]]
        else:
            raise ValueError(f"Unknown component: {comp_name!r}")

    def _sigma_1_std() -> float:
        """Return the posterior std of the scalar σ₁ component."""
        sigma_1_in = [[zero_vp, [zero_vs_IC, zero_vs_M], zero_rho], [zeros1, np.array([1.0])]]
        sigma_1_out = posterior.covariance(sigma_1_in)
        sigma_1_var = float(np.asarray(sigma_1_out[1][1], dtype=float).reshape(-1)[0])
        return float(np.sqrt(max(0.0, sigma_1_var)))

    def _extract_comp(full_out, comp_name: str):
        """Pull the per-component output from the nested covariance result."""
        if comp_name == "vp":
            return full_out[0][0]
        elif comp_name == "vs_IC":
            return full_out[0][1][0]
        elif comp_name == "vs_M":
            return full_out[0][1][1]
        elif comp_name == "rho":
            return full_out[0][2]
        else:
            raise ValueError(f"Unknown component: {comp_name!r}")

    def _probe_std(M_comp, r_centres: np.ndarray, comp_name: str) -> np.ndarray:
        """Estimate std at each r_centre by probing the posterior covariance."""
        n = len(r_centres)
        stds = np.zeros(n)
        domain  = M_comp.function_domain
        dom_len = float(domain.b - domain.a)
        sigma_b = dom_len / n_probes  # bump bandwidth

        for i, r_i in enumerate(r_centres):
            def _normed_bump(r, _r_i=r_i, _sigma=sigma_b):
                vals = np.exp(-0.5 * ((np.asarray(r, dtype=float) - _r_i) / _sigma) ** 2)
                return vals

            r_quad    = np.linspace(float(domain.a), float(domain.b), 512)
            bump_vals = _normed_bump(r_quad)
            norm_sq   = np.trapezoid(bump_vals ** 2, r_quad)
            if norm_sq < 1e-30:
                continue
            norm = np.sqrt(norm_sq)

            def _unit_bump(r, _r_i=r_i, _sigma=sigma_b, _norm=norm):
                return np.exp(-0.5 * ((np.asarray(r, dtype=float) - _r_i) / _sigma) ** 2) / _norm

            bump_fn    = _IFunction(M_comp, evaluate_callable=_unit_bump)
            full_in    = _embed_bump(bump_fn, comp_name)
            cov_out    = posterior.covariance(full_in)
            comp_out   = _extract_comp(cov_out, comp_name)

            out_vals   = np.asarray([comp_out(r)   for r in r_quad], dtype=float)
            in_vals    = np.asarray([_unit_bump(r)  for r in r_quad], dtype=float)
            inner      = np.trapezoid(in_vals * out_vals, r_quad)
            stds[i]    = np.sqrt(max(0.0, inner))

        return stds

    # ── Probe centres ──────────────────────────────────────────────────────
    half_vp   = R        / (2 * n_probes)
    half_vsIC = ICB      / (2 * n_probes)
    half_vsM  = (R - CMB) / (2 * n_probes)
    half_rho  = R        / (2 * n_probes)

    probe_r_vp    = np.linspace(half_vp,          R   - half_vp,   n_probes)
    probe_r_vs_IC = np.linspace(half_vsIC,         ICB - half_vsIC, n_probes)
    probe_r_vs_M  = np.linspace(CMB + half_vsM,    R   - half_vsM,  n_probes)
    probe_r_rho   = np.linspace(half_rho,          R   - half_rho,  n_probes)

    std_vp    = _probe_std(M_vp_space,    probe_r_vp,    "vp")
    std_vs_IC = _probe_std(M_vs_IC_space, probe_r_vs_IC, "vs_IC")
    std_vs_M  = _probe_std(M_vs_M_space,  probe_r_vs_M,  "vs_M")
    std_rho   = _probe_std(M_rho_space,   probe_r_rho,   "rho")
    sigma_1_std = _sigma_1_std()

    d_pred_post = np.asarray(G_st(exp), dtype=float) if G_st is not None else np.zeros(0)

    return _PostBlockDisplayData(
        r_vp=r_vp, r_vs_IC=r_vs_IC, r_vs_M=r_vs_M, r_rho=r_rho,
        mean_vp=mean_vp, mean_vs_IC=mean_vs_IC,
        mean_vs_M=mean_vs_M, mean_rho=mean_rho,
        probe_r_vp=probe_r_vp,    probe_r_vs_IC=probe_r_vs_IC,
        probe_r_vs_M=probe_r_vs_M, probe_r_rho=probe_r_rho,
        std_vp=std_vp, std_vs_IC=std_vs_IC,
        std_vs_M=std_vs_M, std_rho=std_rho,
        sigma_1_mean=sigma_1_mean, sigma_1_std=sigma_1_std,
        d_pred_post=d_pred_post,
    )


def compute_model_curves(
    model_vec,
    specs: "RadialSpecs",
    *,
    n_grid: int = 200,
) -> "_PostBlockDisplayData":
    """Evaluate the radial-coefficient curves of a model-space vector.

    A posterior **sample** (or any model-space element with the same nested
    structure as ``GaussianMeasure.expectation``) is evaluated on the dense
    radial grids used elsewhere, returning a :class:`_PostBlockDisplayData`
    whose ``mean_*`` fields hold the sample's curves and ``sigma_1_mean`` holds
    the scalar CMB-topography coefficient.  Uncertainty fields are zero-filled
    (a single realisation carries no spread); the result is drop-in compatible
    with :class:`model_map_viz.ModelMapViewer`.

    Parameters
    ----------
    model_vec : Vector
        Model-space element, e.g. ``posterior.sample()``.  Nesting:
        ``vec[0][0]`` = vp, ``vec[0][1][0]`` = vs (inner core),
        ``vec[0][1][1]`` = vs (mantle), ``vec[0][2]`` = rho,
        ``vec[1][1][0]`` = scalar σ₁.
    specs : RadialSpecs
        Provides ``earth_radius_km`` / ``icb_radius_km`` / ``cmb_radius_km``.
    n_grid : int
        Radial grid points per component.

    Returns
    -------
    _PostBlockDisplayData
    """
    R   = specs.earth_radius_km
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km

    f_vp    = model_vec[0][0]
    f_vs_IC = model_vec[0][1][0]
    f_vs_M  = model_vec[0][1][1]
    f_rho   = model_vec[0][2]
    sigma_1 = float(np.asarray(model_vec[1][1], dtype=float).reshape(-1)[0])

    r_vp    = np.linspace(0.0, R,   n_grid)
    r_vs_IC = np.linspace(0.0, ICB, n_grid)
    r_vs_M  = np.linspace(CMB, R,   n_grid)
    r_rho   = np.linspace(0.0, R,   n_grid)

    mean_vp    = np.asarray([f_vp(r)    for r in r_vp],    dtype=float)
    mean_vs_IC = np.asarray([f_vs_IC(r) for r in r_vs_IC], dtype=float)
    mean_vs_M  = np.asarray([f_vs_M(r)  for r in r_vs_M],  dtype=float)
    mean_rho   = np.asarray([f_rho(r)   for r in r_rho],   dtype=float)

    z = np.zeros(0)
    return _PostBlockDisplayData(
        r_vp=r_vp, r_vs_IC=r_vs_IC, r_vs_M=r_vs_M, r_rho=r_rho,
        mean_vp=mean_vp, mean_vs_IC=mean_vs_IC,
        mean_vs_M=mean_vs_M, mean_rho=mean_rho,
        probe_r_vp=z, probe_r_vs_IC=z, probe_r_vs_M=z, probe_r_rho=z,
        std_vp=z, std_vs_IC=z, std_vs_M=z, std_rho=z,
        sigma_1_mean=sigma_1, sigma_1_std=0.0,
        d_pred_post=z,
    )


# ---------------------------------------------------------------------------
# Disk-cache helpers
# ---------------------------------------------------------------------------

def _make_cache_key(
    n_basis: int,
    sigma_var: float,
    n_grid: int,
    n_probes: int,
    data_dir: str,
    kernel_dir: str,
) -> str:
    """Return a 16-character hex digest that uniquely identifies a parameter set.

    Any change to the inversion or probing parameters produces a different key,
    preventing stale cached arrays from being used.
    """
    parts = "|".join(
        [str(n_basis), str(sigma_var), str(n_grid), str(n_probes), data_dir, kernel_dir]
    )
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# PosteriorViewer
# ---------------------------------------------------------------------------

class PosteriorViewer:
    """Cache and serve per-block posterior display data.

    Parameters
    ----------
    posterior_dict : dict
        ``{BlockIndex: GaussianMeasure}`` — output of ``solve_all_blocks``.
    forward_dict : dict
        ``{BlockIndex: (G_st, C_D_st, M_st, D_st)}`` — output of
        ``build_block_forward`` / ``full_spectrum_utils``.
    specs : RadialSpecs
        Shared configuration.
    n_grid : int
        Radial grid points per component.
    n_probes : int
        Number of Gaussian bump probes for covariance estimation.
    cache_dir : Path or str, optional
        Directory in which ``.npz`` files are stored.  If ``None`` (default)
        the disk cache is disabled.  Pass a directory path and a matching
        *cache_key* to enable cross-session reuse.
    cache_key : str, optional
        16-character hex string produced by :func:`_make_cache_key`.  Must be
        supplied together with *cache_dir*.
    blocks : list, optional
        Full list of blocks shown by the viewer.  This may be larger than
        ``posterior_dict`` when some blocks are served directly from disk cache.
    """

    def __init__(
        self,
        posterior_dict: dict,
        forward_dict: dict,
        specs: "RadialSpecs",
        *,
        n_grid: int = 200,
        n_probes: int = 20,
        cache_dir: Optional[Path] = None,
        cache_key: Optional[str] = None,
        blocks: Optional[List] = None,
    ) -> None:
        self._posterior_dict = posterior_dict
        self._forward_dict   = forward_dict
        self._specs          = specs
        self._n_grid         = n_grid
        self._n_probes       = n_probes
        self._cache: Dict    = {}
        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir is not None else None
        self._cache_key: Optional[str]  = cache_key
        self._blocks: List = sorted(blocks) if blocks is not None else sorted(posterior_dict.keys())

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def blocks(self) -> list:
        """All blocks shown by the viewer (sorted)."""
        return list(self._blocks)

    def is_computed(self, block) -> bool:
        """Return True if display data for *block* is in the in-memory cache."""
        return block in self._cache

    def has_disk_cache(self, block) -> bool:
        """Return True if a ``.npz`` cache file exists for *block*."""
        if self._cache_dir is None or self._cache_key is None:
            return False
        return self._block_cache_path(block).exists()

    def load_cached_block(self, block) -> bool:
        """Load *block* from memory or disk cache without recomputing.

        Returns
        -------
        bool
            ``True`` if the block is now available in memory, ``False`` if no
            cached representation exists.
        """
        if block in self._cache:
            return True

        disk_data = self._load_from_disk_cache(block)
        if disk_data is None:
            return False

        self._cache[block] = disk_data
        return True

    @property
    def n_computed(self) -> int:
        """Number of blocks whose display data has been cached in memory."""
        return len(self._cache)

    @property
    def n_disk_cached(self) -> int:
        """Number of blocks with an on-disk ``.npz`` cache file."""
        return sum(1 for b in self.blocks if self.has_disk_cache(b))

    def compute_block(
        self,
        block,
        progress_cb: Optional[Callable] = None,
    ) -> "_PostBlockDisplayData":
        """Compute and cache display data for *block* (no-op if already cached).

        Parameters
        ----------
        block : BlockIndex
        progress_cb : callable, optional
            Called with *block* once computation finishes.

        Returns
        -------
        _PostBlockDisplayData
        """
        if block in self._cache:
            return self._cache[block]

        # Try loading from the on-disk cache first.
        disk_data = self._load_from_disk_cache(block)
        if disk_data is not None:
            self._cache[block] = disk_data
            if progress_cb is not None:
                progress_cb(block, from_cache=True)
            return disk_data

        if block not in self._posterior_dict or block not in self._forward_dict:
            raise RuntimeError(
                f"Block {block} is not available in memory and no disk cache exists."
            )

        data = _compute_posterior_display_data(
            block,
            self._posterior_dict[block],
            self._forward_dict,
            self._specs,
            n_grid=self._n_grid,
            n_probes=self._n_probes,
        )
        self._cache[block] = data
        self._save_to_disk_cache(block, data)

        if progress_cb is not None:
            progress_cb(block, from_cache=False)

        return data

    def precompute_all(
        self,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        """Compute display data for every block in the posterior dict."""
        for block in self.blocks:
            self.compute_block(block, progress_cb=progress_cb)

    def get_display_data(self, block) -> "_PostBlockDisplayData":
        """Return cached display data for *block* (must be computed first)."""
        if block not in self._cache:
            raise RuntimeError(
                f"Block {block} has not been computed yet. "
                "Call compute_block(block) or precompute_all() first."
            )
        return self._cache[block]

    # ── Disk-cache private helpers ─────────────────────────────────────────

    def _block_cache_path(self, block) -> Path:
        """Return the ``.npz`` path for *block* under the current cache key."""
        return self._cache_dir / self._cache_key / f"s{block.s}_t{block.t}.npz"

    def _save_to_disk_cache(self, block, data: "_PostBlockDisplayData") -> None:
        """Write *data* to the disk cache; silently skip if caching is disabled."""
        if self._cache_dir is None or self._cache_key is None:
            return
        path = self._block_cache_path(block)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            r_vp=data.r_vp,         r_vs_IC=data.r_vs_IC,
            r_vs_M=data.r_vs_M,     r_rho=data.r_rho,
            mean_vp=data.mean_vp,   mean_vs_IC=data.mean_vs_IC,
            mean_vs_M=data.mean_vs_M, mean_rho=data.mean_rho,
            probe_r_vp=data.probe_r_vp,     probe_r_vs_IC=data.probe_r_vs_IC,
            probe_r_vs_M=data.probe_r_vs_M, probe_r_rho=data.probe_r_rho,
            std_vp=data.std_vp,     std_vs_IC=data.std_vs_IC,
            std_vs_M=data.std_vs_M, std_rho=data.std_rho,
            sigma_1_mean=np.array(data.sigma_1_mean),
            sigma_1_std=np.array(data.sigma_1_std),
            d_pred_post=data.d_pred_post,
        )

    def _load_from_disk_cache(self, block) -> "Optional[_PostBlockDisplayData]":
        """Load and return cached data for *block*, or ``None`` if unavailable."""
        if self._cache_dir is None or self._cache_key is None:
            return None
        path = self._block_cache_path(block)
        if not path.exists():
            return None
        try:
            npz = np.load(path)
            return _PostBlockDisplayData(
                r_vp=npz["r_vp"],         r_vs_IC=npz["r_vs_IC"],
                r_vs_M=npz["r_vs_M"],     r_rho=npz["r_rho"],
                mean_vp=npz["mean_vp"],   mean_vs_IC=npz["mean_vs_IC"],
                mean_vs_M=npz["mean_vs_M"], mean_rho=npz["mean_rho"],
                probe_r_vp=npz["probe_r_vp"],     probe_r_vs_IC=npz["probe_r_vs_IC"],
                probe_r_vs_M=npz["probe_r_vs_M"], probe_r_rho=npz["probe_r_rho"],
                std_vp=npz["std_vp"],     std_vs_IC=npz["std_vs_IC"],
                std_vs_M=npz["std_vs_M"], std_rho=npz["std_rho"],
                sigma_1_mean=float(npz["sigma_1_mean"]),
                sigma_1_std=float(npz["sigma_1_std"]),
                d_pred_post=(
                    npz["d_pred_post"]
                    if "d_pred_post" in npz.files
                    else np.zeros(0)
                ),
            )
        except Exception:
            return None  # corrupted file — fall through to recompute


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_posterior_axes(
    axes,
    display_data: "_PostBlockDisplayData",
    block: "BlockIndex",
    specs: "RadialSpecs",
    *,
    n_sigma: float = 1.0,
) -> None:
    """Draw the 4 posterior panels into an existing axes array.

    Parameters
    ----------
    axes : sequence of Axes, length 4
        Cleared and redrawn in place.
    display_data : _PostBlockDisplayData
    block : BlockIndex
    specs : RadialSpecs
    n_sigma : float
        Width of the uncertainty band in units of standard deviations.
    """
    for ax in axes:
        ax.clear()
        ax.set_axis_on()

    def _plot_panel(ax, r, mean, probe_r, std, label: str, color: str) -> None:
        std_dense = np.interp(r, probe_r, std)
        ax.plot(mean, r, color=color, lw=1.5, label="mean")
        ax.fill_betweenx(
            r,
            mean - n_sigma * std_dense,
            mean + n_sigma * std_dense,
            alpha=0.3,
            color=color,
            label=f"±{n_sigma:.1g}σ",
        )
        ax.set_ylabel("radius (km)")
        ax.set_xlabel(label)
        ax.legend(fontsize=7)

    # Panel 0 — δvp
    _plot_panel(
        axes[0], display_data.r_vp, display_data.mean_vp,
        display_data.probe_r_vp, display_data.std_vp,
        _PARAM_LABELS["vp"], _PARAM_COLORS["vp"],
    )

    # Panel 1 — δvs (IC + mantle combined, gray outer core)
    ax_vs = axes[1]
    icb_km = specs.icb_radius_km
    cmb_km = specs.cmb_radius_km
    ax_vs.axhspan(icb_km, cmb_km, color="lightgray", alpha=0.6, label="outer core")
    # IC
    std_IC = np.interp(display_data.r_vs_IC, display_data.probe_r_vs_IC, display_data.std_vs_IC)
    ax_vs.plot(display_data.mean_vs_IC, display_data.r_vs_IC,
               color=_PARAM_COLORS["vs_IC"], lw=1.5, label="mean")
    ax_vs.fill_betweenx(
        display_data.r_vs_IC,
        display_data.mean_vs_IC - n_sigma * std_IC,
        display_data.mean_vs_IC + n_sigma * std_IC,
        alpha=0.3, color=_PARAM_COLORS["vs_IC"], label=f"±{n_sigma:.1g}σ",
    )
    # Mantle
    std_M = np.interp(display_data.r_vs_M, display_data.probe_r_vs_M, display_data.std_vs_M)
    ax_vs.plot(display_data.mean_vs_M, display_data.r_vs_M,
               color=_PARAM_COLORS["vs_M"], lw=1.5)
    ax_vs.fill_betweenx(
        display_data.r_vs_M,
        display_data.mean_vs_M - n_sigma * std_M,
        display_data.mean_vs_M + n_sigma * std_M,
        alpha=0.3, color=_PARAM_COLORS["vs_M"],
    )
    ax_vs.set_ylabel("radius (km)")
    ax_vs.set_xlabel("δvs (km s⁻¹)")
    ax_vs.legend(fontsize=7)

    # Panel 2 — δρ
    _plot_panel(
        axes[2], display_data.r_rho, display_data.mean_rho,
        display_data.probe_r_rho, display_data.std_rho,
        _PARAM_LABELS["rho"], _PARAM_COLORS["rho"],
    )

    # Panel 3 — scalar σ₁ (CMB topography coefficient)
    axes[3].axhline(
        display_data.sigma_1_mean,
        color=_PARAM_COLORS["sigma_1"],
        lw=2,
        label=f"mean = {display_data.sigma_1_mean:.3f}",
    )
    axes[3].errorbar(
        x=0.5,
        y=display_data.sigma_1_mean,
        yerr=n_sigma * display_data.sigma_1_std,
        fmt="o",
        color=_PARAM_COLORS["sigma_1"],
        capsize=6,
        label=f"±{n_sigma:.1g}σ = {n_sigma * display_data.sigma_1_std:.2f}",
    )
    axes[3].set_xlim(0, 1)
    axes[3].set_xticks([])
    axes[3].set_ylabel(_PARAM_LABELS["sigma_1"])
    axes[3].set_xlabel("σ₁")
    axes[3].legend(fontsize=7)


def _draw_datafit_axes(
    ax,
    d_obs: "np.ndarray",
    d_err: "np.ndarray",
    d_pred_post: "np.ndarray",
    block: "BlockIndex",
    *,
    data_label: str = "data ± σ_D",
) -> None:
    """Draw observed data with error bars and posterior prediction on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cleared and redrawn in place.
    d_obs : np.ndarray, shape (N_d,)
        Observed (or synthetic) data vector.
    d_err : np.ndarray, shape (N_d,)
        1-sigma data noise (sqrt of diagonal of C_D).
    d_pred_post : np.ndarray, shape (N_d,)
        Posterior predictive: ``G @ posterior_mean``.
    block : BlockIndex
    data_label : str
        Legend label for the data points.
    """
    ax.cla()
    N = len(d_obs)
    if N == 0 or len(d_pred_post) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        ax.set_title(f"Data fit — s={block.s}, t={block.t}", fontsize=8)
        return

    idx  = np.arange(N)
    pred = d_pred_post[:N]  # guard against unexpected shape mismatch

    ax.errorbar(idx, d_obs, yerr=d_err,
                fmt="o", ms=3, color="steelblue",
                ecolor="steelblue", elinewidth=0.8,
                capsize=2, alpha=0.8, label=data_label, zorder=3)
    ax.plot(idx, pred, "_", ms=7, color="darkorange",
            markeredgewidth=1.8, label="G m̂ (posterior mean)", zorder=4)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    safe_err = np.where(d_err > 0, d_err, np.inf)
    chi_rms  = float(np.sqrt(np.mean(((d_obs - pred) / safe_err) ** 2)))
    chi_color = (
        "forestgreen" if chi_rms < 0.5
        else ("darkorange" if chi_rms < 1.5 else "red")
    )
    ax.set_xlabel("observation index", fontsize=7)
    ax.set_ylabel("splitting coeff.", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.set_title(
        f"Data fit — s={block.s}, t={block.t}  |  "
        f"RMS (d − Gm̂)/σ_D = {chi_rms:.2f}",
        fontsize=7.5, color=chi_color,
    )


def render_posterior_figure(
    display_data: "_PostBlockDisplayData",
    block: "BlockIndex",
    specs: "RadialSpecs",
    *,
    n_sigma: float = 1.0,
    figsize=(18, 5),
) -> "matplotlib.figure.Figure":
    """Create a new figure with 4 posterior panels for *block*.

    Parameters
    ----------
    display_data : _PostBlockDisplayData
        Pre-computed via :func:`_compute_posterior_display_data`.
    block : BlockIndex
    specs : RadialSpecs
    n_sigma : float
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(
        f"Block (s={block.s}, t={block.t}) — radial posterior", fontsize=13
    )
    _draw_posterior_axes(axes, display_data, block, specs, n_sigma=n_sigma)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MatplotlibPosteriorViewer
# ---------------------------------------------------------------------------

class MatplotlibPosteriorViewer:
    """Standalone Matplotlib control panel for browsing block posteriors.

    Parameters
    ----------
    viewer : PosteriorViewer
    blocks : list of BlockIndex
    specs : RadialSpecs
    plt : module
        The ``matplotlib.pyplot`` module.
    button_class : type
        ``matplotlib.widgets.Button``.
    slider_class : type
        ``matplotlib.widgets.Slider``.
    n_sigma : float
        Initial uncertainty-band width in units of σ.

    Keyboard shortcuts
    ------------------
    ← / →   Previous / next block.
    c       Compute current block.
    a       Precompute all blocks.
    """

    def __init__(
        self,
        viewer: "PosteriorViewer",
        blocks: list,
        specs: "RadialSpecs",
        plt,
        button_class,
        slider_class,
        *,
        n_sigma: float = 1.0,
    ) -> None:
        if not blocks:
            raise ValueError("MatplotlibPosteriorViewer requires at least one block.")

        self.viewer  = viewer
        self.blocks  = list(blocks)
        self.specs   = specs
        self.plt     = plt
        self.Button  = button_class
        self.Slider  = slider_class
        self.n_sigma = n_sigma
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_queue: Optional[queue.Queue] = None
        self._worker_timer = None
        self._active_job: Optional[str] = None
        self._active_block = None

        self.fig, self.axes = self.plt.subplots(1, 4, figsize=(18, 6))
        self.fig.subplots_adjust(
            left=0.05, right=0.98, bottom=0.24, top=0.86, wspace=0.35
        )
        self.status_text = self.fig.text(0.05, 0.91, "", fontsize=9)
        self._worker_timer = self.fig.canvas.new_timer(interval=100)
        self._worker_timer.add_callback(self._poll_worker)

        self._build_widgets()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._redraw()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def current_block(self) -> "BlockIndex":
        """The block currently selected by the slider."""
        index = min(int(round(self.block_slider.val)), len(self.blocks) - 1)
        return self.blocks[index]

    # ── Public API ────────────────────────────────────────────────────────

    def show(self) -> None:
        """Open the interactive Matplotlib window."""
        self.plt.show()

    def save(self, path: Path) -> None:
        """Save the current figure state to *path*."""
        self.fig.savefig(path, dpi=160, bbox_inches="tight")

    def compute_current(self) -> None:
        """Compute display data for the currently selected block and redraw."""
        self.viewer.compute_block(self.current_block)
        self._redraw()

    def precompute_all(self) -> None:
        """Compute display data for every block sequentially and redraw."""
        self.viewer.precompute_all()
        self._redraw()

    # ── Widget construction ───────────────────────────────────────────────

    def _build_widgets(self) -> None:
        block_valmax = max(len(self.blocks) - 1, 1)
        block_ax = self.fig.add_axes([0.18, 0.15, 0.50, 0.03])
        self.block_slider = self.Slider(
            block_ax,
            "block index",
            0,
            block_valmax,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        self.block_slider.on_changed(self._on_block_changed)

        prev_ax = self.fig.add_axes([0.05, 0.14, 0.09, 0.045])
        next_ax = self.fig.add_axes([0.71, 0.14, 0.09, 0.045])
        self.prev_button = self.Button(prev_ax, "previous")
        self.next_button = self.Button(next_ax, "next")
        self.prev_button.on_clicked(lambda _e: self._step_block(-1))
        self.next_button.on_clicked(lambda _e: self._step_block(1))

        nsigma_ax = self.fig.add_axes([0.18, 0.09, 0.30, 0.03])
        self.nsigma_slider = self.Slider(
            nsigma_ax,
            "n_sigma",
            0.5,
            3.0,
            valinit=self.n_sigma,
            valstep=0.1,
        )
        self.nsigma_slider.on_changed(self._on_nsigma_changed)

        compute_ax     = self.fig.add_axes([0.55, 0.07, 0.11, 0.045])
        precompute_ax  = self.fig.add_axes([0.67, 0.07, 0.15, 0.045])
        self.compute_button     = self.Button(compute_ax,    "Compute")
        self.precompute_button  = self.Button(precompute_ax, "Precompute all")
        self.compute_button.on_clicked(self._on_compute)
        self.precompute_button.on_clicked(self._on_precompute_all)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _job_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _clear_worker(self) -> None:
        self._worker_thread = None
        self._worker_queue = None
        self._active_job = None
        self._active_block = None
        if self._worker_timer is not None:
            self._worker_timer.stop()

    def _start_worker(self, target, start_message: str, *, job: str, block=None) -> bool:
        if self._job_running():
            self._set_status("A posterior computation is already running.")
            return False

        self._worker_queue = queue.Queue()
        self._active_job = job
        self._active_block = block
        self._set_status(start_message)

        def _runner() -> None:
            try:
                target(self._worker_queue)
            except Exception as exc:
                self._worker_queue.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self._worker_queue.put(("finished", None))

        self._worker_thread = threading.Thread(target=_runner, daemon=True)
        self._worker_thread.start()
        if self._worker_timer is not None:
            self._worker_timer.start()
        return True

    def _poll_worker(self) -> None:
        if self._worker_queue is None:
            return

        latest_status: Optional[str] = None
        precompute_done = False
        error_message: Optional[str] = None
        finished = False

        while True:
            try:
                event = self._worker_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "error":
                error_message = event[1]
            elif kind == "compute_done":
                pass
            elif kind == "precompute_progress":
                _, completed, total, label = event
                latest_status = f"Precomputing... {completed}/{total} — {label}"
            elif kind == "precompute_done":
                precompute_done = True
            elif kind == "finished":
                finished = True

        if error_message is not None:
            self._clear_worker()
            self._set_status(f"Computation failed — {error_message}")
            return

        if latest_status is not None:
            self._set_status(latest_status)

        if not finished or self._job_running():
            return

        job = self._active_job
        block = self._active_block
        self._clear_worker()

        if job == "compute" and block is not None and self.viewer.is_computed(block):
            if block == self.current_block:
                self._redraw()
            else:
                self._set_status(
                    f"Computed block (s={block.s}, t={block.t}) | "
                    f"{self.viewer.n_computed}/{len(self.blocks)} blocks computed"
                )
            return

        if job == "precompute" and precompute_done:
            self._redraw()

    def _show_placeholder(self, message: str) -> None:
        for ax in self.axes:
            ax.clear()
            ax.set_axis_off()
        self.axes[2].text(
            0.5, 0.5, message,
            ha="center", va="center",
            transform=self.axes[2].transAxes,
            fontsize=11,
        )
        self.fig.suptitle("Posterior viewer", fontsize=10)
        self._set_status(message)

    def _redraw(self) -> None:
        block = self.current_block
        if not self.viewer.is_computed(block) and not self.viewer.load_cached_block(block):
            self._show_placeholder(
                f"Block (s={block.s}, t={block.t}) — not computed.\n"
                "Press 'Compute' or use keyboard shortcut 'c'."
            )
            return

        display_data = self.viewer.get_display_data(block)
        n = float(self.nsigma_slider.val)
        _draw_posterior_axes(self.axes, display_data, block, self.specs, n_sigma=n)
        self.fig.suptitle(
            f"Posterior — block (s={block.s}, t={block.t})", fontsize=10
        )
        n_done = self.viewer.n_computed
        n_total = len(self.blocks)
        self._set_status(
            f"Ready — block (s={block.s}, t={block.t}) | "
            f"{n_done}/{n_total} blocks computed"
        )
        self.fig.canvas.draw_idle()

    # ── Callbacks ────────────────────────────────────────────────────────

    def _on_block_changed(self, _value: float) -> None:
        self._redraw()

    def _on_nsigma_changed(self, _value: float) -> None:
        self.n_sigma = float(self.nsigma_slider.val)
        self._redraw()

    def _step_block(self, step: int) -> None:
        new_index = (int(round(self.block_slider.val)) + step) % len(self.blocks)
        self.block_slider.set_val(new_index)

    def _on_key_press(self, event) -> None:
        if event.key == "left":
            self._step_block(-1)
        elif event.key == "right":
            self._step_block(1)
        elif event.key == "c":
            self._on_compute(None)
        elif event.key == "a":
            self._on_precompute_all(None)

    def _on_compute(self, _event) -> None:
        block = self.current_block
        hint = " (from cache)" if self.viewer.has_disk_cache(block) else ""
        self._start_worker(
            lambda work_queue: self._compute_block_async(block, work_queue),
            f"Loading block (s={block.s}, t={block.t}){hint}...",
            job="compute",
            block=block,
        )

    def _on_precompute_all(self, _event) -> None:
        n_disk = self.viewer.n_disk_cached
        n_total = len(self.blocks)
        hint = f" ({n_disk}/{n_total} already cached)" if n_disk else ""
        self._start_worker(
            self._precompute_all_async,
            f"Precomputing all block posteriors{hint}...",
            job="precompute",
        )

    def _compute_block_async(self, block, work_queue: queue.Queue) -> None:
        self.viewer.compute_block(block)
        work_queue.put(("compute_done", block))

    def _precompute_all_async(self, work_queue: queue.Queue) -> None:
        total = len(self.blocks)
        for completed, block in enumerate(self.blocks, start=1):
            from_cache = self.viewer.has_disk_cache(block) and not self.viewer.is_computed(block)
            self.viewer.compute_block(block)
            tag = " (cache)" if from_cache else ""
            work_queue.put(("precompute_progress", completed, total, f"s={block.s},t={block.t}{tag}"))
        work_queue.put(("precompute_done", total))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the standalone posterior-viewer CLI parser."""
    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=demo_dir / "data" / "normal-mode-data",
        metavar="DIR",
        help="Path to the normal-mode data directory. Default: data/normal-mode-data",
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=demo_dir / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian",
        metavar="DIR",
        help="Path to the normal-mode kernel directory.",
    )
    parser.add_argument(
        "--s-max",
        type=int,
        default=4,
        help="Maximum even harmonic degree to include. Default 4.",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        default=100,
        help="Number of radial basis functions per parameter. Default 100.",
    )
    parser.add_argument(
        "--sigma-var",
        type=float,
        default=100.0,
        help="Scalar CMB topography prior variance σ_var. Default 100.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for solve_all_blocks. Default 1.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=200,
        help="Number of radial evaluation points per panel. Default 200.",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=10,
        help="Number of Gaussian bump probes for covariance estimation. Default 10.",
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=1.0,
        help="Initial uncertainty-band width in units of σ. Default 1.0.",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Precompute all block posteriors before opening the window.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the viewer figure to this path and exit.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=demo_dir / ".posterior_cache",
        metavar="DIR",
        help="Directory for on-disk display-data cache.  Default: .posterior_cache/",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the on-disk cache (always recompute probes).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached display data for the current parameter set and exit.",
    )
    return parser


def main() -> None:
    """Run the standalone Matplotlib posterior viewer."""
    parser = build_parser()
    args = parser.parse_args()

    import matplotlib

    if args.no_show or args.save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog
    from full_spectrum_utils import (
        RadialSpecs,
        enumerate_blocks,
        block_data_split,
        build_block_forward,
        build_shared_bessel_blocks,
        build_block_prior,
        prior_power_spectrum_default,
        solve_all_blocks,
    )
    from intervalinf import ParallelConfig

    print(f"Loading catalog from: {args.kernel_dir}")
    catalog = NormalModeKernelCatalog(str(args.kernel_dir))
    print(f"Loading data registry from: {args.data_dir}")
    reg = NormalModeDataRegistry(str(args.data_dir), mode_filter=catalog.list_modes())

    blocks = enumerate_blocks(reg, s_max=args.s_max)
    split  = block_data_split(reg, blocks)
    print(f"Blocks: {len(blocks)}")

    parallel_cfg = ParallelConfig(enabled=(args.n_jobs > 1), n_jobs=args.n_jobs)
    specs = RadialSpecs(n_basis=args.n_basis, parallel_cfg=parallel_cfg)

    # ── Disk-cache configuration (computed early to skip unnecessary work) ──
    cache_key = _make_cache_key(
        n_basis=args.n_basis,
        sigma_var=args.sigma_var,
        n_grid=args.n_grid,
        n_probes=args.n_probes,
        data_dir=str(args.data_dir.resolve()),
        kernel_dir=str(args.kernel_dir.resolve()),
    )
    cache_dir = None if args.no_cache else args.cache_dir

    if args.clear_cache and cache_dir is not None:
        import shutil
        target = Path(cache_dir) / cache_key
        if target.exists():
            shutil.rmtree(target)
            print(f"Cache cleared: {target}")
        else:
            print(f"No cache found at: {target}")
        return

    # Identify which blocks still need solving (uncached ones).
    def _npz_exists(b: "BlockIndex") -> bool:
        return (
            cache_dir is not None
            and (Path(cache_dir) / cache_key / f"s{b.s}_t{b.t}.npz").exists()
        )

    uncached = [b for b in blocks if not _npz_exists(b)]
    n_cached = len(blocks) - len(uncached)

    if cache_dir is not None:
        print(
            f"Disk cache: {cache_dir} (key {cache_key}) "
            f"— {n_cached}/{len(blocks)} blocks already cached."
        )

    if uncached:
        print(f"Building forward operators ({len(uncached)} uncached blocks)...")
        forward_dict = {
            b: build_block_forward(b.s, b.t, split[b], catalog, specs)
            for b in uncached
        }

        print(f"Building priors ({len(uncached)} uncached blocks)...")
        shared_bessel = build_shared_bessel_blocks(specs)
        prior_dict = {
            b: build_block_prior(
                b.s, b.t, shared_bessel, specs,
                tau_fn=prior_power_spectrum_default,
                sigma_var=args.sigma_var,
            )
            for b in uncached
        }

        print(f"Solving {len(uncached)} uncached blocks (n_jobs={args.n_jobs})...")
        posterior_dict = solve_all_blocks(forward_dict, prior_dict, split, n_jobs=args.n_jobs)
        print("Solve complete.")
    else:
        print("All blocks cached — skipping solve.")
        forward_dict  = {}
        posterior_dict = {}

    viewer = PosteriorViewer(
        posterior_dict,
        forward_dict,
        specs,
        n_grid=args.n_grid,
        n_probes=args.n_probes,
        cache_dir=cache_dir,
        cache_key=cache_key,
        blocks=blocks,
    )

    if uncached and cache_dir is not None:
        print(f"Caching display data for {len(uncached)} newly solved blocks...")
        for i, block in enumerate(uncached, start=1):
            print(f"  [{i}/{len(uncached)}] block (s={block.s}, t={block.t})")
            viewer.compute_block(block)
        print("Display cache complete.")

    app = MatplotlibPosteriorViewer(
        viewer,
        blocks,
        specs,
        plt,
        Button,
        Slider,
        n_sigma=args.n_sigma,
    )

    if args.precompute or args.save:
        app.precompute_all()

    if args.save is not None:
        app.save(args.save)

    if not args.no_show:
        app.show()


if __name__ == "__main__":
    main()
