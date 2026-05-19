"""
full_spectrum_viz.py
====================

Visualisation helpers for the full-spectrum splitting-function PLI example.

Provides four headline figure functions:

    plot_block_posterior           — per-block radial posterior fields (4+1 panels)
    plot_cmb_map                   — synthesised CMB topography map
    plot_equatorial_slice          — equatorial slice at a given radius (gated off by default)
    plot_property_posterior_summary — property posterior bar chart + correlation heatmap

All functions return an open :class:`matplotlib.figure.Figure` object for
display in a notebook; callers may save or close it afterward. ``plt.show()``
is never called internally.

Dependencies
------------
- ``matplotlib`` (always required)
- ``pyshtools`` (required for CMB map and equatorial slice synthesis)
- ``cartopy`` (optional; plain imshow fallback if unavailable)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

if TYPE_CHECKING:
    from full_spectrum_utils import BlockIndex, RadialSpecs
    from pygeoinf import GaussianMeasure


# =============================================================================
# Figure 1 — Per-block radial posterior fields
# =============================================================================

def plot_block_posterior(
    block: "BlockIndex",
    posterior: "GaussianMeasure",
    forward_dict: dict,
    specs: "RadialSpecs",
    *,
    n_grid: int = 200,
    n_probes: int = 20,
) -> matplotlib.figure.Figure:
    """Plot the radial posterior mean and ±1σ uncertainty band for one (s, t) block.

    Produces a 5-panel figure:

    - Panels 1–4: posterior mean function and shaded ±1σ band for vp, vs (IC),
      vs (mantle), and rho.  The vs panel shows both sub-domains on the same
      axes with a grey shaded gap for the outer core where vs = 0.
    - Panel 5: scalar sigma_1 (CMB topography) shown as a horizontal line
      (posterior mean) with a ±1σ error bar derived from the prior variance
      (sigma_var = 100.0).

    The ±1σ uncertainty band for each radial component is estimated by probing
    the posterior covariance operator with ``n_probes`` narrow Gaussian bumps
    distributed uniformly across the component's domain.  Each probe
    $b_i(r) = A \\exp(-(r - r_i)^2 / (2 \\sigma_b^2))$ is normalised so that
    $\\|b_i\\|_{L^2} = 1$.  The pointwise standard deviation estimate at $r_i$
    is $\\hat{\\sigma}(r_i) = \\sqrt{\\max(0, \\langle b_i, C_{\\mathrm{post}}(b_i) \\rangle)}$.

    Parameters
    ----------
    block : BlockIndex
        The (s, t) spectral block to plot.
    posterior : GaussianMeasure
        Posterior measure for this block from :func:`~full_spectrum_utils.solve_block`.
    forward_dict : dict
        ``{BlockIndex: (G_st, C_D_st, M_st, D_st)}`` — used to extract the
        component model spaces for domain information.
    specs : RadialSpecs
        Shared configuration supplying domain bounds
        (``earth_radius_km``, ``icb_radius_km``, ``cmb_radius_km``).
    n_grid : int
        Number of radial grid points for evaluating the posterior mean function.
        Default 200.
    n_probes : int
        Number of narrow-Gaussian probes used to estimate the uncertainty band.
        Default 20.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with 5 panels (4 radial + 1 scalar sigma_1).
    """
    from intervalinf import IntervalDomain
    from intervalinf.core.functions import Function as _IFunction

    R = specs.earth_radius_km
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km

    # ── Unpack model space and expectation ───────────────────────────────────
    G_st, C_D_st, M_st, D_st = forward_dict[block]
    exp = posterior.expectation
    # exp structure: [[f_vp, [f_vs_IC, f_vs_M], f_rho], [sigma_0_arr, sigma_1_arr]]
    f_vp = exp[0][0]
    f_vs_IC = exp[0][1][0]
    f_vs_M = exp[0][1][1]
    f_rho = exp[0][2]
    sigma_1_mean = float(exp[1][1][0])

    # ── Build per-component model spaces for covariance probing ─────────────
    # M_st = HilbertSpaceDirectSum([M_functions, M_euclidean])
    # M_functions = LebesgueSpaceDirectSum([M_vp, M_vs, M_rho])
    # M_vs = LebesgueSpaceDirectSum([M_vs_IC, M_vs_M])   (PartitionedLebesgueSpace.model_space)
    M_functions = M_st.subspaces[0]   # LebesgueSpaceDirectSum([M_vp, M_vs, M_rho])
    M_euclidean = M_st.subspaces[1]   # HilbertSpaceDirectSum([M_sigma_0, M_sigma_1])
    M_vp_space = M_functions.subspaces[0]
    M_vs_space = M_functions.subspaces[1]   # PartitionedLebesgueSpace.model_space
    M_rho_space = M_functions.subspaces[2]

    # For vs, extract the two sub-spaces (IC and mantle)
    # These are accessible via .subspaces on the LebesgueSpaceDirectSum
    M_vs_IC_space = M_vs_space.subspaces[0]
    M_vs_M_space = M_vs_space.subspaces[1]

    # ── Radial grids ─────────────────────────────────────────────────────────
    r_vp = np.linspace(0.0, R, n_grid)
    r_vs_IC = np.linspace(0.0, ICB, n_grid)
    r_vs_M = np.linspace(CMB, R, n_grid)
    r_rho = np.linspace(0.0, R, n_grid)

    # ── Evaluate posterior mean functions ────────────────────────────────────
    mean_vp = np.asarray([f_vp(r) for r in r_vp], dtype=float)
    mean_vs_IC = np.asarray([f_vs_IC(r) for r in r_vs_IC], dtype=float)
    mean_vs_M = np.asarray([f_vs_M(r) for r in r_vs_M], dtype=float)
    mean_rho = np.asarray([f_rho(r) for r in r_rho], dtype=float)

    # ── Uncertainty probing ──────────────────────────────────────────────────
    def _probe_std(M_comp, r_centres, parent_space_selector):
        """Estimate std at each r_centre by probing posterior covariance."""
        n = len(r_centres)
        stds = np.zeros(n)
        domain = M_comp.function_domain
        dom_len = domain.b - domain.a
        sigma_b = dom_len / n_probes

        for i, r_i in enumerate(r_centres):
            # Build a narrow Gaussian bump in the component's space
            def _bump_eval(r, _r_i=r_i, _sigma=sigma_b):
                return np.exp(-0.5 * ((r - _r_i) / _sigma) ** 2)

            # Compute L2 norm for normalisation
            r_quad = np.linspace(float(domain.a), float(domain.b), 512)
            bump_vals = _bump_eval(r_quad)
            norm_sq = np.trapezoid(bump_vals ** 2, r_quad)
            if norm_sq < 1e-30:
                continue
            norm = np.sqrt(norm_sq)

            def _normalised_bump(r, _r_i=r_i, _sigma=sigma_b, _norm=norm):
                return np.exp(-0.5 * ((r - _r_i) / _sigma) ** 2) / _norm

            bump_fn = _IFunction(M_comp, evaluate_callable=_normalised_bump)

            # Embed bump into full model-space input using selector
            full_input = parent_space_selector(bump_fn)
            cov_out_full = posterior.covariance(full_input)
            # Extract the relevant component output
            cov_out_comp = _extract_component_output(cov_out_full, M_comp, parent_space_selector)
            # Compute ⟨b_i, C_post(b_i)⟩
            out_vals = np.asarray([cov_out_comp(r) for r in r_quad], dtype=float)
            in_vals = np.asarray([_normalised_bump(r) for r in r_quad], dtype=float)
            inner = np.trapezoid(in_vals * out_vals, r_quad)
            stds[i] = np.sqrt(max(0.0, inner))

        return stds

    def _extract_component_output(full_out, M_comp, selector):
        """Extract the function corresponding to M_comp from nested output."""
        # The covariance output mirrors the expectation structure:
        # full_out = [[f_vp_out, [f_vs_IC_out, f_vs_M_out], f_rho_out], [s0, s1]]
        # selector encodes which component we're probing — use its label
        return selector.extract_output(full_out)

    # Build a selector class that knows which component to extract
    class _Selector:
        """Helper to embed a component bump into the full model-space structure and extract output."""

        def __init__(self, comp_name):
            self.comp_name = comp_name

        def __call__(self, bump_fn):
            """Return a full model-space input with bump_fn in the right slot."""
            zero_vp = _make_zero_fn(M_vp_space)
            zero_vs_IC = _make_zero_fn(M_vs_IC_space)
            zero_vs_M = _make_zero_fn(M_vs_M_space)
            zero_rho = _make_zero_fn(M_rho_space)
            if self.comp_name == 'vp':
                return [[bump_fn, [zero_vs_IC, zero_vs_M], zero_rho], [np.zeros(1), np.zeros(1)]]
            elif self.comp_name == 'vs_IC':
                return [[zero_vp, [bump_fn, zero_vs_M], zero_rho], [np.zeros(1), np.zeros(1)]]
            elif self.comp_name == 'vs_M':
                return [[zero_vp, [zero_vs_IC, bump_fn], zero_rho], [np.zeros(1), np.zeros(1)]]
            elif self.comp_name == 'rho':
                return [[zero_vp, [zero_vs_IC, zero_vs_M], bump_fn], [np.zeros(1), np.zeros(1)]]
            else:
                raise ValueError(f"Unknown component: {self.comp_name}")

        def extract_output(self, full_out):
            """Extract the component's function from the nested covariance output."""
            if self.comp_name == 'vp':
                return full_out[0][0]
            elif self.comp_name == 'vs_IC':
                return full_out[0][1][0]
            elif self.comp_name == 'vs_M':
                return full_out[0][1][1]
            elif self.comp_name == 'rho':
                return full_out[0][2]
            else:
                raise ValueError(f"Unknown component: {self.comp_name}")

    def _make_zero_fn(M_space):
        return _IFunction(M_space, evaluate_callable=lambda r: np.zeros_like(np.asarray(r, dtype=float)))

    # Probe centres (spaced uniformly, n_probes points)
    probe_r_vp = np.linspace(0.0 + R / (2 * n_probes), R - R / (2 * n_probes), n_probes)
    probe_r_vs_IC = np.linspace(ICB / (2 * n_probes), ICB - ICB / (2 * n_probes), n_probes)
    dom_M_len = R - CMB
    probe_r_vs_M = np.linspace(CMB + dom_M_len / (2 * n_probes), R - dom_M_len / (2 * n_probes), n_probes)
    probe_r_rho = np.linspace(0.0 + R / (2 * n_probes), R - R / (2 * n_probes), n_probes)

    sel_vp = _Selector('vp')
    sel_vs_IC = _Selector('vs_IC')
    sel_vs_M = _Selector('vs_M')
    sel_rho = _Selector('rho')

    std_vp = _probe_std(M_vp_space, probe_r_vp, sel_vp)
    std_vs_IC = _probe_std(M_vs_IC_space, probe_r_vs_IC, sel_vs_IC)
    std_vs_M = _probe_std(M_vs_M_space, probe_r_vs_M, sel_vs_M)
    std_rho = _probe_std(M_rho_space, probe_r_rho, sel_rho)

    # ── Plot ─────────────────────────────────────────────────────────────────
    # Layout: 5 panels — {δvp, δvs-IC, δvs-M, δρ, σ₁}
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Block (s={block.s}, t={block.t}) — radial posterior", fontsize=13)

    def _plot_radial_panel(ax, r_mean, mean, probe_r, std, label, color='steelblue'):
        ax.plot(mean, r_mean, color=color, lw=1.5, label='mean')
        ax.fill_betweenx(
            probe_r,
            np.interp(probe_r, r_mean, mean) - std,
            np.interp(probe_r, r_mean, mean) + std,
            alpha=0.3, color=color, label='±1σ',
        )
        ax.set_ylabel('radius (km)')
        ax.set_xlabel(label)
        ax.legend(fontsize=7)

    # Panel 0: vp
    _plot_radial_panel(axes[0], r_vp, mean_vp, probe_r_vp, std_vp, 'δvp', color='steelblue')

    # Panel 1: vs — inner core
    _plot_radial_panel(axes[1], r_vs_IC, mean_vs_IC, probe_r_vs_IC, std_vs_IC,
                       'δvs (IC)', color='forestgreen')

    # Panel 2: vs — mantle
    _plot_radial_panel(axes[2], r_vs_M, mean_vs_M, probe_r_vs_M, std_vs_M,
                       'δvs (mantle)', color='seagreen')

    # Panel 3: rho
    _plot_radial_panel(axes[3], r_rho, mean_rho, probe_r_rho, std_rho, 'δρ', color='goldenrod')

    # Panel 4: sigma_1 scalar (CMB topography) — prior ±1σ as reference
    sigma_1_std = np.sqrt(specs.sigma_var) if hasattr(specs, 'sigma_var') else 10.0
    axes[4].axhline(sigma_1_mean, color='firebrick', lw=2, label=f'mean = {sigma_1_mean:.3f}')
    axes[4].errorbar(
        x=0.5, y=sigma_1_mean, yerr=sigma_1_std,
        fmt='o', color='firebrick', capsize=6, label=f'±σ_prior = {sigma_1_std:.2f}',
    )
    axes[4].set_xlim(0, 1)
    axes[4].set_xticks([])
    axes[4].set_ylabel('σ₁ (CMB topo)')
    axes[4].set_xlabel('σ₁')
    axes[4].legend(fontsize=7)

    fig.tight_layout()
    return fig


# =============================================================================
# Figure 2 — CMB topography map
# =============================================================================

def plot_cmb_map(
    blocks: list,
    posterior_dict: dict,
    s_max: int,
    *,
    n_lat: int = 90,
    n_lon: int = 180,
) -> matplotlib.figure.Figure:
    """Synthesise and plot the CMB topography map from posterior sigma_1 coefficients.

    Reconstructs the spherical harmonic representation of the CMB boundary
    displacement:

    $$\\delta\\Sigma(\\theta, \\varphi) = \\sum_{s,t} \\sigma_1^{st} Y_s^t(\\theta, \\varphi)$$

    using the posterior mean $\\sigma_1^{st}$ for each block and pyshtools
    for the synthesis.

    Parameters
    ----------
    blocks : list of BlockIndex
        All (s, t) spectral blocks in the inversion.
    posterior_dict : dict
        ``{BlockIndex: GaussianMeasure}`` — posterior measures with accessible
        ``.expectation`` attribute.  The CMB coefficient is extracted as
        ``posterior_dict[block].expectation[1][1][0]``.
    s_max : int
        Maximum degree for the SH synthesis.
    n_lat : int
        Number of latitude grid points for display.  Default 90.
    n_lon : int
        Number of longitude grid points for display.  Default 180.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the CMB topography map and a colorbar.

    Raises
    ------
    ValueError
        If ``blocks`` is empty.
    ImportError
        If ``pyshtools`` is not installed.
    """
    try:
        import pyshtools
    except ImportError as exc:
        raise ImportError(
            "pyshtools is required for CMB map synthesis.  "
            "Install with: pip install pyshtools"
        ) from exc

    if not blocks:
        raise ValueError("blocks must be non-empty to construct the CMB map.")

    # Build the SH coefficient array
    coeffs = np.zeros((2, s_max + 1, s_max + 1))
    for block in blocks:
        s, t = block.s, block.t
        if s > s_max:
            continue
        sigma_1_val = float(posterior_dict[block].expectation[1][1][0])
        # t-index convention (Dahlen & Tromp):
        # t=0 → zonal (m=0): coeffs[0, s, 0]
        # t=2k-1 (odd) → cosine order k: coeffs[0, s, k]
        # t=2k (even, t>0) → sine order k: coeffs[1, s, k]
        if t == 0:
            coeffs[0, s, 0] = sigma_1_val
        elif t % 2 == 1:
            m = (t + 1) // 2
            if m <= s_max:
                coeffs[0, s, m] = sigma_1_val
        else:
            m = t // 2
            if m <= s_max:
                coeffs[1, s, m] = sigma_1_val

    # Synthesise the map via pyshtools
    sh_obj = pyshtools.SHCoeffs.from_array(coeffs, normalization='4pi', lmax=s_max)
    grid = sh_obj.expand(grid='DH', lmax=s_max)
    map_data = grid.data  # shape (nlat_DH, nlon_DH)

    # Build lat/lon axes for the DH grid
    nlat_dh, nlon_dh = map_data.shape
    lats_dh = 90.0 - np.arange(nlat_dh) * 180.0 / nlat_dh
    lons_dh = np.arange(nlon_dh) * 360.0 / nlon_dh

    # Try Cartopy for a Mollweide projection
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection=ccrs.Mollweide())
        lon2d, lat2d = np.meshgrid(lons_dh, lats_dh)
        im = ax.pcolormesh(
            lon2d, lat2d, map_data,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            shading='auto',
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_global()
        ax.set_title(f"CMB topography δΣ (s ≤ {s_max})")
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='δΣ_CMB')
    except ImportError:
        # Fallback: plain imshow
        fig, ax = plt.subplots(figsize=(10, 5))
        extent = [lons_dh[0], lons_dh[-1], lats_dh[-1], lats_dh[0]]
        im = ax.imshow(map_data, extent=extent, aspect='auto', cmap='RdBu_r', origin='upper')
        ax.set_xlabel('longitude (°)')
        ax.set_ylabel('latitude (°)')
        ax.set_title(f"CMB topography δΣ (s ≤ {s_max})")
        plt.colorbar(im, ax=ax, label='δΣ_CMB')

    fig.tight_layout()
    return fig


# =============================================================================
# Figure 3 — Equatorial slice (gated off by default)
# =============================================================================

def plot_equatorial_slice(
    blocks: list,
    posterior_dict: dict,
    s_max: int,
    param: str = 'vs',
    *,
    enabled: bool = False,
    n_lat: int = 90,
    n_lon: int = 180,
    r_km: float = 4000.0,
) -> Optional[matplotlib.figure.Figure]:
    """Synthesise and plot the equatorial slice for a volumetric parameter.

    Evaluates the posterior mean radial function at ``r_km`` for each block
    and synthesises the resulting angular map via pyshtools:

    $$\\delta p(r_{\\mathrm{eq}}, \\theta, \\varphi) = \\sum_{s,t} m_p^{st}(r_{\\mathrm{eq}}) Y_s^t(\\theta, \\varphi)$$

    Parameters
    ----------
    blocks : list of BlockIndex
        All (s, t) spectral blocks.
    posterior_dict : dict
        ``{BlockIndex: GaussianMeasure}`` posterior measures.
    s_max : int
        Maximum degree.
    param : str
        Which parameter to synthesise: ``'vp'``, ``'vs'``, or ``'rho'``.
        Default ``'vs'``.
    enabled : bool
        If ``False`` (default), return ``None`` immediately without computation.
    n_lat : int
        Number of latitude grid points for display.
    n_lon : int
        Number of longitude grid points for display.
    r_km : float
        Radial evaluation point in km.  Default 4000.0.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or ``None`` if ``enabled=False``.

    Raises
    ------
    ImportError
        If ``pyshtools`` is not installed and ``enabled=True``.
    """
    if not enabled:
        return None

    try:
        import pyshtools
    except ImportError as exc:
        raise ImportError(
            "pyshtools is required for equatorial slice synthesis.  "
            "Install with: pip install pyshtools"
        ) from exc

    # Build SH coefficient array from posterior means evaluated at r_km
    coeffs_mean = np.zeros((2, s_max + 1, s_max + 1))

    param_idx = {'vp': 0, 'vs': 1, 'rho': 2}
    if param not in param_idx:
        raise ValueError(f"Unknown param '{param}': choose from 'vp', 'vs', 'rho'.")

    p_idx = param_idx[param]

    for block in blocks:
        s, t = block.s, block.t
        if s > s_max:
            continue
        exp = posterior_dict[block].expectation
        # exp[0] = [f_vp, [f_vs_IC, f_vs_M], f_rho]
        if param == 'vs':
            # Evaluate at r_km: pick IC or mantle branch
            from full_spectrum_utils import RadialSpecs as _RS
            # Fall back to default specs for domain info — the caller should pass specs
            # if needed; here we use a safe default.
            ICB_default = 1217.5
            CMB_default = 3480.0
            if r_km <= ICB_default:
                val = float(exp[0][1][0](r_km))
            elif r_km >= CMB_default:
                val = float(exp[0][1][1](r_km))
            else:
                val = 0.0  # outer core
        else:
            val = float(exp[0][p_idx](r_km))

        if t == 0:
            coeffs_mean[0, s, 0] = val
        elif t % 2 == 1:
            m = (t + 1) // 2
            if m <= s_max:
                coeffs_mean[0, s, m] = val
        else:
            m = t // 2
            if m <= s_max:
                coeffs_mean[1, s, m] = val

    # Synthesise the mean map
    sh_mean = pyshtools.SHCoeffs.from_array(coeffs_mean, normalization='4pi', lmax=s_max)
    grid_mean = sh_mean.expand(grid='DH', lmax=s_max)
    map_mean = grid_mean.data

    nlat_dh, nlon_dh = map_mean.shape
    lats_dh = 90.0 - np.arange(nlat_dh) * 180.0 / nlat_dh
    lons_dh = np.arange(nlon_dh) * 360.0 / nlon_dh

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Equatorial slice δ{param} at r = {r_km:.0f} km (s ≤ {s_max})")

    extent = [lons_dh[0], lons_dh[-1], lats_dh[-1], lats_dh[0]]
    im0 = axes[0].imshow(map_mean, extent=extent, aspect='auto', cmap='RdBu_r', origin='upper')
    axes[0].set_title(f"δ{param} mean at r={r_km:.0f} km")
    axes[0].set_xlabel('longitude (°)')
    axes[0].set_ylabel('latitude (°)')
    plt.colorbar(im0, ax=axes[0], label=f'δ{param}')

    axes[1].set_visible(False)  # std panel placeholder (requires covariance probing)
    axes[1].set_title('std (not computed)')

    fig.tight_layout()
    return fig


# =============================================================================
# Figure 4 — Property posterior diagnostics
# =============================================================================

def plot_property_posterior_summary(
    mu_P: np.ndarray,
    sigma_P: np.ndarray,
    corr_P: np.ndarray,
    target_names: list,
) -> matplotlib.figure.Figure:
    """Plot the property posterior mean ± 1σ bar chart and correlation heatmap.

    Two-panel figure:

    - **Left panel**: horizontal bar chart showing the posterior mean ± 1σ
      for each property target.
    - **Right panel**: $N_p \\times N_p$ correlation heatmap with entries in
      $[-1, 1]$.

    Parameters
    ----------
    mu_P : np.ndarray
        Posterior mean vector, shape ``(N_p,)``.
    sigma_P : np.ndarray
        Posterior standard deviation vector, shape ``(N_p,)``.
    corr_P : np.ndarray
        Posterior correlation matrix, shape ``(N_p, N_p)``.
    target_names : list of str
        Human-readable name for each property, length ``N_p``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two side-by-side panels.
    """
    N_p = len(mu_P)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, N_p * 1.2), max(5, N_p * 0.7)))

    # Left: horizontal bar chart of mean ± sigma
    y_pos = np.arange(N_p)
    axes[0].barh(y_pos, mu_P, xerr=sigma_P, align='center',
                 color='steelblue', alpha=0.7, ecolor='darkblue', capsize=4)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(target_names, fontsize=8)
    axes[0].axvline(0.0, color='k', lw=0.8, ls='--')
    axes[0].set_xlabel('Property posterior mean ± 1σ')
    axes[0].set_title('Property posterior means')
    axes[0].invert_yaxis()

    # Right: correlation heatmap
    im = axes[1].imshow(corr_P, vmin=-1.0, vmax=1.0, cmap='RdBu_r', aspect='auto')
    axes[1].set_xticks(np.arange(N_p))
    axes[1].set_xticklabels(target_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(np.arange(N_p))
    axes[1].set_yticklabels(target_names, fontsize=8)
    axes[1].set_title('Property posterior correlation')
    plt.colorbar(im, ax=axes[1], label='correlation')

    fig.tight_layout()
    return fig
