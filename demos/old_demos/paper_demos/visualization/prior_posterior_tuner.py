"""
prior_posterior_tuner.py
========================

Interactive ipywidgets app for tuning Bessel-Sobolev prior hyperparameters for
a single (s, t) block and immediately computing the Bayesian posterior.

Intended use in a Jupyter notebook
-----------------------------------
::

    import sys, os
    demo_dir = os.path.dirname(os.path.abspath("__file__"))
    sys.path.insert(0, os.path.join(demo_dir, "utils"))
    sys.path.insert(0, os.path.join(demo_dir, "visualization"))

    from prior_posterior_tuner import make_tuner_widget
    # ... load catalog, reg, specs, blocks, split, forward_dict as usual ...

    widget = make_tuner_widget(catalog, reg, specs, blocks, split)
    display(widget)

Workflow
--------
1. Select a spectral block (s, t) from the dropdown.
2. Adjust per-parameter Bessel-Sobolev hyperparameters:
   - ``s_order`` — smoothness order of the Matérn-like covariance.
   - ``length``  — length-scale in km controlling correlation range.
   - ``var``     — overall variance (amplitude²).
   - ``bc``      — boundary conditions at the interval endpoints.
   - ``τ``       — amplitude scaling factor (cheap; updates display instantly).
3. Adjust ``σ_var`` for the CMB topography scalar component.
4. Click **Build Prior** — rebuilds Bessel-Sobolev operators and precomputes
   the prior display data.  Takes 10–60 s depending on ``n_basis``.
5. Tau sliders update the prior preview instantly (no rebuild required).
6. Click **Compute Posterior** — solves the linear Bayesian inference problem
   for the current block using the built prior.  Takes 5–30 s.
7. Adjust the posterior ``n_σ`` slider to change the credible-interval width.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

import _path_setup  # noqa: F401

if TYPE_CHECKING:
    from full_spectrum_utils import BlockIndex, RadialSpecs
    from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BC_OPTIONS: List[tuple] = [
    ("Neumann ↓ / Neumann ↑", "neumann"),
    ("Neumann ↓ / Dirichlet ↑", "mixed_neumann_dirichlet"),
    ("Dirichlet ↓ / Neumann ↑", "mixed_dirichlet_neumann"),
    ("Dirichlet both", "dirichlet"),
]

# Default hyperparameters — mirrors _BESSEL_PARAMS in full_spectrum_utils.py
_DEFAULT_HYPER: Dict[str, dict] = {
    "vp":    {"s_order": 6.0, "length": 20.0, "var": 10.0, "bc": "mixed_neumann_dirichlet"},
    "vs_IC": {"s_order": 4.0, "length": 20.0, "var": 10.0, "bc": "neumann"},
    "vs_M":  {"s_order": 4.0, "length": 20.0, "var": 10.0, "bc": "mixed_neumann_dirichlet"},
    "rho":   {"s_order": 5.0, "length": 25.0, "var": 10.0, "bc": "mixed_neumann_dirichlet"},
}

_DEFAULT_TAUS: Dict[str, float] = {"vp": 1.0, "vs_IC": 1.0, "vs_M": 1.0, "rho": 1.0}

_PARAM_COLORS: Dict[str, str] = {
    "vp":    "steelblue",
    "vs_IC": "coral",
    "vs_M":  "coral",
    "rho":   "seagreen",
}

_PARAMS: List[str] = ["vp", "vs_IC", "vs_M", "rho"]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _build_custom_bessel_blocks(
    specs: "RadialSpecs",
    hyper: Dict[str, dict],
) -> Dict:
    """Build BesselSobolevInverse operators from user-supplied hyperparameters.

    Parameters
    ----------
    specs : RadialSpecs
        Shared configuration (n_basis, domain bounds, integration configs).
    hyper : dict
        ``{param_name: {"s_order": float, "length": float, "var": float, "bc": str}}``
        for each of ``"vp"``, ``"vs_IC"``, ``"vs_M"``, ``"rho"``.

    Returns
    -------
    dict
        ``{param_name: BesselSobolevInverse}``
    """
    from intervalinf import (
        Lebesgue, IntervalDomain, BoundaryConditions, IntegrationConfig,
    )
    from intervalinf.operators import Laplacian, RadialLaplacian, BesselSobolevInverse
    from intervalinf.spaces import WeightedLebesgue
    from full_spectrum_utils import _make_component_space, _radial_bc

    N = specs.n_basis
    cfg = specs.lebesgue_cfg.inner_product
    pcfg = specs.parallel_cfg
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km
    R = specs.earth_radius_km
    domain = IntervalDomain(0, R)

    ref_domains = {
        "vp":    domain,
        "vs_IC": IntervalDomain(0, ICB),
        "vs_M":  IntervalDomain(CMB, R),
        "rho":   domain,
    }
    ref_spaces = {
        p: _make_component_space(d, cfg, pcfg, specs.weighted)
        for p, d in ref_domains.items()
    }

    bessel_cfg = IntegrationConfig(method="trapz", n_points=max(2000, N * 40))
    n_samples = max(512, N * 16)

    result: Dict = {}
    for p, M_ref in ref_spaces.items():
        h = hyper[p]
        s_order = float(h["s_order"])
        length  = float(h["length"])
        var     = float(h["var"])
        bc_str  = str(h["bc"])

        k = float(np.power(var, -0.5 / s_order))
        alpha = (length ** 2) * (k ** 2)
        if specs.weighted:
            effective_bc = _radial_bc(bc_str, ref_domains[p])
            bcs = BoundaryConditions(bc_type=effective_bc)
            L = RadialLaplacian(
                M_ref, bcs, alpha,
                method="spectral", dofs=N, integration_config=cfg, n_samples=n_samples,
            )
        else:
            bcs = BoundaryConditions(bc_type=bc_str)
            L = Laplacian(
                M_ref, bcs, alpha,
                method="spectral", dofs=N, integration_config=cfg, n_samples=n_samples,
            )
        result[p] = BesselSobolevInverse(
            M_ref, M_ref, k, s_order, L,
            dofs=N, n_samples=n_samples, use_fast_transforms=True,
            integration_config=bessel_cfg,
        )
    return result


# ---------------------------------------------------------------------------
# Public widget factory
# ---------------------------------------------------------------------------

def make_tuner_widget(
    catalog,
    reg,
    specs: "RadialSpecs",
    blocks: List["BlockIndex"],
    split: Dict,
    *,
    n_grid: int = 300,
    n_samples: int = 5,
    n_probes: int = 15,
) -> "ipywidgets.VBox":
    """Create an interactive prior-posterior tuning widget.

    Parameters
    ----------
    catalog : NormalModeKernelCatalog
        Sensitivity kernel catalog.
    reg : NormalModeDataRegistry
        Full data registry (used to extract per-block data vectors via *split*).
    specs : RadialSpecs
        Shared configuration (n_basis, domain bounds, integration configs).
    blocks : list of BlockIndex
        Available spectral blocks for the block dropdown.
    split : dict
        ``{BlockIndex: NormalModeDataRegistry}`` — per-block sub-registries.
    n_grid : int
        Radial grid resolution for prior/posterior display.  Default 300.
    n_samples : int
        Number of prior sample curves to precompute.  Default 5.
    n_probes : int
        Number of Gaussian bump probes for posterior covariance estimation.
        Default 15.

    Returns
    -------
    ipywidgets.VBox
        Self-contained widget ready to be passed to ``display()``.
    """
    import ipywidgets as widgets
    from IPython.display import display as _display, clear_output

    from full_spectrum_utils import (
        build_block_forward,
        build_block_prior,
        solve_block,
    )
    from prior_viz import PriorViewer, _draw_prior_axes
    from posterior_viz import PosteriorViewer, _draw_posterior_axes

    # ── Mutable state ─────────────────────────────────────────────────────────
    state: Dict = {
        "viewer":        None,   # PriorViewer (set after Build Prior)
        "shared_bessel": None,   # cached BesselSobolevInverse dict
        "post_viewer":   None,   # PosteriorViewer (set after Compute Posterior)
        "forward_dict":  {},     # {BlockIndex: (G, C_D, M, D)}  — persists across rebuilds
    }

    # ── Block selector ────────────────────────────────────────────────────────
    block_dd = widgets.Dropdown(
        options=[(f"s={b.s}, t={b.t}", b) for b in blocks],
        description="Block:",
        layout=widgets.Layout(width="230px"),
    )

    # ── Per-parameter hyperparameter controls ─────────────────────────────────
    def _make_param_controls(pname: str, defaults: dict) -> tuple:
        """Return (container_box, dict_of_widgets) for one parameter."""
        color = _PARAM_COLORS.get(pname, "black")
        header = widgets.HTML(
            f"<b style='font-size:12px;color:{color};padding-right:8px'>"
            f"{pname}</b>",
            layout=widgets.Layout(width="58px"),
        )
        sl_order = widgets.FloatSlider(
            value=defaults["s_order"], min=1.0, max=12.0, step=0.5,
            description="order:", continuous_update=False,
            style={"description_width": "46px"},
            layout=widgets.Layout(width="240px"),
        )
        sl_length = widgets.FloatSlider(
            value=defaults["length"], min=5.0, max=300.0, step=5.0,
            description="length:", continuous_update=False,
            style={"description_width": "48px"},
            layout=widgets.Layout(width="250px"),
        )
        sl_var = widgets.FloatLogSlider(
            value=defaults["var"], base=10, min=-1.0, max=3.0, step=0.1,
            description="var:", continuous_update=False,
            style={"description_width": "36px"},
            layout=widgets.Layout(width="230px"),
        )
        bc_dd = widgets.Dropdown(
            options=_BC_OPTIONS,
            value=defaults["bc"],
            description="bc:",
            style={"description_width": "24px"},
            layout=widgets.Layout(width="260px"),
        )
        sl_tau = widgets.FloatSlider(
            value=_DEFAULT_TAUS[pname], min=0.0, max=10.0, step=0.05,
            description=f"τ_{pname}:", continuous_update=True,
            style={"description_width": "60px"},
            layout=widgets.Layout(width="270px"),
        )
        row1 = widgets.HBox([header, sl_order, sl_length, sl_var, bc_dd])
        row2 = widgets.HBox([
            widgets.Label("", layout=widgets.Layout(width="60px")),
            sl_tau,
        ])
        box = widgets.VBox(
            [row1, row2],
            layout=widgets.Layout(
                border="1px solid #e0e0e0",
                padding="4px",
                margin="2px 0",
            ),
        )
        return box, dict(order=sl_order, length=sl_length, var=sl_var, bc=bc_dd, tau=sl_tau)

    ctrl: Dict[str, Dict] = {}
    param_boxes = []
    for p in _PARAMS:
        box, cw = _make_param_controls(p, _DEFAULT_HYPER[p])
        ctrl[p] = cw
        param_boxes.append(box)

    # ── Global controls ────────────────────────────────────────────────────────
    sl_sigma_var = widgets.FloatLogSlider(
        value=100.0, base=10, min=-1.0, max=5.0, step=0.1,
        description="σ_var:", continuous_update=False,
        style={"description_width": "46px"},
        layout=widgets.Layout(width="260px"),
    )
    sl_n_sigma_prior = widgets.FloatSlider(
        value=2.0, min=0.5, max=5.0, step=0.5,
        description="n_σ:", continuous_update=True,
        style={"description_width": "36px"},
        layout=widgets.Layout(width="200px"),
    )
    sl_n_sigma_post = widgets.FloatSlider(
        value=1.0, min=0.5, max=5.0, step=0.5,
        description="n_σ:", continuous_update=True,
        style={"description_width": "36px"},
        layout=widgets.Layout(width="200px"),
    )

    btn_build   = widgets.Button(
        description="▶ Build Prior", button_style="primary",
        layout=widgets.Layout(width="140px"),
    )
    btn_resample = widgets.Button(
        description="⟳ Resample", layout=widgets.Layout(width="110px"),
    )
    btn_compute = widgets.Button(
        description="▶ Compute Posterior", button_style="warning",
        layout=widgets.Layout(width="190px"), disabled=True,
    )

    lbl_build   = widgets.Label("Status: not built")
    lbl_compute = widgets.Label("Status: not computed")

    out_prior = widgets.Output(layout=widgets.Layout(width="100%", min_height="200px"))
    out_post  = widgets.Output(layout=widgets.Layout(width="100%", min_height="200px"))

    # ── Helper: collect current hyperparameter values ─────────────────────────
    def _get_hyper() -> Dict[str, dict]:
        return {
            p: {
                "s_order": ctrl[p]["order"].value,
                "length":  ctrl[p]["length"].value,
                "var":     ctrl[p]["var"].value,
                "bc":      ctrl[p]["bc"].value,
            }
            for p in _PARAMS
        }

    def _get_taus() -> Dict[str, float]:
        return {p: ctrl[p]["tau"].value for p in _PARAMS}

    # ── Draw helpers ──────────────────────────────────────────────────────────
    def _redraw_prior() -> None:
        v = state["viewer"]
        if v is None or not v.is_precomputed:
            return
        block = block_dd.value
        taus  = _get_taus()
        data  = v.get_display_data(
            block,
            tau_vp=taus["vp"],
            tau_vs_IC=taus["vs_IC"],
            tau_vs_M=taus["vs_M"],
            tau_rho=taus["rho"],
            sigma_var=sl_sigma_var.value,
        )
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
        fig.suptitle(f"Prior — s={block.s}, t={block.t}", fontsize=9)
        _draw_prior_axes(axes, data, block, specs, n_sigma=sl_n_sigma_prior.value)
        fig.tight_layout()
        with out_prior:
            clear_output(wait=True)
            _display(fig)
        plt.close(fig)

    def _redraw_posterior() -> None:
        pv = state["post_viewer"]
        if pv is None:
            return
        block = block_dd.value
        if not pv.is_computed(block):
            return
        data = pv.get_display_data(block)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
        fig.suptitle(f"Posterior — s={block.s}, t={block.t}", fontsize=9)
        _draw_posterior_axes(axes, data, block, specs, n_sigma=sl_n_sigma_post.value)
        fig.tight_layout()
        with out_post:
            clear_output(wait=True)
            _display(fig)
        plt.close(fig)

    # ── Build Prior button ────────────────────────────────────────────────────
    def _on_build(_btn) -> None:
        btn_build.disabled = True
        btn_compute.disabled = True
        lbl_build.value = "Status: building Bessel operators…"
        hyper = _get_hyper()

        def _run() -> None:
            try:
                lbl_build.value = "Status: building operators…"
                shared_bessel = _build_custom_bessel_blocks(specs, hyper)
                lbl_build.value = "Status: precomputing prior…"
                viewer = PriorViewer(
                    shared_bessel, specs, n_grid=n_grid, n_samples=n_samples
                )
                viewer.precompute()
                state["viewer"]        = viewer
                state["shared_bessel"] = shared_bessel
                # Ensure the forward operator for the current block is ready
                block = block_dd.value
                if block not in state["forward_dict"]:
                    lbl_build.value = "Status: building forward operator…"
                    state["forward_dict"][block] = build_block_forward(
                        block.s, block.t, split[block], catalog, specs,
                    )
                lbl_build.value = "Status: ✓ prior built"
                btn_compute.disabled = False
                _redraw_prior()
            except Exception as exc:
                import traceback
                lbl_build.value = f"Status: ✗ {exc}"
                traceback.print_exc()
            finally:
                btn_build.disabled = False

        threading.Thread(target=_run, daemon=True).start()

    # ── Compute Posterior button ──────────────────────────────────────────────
    def _on_compute(_btn) -> None:
        btn_compute.disabled = True
        lbl_compute.value = "Status: assembling prior…"
        shared_bessel = state["shared_bessel"]
        taus = _get_taus()
        sigma_var = sl_sigma_var.value
        block = block_dd.value

        def _run() -> None:
            try:
                # Ensure forward operator exists for this block
                if block not in state["forward_dict"]:
                    lbl_compute.value = "Status: building forward operator…"
                    state["forward_dict"][block] = build_block_forward(
                        block.s, block.t, split[block], catalog, specs,
                    )
                G_st, C_D_st, M_st, D_st = state["forward_dict"][block]

                # Build prior with current taus (reuses pre-built Bessel ops)
                def _tau_fn(p: str, _s: int) -> float:
                    return taus.get(p, 1.0)

                prior_st = build_block_prior(
                    block.s, block.t, shared_bessel, specs,
                    tau_fn=_tau_fn,
                    sigma_var=sigma_var,
                )

                # Solve
                lbl_compute.value = "Status: solving…"
                d_st = split[block].data_vector
                posterior_st = solve_block(
                    block.s, block.t, G_st, C_D_st, prior_st, d_st
                )

                # Build PosteriorViewer for just this block
                lbl_compute.value = "Status: probing covariance…"
                pv = PosteriorViewer(
                    {block: posterior_st},
                    state["forward_dict"],
                    specs,
                    n_grid=n_grid,
                    n_probes=n_probes,
                )
                pv.compute_block(block)
                state["post_viewer"] = pv
                lbl_compute.value = "Status: ✓ posterior computed"
                _redraw_posterior()
            except Exception as exc:
                import traceback
                lbl_compute.value = f"Status: ✗ {exc}"
                traceback.print_exc()
            finally:
                btn_compute.disabled = False

        threading.Thread(target=_run, daemon=True).start()

    # ── Resample button ───────────────────────────────────────────────────────
    def _on_resample(_btn) -> None:
        v = state["viewer"]
        if v is not None and v.is_precomputed:
            v.resample()
            _redraw_prior()

    btn_build.on_click(_on_build)
    btn_compute.on_click(_on_compute)
    btn_resample.on_click(_on_resample)

    # ── Instant updates (τ and n_σ sliders) ───────────────────────────────────
    for p in _PARAMS:
        ctrl[p]["tau"].observe(lambda _change: _redraw_prior(), names="value")
    sl_n_sigma_prior.observe(lambda _change: _redraw_prior(), names="value")
    sl_n_sigma_post.observe(lambda _change: _redraw_posterior(), names="value")

    # ── Block change: invalidate cached posterior, redraw prior if available ──
    def _on_block_change(change) -> None:
        with out_post:
            clear_output()
        lbl_compute.value = "Status: block changed — recompute needed"
        btn_compute.disabled = (
            state["viewer"] is None or not state["viewer"].is_precomputed
        )
        _redraw_prior()

    block_dd.observe(_on_block_change, names="value")

    # ── Layout ────────────────────────────────────────────────────────────────
    _hr = lambda: widgets.HTML("<hr style='border:0;border-top:1px solid #d0d0d0;margin:6px 0'/>")

    hyper_section = widgets.VBox(
        [
            widgets.HTML("<b>Hyperparameters</b> "
                         "<span style='font-size:11px;color:#888'>"
                         "(order / length / var / bc — rebuild required)</span>"),
            *param_boxes,
            widgets.HBox([sl_sigma_var]),
        ],
        layout=widgets.Layout(
            border="1px solid #c0c0c0", padding="6px", margin="4px 0",
        ),
    )

    prior_controls = widgets.VBox([
        widgets.HBox([btn_build, btn_resample, lbl_build]),
        widgets.HBox([
            widgets.Label("Credible interval:", layout=widgets.Layout(width="120px")),
            sl_n_sigma_prior,
        ]),
    ])

    post_controls = widgets.VBox([
        widgets.HBox([btn_compute, lbl_compute]),
        widgets.HBox([
            widgets.Label("Credible interval:", layout=widgets.Layout(width="120px")),
            sl_n_sigma_post,
        ]),
    ])

    return widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='margin:4px 0 8px 0'>Prior–Posterior Tuner</h3>"
            ),
            block_dd,
            _hr(),
            hyper_section,
            _hr(),
            prior_controls,
            out_prior,
            _hr(),
            post_controls,
            out_post,
        ],
        layout=widgets.Layout(padding="8px"),
    )
