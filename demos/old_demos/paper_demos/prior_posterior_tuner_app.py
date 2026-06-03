#!/usr/bin/env python3
"""
prior_posterior_tuner_app.py
============================

Standalone Tkinter GUI for tuning Bessel-Sobolev prior hyperparameters and
computing the Bayesian posterior for a single (s, t) spectral block.

Usage
-----
    cd /path/to/paper_demos

    python prior_posterior_tuner_app.py
    python prior_posterior_tuner_app.py --n-basis 50 --s-max 6
    python prior_posterior_tuner_app.py \\
        --data-dir  data/normal-mode-data \\
        --kernel-dir data/normal-mode-kernels/kernels-all_PREM-layers_Adrian

The default data paths assume you run from the paper_demos/ directory.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Add local helper folders to sys.path so we can import demo modules.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "utils"))
sys.path.insert(0, str(_HERE / "visualization"))

# ---------------------------------------------------------------------------
# Set Matplotlib backend before *any* pyplot import.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ---------------------------------------------------------------------------
# Workspace root detection (for europa commands)
# ---------------------------------------------------------------------------

def _find_workspace_root() -> Path:
    """Walk up from this file until .europa.yml is found."""
    p = Path(__file__).resolve().parent
    while p.parent != p:
        if (p / ".europa.yml").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parent  # fallback: paper_demos dir


def _is_europa_running(status_text: str) -> bool:
    """Return True if europa status output indicates a running job."""
    txt = status_text.lower()
    return "running" in txt or "pid:" in txt


_WORKSPACE_ROOT = _find_workspace_root()


# ---------------------------------------------------------------------------
# Minimal viewer wrapper for NPZ-loaded posterior display data
# ---------------------------------------------------------------------------

class _EuropaDisplayViewer:
    """Wraps _PostBlockDisplayData objects loaded from a europa NPZ result."""

    def __init__(self, data_dict: dict) -> None:
        self._data = data_dict

    def is_computed(self, block) -> bool:
        return block in self._data

    def get_display_data(self, block):
        return self._data[block]


def _build_scaled_data_noise_measure(data_space, error_vector: np.ndarray,
                                     noise_std_multiplier: float):
    """Return a diagonal data-noise measure with scaled standard deviations."""
    from pygeoinf import GaussianMeasure

    multiplier = float(noise_std_multiplier)
    if multiplier <= 0.0:
        raise ValueError("Data-noise multiplier must be positive.")

    scaled_std = multiplier * np.asarray(error_vector, dtype=float)
    covariance_matrix = np.diag(scaled_std ** 2)
    return GaussianMeasure.from_covariance_matrix(
        data_space,
        covariance_matrix,
        expectation=np.zeros(data_space.dim),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAMS: List[str] = ["vp", "vs_IC", "vs_M", "rho"]

_DEFAULT_HYPER: Dict[str, dict] = {
    "vp":    {"s_order": 6.0, "length": 20.0, "log_var": 1.0, "bc": "mixed_neumann_dirichlet"},
    "vs_IC": {"s_order": 4.0, "length": 20.0, "log_var": 1.0, "bc": "neumann"},
    "vs_M":  {"s_order": 4.0, "length": 20.0, "log_var": 1.0, "bc": "mixed_neumann_dirichlet"},
    "rho":   {"s_order": 5.0, "length": 25.0, "log_var": 1.0, "bc": "mixed_neumann_dirichlet"},
}

_DEFAULT_TAUS: Dict[str, float] = {"vp": 1.0, "vs_IC": 1.0, "vs_M": 1.0, "rho": 1.0}

_PARAM_BG: Dict[str, str] = {
    "vp":    "#4682b4",
    "vs_IC": "#cd5c5c",
    "vs_M":  "#cd5c5c",
    "rho":   "#3a8c5a",
}

_BC_OPTIONS: List[tuple] = [
    ("Neumann ↓ / Neumann ↑",    "neumann"),
    ("Neumann ↓ / Dirichlet ↑",  "mixed_neumann_dirichlet"),
    ("Dirichlet ↓ / Neumann ↑",  "mixed_dirichlet_neumann"),
    ("Dirichlet both",            "dirichlet"),
]
_BC_LABEL_TO_VAL = dict(_BC_OPTIONS)
_BC_VAL_TO_LABEL = {v: k for k, v in _BC_LABEL_TO_VAL.items()}


# ---------------------------------------------------------------------------
# App class
# ---------------------------------------------------------------------------

class TunerApp:
    """Main application window."""

    def __init__(
        self,
        root: tk.Tk,
        catalog,
        reg,
        specs,
        blocks,
        split,
        *,
        n_grid: int = 300,
        n_samples: int = 5,
        n_probes: int = 15,
    ) -> None:
        self.root      = root
        self.catalog   = catalog
        self.reg       = reg
        self.specs     = specs
        self.blocks    = blocks
        self.split     = split
        self.n_grid    = n_grid
        self.n_samples = n_samples
        self.n_probes  = n_probes

        # Mutable state (accessed from both main thread and worker threads)
        self._viewer:        Optional[object] = None
        self._shared_bessel: Optional[dict]   = None
        self._post_viewer:   Optional[object] = None
        self._forward_dict:  dict             = {}
        self._posterior_noise_multiplier: Optional[float] = None
        self._building   = False
        self._computing  = False
        self._computing_all = False

        # Shared per-block posterior cache (real-data path), used by both the
        # "Compute All Blocks" (inversion) and "Compute Inference" buttons so
        # neither repeats the other's solves. Keyed per block on everything the
        # posterior depends on; see _block_posterior_key. The single-block
        # "Compute Posterior" button does NOT use this cache.
        self._block_posteriors: Dict[object, object] = {}   # block → GaussianMeasure
        self._block_keys:       Dict[object, tuple]  = {}    # block → cache key
        # Property-operator cache, keyed on the target-set identity so a
        # targets-only change rebuilds operators without re-solving any block.
        self._prop_ops:     Optional[dict] = None
        self._prop_ops_key: Optional[tuple] = None

        # Synthetic data state
        from synthetic_data import SynthConfig
        self._synth_config = SynthConfig()        # noise_scale=1, null_amp=0, seed=0
        self._synth_pm:     dict           = {}   # keyed block → SynthPosteriorMean (cleared on prior rebuild only)
        self._synth_cores:  dict           = {}   # keyed (block, null_amp, seed)
        self._synth_states: dict           = {}   # keyed block → last SynthState
        self._synth_win:    Optional[tk.Toplevel] = None

        # Bayesian-inference (property pushforward) state
        self._targets:           Optional[list]        = None   # lazily built default targets
        self._inferring                                = False
        self._target_win:        Optional[tk.Toplevel] = None
        self._inference_win:     Optional[tk.Toplevel] = None
        self._property_posterior: Optional[object]     = None
        self._map_win:           Optional[tk.Toplevel] = None
        self._posterior_sample_curves: list            = []   # list of {block: _PostBlockDisplayData}
        root.title("Prior–Posterior Tuner")
        root.geometry("1520x860")
        root.minsize(1100, 640)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Top-level layout: left control panel | right figure panels."""
        pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5,
                            sashrelief=tk.RAISED, bg="#cccccc")
        pw.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        left  = ttk.Frame(pw, width=390)
        right = ttk.Frame(pw)
        pw.add(left,  minsize=370)
        pw.add(right, minsize=800)

        self._build_controls(left)
        self._build_plots(right)

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Build the scrollable left-side control panel."""
        # Scrollable canvas + scrollbar
        canvas = tk.Canvas(parent, highlightthickness=0, bg="#f5f5f5")
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=(6, 4))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(win_id, width=event.width)

        inner.bind("<Configure>", _resize_canvas)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))

        # Mousewheel
        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        for w in (canvas, inner):
            w.bind("<MouseWheel>", _on_wheel)
            w.bind("<Button-4>",   lambda e: canvas.yview_scroll(-1, "units"))
            w.bind("<Button-5>",   lambda e: canvas.yview_scroll( 1, "units"))

        self._populate_controls(inner)

    def _populate_controls(self, f: ttk.Frame) -> None:
        """Fill the scrollable inner frame with all controls."""
        row = 0

        ttk.Label(f, text="Prior – Posterior Tuner",
                  font=("Helvetica", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 6))
        row += 1

        # ── Block selector ──────────────────────────────────────────────────
        ttk.Label(f, text="Block:").grid(row=row, column=0, sticky="w")
        self._block_var = tk.StringVar()
        labels = [f"s={b.s}, t={b.t}" for b in self.blocks]
        self._block_combo = ttk.Combobox(f, textvariable=self._block_var,
                                         values=labels, state="readonly", width=18)
        self._block_combo.current(0)
        self._block_combo.grid(row=row, column=1, columnspan=2,
                               sticky="w", padx=4, pady=2)
        self._block_combo.bind("<<ComboboxSelected>>", self._on_block_change)
        row += 1

        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1
        # ── Model space mode ─────────────────────────────────────────────────
        self._weighted_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f,
            text="Weighted L²(r²) + RadialLaplacian",
            variable=self._weighted_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Label(
            f,
            text="⟨f,g⟩ = ∫fg r²dr (spherical geometry). Takes effect on next ▶ Build Prior.",
            foreground="gray", wraplength=340, font=("Helvetica", 8),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 3))
        row += 1

        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1
        # ── CPU count ───────────────────────────────────────────────────────
        cpu_row = ttk.Frame(f)
        cpu_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(cpu_row, text="CPUs (n_jobs):").pack(side=tk.LEFT)
        n_cpu = os.cpu_count() or 4
        self._n_jobs_var = tk.IntVar(value=1)
        ttk.Spinbox(cpu_row, from_=1, to=n_cpu, textvariable=self._n_jobs_var,
                    width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(cpu_row, text=f"(1\u2013{n_cpu}; for operator assembly)",
                  foreground="gray").pack(side=tk.LEFT)
        row += 1

        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1
        # ── Hyperparameter controls (one section per component) ─────────────
        ttk.Label(f, text="Hyperparameters",
                  font=("Helvetica", 9, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        row += 1

        self._ctrl: Dict[str, Dict[str, tk.Variable]] = {}

        for p in _PARAMS:
            row = self._add_param_section(f, row, p)

        # ── sigma_var ───────────────────────────────────────────────────────
        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1

        self._sv_logvar = tk.DoubleVar(value=2.0)   # 10^2 = 100
        row = self._add_log_slider(f, row, "σ_var:", self._sv_logvar,
                                   lo=-1.0, hi=5.0)

        self._data_noise_mult = tk.DoubleVar(value=1.0)
        row = self._add_slider(
            f,
            row,
            "data noise × σ_D:",
            self._data_noise_mult,
            lo=0.1,
            hi=10.0,
            res=0.05,
            fmt="{:.2f}",
        )

        # ── n_σ prior ───────────────────────────────────────────────────────
        self._nsig_prior = tk.DoubleVar(value=2.0)
        self._nsig_prior.trace_add("write", lambda *_: self.root.after(0, self._redraw_prior))
        row = self._add_slider(f, row, "n_σ (prior):", self._nsig_prior,
                               lo=0.5, hi=5.0, res=0.5, fmt="{:.1f}")

        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1

        # ── Build Prior + Resample ──────────────────────────────────────────
        btn_row = ttk.Frame(f)
        btn_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        self._btn_build = ttk.Button(btn_row, text="▶ Build Prior",
                                     command=self._on_build)
        self._btn_build.pack(side=tk.LEFT)
        self._btn_resample = ttk.Button(btn_row, text="⟳ Resample",
                                        command=self._on_resample)
        self._btn_resample.pack(side=tk.LEFT, padx=6)
        row += 1

        self._lbl_build = ttk.Label(f, text="Status: not built",
                                    foreground="gray", wraplength=340)
        self._lbl_build.grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        row += 1

        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1

        # ── n_σ posterior ───────────────────────────────────────────────────
        self._nsig_post = tk.DoubleVar(value=1.0)
        self._nsig_post.trace_add("write", lambda *_: self.root.after(0, self._redraw_posterior))
        row = self._add_slider(f, row, "n_σ (posterior):", self._nsig_post,
                               lo=0.5, hi=5.0, res=0.5, fmt="{:.1f}")

        # ── Compute mode ──────────────────────────────────────────────────
        self._compute_mode_var = tk.StringVar(value="local")
        mode_row = ttk.Frame(f)
        mode_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(mode_row, text="Compute on:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Local",  variable=self._compute_mode_var,
                        value="local",  command=self._on_mode_change).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Radiobutton(mode_row, text="Europa", variable=self._compute_mode_var,
                        value="europa", command=self._on_mode_change).pack(side=tk.LEFT, padx=2)
        row += 1

        # Europa VPN row (hidden by default; shown when Europa mode selected)
        self._europa_ctrl_row = ttk.Frame(f)
        self._europa_ctrl_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        self._btn_vpn = ttk.Button(self._europa_ctrl_row, text="Check VPN",
                                    command=self._on_vpn_check)
        self._btn_vpn.pack(side=tk.LEFT)
        self._lbl_vpn = ttk.Label(self._europa_ctrl_row, text="", foreground="gray")
        self._lbl_vpn.pack(side=tk.LEFT, padx=6)
        self._europa_ctrl_row.grid_remove()  # hidden until Europa mode is chosen
        row += 1

        # ── Data mode (Real / Synthetic) ──────────────────────────
        self._data_mode_var = tk.StringVar(value="real")
        data_row = ttk.Frame(f)
        data_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        ttk.Label(data_row, text="Data:").pack(side=tk.LEFT)
        ttk.Radiobutton(data_row, text="Real", variable=self._data_mode_var,
                        value="real", command=self._on_data_mode_change
                        ).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Radiobutton(data_row, text="Synthetic", variable=self._data_mode_var,
                        value="synthetic", command=self._on_data_mode_change
                        ).pack(side=tk.LEFT, padx=2)
        self._btn_synth_cfg = ttk.Button(data_row, text="Configure…",
                                         command=self._on_open_synth_config,
                                         state="disabled")
        self._btn_synth_cfg.pack(side=tk.LEFT, padx=8)
        row += 1

        # ── Compute Posterior ───────────────────────────────────────────────
        self._btn_compute = ttk.Button(f, text="▶ Compute Posterior",
                                       command=self._on_compute, state="disabled")
        self._btn_compute.grid(row=row, column=0, columnspan=3,
                               sticky="w", pady=2)
        row += 1

        self._lbl_compute = ttk.Label(f, text="Status: not computed",
                                      foreground="gray", wraplength=340)
        self._lbl_compute.grid(row=row, column=0, columnspan=3,
                                sticky="w", pady=1)
        row += 1

        # ── Compute All Blocks ──────────────────────────────────────────────
        self._btn_compute_all = ttk.Button(
            f, text="▶▶ Compute All Blocks",
            command=self._on_compute_all, state="disabled")
        self._btn_compute_all.grid(row=row, column=0, columnspan=3,
                                   sticky="w", pady=2)
        row += 1

        ttk.Label(
            f,
            text="Solves every (s,t) block with the current prior, then use "
                 "the Block dropdown to browse each posterior.",
            foreground="gray", wraplength=340, font=("Helvetica", 8),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 1))
        row += 1

        # Number of posterior realisations cached for the model-map viewer
        # (local compute only — europa returns display data without covariance).
        ttk.Label(f, text="Posterior samples:").grid(
            row=row, column=0, sticky="w")
        self._n_samples_var = tk.IntVar(value=8)
        ttk.Spinbox(f, from_=0, to=64, increment=1, width=5,
                    textvariable=self._n_samples_var).grid(
            row=row, column=1, sticky="w")
        row += 1

        self._lbl_compute_all = ttk.Label(f, text="Status: not computed",
                                          foreground="gray", wraplength=340)
        self._lbl_compute_all.grid(row=row, column=0, columnspan=3,
                                   sticky="w", pady=1)
        row += 1

        # ── Bayesian inference (property pushforward) ───────────────────────
        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        row += 1
        ttk.Label(f, text="Bayesian inference (pushforward)",
                  font=("Helvetica", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        row += 1
        ttk.Label(
            f,
            text="Pushes the per-block posteriors through the target "
                 "property operators to obtain the property posterior.",
            foreground="gray", wraplength=340, font=("Helvetica", 8),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 2))
        row += 1

        self._btn_targets = ttk.Button(f, text="🎯 Show Target Kernels",
                                       command=self._on_show_targets)
        self._btn_targets.grid(row=row, column=0, columnspan=3,
                               sticky="w", pady=2)
        row += 1

        self._btn_infer = ttk.Button(f, text="▶ Compute Inference",
                                     command=self._on_compute_inference)
        self._btn_infer.grid(row=row, column=0, columnspan=3,
                             sticky="w", pady=2)
        row += 1

        self._lbl_infer = ttk.Label(f, text="Status: not computed",
                                    foreground="gray", wraplength=340)
        self._lbl_infer.grid(row=row, column=0, columnspan=3,
                             sticky="w", pady=1)
        row += 1

        # ── Posterior-mean model maps (needs all blocks computed) ───────────
        self._btn_maps = ttk.Button(
            f, text="🗺 Show Model Maps",
            command=self._on_show_maps, state="disabled")
        self._btn_maps.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        ttk.Label(
            f,
            text="Geographic maps of the posterior-mean model (vp, vs, ρ depth "
                 "slices + CMB topography). Requires Compute All Blocks first.",
            foreground="gray", wraplength=340, font=("Helvetica", 8),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 1))
        row += 1

        self._lbl_maps = ttk.Label(f, text="Status: not available",
                                   foreground="gray", wraplength=340)
        self._lbl_maps.grid(row=row, column=0, columnspan=3, sticky="w", pady=1)

    def _add_param_section(self, f: ttk.Frame, row: int, p: str) -> int:
        """Add a colored header + (order, length, var, bc, τ) row for parameter p."""
        d = _DEFAULT_HYPER[p]
        color = _PARAM_BG[p]

        # Colored header label
        hdr = tk.Label(f, text=f"  {p}  ", bg=color, fg="white",
                       font=("Helvetica", 9, "bold"), anchor="w")
        hdr.grid(row=row, column=0, columnspan=3, sticky="ew",
                 padx=0, pady=(5, 1))
        row += 1

        cv: Dict[str, tk.Variable] = {}

        # s_order
        v_so = tk.DoubleVar(value=d["s_order"])
        cv["s_order"] = v_so
        row = self._add_slider(f, row, "  order:", v_so,
                               lo=1.0, hi=12.0, res=0.5, fmt="{:.1f}",
                               indent=True)

        # length
        v_len = tk.DoubleVar(value=d["length"])
        cv["length"] = v_len
        row = self._add_slider(f, row, "  length (km):", v_len,
                               lo=5.0, hi=300.0, res=5.0, fmt="{:.0f}",
                               indent=True)

        # var (log scale)
        v_lv = tk.DoubleVar(value=d["log_var"])
        cv["log_var"] = v_lv
        row = self._add_log_slider(f, row, "  var:", v_lv,
                                   lo=-1.0, hi=3.0, indent=True)

        # bc dropdown
        v_bc = tk.StringVar(value=_BC_VAL_TO_LABEL[d["bc"]])
        cv["bc_label"] = v_bc
        ttk.Label(f, text="  bc:").grid(row=row, column=0, sticky="w", padx=(12, 0))
        bc_combo = ttk.Combobox(f, textvariable=v_bc,
                                values=[lbl for lbl, _ in _BC_OPTIONS],
                                state="readonly", width=26)
        bc_combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=4, pady=1)
        row += 1

        # τ (instant redraw)
        v_tau = tk.DoubleVar(value=_DEFAULT_TAUS[p])
        cv["tau"] = v_tau
        v_tau.trace_add("write", lambda *_: self.root.after(0, self._redraw_prior))
        row = self._add_slider(f, row, f"  τ_{p}:", v_tau,
                               lo=0.0, hi=10.0, res=0.05, fmt="{:.2f}",
                               indent=True)

        self._ctrl[p] = cv
        return row

    def _add_slider(
        self, f: ttk.Frame, row: int, label: str,
        var: tk.DoubleVar, lo: float, hi: float, res: float, fmt: str,
        indent: bool = False,
    ) -> int:
        """Add label + Scale + value display.  Returns next row."""
        pad_l = 12 if indent else 0
        ttk.Label(f, text=label).grid(
            row=row, column=0, sticky="w", padx=(pad_l, 0))

        val_lbl = ttk.Label(f, text=fmt.format(var.get()), width=7, anchor="e")
        var.trace_add("write", lambda *_: val_lbl.config(text=fmt.format(var.get())))

        sl = tk.Scale(f, variable=var, from_=lo, to=hi, resolution=res,
                      orient=tk.HORIZONTAL, showvalue=False,
                      length=165, bd=0, sliderrelief=tk.FLAT)
        sl.grid(row=row, column=1, sticky="w", padx=4, pady=1)
        val_lbl.grid(row=row, column=2, sticky="w")
        return row + 1

    def _add_log_slider(
        self, f: ttk.Frame, row: int, label: str,
        log_var: tk.DoubleVar, lo: float, hi: float,
        indent: bool = False,
    ) -> int:
        """Add a log10-scale slider (slider = log10 of actual value)."""
        pad_l = 12 if indent else 0
        ttk.Label(f, text=label).grid(
            row=row, column=0, sticky="w", padx=(pad_l, 0))

        val_lbl = ttk.Label(f, text=f"{10**log_var.get():.3g}", width=7, anchor="e")
        log_var.trace_add("write",
                          lambda *_: val_lbl.config(text=f"{10**log_var.get():.3g}"))

        sl = tk.Scale(f, variable=log_var, from_=lo, to=hi, resolution=0.1,
                      orient=tk.HORIZONTAL, showvalue=False,
                      length=165, bd=0, sliderrelief=tk.FLAT)
        sl.grid(row=row, column=1, sticky="w", padx=4, pady=1)
        val_lbl.grid(row=row, column=2, sticky="w")
        return row + 1

    # ── Right-side plots ─────────────────────────────────────────────────────

    def _build_plots(self, parent: ttk.Frame) -> None:
        """Embed two matplotlib figures (prior top, posterior bottom)."""
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Prior figure
        prior_frame = ttk.LabelFrame(parent, text="Prior")
        prior_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        prior_frame.rowconfigure(1, weight=1)
        prior_frame.columnconfigure(0, weight=1)

        self._fig_prior = Figure(figsize=(13, 3.4), tight_layout=True)
        self._axes_prior = self._fig_prior.subplots(1, 4)
        self._fig_prior.suptitle("Prior (not yet built)", fontsize=9, color="gray")

        self._canvas_prior = FigureCanvasTkAgg(self._fig_prior, master=prior_frame)
        toolbar_p = NavigationToolbar2Tk(self._canvas_prior, prior_frame, pack_toolbar=False)
        toolbar_p.grid(row=0, column=0, sticky="ew")
        self._canvas_prior.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._canvas_prior.draw()

        # Posterior figure (parameter panels top row + data-fit panel bottom row)
        post_frame = ttk.LabelFrame(parent, text="Posterior")
        post_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        post_frame.rowconfigure(1, weight=1)
        post_frame.columnconfigure(0, weight=1)

        from matplotlib.gridspec import GridSpec as _GS
        self._fig_post = Figure(figsize=(13, 5.0), tight_layout=True)
        _gs_post = _GS(2, 4, figure=self._fig_post,
                       height_ratios=[2.2, 1.0], hspace=0.65)
        self._axes_post = [self._fig_post.add_subplot(_gs_post[0, i])
                           for i in range(4)]
        self._ax_datafit = self._fig_post.add_subplot(_gs_post[1, :])
        self._ax_datafit.text(0.5, 0.5, "no data yet",
                              transform=self._ax_datafit.transAxes,
                              ha="center", va="center", fontsize=8, color="gray")
        self._fig_post.suptitle("Posterior (not yet computed)", fontsize=9, color="gray")

        self._canvas_post = FigureCanvasTkAgg(self._fig_post, master=post_frame)
        toolbar_q = NavigationToolbar2Tk(self._canvas_post, post_frame, pack_toolbar=False)
        toolbar_q.grid(row=0, column=0, sticky="ew")
        self._canvas_post.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._canvas_post.draw()

    # ── State helpers ────────────────────────────────────────────────────────

    def _current_block(self):
        return self.blocks[self._block_combo.current()]

    def _get_hyper(self) -> Dict[str, dict]:
        return {
            p: {
                "s_order": self._ctrl[p]["s_order"].get(),
                "length":  self._ctrl[p]["length"].get(),
                "var":     10 ** self._ctrl[p]["log_var"].get(),
                "bc":      _BC_LABEL_TO_VAL[self._ctrl[p]["bc_label"].get()],
            }
            for p in _PARAMS
        }

    def _get_taus(self) -> Dict[str, float]:
        return {p: self._ctrl[p]["tau"].get() for p in _PARAMS}

    def _current_data_noise_multiplier(self) -> float:
        return float(self._data_noise_mult.get())

    def _current_specs(self):
        """Return a copy of specs with the current n_jobs and weighted settings."""
        from intervalinf import ParallelConfig
        n_jobs = int(self._n_jobs_var.get())
        return dataclasses.replace(
            self.specs,
            parallel_cfg=ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs),
            weighted=bool(self._weighted_var.get()),
        )

    # ── Compute mode callbacks ────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        if self._compute_mode_var.get() == "europa":
            self._btn_compute.config(text="▶ Submit to Europa")
            self._btn_compute_all.config(text="▶▶ Submit All Blocks to Europa")
            self._europa_ctrl_row.grid()
        else:
            self._btn_compute.config(text="▶ Compute Posterior")
            self._btn_compute_all.config(text="▶▶ Compute All Blocks")
            self._europa_ctrl_row.grid_remove()

    # ── Synthetic-data callbacks ───────────────────────────────────

    def _on_data_mode_change(self) -> None:
        is_synth = self._data_mode_var.get() == "synthetic"
        self._btn_synth_cfg.config(state=("normal" if is_synth else "disabled"))
        # Toggle overlay on existing posterior plot if any
        self.root.after(0, self._redraw_posterior)

    def _on_open_synth_config(self) -> None:
        if self._synth_win is not None:
            try:
                self._synth_win.lift()
                return
            except tk.TclError:
                self._synth_win = None

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.gridspec import GridSpec
        from synthetic_data import SynthConfig

        win = tk.Toplevel(self.root)
        win.title("Synthetic data configuration")
        win.geometry("1060x680")
        win.resizable(True, True)
        self._synth_win = win

        # ── Top controls frame ────────────────────────────────────────────
        ctrl = ttk.Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        cfg = self._synth_config
        noise_var = tk.DoubleVar(value=float(cfg.noise_scale))
        null_var  = tk.DoubleVar(value=float(cfg.null_amp))
        seed_var  = tk.IntVar(value=int(cfg.seed))

        pad = dict(padx=6, pady=2)
        ttk.Label(ctrl, text="True model: posterior mean from real data + current prior",
                  foreground="gray").grid(row=0, column=0, columnspan=6,
                                          sticky="w", **pad)

        ttk.Label(ctrl, text="Noise scale (× σ_D):"
                  ).grid(row=1, column=0, sticky="w", **pad)
        tk.Scale(ctrl, variable=noise_var, from_=0.0, to=5.0, resolution=0.05,
                 orient=tk.HORIZONTAL, length=200, showvalue=True
                 ).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(ctrl, text="Null-space amplitude:"
                  ).grid(row=1, column=2, sticky="w", padx=(16, 6), pady=2)
        tk.Scale(ctrl, variable=null_var, from_=0.0, to=3.0, resolution=0.05,
                 orient=tk.HORIZONTAL, length=200, showvalue=True
                 ).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(ctrl, text="Random seed:"
                  ).grid(row=1, column=4, sticky="w", padx=(16, 6), pady=2)
        ttk.Entry(ctrl, textvariable=seed_var, width=8
                  ).grid(row=1, column=5, sticky="w", **pad)

        status_lbl = ttk.Label(ctrl, text="Click 'Apply + Preview' to generate.",
                               foreground="gray")
        status_lbl.grid(row=2, column=0, columnspan=4, sticky="w", **pad)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=2, column=4, columnspan=2, sticky="e", pady=4)

        # ── Matplotlib figure ──────────────────────────────────────────────
        fig = Figure(figsize=(10.5, 5.0), tight_layout=True)
        gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       height_ratios=[1.3, 1.0])
        ax_vp   = fig.add_subplot(gs[0, 0])
        ax_vs   = fig.add_subplot(gs[0, 1])   # combined IC + mantle
        ax_rho  = fig.add_subplot(gs[0, 2])
        ax_data = fig.add_subplot(gs[1, :])

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True,
                                    padx=4, pady=4)

        # Placeholder text on first open
        for ax, title in zip([ax_vp, ax_vs, ax_rho],
                              ["δvp", "δvs", "δρ"]):
            ax.set_title(title, fontsize=8)
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")
        ax_data.text(0.5, 0.5, "no data", transform=ax_data.transAxes,
                     ha="center", va="center", fontsize=8, color="gray")
        ax_data.set_title("Data: real vs synthetic", fontsize=8)
        canvas.draw()

        # ── Plot helpers ───────────────────────────────────────────────────
        _PARAM_COLORS = {
            "vp": "steelblue", "vs_IC": "forestgreen",
            "vs_M": "seagreen", "rho": "goldenrod",
        }

        def _draw_model(synth) -> None:
            """Redraw the three true-model profile panels."""
            specs = self._current_specs()
            R, ICB, CMB = (specs.earth_radius_km,
                           specs.icb_radius_km, specs.cmb_radius_km)

            # vp panel
            ax_vp.cla()
            ax_vp.plot(synth.true_vp, synth.r_vp,
                       color=_PARAM_COLORS["vp"], linewidth=1.4, label="true model")
            ax_vp.axvline(0, color="gray", linewidth=0.6, linestyle=":")
            ax_vp.set_xlabel("δvp (km/s)", fontsize=7)
            ax_vp.set_ylabel("r (km)", fontsize=7)
            ax_vp.set_ylim(0.0, R)
            ax_vp.tick_params(labelsize=6)
            ax_vp.set_title("δvp", fontsize=8)
            ax_vp.legend(fontsize=6)

            # Combined vs panel: IC + outer-core gap + mantle
            ax_vs.cla()
            ax_vs.plot(synth.true_vs_IC, synth.r_vs_IC,
                       color=_PARAM_COLORS["vs_IC"], linewidth=1.4, label="IC")
            ax_vs.axhspan(ICB, CMB, color="lightgray", alpha=0.55, zorder=0)
            ax_vs.text(0.0, 0.5 * (ICB + CMB), "outer core",
                       transform=ax_vs.get_yaxis_transform(),
                       fontsize=6, color="gray", va="center", ha="left")
            ax_vs.plot(synth.true_vs_M, synth.r_vs_M,
                       color=_PARAM_COLORS["vs_M"], linewidth=1.4, label="mantle")
            ax_vs.axvline(0, color="gray", linewidth=0.6, linestyle=":")
            ax_vs.set_xlabel("δvs (km/s)", fontsize=7)
            ax_vs.set_ylabel("r (km)", fontsize=7)
            ax_vs.set_ylim(0.0, R)
            ax_vs.tick_params(labelsize=6)
            ax_vs.set_title("δvs", fontsize=8)
            ax_vs.legend(fontsize=6)

            # rho panel
            ax_rho.cla()
            ax_rho.plot(synth.true_rho, synth.r_rho,
                        color=_PARAM_COLORS["rho"], linewidth=1.4, label="true model")
            ax_rho.axvline(0, color="gray", linewidth=0.6, linestyle=":")
            ax_rho.set_xlabel("δρ (g/cm³)", fontsize=7)
            ax_rho.set_ylabel("r (km)", fontsize=7)
            ax_rho.set_ylim(0.0, R)
            ax_rho.tick_params(labelsize=6)
            ax_rho.set_title("δρ", fontsize=8)
            ax_rho.legend(fontsize=6)

        def _draw_data(synth, d_real, err_real) -> None:
            """Redraw the data comparison panel."""
            ax_data.cla()
            N = len(d_real)
            idx = np.arange(N)
            noise_scale = synth.config.noise_scale
            ax_data.errorbar(idx, d_real, yerr=err_real,
                             fmt="o", ms=3, color="steelblue",
                             ecolor="steelblue", elinewidth=0.8,
                             capsize=2, alpha=0.8, label="real data ± σ_D",
                             zorder=3)
            ax_data.errorbar(idx + 0.25, synth.d_synth,
                             yerr=noise_scale * err_real,
                             fmt="s", ms=3, color="darkorange",
                             ecolor="darkorange", elinewidth=0.8,
                             capsize=2, alpha=0.8,
                             label=f"synth ± {noise_scale:.2f}σ_D",
                             zorder=3)
            ax_data.plot(idx + 0.125, synth.d_clean, "_",
                         ms=6, color="black", markeredgewidth=1.5,
                         label="G m_true (noiseless)", zorder=4)
            ax_data.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax_data.set_xlabel("observation index", fontsize=7)
            ax_data.set_ylabel("splitting coeff.", fontsize=7)
            ax_data.tick_params(labelsize=6)
            ax_data.legend(fontsize=6, ncol=3)

            # ── Data-fit quality: RMS standardized residual for G m vs d_real
            block_now = self._current_block()
            pm_now = self._synth_pm.get(block_now)
            d_pred = pm_now.d_clean_base if pm_now is not None else synth.d_clean
            safe_err = np.where(err_real > 0, err_real, np.inf)
            chi2 = float(np.mean(((d_real - d_pred) / safe_err) ** 2))
            chi_rms = float(np.sqrt(chi2))
            target = getattr(pm_now, "fit_target_chi_rms", None) if pm_now else None
            lam = getattr(pm_now, "fit_lambda", None) if pm_now else None
            if target is None:
                chi2_label = f"RMS σ = {chi_rms:.2f}  (G m vs d_real)"
                chi2_color = ("forestgreen" if chi_rms < 1e-2
                              else ("darkorange" if chi_rms < 2.0 else "red"))
            else:
                lam_label = "" if lam is None else f", λ={lam:.1e}"
                chi2_label = f"RMS σ = {chi_rms:.2f} / {target:.1f}{lam_label}"
                chi2_color = ("forestgreen" if chi_rms <= target
                              else ("darkorange" if chi_rms <= 1.5 * target else "red"))
            ax_data.set_title(
                f"Data: real vs synthetic   |   {chi2_label}",
                fontsize=7.5, color=chi2_color)

        # If we already have a cached state for the current block, draw it now
        block = self._current_block()
        cached = self._synth_states.get(block)
        if cached is not None:
            try:
                d_real   = self.split[block].data_vector
                err_real = self.split[block].error_vector
                _draw_model(cached)
                _draw_data(cached, d_real, err_real)
                canvas.draw()
                status_lbl.config(text="Showing cached state.", foreground="green")
            except Exception:
                pass

        # ── Apply + Preview ────────────────────────────────────────────────
        def _on_apply():
            from synthetic_data import SynthConfig

            new_cfg = SynthConfig(
                noise_scale=float(noise_var.get()),
                null_amp=float(null_var.get()),
                seed=int(seed_var.get()),
            )
            self._synth_config = new_cfg
            # Note: do NOT call _invalidate_synth_cache() here.
            # Caches are keyed on their inputs and remain valid until the prior changes.
            status_lbl.config(text="Computing…", foreground="darkorange")
            btn_apply.state(["disabled"])

            # Check prerequisites
            block_   = self._current_block()
            if self._shared_bessel is None:
                status_lbl.config(
                    text="⚠ Build the prior first, then reopen this window.",
                    foreground="red")
                btn_apply.state(["!disabled"])
                return

            specs_   = self._current_specs()
            hyper_   = self._get_hyper()
            taus_    = self._get_taus()
            sigma_var_ = 10 ** self._sv_logvar.get()
            d_real_  = self.split[block_].data_vector
            err_real_= self.split[block_].error_vector
            shared_  = self._shared_bessel

            def _run():
                try:
                    from full_spectrum_utils import (
                        build_block_forward, build_block_prior)
                    from synthetic_data import (
                        compute_least_norm, compute_synth_core,
                        synth_state_from_core)

                    if block_ not in self._forward_dict:
                        self._forward_dict[block_] = build_block_forward(
                            block_.s, block_.t, self.split[block_],
                            self.catalog, specs_)
                    G_st, C_D_st, _, _ = self._forward_dict[block_]

                    # ── Level 1: least-norm cache (prior-independent) ───────
                    # C_ref-weighted target-fit smoother:
                    # m = C_ref G*(G C_ref G* + lambda C_D)^{-1} d.
                    # lambda is chosen for RMS standardized residual <= 2.
                    # Uses a fixed reference prior (tau=1) so the synthetic true
                    # model is independent of the tunable hyperparameters.
                    # Cache is invalidated when shared_bessel changes (new k/s).
                    pm = self._synth_pm.get(block_)
                    if pm is None:
                        win.after(0, lambda: status_lbl.config(
                            text="Computing smooth least-norm solution (C·G*(GCG*)⁻¹d)…",
                            foreground="darkorange"))
                        # Build fixed reference covariance (tau=1 for all params)
                        prior_ref = build_block_prior(
                            block_.s, block_.t, shared_, specs_,
                            tau_fn=lambda p, _s: 1.0, sigma_var=1.0)
                        pm = compute_least_norm(
                            block_, G_st, d_real_, specs_,
                            n_grid=self.n_grid,
                            C_ref=prior_ref.covariance,
                            data_std=err_real_,
                            target_chi_rms=2.0)
                        self._synth_pm[block_] = pm
                    prior_st = None  # not needed unless null_amp > 0

                    # ── Level 2: core cache (per null_amp + seed) ─────────
                    # Reuses pm; only runs when null_amp or seed changes.
                    core_key = (block_, new_cfg.null_amp, new_cfg.seed)
                    core = self._synth_cores.get(core_key)
                    if core is None:
                        if new_cfg.null_amp > 0.0 and prior_st is None:
                            def _tau_fn(p, _s):
                                return taus_.get(p, 1.0)
                            prior_st = build_block_prior(
                                block_.s, block_.t, shared_, specs_,
                                tau_fn=_tau_fn, sigma_var=sigma_var_)
                        win.after(0, lambda: status_lbl.config(
                            text="Drawing noise sample…" if new_cfg.null_amp == 0
                            else "Computing null-space perturbation…",
                            foreground="darkorange"))
                        core = compute_synth_core(
                            block_, G_st, C_D_st, prior_st, d_real_, specs_,
                            null_amp=new_cfg.null_amp, seed=new_cfg.seed,
                            n_grid=self.n_grid, posterior_mean=pm)
                        self._synth_cores[core_key] = core
                    else:
                        win.after(0, lambda: status_lbl.config(
                            text="Scaling noise (cached model)…",
                            foreground="steelblue"))

                    # ── Level 3: state (per noise_scale) — always cheap ───
                    state = synth_state_from_core(core, new_cfg)
                    self._synth_states[block_] = state

                    def _update():
                        try:
                            _draw_model(state)
                            _draw_data(state, d_real_, err_real_)
                            canvas.draw()
                            status_lbl.config(
                                text="✓ Applied.",
                                foreground="green")
                        except tk.TclError:
                            pass  # window was closed
                    win.after(0, _update)
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    msg = f"✗ {exc}"
                    win.after(0, lambda m=msg: status_lbl.config(
                        text=m, foreground="red"))
                finally:
                    win.after(0, lambda: btn_apply.state(["!disabled"]))

            threading.Thread(target=_run, daemon=True).start()

        def _on_close():
            self._synth_win = None
            win.destroy()

        btn_apply = ttk.Button(btn_frame, text="Apply + Preview",
                               command=_on_apply)
        btn_apply.pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Close", command=_on_close
                   ).pack(side=tk.LEFT, padx=6)

        win.protocol("WM_DELETE_WINDOW", _on_close)

    def _invalidate_synth_cache(self) -> None:
        """Clear synth states and cores. Does NOT clear posterior-mean cache."""
        self._synth_states.clear()
        self._synth_cores.clear()

    def _ensure_synth_state(self, block, G_st, C_D_st, prior_st, d_real, specs):
        """Generate (or retrieve cached) synthetic state for this block + config."""
        state = self._synth_states.get(block)
        if state is not None and state.config == self._synth_config:
            return state
        from synthetic_data import generate_synth_state
        state = generate_synth_state(
            block, G_st, C_D_st, prior_st, d_real, specs,
            config=self._synth_config, n_grid=self.n_grid,
        )
        self._synth_states[block] = state
        return state

    def _overlay_true_model(self, synth) -> None:
        """Overlay dashed black true-model lines on the posterior axes."""
        axes = self._axes_post
        style       = dict(color="black", linestyle="--", linewidth=1.2, zorder=10)
        style_lbl   = dict(style, label="true")
        axes[0].plot(synth.true_vp,    synth.r_vp,    **style_lbl)
        axes[1].plot(synth.true_vs_IC, synth.r_vs_IC, **style_lbl)
        axes[1].plot(synth.true_vs_M,  synth.r_vs_M,  **style)
        axes[2].plot(synth.true_rho,   synth.r_rho,   **style_lbl)
        axes[3].axhline(synth.sigma_1_true, color="black", linestyle="--",
                        linewidth=1.2, zorder=10,
                        label=f"true σ₁ = {synth.sigma_1_true:.3f}")
        for ax in axes:
            ax.legend(fontsize=7)

    def _on_vpn_check(self) -> None:
        self._lbl_vpn.config(text="checking…", foreground="darkorange")
        self._btn_vpn.state(["disabled"])

        def _run():
            try:
                result = subprocess.run(
                    ["europa", "status"],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    # Show first non-empty line of output
                    first = next(
                        (l.strip() for l in result.stdout.splitlines() if l.strip()),
                        "OK",
                    )
                    self.root.after(0, lambda t=first: self._lbl_vpn.config(
                        text=f"✓ {t}", foreground="green"))
                else:
                    err = (result.stderr or result.stdout or "unknown error").strip()
                    self.root.after(0, lambda e=err: self._lbl_vpn.config(
                        text=f"✗ {e[:60]}", foreground="red"))
            except FileNotFoundError:
                self.root.after(0, lambda: self._lbl_vpn.config(
                    text="✗ 'europa' not found in PATH", foreground="red"))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self._lbl_vpn.config(
                    text="✗ timed out — VPN connected?", foreground="red"))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._lbl_vpn.config(
                    text=f"✗ {e[:60]}", foreground="red"))
            finally:
                self.root.after(0, lambda: self._btn_vpn.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    # ── Figure draw helpers (must run on main thread) ─────────────────────────

    def _redraw_prior(self) -> None:
        from prior_viz import _draw_prior_axes
        v = self._viewer
        if v is None or not v.is_precomputed:
            return
        block = self._current_block()
        taus  = self._get_taus()
        data  = v.get_display_data(
            block,
            tau_vp=taus["vp"], tau_vs_IC=taus["vs_IC"],
            tau_vs_M=taus["vs_M"], tau_rho=taus["rho"],
            sigma_var=10 ** self._sv_logvar.get(),
        )
        for ax in self._axes_prior:
            ax.cla()
        _draw_prior_axes(self._axes_prior, data, block, self.specs,
                         n_sigma=self._nsig_prior.get())
        self._fig_prior.suptitle(f"Prior — s={block.s}, t={block.t}", fontsize=9)
        self._fig_prior.tight_layout()
        self._canvas_prior.draw_idle()

    def _redraw_posterior(self) -> None:
        from posterior_viz import _draw_posterior_axes, _draw_datafit_axes
        pv = self._post_viewer
        if pv is None:
            return
        block = self._current_block()
        if not pv.is_computed(block):
            return
        data = pv.get_display_data(block)
        for ax in self._axes_post:
            ax.cla()
        _draw_posterior_axes(self._axes_post, data, block, self.specs,
                             n_sigma=self._nsig_post.get())
        # Overlay true model in synthetic mode
        if self._data_mode_var.get() == "synthetic":
            synth = self._synth_states.get(block)
            if synth is not None:
                self._overlay_true_model(synth)

        # ── Data-fit panel ──────────────────────────────────────────────────
        noise_mult = (
            self._posterior_noise_multiplier
            if self._posterior_noise_multiplier is not None
            else self._current_data_noise_multiplier()
        )
        d_obs = self.split[block].data_vector
        d_err = noise_mult * self.split[block].error_vector
        data_label = f"data ± {noise_mult:.2g}σ_D"
        if self._data_mode_var.get() == "synthetic":
            synth = self._synth_states.get(block)
            if synth is not None and len(synth.d_synth) > 0:
                d_obs = synth.d_synth
                data_label = f"d_synth ± {noise_mult:.2g}σ_D (assumed)"
        _draw_datafit_axes(self._ax_datafit, d_obs, d_err,
                           data.d_pred_post, block, data_label=data_label)

        mode_tag = "  [synthetic]" if self._data_mode_var.get() == "synthetic" else ""
        self._fig_post.suptitle(
            f"Posterior — s={block.s}, t={block.t}{mode_tag}", fontsize=9)
        self._fig_post.tight_layout()
        self._canvas_post.draw_idle()

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_build(self) -> None:
        if self._building:
            return
        self._building = True
        self._btn_build.state(["disabled"])
        self._btn_compute.state(["disabled"])
        self._lbl_build.config(text="Status: starting…", foreground="darkorange")
        hyper = self._get_hyper()
        specs = self._current_specs()

        def _run() -> None:
            try:
                from prior_posterior_tuner import _build_custom_bessel_blocks
                from prior_viz import PriorViewer
                from full_spectrum_utils import build_block_forward

                self.root.after(0, lambda: self._lbl_build.config(
                    text="Status: building Bessel-Sobolev operators…",
                    foreground="darkorange"))
                shared_bessel = _build_custom_bessel_blocks(specs, hyper)

                self.root.after(0, lambda: self._lbl_build.config(
                    text="Status: precomputing prior eigenpairs…",
                    foreground="darkorange"))
                viewer = PriorViewer(shared_bessel, specs,
                                     n_grid=self.n_grid, n_samples=self.n_samples)
                viewer.precompute()

                self._viewer        = viewer
                self._shared_bessel = shared_bessel

                # Pre-cache forward operator for the currently selected block
                block = self._current_block()
                if block not in self._forward_dict:
                    self.root.after(0, lambda: self._lbl_build.config(
                        text="Status: building forward operator…",
                        foreground="darkorange"))
                    self._forward_dict[block] = build_block_forward(
                        block.s, block.t, self.split[block], self.catalog, specs)

                self.root.after(0, self._on_build_done)

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_build.config(
                    text=m, foreground="red"))
            finally:
                self._building = False
                self.root.after(0, lambda: self._btn_build.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _on_build_done(self) -> None:
        # Prior changed → synth states and cores must be regenerated.
        # _synth_pm also clears because the reference covariance (tau=1) is
        # built from shared_bessel, which may have new k/s parameters.
        self._synth_pm.clear()
        self._invalidate_synth_cache()
        # Prior rebuilt → the shared block-posterior cache is logically
        # invalidated (its key folds in id(shared_bessel)); drop stale entries
        # so memory does not grow across rebuilds. Property operators do not
        # depend on the prior and are left intact.
        self._block_posteriors.clear()
        self._block_keys.clear()
        # Forward operators must also be cleared: when weighted mode changes,
        # the model space type changes (plain Lebesgue ↔ WeightedLebesgue),
        # which changes the adjoint. A stale flat-mode forward paired with a
        # weighted-mode prior causes domain mismatch in the posterior solve.
        self._forward_dict.clear()
        self._lbl_build.config(text="Status: ✓ prior built", foreground="green")
        self._btn_compute.state(["!disabled"])
        self._btn_compute_all.state(["!disabled"])
        self._redraw_prior()

    def _on_compute(self) -> None:
        """Route to local or europa compute based on current mode."""
        if self._compute_mode_var.get() == "europa":
            self._on_compute_europa()
        else:
            self._on_compute_local()

    # ── Compute All Blocks ────────────────────────────────────────────────────

    def _on_compute_all(self) -> None:
        """Route the all-block posterior solve to local or europa."""
        if self._computing_all:
            return
        if self._compute_mode_var.get() == "europa":
            self._on_compute_all_europa()
        else:
            self._on_compute_all_local()

    # ── Shared per-block posterior cache ──────────────────────────────────────

    @staticmethod
    def _block_posterior_key(shared_bessel, taus: Dict[str, float],
                             sigma_var: float, noise_mult: float) -> tuple:
        """Cache key capturing everything a real-data block posterior depends on.

        ``id(shared_bessel)`` folds in the Bessel-Sobolev hyperparameters: a
        rebuilt prior is a fresh object, so all keys miss automatically. The
        forward operator and observed data are fixed per block (real data), so
        they are not part of the key.
        """
        return (id(shared_bessel),
                frozenset(taus.items()),
                float(sigma_var),
                float(noise_mult))

    def _ensure_block_posteriors(self, blocks, *, shared_bessel, taus,
                                 sigma_var, noise_mult, specs,
                                 progress=None) -> dict:
        """Return ``{block: GaussianMeasure}`` for *blocks*, solving only those
        whose cached posterior is stale or missing.

        Shared by the inversion ("Compute All Blocks") and inference paths so a
        block is never solved twice for the same prior/noise. ``progress`` is an
        optional ``callable(done, total, block)`` for status updates.
        """
        from full_spectrum_utils import (
            build_block_forward, build_block_prior, solve_block,
        )

        key = self._block_posterior_key(shared_bessel, taus, sigma_var, noise_mult)

        def _tau_fn(p: str, _s: int) -> float:
            return taus.get(p, 1.0)

        out: dict = {}
        total = len(blocks)
        for k, block in enumerate(blocks):
            if self._block_keys.get(block) == key and block in self._block_posteriors:
                out[block] = self._block_posteriors[block]   # cache hit
                continue
            if progress is not None:
                progress(k, total, block)
            if block not in self._forward_dict:
                self._forward_dict[block] = build_block_forward(
                    block.s, block.t, self.split[block], self.catalog, specs)
            G_st, _C_D_st, _M, D_st = self._forward_dict[block]
            C_D_assumed = _build_scaled_data_noise_measure(
                D_st, self.split[block].error_vector, noise_mult)
            prior_st = build_block_prior(
                block.s, block.t, shared_bessel, specs,
                tau_fn=_tau_fn, sigma_var=sigma_var)
            d_st = self.split[block].data_vector
            post = solve_block(block.s, block.t, G_st, C_D_assumed, prior_st, d_st)
            self._block_posteriors[block] = post
            self._block_keys[block] = key
            out[block] = post
        return out

    def _ensure_property_operators(self, targets, specs, s_max) -> dict:
        """Return per-block property operators, rebuilding only when the target
        set changed. Property operators do not depend on the prior or noise."""
        from full_spectrum_utils import build_property_operator
        key = (id(targets), tuple(self.blocks), s_max)
        if self._prop_ops_key == key and self._prop_ops is not None:
            return self._prop_ops
        prop_ops = build_property_operator(
            targets, self.blocks, self._forward_dict, specs, s_max)
        self._prop_ops = prop_ops
        self._prop_ops_key = key
        return prop_ops

    def _on_compute_all_local(self) -> None:
        """Solve every (s,t) block with the current prior and cache all
        posterior display data so the Block dropdown can browse each one.

        Uses real data with the current scaled data-noise; synthetic mode is a
        per-block workflow and is not applied here.
        """
        if self._computing_all:
            return
        self._computing_all = True
        self._btn_compute_all.state(["disabled"])
        self._btn_compute.state(["disabled"])
        self._lbl_compute_all.config(text="Status: starting…",
                                     foreground="darkorange")

        specs      = self._current_specs()
        hyper      = self._get_hyper()
        taus       = self._get_taus()
        sigma_var  = 10 ** self._sv_logvar.get()
        noise_mult = self._current_data_noise_multiplier()
        n_samples  = max(0, int(self._n_samples_var.get()))

        def _run() -> None:
            try:
                from posterior_viz import PosteriorViewer

                # Ensure shared Bessel-Sobolev blocks exist (build if needed).
                if self._shared_bessel is None:
                    from prior_posterior_tuner import _build_custom_bessel_blocks
                    self.root.after(0, lambda: self._lbl_compute_all.config(
                        text="Status: building Bessel-Sobolev operators…",
                        foreground="darkorange"))
                    self._shared_bessel = _build_custom_bessel_blocks(specs, hyper)
                shared_bessel = self._shared_bessel

                # Solve every block via the shared cache (real data + current
                # prior + scaled noise). Blocks already solved for this exact
                # prior/noise are reused; only stale/missing ones are re-solved.
                n_blocks = len(self.blocks)

                def _progress(done, total, b):
                    self.root.after(0, lambda: self._lbl_compute_all.config(
                        text=(f"Status: solving block {done + 1}/{total} "
                              f"(s={b.s}, t={b.t})…"),
                        foreground="darkorange"))

                posterior_dict = self._ensure_block_posteriors(
                    list(self.blocks), shared_bessel=shared_bessel, taus=taus,
                    sigma_var=sigma_var, noise_mult=noise_mult, specs=specs,
                    progress=_progress)

                # Build a single viewer over all blocks and probe covariances.
                pv = PosteriorViewer(
                    posterior_dict, self._forward_dict, specs,
                    n_grid=self.n_grid, n_probes=self.n_probes,
                    blocks=list(self.blocks),
                )
                for k, block in enumerate(self.blocks):
                    self.root.after(0, lambda k=k, b=block: self._lbl_compute_all.config(
                        text=(f"Status: probing covariance {k + 1}/{n_blocks} "
                              f"(s={b.s}, t={b.t})…"),
                        foreground="darkorange"))
                    pv.compute_block(block)

                self._post_viewer = pv
                self._posterior_noise_multiplier = noise_mult

                # ── Draw cached posterior realisations (model-map viewer) ───
                sample_curves: list = []
                if n_samples > 0:
                    import numpy as _np
                    from posterior_viz import compute_model_curves
                    _np.random.seed(0)  # reproducible realisations
                    # One sample index = one realisation of the whole model;
                    # draw n_samples per block, then transpose into realisations.
                    per_block_samples: dict = {}
                    for k, block in enumerate(self.blocks):
                        self.root.after(0, lambda k=k, b=block: self._lbl_compute_all.config(
                            text=(f"Status: sampling block {k + 1}/{n_blocks} "
                                  f"(s={b.s}, t={b.t})…"),
                            foreground="darkorange"))
                        per_block_samples[block] = posterior_dict[block].samples(n_samples)
                    for j in range(n_samples):
                        realisation = {
                            block: compute_model_curves(
                                per_block_samples[block][j], specs,
                                n_grid=self.n_grid)
                            for block in self.blocks
                        }
                        sample_curves.append(realisation)
                self._posterior_sample_curves = sample_curves

                self.root.after(0, self._on_compute_all_done)

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_compute_all.config(
                    text=m, foreground="red"))
            finally:
                self._computing_all = False
                self.root.after(0, lambda: self._btn_compute_all.state(["!disabled"]))
                self.root.after(0, lambda: self._btn_compute.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _on_compute_all_done(self) -> None:
        pv = self._post_viewer
        n = getattr(pv, "n_computed", len(self.blocks))
        self._lbl_compute_all.config(
            text=(f"Status: ✓ all {n} blocks computed — "
                  f"use the Block dropdown to browse"),
            foreground="green")
        # Posterior-mean model maps are now available.
        self._btn_maps.state(["!disabled"])
        n_real = len(self._posterior_sample_curves)
        if n_real > 0:
            self._lbl_maps.config(
                text=(f"Status: ready — {n_real} posterior samples cached "
                      f"(click 🗺 Show Model Maps)"),
                foreground="gray")
        else:
            self._lbl_maps.config(
                text="Status: ready — click 🗺 Show Model Maps (mean only)",
                foreground="gray")
        # Real data is now the basis for the cached posteriors.
        if self._data_mode_var.get() != "synthetic":
            self._redraw_posterior()

    def _on_compute_all_europa(self) -> None:
        """Submit an all-block posterior solve to europa, poll, pull, and
        cache every block's display data so the dropdown can browse them."""
        if self._computing_all:
            return
        self._computing_all = True
        self._btn_compute_all.state(["disabled"])
        self._btn_compute.state(["disabled"])
        self._lbl_compute_all.config(text="Status: preparing europa job…",
                                     foreground="darkorange")

        taus      = self._get_taus()
        sigma_var = 10 ** self._sv_logvar.get()
        data_noise_multiplier = self._current_data_noise_multiplier()
        hyper     = self._get_hyper()
        n_jobs    = int(self._n_jobs_var.get())
        n_basis   = self.specs.n_basis
        s_max     = max(b.s for b in self.blocks)

        def _run() -> None:
            try:
                from posterior_viz import _PostBlockDisplayData

                params = {
                    "task":      "all_blocks",
                    "n_basis":   n_basis,
                    "s_max":     s_max,
                    "hyper":     hyper,
                    "taus":      taus,
                    "sigma_var": sigma_var,
                    "data_noise_multiplier": data_noise_multiplier,
                    "n_jobs":    n_jobs,
                    "n_grid":    self.n_grid,
                    "n_probes":  self.n_probes,
                    "data_mode": "real",
                }
                _demo = (_WORKSPACE_ROOT
                         / "intervalinf/demos/old_demos/paper_demos")
                params_path = _demo / "tuner_all_params.json"
                npz_path    = _demo / "tuner_all_result.npz"

                if npz_path.exists():
                    npz_path.unlink()
                with open(params_path, "w") as fh:
                    json.dump(params, fh, indent=2)

                self.root.after(0, lambda: self._lbl_compute_all.config(
                    text="Status: pushing to europa (europa submit)…",
                    foreground="darkorange"))

                _rel_script = "intervalinf/demos/old_demos/paper_demos/prior_posterior_tuner_remote.py"
                _rel_params = "intervalinf/demos/old_demos/paper_demos/tuner_all_params.json"
                _rel_out    = "intervalinf/demos/old_demos/paper_demos/tuner_all_result.npz"
                cmd = (f"python {_rel_script}"
                       f" --params-json {_rel_params}"
                       f" --out-npz {_rel_out}")
                result = subprocess.run(
                    ["europa", "submit", cmd],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"europa submit failed:\n{result.stderr or result.stdout}")

                prev_running = False
                for attempt in range(720):   # max ~2 h at 10 s intervals
                    time.sleep(10)
                    st = subprocess.run(
                        ["europa", "status"],
                        cwd=str(_WORKSPACE_ROOT),
                        capture_output=True, text=True, timeout=30,
                    )
                    is_running = _is_europa_running(st.stdout)
                    elapsed = (attempt + 1) * 10
                    msg = f"Status: europa running all blocks… (~{elapsed} s elapsed)"
                    self.root.after(0, lambda m=msg: self._lbl_compute_all.config(
                        text=m, foreground="darkorange"))
                    if prev_running and not is_running:
                        break
                    if attempt == 0 and not is_running:
                        break
                    prev_running = is_running

                self.root.after(0, lambda: self._lbl_compute_all.config(
                    text="Status: pulling results from europa…",
                    foreground="darkorange"))
                pull = subprocess.run(
                    ["europa", "pull"],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=120,
                )
                if pull.returncode != 0:
                    raise RuntimeError(
                        f"europa pull failed:\n{pull.stderr or pull.stdout}")
                if not npz_path.exists():
                    subprocess.run(["europa", "logs", "30"], cwd=str(_WORKSPACE_ROOT))
                    raise RuntimeError(
                        "tuner_all_result.npz not found after pull — "
                        "check europa logs for errors")

                # ── Reconstruct per-block display data from prefixed keys ───
                data = np.load(str(npz_path))
                block_lookup = {(b.s, b.t): b for b in self.blocks}
                labels = [str(x) for x in data["blocks"]]
                display_dict: dict = {}
                for lab in labels:
                    s_str, t_str = lab.split(",")
                    s_i, t_i = int(s_str), int(t_str)
                    block = block_lookup.get((s_i, t_i))
                    if block is None:
                        continue
                    pre = f"s{s_i}_t{t_i}__"
                    display_dict[block] = _PostBlockDisplayData(
                        r_vp=data[pre + "r_vp"],     r_vs_IC=data[pre + "r_vs_IC"],
                        r_vs_M=data[pre + "r_vs_M"], r_rho=data[pre + "r_rho"],
                        mean_vp=data[pre + "mean_vp"],     mean_vs_IC=data[pre + "mean_vs_IC"],
                        mean_vs_M=data[pre + "mean_vs_M"], mean_rho=data[pre + "mean_rho"],
                        probe_r_vp=data[pre + "probe_r_vp"],     probe_r_vs_IC=data[pre + "probe_r_vs_IC"],
                        probe_r_vs_M=data[pre + "probe_r_vs_M"], probe_r_rho=data[pre + "probe_r_rho"],
                        std_vp=data[pre + "std_vp"],     std_vs_IC=data[pre + "std_vs_IC"],
                        std_vs_M=data[pre + "std_vs_M"], std_rho=data[pre + "std_rho"],
                        sigma_1_mean=float(data[pre + "sigma_1_mean"]),
                        sigma_1_std=float(data[pre + "sigma_1_std"]),
                        d_pred_post=data[pre + "d_pred_post"],
                    )

                self._post_viewer = _EuropaDisplayViewer(display_dict)
                self._posterior_noise_multiplier = data_noise_multiplier
                # Europa returns display data without the posterior covariance,
                # so no realisation samples are available for the map viewer.
                self._posterior_sample_curves = []
                self.root.after(0, self._on_compute_all_done)

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_compute_all.config(
                    text=m, foreground="red"))
            finally:
                self._computing_all = False
                self.root.after(0, lambda: self._btn_compute_all.state(["!disabled"]))
                self.root.after(0, lambda: self._btn_compute.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    # ── Bayesian inference (property pushforward) ─────────────────────────────

    def _ensure_targets(self) -> list:
        """Lazily build and cache the default property-target set."""
        if self._targets is None:
            from target_kernel_viz import _make_default_targets
            s_max = max(b.s for b in self.blocks)
            self._targets = _make_default_targets(self.specs, s_max)
        return self._targets

    @staticmethod
    def _target_label(idx: int, target) -> str:
        """Human-readable label for a property target."""
        from property_targets import CapBulkTarget, BoxcarBulkTarget
        loc = f"lat={target.lat_deg:.0f}°, lon={target.lon_deg:.0f}°"
        cap = f"cap={target.cap_radius_deg:.0f}°"
        if isinstance(target, CapBulkTarget):
            return (f"{idx}: {target.param} cap-bump "
                    f"({loc}, r₀={target.r0_km:.0f} km, {cap})")
        if isinstance(target, BoxcarBulkTarget):
            return (f"{idx}: {target.param} cap-box "
                    f"({loc}, r∈[{target.r_low_km:.0f},{target.r_high_km:.0f}] km, {cap})")
        return f"{idx}: CMB cap ({loc}, {cap})"

    def _on_show_targets(self) -> None:
        """Open the interactive target-kernel viewer in a separate window."""
        if self._target_win is not None:
            try:
                self._target_win.lift()
                return
            except tk.TclError:
                self._target_win = None

        self._lbl_infer.config(text="Status: building target kernels…",
                               foreground="darkorange")
        self._btn_targets.state(["disabled"])
        s_max = max(b.s for b in self.blocks)
        targets = self._ensure_targets()

        def _run() -> None:
            try:
                from target_kernel_viz import TargetKernelViewer
                engine = TargetKernelViewer(
                    targets, self.blocks, s_max=s_max,
                    earth_radius_km=self.specs.earth_radius_km,
                    icb_radius_km=self.specs.icb_radius_km,
                    cmb_radius_km=self.specs.cmb_radius_km,
                )
                self.root.after(0, lambda: self._open_target_window(engine))
            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_infer.config(
                    text=m, foreground="red"))
            finally:
                self.root.after(0, lambda: self._btn_targets.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _open_target_window(self, engine) -> None:
        """Embed the target-kernel viewer (TargetKernelViewer engine) in Tk."""
        import matplotlib.pyplot as plt
        from target_kernel_viz import _draw_geographic_axes, _draw_st_axes
        from property_targets import CapBulkTarget, BoxcarBulkTarget

        win = tk.Toplevel(self.root)
        win.title("Target kernels")
        win.geometry("1180x720")
        self._target_win = win

        def _on_close() -> None:
            self._target_win = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        state = {"idx": 0, "mode": "geo", "recon": "smax"}

        ctrl = ttk.Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Label(ctrl, text="Target:").pack(side=tk.LEFT)
        labels = [self._target_label(i, t) for i, t in enumerate(engine.targets)]
        tvar = tk.StringVar(value=labels[0])
        combo = ttk.Combobox(ctrl, textvariable=tvar, values=labels,
                             state="readonly", width=42)
        combo.current(0)
        combo.pack(side=tk.LEFT, padx=4)

        fig = Figure(figsize=(11, 6), tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True,
                                    padx=4, pady=4)

        def _redraw() -> None:
            idx = state["idx"]
            is_bulk = isinstance(engine.targets[idx], (CapBulkTarget, BoxcarBulkTarget))
            fig.clf()
            if is_bulk:
                ax_main = fig.add_subplot(1, 2, 1)
                ax_side = fig.add_subplot(1, 2, 2)
            else:
                ax_main = fig.add_subplot(1, 1, 1)
                ax_side = None
            if state["mode"] == "geo":
                _draw_geographic_axes(ax_main, ax_side, engine, idx,
                                      reconstruction=state["recon"], plt=plt)
            else:
                _draw_st_axes(ax_main, ax_side, engine, idx, plt=plt)
            canvas.draw_idle()

        def _on_select(_evt=None) -> None:
            state["idx"] = combo.current()
            _redraw()

        combo.bind("<<ComboboxSelected>>", _on_select)

        mode_btn = ttk.Button(ctrl, text="Switch to s-t view")

        def _toggle_mode() -> None:
            if state["mode"] == "geo":
                state["mode"] = "st"
                mode_btn.config(text="Switch to geographic view")
                geo_btn.state(["disabled"])
            else:
                state["mode"] = "geo"
                mode_btn.config(text="Switch to s-t view")
                geo_btn.state(["!disabled"])
            _redraw()

        mode_btn.config(command=_toggle_mode)
        mode_btn.pack(side=tk.LEFT, padx=(16, 4))

        geo_btn = ttk.Button(ctrl, text="Geo: full s≤s_max")

        def _toggle_recon() -> None:
            state["recon"] = "blocks" if state["recon"] == "smax" else "smax"
            geo_btn.config(text="Geo: model blocks only"
                           if state["recon"] == "blocks" else "Geo: full s≤s_max")
            _redraw()

        geo_btn.config(command=_toggle_recon)
        geo_btn.pack(side=tk.LEFT, padx=4)

        _redraw()
        self._lbl_infer.config(text="Status: target kernels open",
                               foreground="green")

    # ── Posterior-mean model maps ─────────────────────────────────────────────

    def _on_show_maps(self) -> None:
        """Open the posterior-mean model-map viewer (needs all blocks solved)."""
        if self._map_win is not None:
            try:
                self._map_win.lift()
                return
            except tk.TclError:
                self._map_win = None

        pv = self._post_viewer
        if pv is None:
            self._lbl_maps.config(
                text="Status: ✗ compute all blocks first", foreground="red")
            return

        # Collect cached per-block display data.
        display_dict = {}
        for block in self.blocks:
            if pv.is_computed(block):
                display_dict[block] = pv.get_display_data(block)
        if not display_dict:
            self._lbl_maps.config(
                text="Status: ✗ no computed blocks found", foreground="red")
            return

        s_max = max(b.s for b in display_dict)
        try:
            from model_map_viz import ModelMapViewer

            def _make_engine(dd_dict):
                return ModelMapViewer(
                    dd_dict, s_max=s_max,
                    earth_radius_km=self.specs.earth_radius_km,
                    icb_radius_km=self.specs.icb_radius_km,
                    cmb_radius_km=self.specs.cmb_radius_km,
                )

            # First realisation is always the posterior mean; subsequent
            # entries are cached sample realisations (local compute only).
            realisations = [("Mean", _make_engine(display_dict))]
            for k, sample_dict in enumerate(self._posterior_sample_curves):
                # Restrict each realisation to the blocks that were computed.
                sub = {b: sample_dict[b] for b in display_dict if b in sample_dict}
                if sub:
                    realisations.append((f"Sample {k + 1}", _make_engine(sub)))

            self._open_model_map_window(realisations)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self._lbl_maps.config(text=f"Status: ✗ {exc}", foreground="red")

    def _open_model_map_window(self, realisations) -> None:
        """Embed the ModelMapViewer engine(s) in an interactive Tk window with a
        component selector, depth slider, and (when several posterior samples
        are cached) a realisation selector.

        Parameters
        ----------
        realisations : list[tuple[str, ModelMapViewer]]
            ``(label, engine)`` pairs.  The first entry is the posterior mean;
            any further entries are cached sample realisations.
        """
        import matplotlib.pyplot as plt
        from model_map_viz import _draw_model_map

        win = tk.Toplevel(self.root)
        win.title("Posterior model maps")
        win.geometry("900x760")
        self._map_win = win

        def _on_close() -> None:
            self._map_win = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        labels = [lbl for lbl, _ in realisations]
        label_to_index = {lbl: idx for idx, lbl in enumerate(labels)}
        engine = realisations[0][1]
        R = engine.earth_radius_km
        state = {
            "component": "vp",
            "radius": R,
            "engine": engine,
            "real_index": 0,
            "real_label": labels[0],
        }

        ctrl = ttk.Frame(win)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        ttk.Label(ctrl, text="Component:").pack(side=tk.LEFT)
        comp_var = tk.StringVar(value="vp")
        comp_combo = ttk.Combobox(
            ctrl, textvariable=comp_var,
            values=["vp", "vs", "rho", "CMB"], state="readonly", width=6)
        comp_combo.current(0)
        comp_combo.pack(side=tk.LEFT, padx=4)

        # Realisation selector (only meaningful when samples are cached).
        ttk.Label(ctrl, text="Realisation:").pack(side=tk.LEFT, padx=(16, 2))
        real_var = tk.StringVar(value=realisations[0][0])
        prev_btn = ttk.Button(ctrl, text="<", width=2)
        prev_btn.pack(side=tk.LEFT, padx=(0, 2))
        real_combo = ttk.Combobox(
            ctrl, textvariable=real_var,
            values=labels,
            state="readonly", width=10)
        real_combo.current(0)
        real_combo.pack(side=tk.LEFT, padx=4)
        next_btn = ttk.Button(ctrl, text=">", width=2)
        next_btn.pack(side=tk.LEFT, padx=(2, 4))
        initial_real_status = "mean only" if len(realisations) == 1 else labels[0]
        real_status = ttk.Label(
            ctrl, text=initial_real_status, width=12, foreground="gray")
        real_status.pack(side=tk.LEFT, padx=(0, 2))
        if len(realisations) == 1:
            real_combo.state(["disabled"])
            prev_btn.state(["disabled"])
            next_btn.state(["disabled"])

        ttk.Label(ctrl, text="Radius (km):").pack(side=tk.LEFT, padx=(16, 2))
        radius_var = tk.DoubleVar(value=R)
        radius_scale = ttk.Scale(
            ctrl, from_=0.0, to=R, orient=tk.HORIZONTAL, length=260,
            variable=radius_var)
        radius_scale.pack(side=tk.LEFT, padx=4)
        depth_lbl = ttk.Label(ctrl, text="", width=34)
        depth_lbl.pack(side=tk.LEFT, padx=4)

        fig = Figure(figsize=(8, 6), tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True,
                                    padx=4, pady=4)

        def _update_depth_label() -> None:
            if state["component"] == "CMB":
                depth_lbl.config(text="(CMB surface)")
            else:
                r = state["radius"]
                depth_lbl.config(text=f"r={r:.0f} km  depth={R - r:.0f} km")

        def _redraw() -> None:
            _update_depth_label()
            radius = None if state["component"] == "CMB" else state["radius"]
            try:
                _draw_model_map(fig, state["engine"], state["component"],
                                radius, plt=plt)
                if state["real_label"] != "Mean":
                    for ax in fig.axes:
                        title = ax.get_title()
                        if title:
                            ax.set_title(title.replace(
                                "posterior mean", state["real_label"]))
            except Exception as exc:
                import traceback; traceback.print_exc()
                fig.clf()
                ax = fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center",
                        wrap=True, transform=ax.transAxes)
            canvas.draw_idle()

        def _on_comp(_evt=None) -> None:
            state["component"] = comp_var.get()
            if state["component"] == "CMB":
                radius_scale.state(["disabled"])
            else:
                radius_scale.state(["!disabled"])
            _redraw()

        comp_combo.bind("<<ComboboxSelected>>", _on_comp)

        def _set_realisation(index: int) -> None:
            if len(realisations) == 1:
                real_status.config(text="mean only")
                return
            index %= len(realisations)
            label, selected_engine = realisations[index]
            state["real_index"] = index
            state["real_label"] = label
            state["engine"] = selected_engine
            if real_var.get() != label:
                real_var.set(label)
            real_combo.current(index)
            real_status.config(text=label)
            self._lbl_maps.config(
                text=f"Status: model maps open — {label}",
                foreground="green")
            _redraw()

        def _on_real(_evt=None) -> None:
            label = real_var.get()
            index = label_to_index.get(label, real_combo.current())
            if index < 0:
                index = state["real_index"]
            _set_realisation(index)

        real_combo.bind("<<ComboboxSelected>>", _on_real)
        real_combo.bind("<Return>", _on_real)
        prev_btn.config(command=lambda: _set_realisation(
            state["real_index"] - 1))
        next_btn.config(command=lambda: _set_realisation(
            state["real_index"] + 1))

        def _on_radius(_evt=None) -> None:
            state["radius"] = float(radius_var.get())
            _update_depth_label()

        # Update the label live while dragging, but only redraw on release to
        # avoid expensive SH expansions on every pixel of slider motion.
        radius_scale.configure(command=lambda _v: _on_radius())
        radius_scale.bind("<ButtonRelease-1>", lambda _e: _redraw())

        _redraw()
        self._lbl_maps.config(
            text=f"Status: model maps open — {state['real_label']}",
            foreground="green")

    def _on_compute_inference(self) -> None:
        """Route inference to the local or europa compute backend."""
        if self._inferring:
            return
        if self._compute_mode_var.get() == "europa":
            self._on_compute_inference_europa()
        else:
            self._on_compute_inference_local()

    def _on_compute_inference_local(self) -> None:
        """Solve every block with the current prior, then push the per-block
        posteriors through the target property operators to obtain the property
        posterior (Bayesian inference = pushforward of the inversion)."""
        if self._inferring:
            return
        self._inferring = True
        self._btn_infer.state(["disabled"])
        self._lbl_infer.config(text="Status: starting inference…",
                               foreground="darkorange")

        specs     = self._current_specs()
        hyper     = self._get_hyper()
        taus      = self._get_taus()
        sigma_var = 10 ** self._sv_logvar.get()
        noise_mult = self._current_data_noise_multiplier()
        s_max     = max(b.s for b in self.blocks)
        targets   = self._ensure_targets()

        def _run() -> None:
            try:
                from full_spectrum_utils import assemble_property_posterior

                # 1. Ensure shared Bessel-Sobolev blocks exist (build if needed).
                if self._shared_bessel is None:
                    from prior_posterior_tuner import _build_custom_bessel_blocks
                    self.root.after(0, lambda: self._lbl_infer.config(
                        text="Status: building Bessel-Sobolev operators…",
                        foreground="darkorange"))
                    self._shared_bessel = _build_custom_bessel_blocks(specs, hyper)
                shared_bessel = self._shared_bessel

                # 2. Solve every block via the shared cache. When inference is
                #    pressed after "Compute All Blocks" with an unchanged prior
                #    and noise, every block is a cache hit (zero re-solve). When
                #    pressed directly, blocks are solved here and stored so a
                #    later inversion reuses them. Only the per-block posteriors
                #    are computed — the model-space covariance probing and
                #    sampling done by "Compute All Blocks" are skipped.
                def _progress(done, total, b):
                    self.root.after(0, lambda: self._lbl_infer.config(
                        text=(f"Status: solving block {done + 1}/{total} "
                              f"(s={b.s}, t={b.t})…"),
                        foreground="darkorange"))

                posterior_dict = self._ensure_block_posteriors(
                    list(self.blocks), shared_bessel=shared_bessel, taus=taus,
                    sigma_var=sigma_var, noise_mult=noise_mult, specs=specs,
                    progress=_progress)

                # 3. Build per-block property operators (cached on target-set
                #    identity) and assemble the property posterior via
                #    pushforward of the per-block Gaussian measures.
                self.root.after(0, lambda: self._lbl_infer.config(
                    text="Status: building property operators…",
                    foreground="darkorange"))
                prop_ops = self._ensure_property_operators(targets, specs, s_max)

                self.root.after(0, lambda: self._lbl_infer.config(
                    text="Status: assembling property posterior (pushforward)…",
                    foreground="darkorange"))
                prop_post = assemble_property_posterior(prop_ops, posterior_dict)
                self._property_posterior = prop_post

                # 4. Extract mean, std, correlation.
                N_p = len(targets)
                mu_P = np.asarray(prop_post.expectation, dtype=float)
                C_P = np.column_stack([
                    np.asarray(prop_post.covariance(np.eye(N_p)[:, j]), dtype=float)
                    for j in range(N_p)
                ])
                C_P = 0.5 * (C_P + C_P.T)
                sigma_P = np.sqrt(np.clip(np.diag(C_P), 0.0, None))
                denom = np.outer(sigma_P, sigma_P)
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr_P = np.where(denom > 0.0, C_P / denom, 0.0)
                names = [self._target_label(i, t) for i, t in enumerate(targets)]

                self.root.after(0, lambda: self._open_inference_window(
                    mu_P, sigma_P, corr_P, names))

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_infer.config(
                    text=m, foreground="red"))
            finally:
                self._inferring = False
                self.root.after(0, lambda: self._btn_infer.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _open_inference_window(self, mu_P, sigma_P, corr_P, names) -> None:
        """Display the property posterior summary (bar chart + correlation)."""
        from full_spectrum_viz import plot_property_posterior_summary

        if self._inference_win is not None:
            try:
                self._inference_win.destroy()
            except tk.TclError:
                pass
            self._inference_win = None

        win = tk.Toplevel(self.root)
        win.title("Bayesian inference — property posterior")
        win.geometry("1120x640")
        self._inference_win = win

        def _on_close() -> None:
            self._inference_win = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        fig = plot_property_posterior_summary(mu_P, sigma_P, corr_P, names)
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True,
                                    padx=4, pady=4)
        canvas.draw()
        self._lbl_infer.config(text="Status: ✓ inference complete",
                               foreground="green")


    def _on_compute_inference_europa(self) -> None:
        """Submit the all-block inference (property pushforward) as a europa job
        and poll for the property-posterior result."""
        if self._inferring:
            return
        self._inferring = True
        self._btn_infer.state(["disabled"])
        self._lbl_infer.config(text="Status: preparing europa job…",
                               foreground="darkorange")

        taus      = self._get_taus()
        sigma_var = 10 ** self._sv_logvar.get()
        data_noise_multiplier = self._current_data_noise_multiplier()
        hyper     = self._get_hyper()
        n_jobs    = int(self._n_jobs_var.get())
        n_basis   = self.specs.n_basis
        s_max     = max(b.s for b in self.blocks)

        def _run() -> None:
            try:
                # ── 1. Write params JSON (task="inference") ─────────────────
                params = {
                    "task":      "inference",
                    "n_basis":   n_basis,
                    "s_max":     s_max,
                    "hyper":     hyper,
                    "taus":      taus,
                    "sigma_var": sigma_var,
                    "data_noise_multiplier": data_noise_multiplier,
                    "n_jobs":    n_jobs,
                    "n_grid":    self.n_grid,
                    "n_probes":  self.n_probes,
                    "data_mode": "real",
                }
                _demo = (_WORKSPACE_ROOT
                         / "intervalinf/demos/old_demos/paper_demos")
                params_path = _demo / "tuner_inference_params.json"
                npz_path    = _demo / "tuner_inference_result.npz"

                if npz_path.exists():
                    npz_path.unlink()

                with open(params_path, "w") as fh:
                    json.dump(params, fh, indent=2)

                self.root.after(0, lambda: self._lbl_infer.config(
                    text="Status: pushing to europa (europa submit)…",
                    foreground="darkorange"))

                # ── 2. Submit ────────────────────────────────────────────────
                _rel_script = ("intervalinf/demos/old_demos/paper_demos/"
                               "prior_posterior_tuner_remote.py")
                _rel_params = ("intervalinf/demos/old_demos/paper_demos/"
                               "tuner_inference_params.json")
                _rel_out    = ("intervalinf/demos/old_demos/paper_demos/"
                               "tuner_inference_result.npz")
                cmd = (f"python {_rel_script}"
                       f" --params-json {_rel_params}"
                       f" --out-npz {_rel_out}")
                result = subprocess.run(
                    ["europa", "submit", cmd],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"europa submit failed:\n"
                        f"{result.stderr or result.stdout}")

                # ── 3. Poll for completion ───────────────────────────────────
                prev_running = False
                for attempt in range(720):   # max ~2 h at 10 s intervals
                    time.sleep(10)
                    st = subprocess.run(
                        ["europa", "status"],
                        cwd=str(_WORKSPACE_ROOT),
                        capture_output=True, text=True, timeout=30,
                    )
                    is_running = _is_europa_running(st.stdout)
                    elapsed = (attempt + 1) * 10
                    msg = f"Status: europa running… (~{elapsed} s elapsed)"
                    self.root.after(0, lambda m=msg: self._lbl_infer.config(
                        text=m, foreground="darkorange"))
                    if prev_running and not is_running:
                        break
                    if attempt == 0 and not is_running:
                        break
                    prev_running = is_running

                # ── 4. Pull results ──────────────────────────────────────────
                self.root.after(0, lambda: self._lbl_infer.config(
                    text="Status: pulling results from europa…",
                    foreground="darkorange"))
                pull = subprocess.run(
                    ["europa", "pull"],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=120,
                )
                if pull.returncode != 0:
                    raise RuntimeError(
                        f"europa pull failed:\n{pull.stderr or pull.stdout}")

                if not npz_path.exists():
                    subprocess.run(["europa", "logs", "30"],
                                   cwd=str(_WORKSPACE_ROOT))
                    raise RuntimeError(
                        "tuner_inference_result.npz not found after pull — "
                        "check europa logs for errors")

                # ── 5. Load NPZ and display ──────────────────────────────────
                data = np.load(str(npz_path), allow_pickle=False)
                mu_P    = np.asarray(data["mu_P"], dtype=float)
                sigma_P = np.asarray(data["sigma_P"], dtype=float)
                corr_P  = np.asarray(data["corr_P"], dtype=float)
                names   = [str(n) for n in data["names"]]
                self._property_posterior = None  # remote result; no measure obj

                self.root.after(0, lambda: self._open_inference_window(
                    mu_P, sigma_P, corr_P, names))

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_infer.config(
                    text=m, foreground="red"))
            finally:
                self._inferring = False
                self.root.after(0, lambda: self._btn_infer.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _on_compute_local(self) -> None:
        if self._computing:
            return
        self._computing = True
        self._btn_compute.state(["disabled"])
        self._lbl_compute.config(text="Status: starting…", foreground="darkorange")
        shared_bessel = self._shared_bessel
        taus      = self._get_taus()
        sigma_var = 10 ** self._sv_logvar.get()
        data_noise_multiplier = self._current_data_noise_multiplier()
        block     = self._current_block()
        specs     = self._current_specs()

        def _run() -> None:
            try:
                from full_spectrum_utils import build_block_forward, build_block_prior, solve_block
                from posterior_viz import PosteriorViewer

                if block not in self._forward_dict:
                    self.root.after(0, lambda: self._lbl_compute.config(
                        text="Status: building forward operator…",
                        foreground="darkorange"))
                    self._forward_dict[block] = build_block_forward(
                        block.s, block.t, self.split[block], self.catalog, specs)

                G_st, C_D_st, _M, D_st = self._forward_dict[block]
                C_D_assumed = _build_scaled_data_noise_measure(
                    D_st,
                    self.split[block].error_vector,
                    data_noise_multiplier,
                )

                def _tau_fn(p: str, _s: int) -> float:
                    return taus.get(p, 1.0)

                self.root.after(0, lambda: self._lbl_compute.config(
                    text="Status: assembling prior…", foreground="darkorange"))
                prior_st = build_block_prior(
                    block.s, block.t, shared_bessel, specs,
                    tau_fn=_tau_fn, sigma_var=sigma_var)

                self.root.after(0, lambda: self._lbl_compute.config(
                    text="Status: solving Bayesian system…", foreground="darkorange"))
                d_real = self.split[block].data_vector
                if self._data_mode_var.get() == "synthetic":
                    self.root.after(0, lambda: self._lbl_compute.config(
                        text="Status: generating synthetic data…",
                        foreground="darkorange"))
                    synth = self._ensure_synth_state(
                        block, G_st, C_D_st, prior_st, d_real, specs)
                    d_st = synth.d_synth
                else:
                    d_st = d_real
                posterior_st = solve_block(
                    block.s, block.t, G_st, C_D_assumed, prior_st, d_st,
                )

                self.root.after(0, lambda: self._lbl_compute.config(
                    text="Status: probing posterior covariance…", foreground="darkorange"))
                pv = PosteriorViewer(
                    {block: posterior_st},
                    self._forward_dict,
                    specs,
                    n_grid=self.n_grid,
                    n_probes=self.n_probes,
                )
                pv.compute_block(block)
                self._post_viewer = pv
                self._posterior_noise_multiplier = data_noise_multiplier

                self.root.after(0, self._on_compute_done)

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_compute.config(
                    text=m, foreground="red"))
            finally:
                self._computing = False
                self.root.after(0, lambda: self._btn_compute.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _on_compute_europa(self) -> None:
        """Submit the posterior computation as a europa job and poll for results."""
        if self._computing:
            return
        self._computing = True
        self._btn_compute.state(["disabled"])
        self._lbl_compute.config(text="Status: preparing europa job…", foreground="darkorange")

        block     = self._current_block()
        taus      = self._get_taus()
        sigma_var = 10 ** self._sv_logvar.get()
        data_noise_multiplier = self._current_data_noise_multiplier()
        hyper     = self._get_hyper()
        n_jobs    = int(self._n_jobs_var.get())
        n_basis   = self.specs.n_basis
        s_max     = max(b.s for b in self.blocks)

        def _run() -> None:
            try:
                from posterior_viz import _PostBlockDisplayData

                # ── 1. Write params JSON ────────────────────────────────────
                params = {
                    "n_basis":   n_basis,
                    "s_max":     s_max,
                    "block":     {"s": block.s, "t": block.t},
                    "hyper":     hyper,
                    "taus":      taus,
                    "sigma_var": sigma_var,
                    "data_noise_multiplier": data_noise_multiplier,
                    "n_jobs":    n_jobs,
                    "n_grid":    self.n_grid,
                    "n_probes":  self.n_probes,
                    "data_mode": self._data_mode_var.get(),
                    "synth": {
                        "noise_scale": float(self._synth_config.noise_scale),
                        "null_amp":    float(self._synth_config.null_amp),
                        "seed":        int(self._synth_config.seed),
                    },
                }
                _demo = (_WORKSPACE_ROOT
                         / "intervalinf/demos/old_demos/paper_demos")
                params_path = _demo / "tuner_params.json"
                npz_path    = _demo / "tuner_result.npz"

                if npz_path.exists():
                    npz_path.unlink()

                with open(params_path, "w") as fh:
                    json.dump(params, fh, indent=2)

                self.root.after(0, lambda: self._lbl_compute.config(
                    text="Status: pushing to europa (europa submit)…",
                    foreground="darkorange"))

                # ── 2. Submit (europa submit does push + background run) ────
                _rel_script  = "intervalinf/demos/old_demos/paper_demos/prior_posterior_tuner_remote.py"
                _rel_params  = "intervalinf/demos/old_demos/paper_demos/tuner_params.json"
                _rel_out     = "intervalinf/demos/old_demos/paper_demos/tuner_result.npz"
                cmd = (f"python {_rel_script}"
                       f" --params-json {_rel_params}"
                       f" --out-npz {_rel_out}")
                result = subprocess.run(
                    ["europa", "submit", cmd],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"europa submit failed:\n{result.stderr or result.stdout}")

                # ── 3. Poll for completion ──────────────────────────────────
                prev_running = False
                for attempt in range(720):   # max ~2 h at 10 s intervals
                    time.sleep(10)
                    st = subprocess.run(
                        ["europa", "status"],
                        cwd=str(_WORKSPACE_ROOT),
                        capture_output=True, text=True, timeout=30,
                    )
                    is_running = _is_europa_running(st.stdout)
                    elapsed = (attempt + 1) * 10
                    msg = f"Status: europa running… (~{elapsed} s elapsed)"
                    self.root.after(0, lambda m=msg: self._lbl_compute.config(
                        text=m, foreground="darkorange"))
                    if prev_running and not is_running:
                        break   # job just stopped
                    if attempt == 0 and not is_running:
                        # Job was either ultra-fast or failed to start;
                        # try pull anyway and let the NPZ check catch it.
                        break
                    prev_running = is_running

                # ── 4. Pull results ─────────────────────────────────────────
                self.root.after(0, lambda: self._lbl_compute.config(
                    text="Status: pulling results from europa…",
                    foreground="darkorange"))
                pull = subprocess.run(
                    ["europa", "pull"],
                    cwd=str(_WORKSPACE_ROOT),
                    capture_output=True, text=True, timeout=120,
                )
                if pull.returncode != 0:
                    raise RuntimeError(
                        f"europa pull failed:\n{pull.stderr or pull.stdout}")

                if not npz_path.exists():
                    # Pull logs to help diagnose
                    subprocess.run(["europa", "logs", "30"], cwd=str(_WORKSPACE_ROOT))
                    raise RuntimeError(
                        "tuner_result.npz not found after pull — "
                        "check europa logs for errors")

                # ── 5. Load NPZ and reconstruct display data ────────────────
                data = np.load(str(npz_path))
                display_data = _PostBlockDisplayData(
                    r_vp=data["r_vp"],    r_vs_IC=data["r_vs_IC"],
                    r_vs_M=data["r_vs_M"], r_rho=data["r_rho"],
                    mean_vp=data["mean_vp"],    mean_vs_IC=data["mean_vs_IC"],
                    mean_vs_M=data["mean_vs_M"], mean_rho=data["mean_rho"],
                    probe_r_vp=data["probe_r_vp"],    probe_r_vs_IC=data["probe_r_vs_IC"],
                    probe_r_vs_M=data["probe_r_vs_M"], probe_r_rho=data["probe_r_rho"],
                    std_vp=data["std_vp"],    std_vs_IC=data["std_vs_IC"],
                    std_vs_M=data["std_vs_M"], std_rho=data["std_rho"],
                    sigma_1_mean=float(data["sigma_1_mean"]),
                    sigma_1_std=float(data["sigma_1_std"]),
                    d_pred_post=(
                        data["d_pred_post"]
                        if "d_pred_post" in data.files
                        else np.zeros(0)
                    ),
                )
                self._post_viewer = _EuropaDisplayViewer({block: display_data})
                self._posterior_noise_multiplier = data_noise_multiplier

                # If remote produced true-model arrays, cache a SynthState so
                # the overlay works on next redraw.
                if "true_vp" in data.files:
                    from synthetic_data import SynthConfig, SynthState
                    synth_state = SynthState(
                        d_synth=np.zeros(0),  # not needed locally for overlay
                        d_clean=np.zeros(0),  # likewise
                        config=SynthConfig(
                            noise_scale=float(data["synth_noise_scale"]),
                            null_amp=float(data["synth_null_amp"]),
                            seed=int(data["synth_seed"]),
                        ),
                        r_vp=data["r_vp"],       r_vs_IC=data["r_vs_IC"],
                        r_vs_M=data["r_vs_M"],   r_rho=data["r_rho"],
                        true_vp=data["true_vp"], true_vs_IC=data["true_vs_IC"],
                        true_vs_M=data["true_vs_M"], true_rho=data["true_rho"],
                        sigma_1_true=float(data["sigma_1_true"]),
                    )
                    self._synth_states[block] = synth_state

                self.root.after(0, self._on_compute_done)

            except Exception as exc:
                import traceback; traceback.print_exc()
                msg = f"Status: ✗ {exc}"
                self.root.after(0, lambda m=msg: self._lbl_compute.config(
                    text=m, foreground="red"))
            finally:
                self._computing = False
                self.root.after(0, lambda: self._btn_compute.state(["!disabled"]))

        threading.Thread(target=_run, daemon=True).start()

    def _on_compute_done(self) -> None:
        self._lbl_compute.config(text="Status: ✓ posterior computed", foreground="green")
        self._redraw_posterior()

    def _on_resample(self) -> None:
        v = self._viewer
        if v is not None and v.is_precomputed:
            v.resample()
            self._redraw_prior()

    def _on_block_change(self, _event=None) -> None:
        block = self._current_block()
        pv = self._post_viewer

        # If the posterior for this block is already cached (e.g. after
        # "Compute All Blocks"), redraw it directly instead of forcing a
        # recompute.
        cached = False
        if pv is not None:
            if pv.is_computed(block):
                cached = True
            elif hasattr(pv, "load_cached_block") and pv.load_cached_block(block):
                cached = True

        if cached and self._data_mode_var.get() != "synthetic":
            self._lbl_compute.config(
                text=f"Status: ✓ showing cached posterior (s={block.s}, t={block.t})",
                foreground="green")
            self._redraw_posterior()
            self._redraw_prior()
            if self._viewer is not None and self._viewer.is_precomputed:
                self._btn_compute.state(["!disabled"])
            return

        # Otherwise invalidate posterior display for the new block.
        for ax in self._axes_post:
            ax.cla()
        self._ax_datafit.cla()
        self._fig_post.suptitle("Posterior (block changed — recompute needed)",
                                fontsize=9, color="gray")
        self._canvas_post.draw_idle()
        self._lbl_compute.config(
            text="Status: block changed — recompute needed", foreground="darkorange")
        if self._viewer is not None and self._viewer.is_precomputed:
            self._btn_compute.state(["!disabled"])
        self._redraw_prior()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prior–Posterior Tuner — interactive Tkinter GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/normal-mode-data"),
        help="Directory containing normal-mode observation data.",
    )
    parser.add_argument(
        "--kernel-dir", type=Path,
        default=Path("data/normal-mode-kernels/kernels-all_PREM-layers_Adrian"),
        help="Directory containing sensitivity kernels.",
    )
    parser.add_argument("--n-basis",   type=int, default=50,
                        help="Number of basis functions (radial resolution).")
    parser.add_argument("--s-max",     type=int, default=6,
                        help="Maximum angular degree for block enumeration.")
    parser.add_argument("--n-grid",    type=int, default=300,
                        help="Radial evaluation grid points for display.")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of prior sample curves to draw.")
    parser.add_argument("--n-probes",  type=int, default=15,
                        help="Number of Gaussian bump probes for posterior covariance.")
    args = parser.parse_args()

    # Local imports (after sys.path was set at module top)
    from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog
    from full_spectrum_utils import RadialSpecs, enumerate_blocks, block_data_split
    from intervalinf import ParallelConfig

    data_dir   = args.data_dir.resolve()
    kernel_dir = args.kernel_dir.resolve()

    print(f"Loading catalog  : {kernel_dir}")
    catalog = NormalModeKernelCatalog(str(kernel_dir))
    print(f"Loading registry : {data_dir}")
    reg = NormalModeDataRegistry(str(data_dir), mode_filter=catalog.list_modes())

    parallel_cfg = ParallelConfig(enabled=False)
    specs  = RadialSpecs(n_basis=args.n_basis, parallel_cfg=parallel_cfg)
    blocks = enumerate_blocks(reg, s_max=args.s_max)
    split  = block_data_split(reg, blocks)

    print(f"Ready: {len(blocks)} blocks  (n_basis={args.n_basis}, s_max={args.s_max})")

    root = tk.Tk()
    TunerApp(root, catalog, reg, specs, blocks, split,
             n_grid=args.n_grid,
             n_samples=args.n_samples,
             n_probes=args.n_probes)
    root.mainloop()


if __name__ == "__main__":
    main()
