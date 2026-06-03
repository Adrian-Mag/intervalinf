#!/usr/bin/env python3
"""
prior_posterior_tuner_remote.py
================================

Remote-side compute script executed on europa by the tuner GUI.

Reads a JSON params file written by the tuner app, runs the full Bayesian
posterior pipeline for a single spectral block, and saves the posterior
display data as an NPZ file that the GUI loads after ``europa pull``.

Usage (invoked automatically via ``europa submit`` — not called by hand)
-----------------------------------------------------------------------
    python prior_posterior_tuner_remote.py \\
        --params-json <path>/tuner_params.json \\
        --out-npz     <path>/tuner_result.npz

Params JSON schema
------------------
{
    "n_basis":   <int>,
    "s_max":     <int>,
    "block":     {"s": <int>, "t": <int>},
    "hyper":     {
        "vp":    {"s_order": f, "length": f, "var": f, "bc": str},
        "vs_IC": {...},
        "vs_M":  {...},
        "rho":   {...}
    },
    "taus":      {"vp": f, "vs_IC": f, "vs_M": f, "rho": f},
    "sigma_var": <float>,
    "data_noise_multiplier": <float>,
    "n_jobs":    <int>,
    "n_grid":    <int>,
    "n_probes":  <int>
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add local helper folders to sys.path.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "utils"))
sys.path.insert(0, str(_HERE / "visualization"))


def _build_scaled_data_noise_measure(data_space, error_vector, noise_std_multiplier):
    """Return a diagonal data-noise measure with scaled standard deviations."""
    import numpy as np
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


def _target_label(idx: int, target) -> str:
    """Human-readable label for a property target (mirrors the GUI)."""
    from property_targets import BulkTarget
    if isinstance(target, BulkTarget):
        return (f"{idx}: {target.param} bulk "
                f"(lat={target.lat_deg:.0f}°, lon={target.lon_deg:.0f}°, "
                f"r₀={target.r0_km:.0f} km)")
    return (f"{idx}: CMB topo "
            f"(lat={target.lat_deg:.0f}°, lon={target.lon_deg:.0f}°)")


def _run_inference(
    all_blocks, split, catalog, specs, s_max, hyper, taus,
    sigma_var, data_noise_multiplier, out_path,
) -> None:
    """Solve every block, then push the per-block posteriors through the target
    property operators to obtain the property posterior (pushforward).

    Saves an NPZ with ``mu_P``, ``sigma_P``, ``corr_P`` and ``names``.
    """
    import numpy as np

    from full_spectrum_utils import (
        build_block_forward, build_block_prior, solve_block,
        build_property_operator, assemble_property_posterior,
    )
    from prior_posterior_tuner import _build_custom_bessel_blocks
    from target_kernel_viz import _make_default_targets

    print("[remote] inference task: building Bessel-Sobolev operators…",
          flush=True)
    shared_bessel = _build_custom_bessel_blocks(specs, hyper)

    def _tau_fn(p: str, _s: int) -> float:
        return taus.get(p, 1.0)

    targets = _make_default_targets(specs, s_max)

    forward_dict: dict = {}
    posterior_dict: dict = {}
    n_blocks = len(all_blocks)
    for k, block in enumerate(all_blocks):
        print(f"[remote] solving block {k + 1}/{n_blocks} "
              f"(s={block.s}, t={block.t})…", flush=True)
        forward_dict[block] = build_block_forward(
            block.s, block.t, split[block], catalog, specs)
        G_st, C_D_st, _M, D_st = forward_dict[block]
        C_D_assumed = _build_scaled_data_noise_measure(
            D_st, split[block].error_vector, data_noise_multiplier)
        prior_st = build_block_prior(
            block.s, block.t, shared_bessel, specs,
            tau_fn=_tau_fn, sigma_var=sigma_var)
        d_st = split[block].data_vector
        posterior_dict[block] = solve_block(
            block.s, block.t, G_st, C_D_assumed, prior_st, d_st)

    print("[remote] building property operators…", flush=True)
    prop_ops = build_property_operator(
        targets, all_blocks, forward_dict, specs, s_max)

    print("[remote] assembling property posterior (pushforward)…", flush=True)
    prop_post = assemble_property_posterior(prop_ops, posterior_dict)

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
    names = [_target_label(i, t) for i, t in enumerate(targets)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        mu_P=mu_P, sigma_P=sigma_P, corr_P=corr_P, names=np.array(names),
    )
    print(f"[remote] inference result saved → {out_path}", flush=True)


def _block_key(block) -> str:
    """Per-block NPZ key prefix, e.g. ``s2_t1__``."""
    return f"s{block.s}_t{block.t}__"


def _run_all_blocks(
    all_blocks, split, catalog, specs, hyper, taus,
    sigma_var, data_noise_multiplier, n_grid, n_probes, out_path,
) -> None:
    """Solve every (s,t) block and compute per-block posterior display data.

    Saves a single NPZ holding all blocks' display arrays under per-block
    prefixed keys (see :func:`_block_key`) plus a ``blocks`` index array of
    ``"s,t"`` strings so the GUI can reconstruct each block's display data.
    """
    import numpy as np

    from full_spectrum_utils import (
        build_block_forward, build_block_prior, solve_block,
    )
    from prior_posterior_tuner import _build_custom_bessel_blocks
    from posterior_viz import _compute_posterior_display_data

    print("[remote] all-blocks task: building Bessel-Sobolev operators…",
          flush=True)
    shared_bessel = _build_custom_bessel_blocks(specs, hyper)

    def _tau_fn(p: str, _s: int) -> float:
        return taus.get(p, 1.0)

    forward_dict: dict = {}
    save_kwargs: dict = {}
    block_labels: list = []
    n_blocks = len(all_blocks)
    for k, block in enumerate(all_blocks):
        print(f"[remote] solving block {k + 1}/{n_blocks} "
              f"(s={block.s}, t={block.t})…", flush=True)
        forward_dict[block] = build_block_forward(
            block.s, block.t, split[block], catalog, specs)
        G_st, _C_D_st, _M, D_st = forward_dict[block]
        C_D_assumed = _build_scaled_data_noise_measure(
            D_st, split[block].error_vector, data_noise_multiplier)
        prior_st = build_block_prior(
            block.s, block.t, shared_bessel, specs,
            tau_fn=_tau_fn, sigma_var=sigma_var)
        d_st = split[block].data_vector
        posterior_st = solve_block(
            block.s, block.t, G_st, C_D_assumed, prior_st, d_st)

        print(f"[remote] probing covariance for block {k + 1}/{n_blocks} "
              f"(n_grid={n_grid}, n_probes={n_probes})…", flush=True)
        dd = _compute_posterior_display_data(
            block, posterior_st, forward_dict, specs,
            n_grid=n_grid, n_probes=n_probes,
        )
        pre = _block_key(block)
        save_kwargs.update({
            pre + "r_vp": dd.r_vp,           pre + "r_vs_IC": dd.r_vs_IC,
            pre + "r_vs_M": dd.r_vs_M,       pre + "r_rho": dd.r_rho,
            pre + "mean_vp": dd.mean_vp,     pre + "mean_vs_IC": dd.mean_vs_IC,
            pre + "mean_vs_M": dd.mean_vs_M, pre + "mean_rho": dd.mean_rho,
            pre + "probe_r_vp": dd.probe_r_vp,
            pre + "probe_r_vs_IC": dd.probe_r_vs_IC,
            pre + "probe_r_vs_M": dd.probe_r_vs_M,
            pre + "probe_r_rho": dd.probe_r_rho,
            pre + "std_vp": dd.std_vp,       pre + "std_vs_IC": dd.std_vs_IC,
            pre + "std_vs_M": dd.std_vs_M,   pre + "std_rho": dd.std_rho,
            pre + "sigma_1_mean": np.array(dd.sigma_1_mean),
            pre + "sigma_1_std": np.array(dd.sigma_1_std),
            pre + "d_pred_post": dd.d_pred_post,
        })
        block_labels.append(f"{block.s},{block.t}")

    save_kwargs["blocks"] = np.array(block_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **save_kwargs)
    print(f"[remote] all-blocks display data saved ({n_blocks} blocks) "
          f"→ {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote posterior compute script for the tuner GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params-json", required=True, type=Path,
                        help="Path to the JSON params file written by the tuner app.")
    parser.add_argument("--out-npz", required=True, type=Path,
                        help="Path where the output NPZ display data will be saved.")
    args = parser.parse_args()

    params_path: Path = args.params_json.resolve()
    out_path:    Path = args.out_npz.resolve()

    # ── Load params ─────────────────────────────────────────────────────────
    with open(params_path) as fh:
        params = json.load(fh)

    n_basis   = int(params["n_basis"])
    s_max     = int(params["s_max"])
    block_st  = params.get("block")    # {"s": int, "t": int} (None for inference)
    hyper     = params["hyper"]        # per-param dict of hyperparameters
    taus      = params["taus"]         # per-param tau scaling
    sigma_var = float(params["sigma_var"])
    data_noise_multiplier = float(params.get("data_noise_multiplier", 1.0))
    n_jobs    = int(params.get("n_jobs", 1))
    n_grid    = int(params.get("n_grid", 300))
    n_probes  = int(params.get("n_probes", 15))
    data_mode = str(params.get("data_mode", "real"))
    synth_in  = params.get("synth", {}) or {}
    synth_noise_scale = float(synth_in.get("noise_scale", 1.0))
    synth_null_amp    = float(synth_in.get("null_amp",    0.0))
    synth_seed        = int(synth_in.get("seed",          0))

    _blk_desc = (f"s={block_st['s']}, t={block_st['t']}" if block_st
                 else "(all-block inference)")
    print(f"[remote] params loaded: block {_blk_desc}, "
          f"n_basis={n_basis}, n_jobs={n_jobs}, data_mode={data_mode}", flush=True)

    # ── Imports (after sys.path set) ─────────────────────────────────────────
    import numpy as np

    from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog
    from full_spectrum_utils import (
        RadialSpecs, enumerate_blocks, block_data_split,
        build_block_forward, build_block_prior, solve_block,
    )
    from prior_posterior_tuner import _build_custom_bessel_blocks
    from posterior_viz import _compute_posterior_display_data
    from intervalinf import ParallelConfig

    # ── Load data ────────────────────────────────────────────────────────────
    data_dir   = _HERE / "data" / "normal-mode-data"
    kernel_dir = (_HERE / "data" / "normal-mode-kernels"
                  / "kernels-all_PREM-layers_Adrian")

    print(f"[remote] loading catalog  : {kernel_dir}", flush=True)
    catalog = NormalModeKernelCatalog(str(kernel_dir))
    print(f"[remote] loading registry : {data_dir}", flush=True)
    reg = NormalModeDataRegistry(str(data_dir), mode_filter=catalog.list_modes())

    # ── Build specs ──────────────────────────────────────────────────────────
    parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
    specs = RadialSpecs(n_basis=n_basis, parallel_cfg=parallel_cfg)

    all_blocks = enumerate_blocks(reg, s_max=s_max)
    split      = block_data_split(reg, all_blocks)

    # ── Inference task (all-block pushforward) ──────────────────────────────
    task = str(params.get("task", "posterior"))
    if task == "inference":
        _run_inference(
            all_blocks, split, catalog, specs, s_max, hyper, taus,
            sigma_var, data_noise_multiplier, out_path,
        )
        return

    if task == "all_blocks":
        _run_all_blocks(
            all_blocks, split, catalog, specs, hyper, taus,
            sigma_var, data_noise_multiplier, n_grid, n_probes, out_path,
        )
        return

    # Find target block
    target_block = None
    for b in all_blocks:
        if b.s == block_st["s"] and b.t == block_st["t"]:
            target_block = b
            break
    if target_block is None:
        raise ValueError(f"Block s={block_st['s']}, t={block_st['t']} not found "
                         f"among {len(all_blocks)} enumerated blocks.")

    print(f"[remote] target block: {target_block}", flush=True)

    # ── Build operators ──────────────────────────────────────────────────────
    print("[remote] building custom Bessel-Sobolev operators…", flush=True)
    shared_bessel = _build_custom_bessel_blocks(specs, hyper)

    print("[remote] building forward operator…", flush=True)
    forward_dict: dict = {}
    forward_dict[target_block] = build_block_forward(
        target_block.s, target_block.t, split[target_block], catalog, specs)

    G_st, C_D_st, _M, D_st = forward_dict[target_block]
    C_D_assumed = _build_scaled_data_noise_measure(
        D_st,
        split[target_block].error_vector,
        data_noise_multiplier,
    )

    # ── Build prior ──────────────────────────────────────────────────────────
    def _tau_fn(p: str, _s: int) -> float:
        return taus.get(p, 1.0)

    print("[remote] assembling prior…", flush=True)
    prior_st = build_block_prior(
        target_block.s, target_block.t, shared_bessel, specs,
        tau_fn=_tau_fn, sigma_var=sigma_var)

    # ── Solve ────────────────────────────────────────────────────────────────
    d_real = split[target_block].data_vector
    synth_state = None
    if data_mode == "synthetic":
        from synthetic_data import SynthConfig, generate_synth_state
        print(f"[remote] generating synthetic data "
              f"(noise_scale={synth_noise_scale}, null_amp={synth_null_amp}, "
              f"seed={synth_seed})…", flush=True)
        synth_state = generate_synth_state(
            target_block, G_st, C_D_st, prior_st, d_real, specs,
            config=SynthConfig(
                noise_scale=synth_noise_scale,
                null_amp=synth_null_amp,
                seed=synth_seed,
            ),
            n_grid=n_grid,
        )
        d_st = synth_state.d_synth
    else:
        d_st = d_real

    print("[remote] solving Bayesian system…", flush=True)
    posterior_st = solve_block(
        target_block.s, target_block.t, G_st, C_D_assumed, prior_st, d_st)

    # ── Compute display data ─────────────────────────────────────────────────
    print(f"[remote] probing posterior covariance (n_grid={n_grid}, "
          f"n_probes={n_probes})…", flush=True)
    dd = _compute_posterior_display_data(
        target_block, posterior_st, forward_dict, specs,
        n_grid=n_grid, n_probes=n_probes,
    )

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        r_vp=dd.r_vp,         r_vs_IC=dd.r_vs_IC,
        r_vs_M=dd.r_vs_M,     r_rho=dd.r_rho,
        mean_vp=dd.mean_vp,   mean_vs_IC=dd.mean_vs_IC,
        mean_vs_M=dd.mean_vs_M, mean_rho=dd.mean_rho,
        probe_r_vp=dd.probe_r_vp,     probe_r_vs_IC=dd.probe_r_vs_IC,
        probe_r_vs_M=dd.probe_r_vs_M, probe_r_rho=dd.probe_r_rho,
        std_vp=dd.std_vp,     std_vs_IC=dd.std_vs_IC,
        std_vs_M=dd.std_vs_M, std_rho=dd.std_rho,
        sigma_1_mean=np.array(dd.sigma_1_mean),
        sigma_1_std=np.array(dd.sigma_1_std),
        d_pred_post=dd.d_pred_post,
    )
    if synth_state is not None:
        save_kwargs.update(
            true_vp=synth_state.true_vp,
            true_vs_IC=synth_state.true_vs_IC,
            true_vs_M=synth_state.true_vs_M,
            true_rho=synth_state.true_rho,
            sigma_1_true=np.array(synth_state.sigma_1_true),
            synth_noise_scale=np.array(synth_noise_scale),
            synth_null_amp=np.array(synth_null_amp),
            synth_seed=np.array(synth_seed),
        )
    np.savez(str(out_path), **save_kwargs)
    print(f"[remote] display data saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
