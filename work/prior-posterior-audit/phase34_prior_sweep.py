#!/usr/bin/env python3
"""
phase34_prior_sweep.py  — Phases 3 & 4 combined
================================================
Phase 3 (sweep design) + Phase 4 (execution) for
MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT.

Three focused sweeps (block s=0, t=0, n_basis=50):

  SWEEP A — Variance scale (keep shape / s_order / length fixed)
     alpha in log-space from 0.001 to 1000
     var_eff = alpha x var_baseline  for every parameter

  SWEEP B — Smoothness order (var & length at baseline)
     s_order in [1, 2, 3, 4, 5, 6, 7, 8] for all parameters jointly

  SWEEP C — Length scale factor (s_order & var at baseline)
     length = scale x length_baseline for all parameters

Saves:
  processed/phase34_prior_sweep.json
  figures/phase34_var_sweep.png, phase34_smooth_sweep.png, phase34_length_sweep.png
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEMOS  = Path("/home/adrian/PhD/Inferences/intervalinf/demos/old_demos/paper_demos")
_UTILS  = _DEMOS / "utils"
_OUTDIR = Path("/home/adrian/PhD/Inferences/intervalinf/work/prior-posterior-audit")
sys.path.insert(0, str(_UTILS))

from full_spectrum_utils import (
    RadialSpecs, BlockIndex, enumerate_blocks, block_data_split,
    build_block_forward, build_block_prior, solve_block,
)
from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog
from prior_posterior_tuner import _build_custom_bessel_blocks
from posterior_viz import _compute_posterior_display_data
from intervalinf import ParallelConfig

# Load params
with open(_DEMOS / "tuner_params.json") as fh:
    params = json.load(fh)

n_basis    = int(params["n_basis"])
s_max      = int(params.get("s_max", 6))
hyper_base = params["hyper"]
taus       = params["taus"]
sigma_var  = float(params["sigma_var"])
n_jobs     = int(params.get("n_jobs", 1))
n_grid     = int(params.get("n_grid", 300))
n_probes   = int(params.get("n_probes", 15))
target_block = BlockIndex(s=params["block"]["s"], t=params["block"]["t"])

DATA_DIR   = _DEMOS / "data" / "normal-mode-data"
KERNEL_DIR = _DEMOS / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian"

# Build forward once
print("[phase34] loading data & building forward problem ...")
t0 = time.perf_counter()
catalog = NormalModeDataRegistry(str(DATA_DIR), mode_filter=NormalModeKernelCatalog(str(KERNEL_DIR)).list_modes())
catalog_k = NormalModeKernelCatalog(str(KERNEL_DIR))
reg       = NormalModeDataRegistry(str(DATA_DIR), mode_filter=catalog_k.list_modes())
all_blk   = enumerate_blocks(reg, s_max=s_max)
split     = block_data_split(reg, all_blk)
print(f"[phase34] data loaded {time.perf_counter()-t0:.1f}s")

parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
specs = RadialSpecs(n_basis=n_basis, parallel_cfg=parallel_cfg)

t0 = time.perf_counter()
G_st, C_D_st, M_st, D_st = build_block_forward(
    target_block.s, target_block.t, split[target_block], catalog_k, specs)
forward_dict = {target_block: (G_st, C_D_st, M_st, D_st)}
print(f"[phase34] forward built {time.perf_counter()-t0:.1f}s")

d        = np.asarray(split[target_block].data_vector, float)
err      = np.asarray(split[target_block].error_vector, float)
N_d      = len(d)
safe_err = np.where(err > 0, err, np.inf)
chi_null = float(np.sqrt(np.mean((d / safe_err) ** 2)))
print(f"[phase34] N_d={N_d}  chi_null={chi_null:.4f}")

def _tau_fn(p, _s): return taus.get(p, 1.0)

def _chi_rms(posterior):
    mu   = posterior.expectation
    pred = np.asarray(G_st(mu), float)
    return float(np.sqrt(np.mean(((pred - d) / safe_err) ** 2)))


def run_config(hyper_cfg, label=""):
    t0 = time.perf_counter()
    shared = _build_custom_bessel_blocks(specs, hyper_cfg)
    prior  = build_block_prior(
        target_block.s, target_block.t, shared, specs,
        tau_fn=_tau_fn, sigma_var=sigma_var)
    t_prior = time.perf_counter() - t0

    t0 = time.perf_counter()
    post = solve_block(target_block.s, target_block.t, G_st, C_D_st, prior, d)
    t_solve = time.perf_counter() - t0

    chi = _chi_rms(post)

    # Probe posterior covariance
    t0 = time.perf_counter()
    disp_post  = _compute_posterior_display_data(
        target_block, post, forward_dict, specs, n_grid=n_grid, n_probes=n_probes)
    disp_prior = _compute_posterior_display_data(
        target_block, prior, forward_dict, specs, n_grid=n_grid, n_probes=n_probes)
    t_probe = time.perf_counter() - t0

    prior_std_vp   = float(np.mean(disp_prior.std_vp))
    post_std_vp    = float(np.mean(disp_post.std_vp))
    contraction_vp = 1.0 - post_std_vp / prior_std_vp if prior_std_vp > 0 else float("nan")

    # Roughness: normalised rms of gradient of posterior mean vp
    mu_vp   = disp_post.mean_vp
    r_vp    = disp_post.r_vp
    grad_vp = np.gradient(mu_vp, r_vp)
    amp_rms = float(np.sqrt(np.mean(mu_vp ** 2)))
    roughness_vp = float(np.sqrt(np.mean(grad_vp ** 2)) / max(amp_rms, 1e-30))

    total = t_prior + t_solve + t_probe
    metrics = dict(
        chi_rms=chi,
        prior_std_vp=prior_std_vp,
        post_std_vp=post_std_vp,
        contraction_vp=contraction_vp,
        amp_rms_vp=amp_rms,
        roughness_vp=roughness_vp,
        t_prior_s=t_prior,
        t_solve_s=t_solve,
        t_probe_s=t_probe,
    )
    print(f"  [{label:16s}] chi={chi:.4f}  post_std={post_std_vp:.4g}  "
          f"rough={roughness_vp:.4g}  ({total:.0f}s)")
    return metrics


# SWEEP A: Variance scale
print("\n" + "="*65)
print("SWEEP A: variance scale factor")
print("="*65)

alpha_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
base_vars  = {p: hyper_base[p]["var"] for p in hyper_base}

sweep_A = []
for alpha in alpha_vals:
    hyper_cfg = {p: dict(hyper_base[p]) for p in hyper_base}
    for p in hyper_cfg:
        hyper_cfg[p]["var"] = base_vars[p] * alpha
    m = run_config(hyper_cfg, label=f"alpha={alpha:.4g}")
    m["alpha"] = alpha
    sweep_A.append(m)

# SWEEP B: Smoothness order
print("\n" + "="*65)
print("SWEEP B: s_order (var & length at baseline)")
print("="*65)

sorder_vals = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
sweep_B = []
for sord in sorder_vals:
    hyper_cfg = {p: dict(hyper_base[p]) for p in hyper_base}
    for p in hyper_cfg:
        hyper_cfg[p]["s_order"] = float(sord)
    m = run_config(hyper_cfg, label=f"s_order={sord}")
    m["s_order"] = float(sord)
    sweep_B.append(m)

# SWEEP C: Length scale
print("\n" + "="*65)
print("SWEEP C: length scale factor (s_order & var at baseline)")
print("="*65)

base_lengths  = {p: hyper_base[p]["length"] for p in hyper_base}
length_scales = [0.1, 0.2, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
sweep_C = []
for ls in length_scales:
    hyper_cfg = {p: dict(hyper_base[p]) for p in hyper_base}
    for p in hyper_cfg:
        hyper_cfg[p]["length"] = base_lengths[p] * ls
    m = run_config(hyper_cfg, label=f"length_x{ls}")
    m["length_scale"] = ls
    sweep_C.append(m)

# Figures
out_figs = _OUTDIR / "figures"

def _make_3panel(records, key, xlabel, title_stem, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Sweep: {title_stem}", fontsize=11)
    xs   = [r[key] for r in records]
    chi  = [r["chi_rms"] for r in records]
    ctr  = [r["contraction_vp"] for r in records]
    rgh  = [r["roughness_vp"] for r in records]
    for ax, ys, col, ylabel, ttl in zip(
        axes,
        [chi, ctr, rgh],
        ["steelblue", "darkorange", "forestgreen"],
        ["chi_rms", "contraction (vp)", "roughness"],
        ["chi_rms", "posterior contraction", "posterior roughness"],
    ):
        ax.plot(xs, ys, "o-", color=col, lw=1.8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{ttl} vs {xlabel}")
    axes[0].axhline(chi_null, color="red",   ls="--", lw=1, label=f"null={chi_null:.2f}")
    axes[0].axhline(1.0,      color="green", ls="--", lw=1, label="chi=1")
    axes[0].legend(fontsize=8)
    axes[1].set_ylim([-0.1, 1.1])
    plt.tight_layout()
    fig.savefig(str(out_figs / fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase34] saved {fname}")

# semi-log x for sweeps A and C
def _make_3panel_semilogx(records, key, xlabel, title_stem, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Sweep: {title_stem}", fontsize=11)
    xs   = [r[key] for r in records]
    chi  = [r["chi_rms"] for r in records]
    ctr  = [r["contraction_vp"] for r in records]
    rgh  = [r["roughness_vp"] for r in records]
    for ax, ys, col, ylabel, ttl in zip(
        axes,
        [chi, ctr, rgh],
        ["steelblue", "darkorange", "forestgreen"],
        ["chi_rms", "contraction (vp)", "roughness"],
        ["chi_rms", "posterior contraction", "posterior roughness"],
    ):
        ax.semilogx(xs, ys, "o-", color=col, lw=1.8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{ttl} vs {xlabel}")
    axes[0].axhline(chi_null, color="red",   ls="--", lw=1, label=f"null={chi_null:.2f}")
    axes[0].axhline(1.0,      color="green", ls="--", lw=1, label="chi=1")
    axes[0].legend(fontsize=8)
    axes[1].set_ylim([-0.1, 1.1])
    plt.tight_layout()
    fig.savefig(str(out_figs / fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase34] saved {fname}")

_make_3panel_semilogx(sweep_A, "alpha", "alpha (var scale)",
                      "A: variance scale", "phase34_var_sweep.png")
_make_3panel(sweep_B, "s_order", "s_order",
             "B: smoothness order", "phase34_smooth_sweep.png")
_make_3panel_semilogx(sweep_C, "length_scale", "length scale factor",
                      "C: length scale", "phase34_length_sweep.png")

# Save
out_proc = _OUTDIR / "processed"
results = {
    "block":   {"s": target_block.s, "t": target_block.t},
    "n_basis": n_basis, "N_d": N_d, "chi_null": chi_null,
    "hyper_base": hyper_base,
    "sweep_A_var_scale":  sweep_A,
    "sweep_B_smoothness": sweep_B,
    "sweep_C_length":     sweep_C,
}
with open(out_proc / "phase34_prior_sweep.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("[phase34] phase34_prior_sweep.json saved")

# Summary
print("\n" + "="*75)
print("PHASE 3+4 SUMMARY")
print("="*75)
print(f"chi_null={chi_null:.4f}")
print(f"\nSWEEP A:  {'alpha':>10} | {'chi_rms':>8} | {'contraction':>12} | {'roughness':>12}")
for r in sweep_A:
    print(f"          {r['alpha']:>10.4g} | {r['chi_rms']:>8.4f} | {r['contraction_vp']:>12.4f} | {r['roughness_vp']:>12.4g}")
print(f"\nSWEEP B:  {'s_order':>8} | {'chi_rms':>8} | {'contraction':>12} | {'roughness':>12}")
for r in sweep_B:
    print(f"          {r['s_order']:>8.1f} | {r['chi_rms']:>8.4f} | {r['contraction_vp']:>12.4f} | {r['roughness_vp']:>12.4g}")
print(f"\nSWEEP C:  {'scale':>8} | {'chi_rms':>8} | {'contraction':>12} | {'roughness':>12}")
for r in sweep_C:
    print(f"          {r['length_scale']:>8.3f} | {r['chi_rms']:>8.4f} | {r['contraction_vp']:>12.4f} | {r['roughness_vp']:>12.4g}")
print("="*75)
print("[phase34] COMPLETE")
