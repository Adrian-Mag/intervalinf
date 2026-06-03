#!/usr/bin/env python3
"""
phase6_cross_check.py
=====================
Phase 6 of MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT.

Cross-check key findings from block (0,0) on two secondary blocks:
  - (2, 0)
  - (2, 3)

For each block:
  1. Baseline chi_rms and contraction (bump probe)
  2. rank(GCG*, 1e-8) and info_ratio statistics
  3. Variance sweep (5 alpha values): chi_rms(alpha) curve
  4. Compare to block (0,0)

Purpose: determine whether the rank-deficiency and chi_rms floor are
universal features of this inference problem or specific to block (0,0).

Saves:
  processed/phase6_cross_check.json
  figures/phase6_comparison.png
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

DATA_DIR   = _DEMOS / "data" / "normal-mode-data"
KERNEL_DIR = _DEMOS / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian"

print("[phase6] loading data ...")
catalog_k   = NormalModeKernelCatalog(str(KERNEL_DIR))
reg         = NormalModeDataRegistry(str(DATA_DIR), mode_filter=catalog_k.list_modes())
all_blk     = enumerate_blocks(reg, s_max=s_max)
split       = block_data_split(reg, all_blk)
available   = list(split.keys())
print(f"[phase6] available blocks: {available}")

parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
specs        = RadialSpecs(n_basis=n_basis, parallel_cfg=parallel_cfg)

def _tau_fn(p, _s): return taus.get(p, 1.0)

# Target blocks for cross-check
candidate_blocks = [BlockIndex(2, 0), BlockIndex(2, 3)]
test_blocks = [b for b in candidate_blocks if b in split]
if not test_blocks:
    # Fall back to the 2nd and 3rd available blocks
    test_blocks = available[1:3] if len(available) >= 3 else available[1:]
print(f"[phase6] cross-check blocks: {test_blocks}")

# Load block(0,0) reference from processed files
with open(_OUTDIR / "processed" / "baseline_metrics.json") as fh:
    ref_metrics = json.load(fh)
print(f"[phase6] reference block(0,0): chi_rms={ref_metrics.get('chi_rms_pygeoinf', '?')}")


def analyze_block(blk: BlockIndex) -> dict:
    print(f"\n[phase6] ===== block ({blk.s},{blk.t}) =====")
    if blk not in split:
        print(f"  [phase6] block {blk} not in split, skipping")
        return {"block": {"s": blk.s, "t": blk.t}, "available": False}

    t0 = time.perf_counter()
    G_st, C_D_st, M_st, D_st = build_block_forward(
        blk.s, blk.t, split[blk], catalog_k, specs)
    forward_dict = {blk: (G_st, C_D_st, M_st, D_st)}
    d   = np.asarray(split[blk].data_vector, float)
    err = np.asarray(split[blk].error_vector, float)
    N_d = len(d)
    safe_err = np.where(err > 0, err, np.inf)
    chi_null = float(np.sqrt(np.mean((d / safe_err) ** 2)))
    print(f"  N_d={N_d}  chi_null={chi_null:.4f}  forward built {time.perf_counter()-t0:.1f}s")

    # Baseline prior + posterior
    shared = _build_custom_bessel_blocks(specs, hyper_base)
    prior  = build_block_prior(blk.s, blk.t, shared, specs,
                               tau_fn=_tau_fn, sigma_var=sigma_var)
    post   = solve_block(blk.s, blk.t, G_st, C_D_st, prior, d)
    mu_post = post.expectation
    pred    = np.asarray(G_st(mu_post), float)
    chi_baseline = float(np.sqrt(np.mean(((pred - d) / safe_err) ** 2)))

    # Bump-probe contraction
    disp_post  = _compute_posterior_display_data(blk, post, forward_dict, specs,
                                                  n_grid=n_grid, n_probes=n_probes)
    disp_prior = _compute_posterior_display_data(blk, prior, forward_dict, specs,
                                                  n_grid=n_grid, n_probes=n_probes)
    prior_std_vp  = float(np.mean(disp_prior.std_vp))
    post_std_vp   = float(np.mean(disp_post.std_vp))
    contraction   = 1.0 - post_std_vp / prior_std_vp if prior_std_vp > 0 else float("nan")
    print(f"  chi_rms={chi_baseline:.4f}  prior_std={prior_std_vp:.4g}  "
          f"post_std={post_std_vp:.4g}  contraction={contraction:.4f}")

    # Dense N for rank analysis
    C_D_diag = err ** 2
    GCG = np.empty((N_d, N_d), float)
    for j in range(N_d):
        e_j = np.zeros(N_d); e_j[j] = 1.0
        GCG[:, j] = np.asarray(G_st(prior.covariance(G_st.adjoint(e_j))), float)
    GCG = 0.5 * (GCG + GCG.T)
    N_mat = GCG + np.diag(C_D_diag)

    eigvals_GCG, eigvecs_GCG = np.linalg.eigh(GCG)
    eigvals_GCG = eigvals_GCG[::-1]
    eigvecs_GCG = eigvecs_GCG[:, ::-1]

    eigvals_N_full, eigvecs_N = np.linalg.eigh(N_mat)
    eigvals_N_full = eigvals_N_full[::-1]
    eigvecs_N      = eigvecs_N[:, ::-1]

    rank_GCG_1e8 = int(np.sum(eigvals_GCG > eigvals_GCG[0] * 1e-8))
    cond_N = float(eigvals_N_full[0] / max(eigvals_N_full[-1], 1e-300))

    lambda_signal = np.array([float(eigvecs_N[:, j] @ GCG @ eigvecs_N[:, j])
                               for j in range(N_d)])
    lambda_noise  = np.array([float(eigvecs_N[:, j] @ (C_D_diag * eigvecs_N[:, j]))
                               for j in range(N_d)])
    info_ratio    = lambda_signal / np.maximum(lambda_noise, 1e-300)
    contraction_dir = lambda_signal / (lambda_signal + lambda_noise)

    n_ctr95 = int(np.sum(contraction_dir > 0.95))
    n_ctr99 = int(np.sum(contraction_dir > 0.99))
    info_med = float(np.median(info_ratio))
    info_frac_gt1 = float(np.mean(info_ratio > 1.0))
    print(f"  rank(GCG*,1e-8)={rank_GCG_1e8}/{N_d}  cond(N)={cond_N:.3g}")
    print(f"  info_ratio median={info_med:.4g}  frac>1={info_frac_gt1:.3f}")
    print(f"  dirs contraction>95%: {n_ctr95}/{N_d}")

    # Variance sweep (5 alpha values: 0.01, 0.1, 1, 10, 100)
    base_vars  = {p: hyper_base[p]["var"] for p in hyper_base}
    alpha_vals = [0.01, 0.1, 1.0, 10.0, 100.0]
    var_sweep  = []
    for alpha in alpha_vals:
        hyper_cfg = {p: dict(hyper_base[p]) for p in hyper_base}
        for p in hyper_cfg:
            hyper_cfg[p]["var"] = base_vars[p] * alpha
        sh2  = _build_custom_bessel_blocks(specs, hyper_cfg)
        pr2  = build_block_prior(blk.s, blk.t, sh2, specs, tau_fn=_tau_fn, sigma_var=sigma_var)
        po2  = solve_block(blk.s, blk.t, G_st, C_D_st, pr2, d)
        pred2 = np.asarray(G_st(po2.expectation), float)
        chi2  = float(np.sqrt(np.mean(((pred2 - d) / safe_err) ** 2)))
        var_sweep.append({"alpha": alpha, "chi_rms": chi2})
        print(f"  alpha={alpha:.3g}: chi_rms={chi2:.4f}")

    return {
        "block": {"s": blk.s, "t": blk.t},
        "available": True,
        "N_d": N_d,
        "chi_null": chi_null,
        "chi_rms_baseline": chi_baseline,
        "prior_std_vp": prior_std_vp,
        "post_std_vp":  post_std_vp,
        "contraction_vp": contraction,
        "rank_GCG_1e8": rank_GCG_1e8,
        "cond_N": cond_N,
        "info_ratio_median": info_med,
        "info_ratio_frac_gt1": info_frac_gt1,
        "n_dirs_contraction_gt95": n_ctr95,
        "n_dirs_contraction_gt99": n_ctr99,
        "var_sweep": var_sweep,
        "eigvals_GCG_top10": eigvals_GCG[:10].tolist(),
    }


cross_results = {}
for blk in test_blocks:
    cross_results[f"block_{blk.s}_{blk.t}"] = analyze_block(blk)

# ── Comparison figure ──────────────────────────────────────────────────────
out_figs = _OUTDIR / "figures"
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Phase 6: Cross-check — baseline vs secondary blocks", fontsize=11)

# Reference block (0,0) data
ref_chi     = float(ref_metrics.get("chi_rms_pygeoinf", 5.34))
ref_rank    = 80   # from Phase 5
ref_info    = 0.167  # from Phase 5

labels = [f"(0,0)\nref"]
chis   = [ref_chi]
ranks  = [ref_rank]
infos  = [ref_info]
for k, v in cross_results.items():
    if v.get("available", False):
        labels.append(f"({v['block']['s']},{v['block']['t']})")
        chis.append(v["chi_rms_baseline"])
        ranks.append(v["rank_GCG_1e8"])
        infos.append(v["info_ratio_median"])

x = np.arange(len(labels))
axes[0].bar(x, chis, color=["steelblue"] + ["darkorange"]*(len(chis)-1))
axes[0].axhline(1.0, color="green", ls="--", lw=1, label="chi=1")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel("chi_rms"); axes[0].set_title("Baseline chi_rms per block")
axes[0].legend(fontsize=8)

axes[1].bar(x, ranks, color=["steelblue"] + ["darkorange"]*(len(ranks)-1))
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel("rank(GCG*, 1e-8)"); axes[1].set_title("Effective rank of GCG*")

axes[2].bar(x, infos, color=["steelblue"] + ["darkorange"]*(len(infos)-1))
axes[2].axhline(1.0, color="red", ls="--", lw=1, label="signal=noise")
axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=9)
axes[2].set_ylabel("median info_ratio"); axes[2].set_title("Median information per N-dir")
axes[2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(str(out_figs / "phase6_comparison.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("[phase6] phase6_comparison.png saved")

# ── Save ────────────────────────────────────────────────────────────────────
out_proc = _OUTDIR / "processed"
results_save = {
    "reference_block": ref_metrics,
    "cross_check": cross_results,
    "test_blocks": [[b.s, b.t] for b in test_blocks],
}
with open(out_proc / "phase6_cross_check.json", "w") as fh:
    json.dump(results_save, fh, indent=2)
print("[phase6] phase6_cross_check.json saved")

print("\n" + "="*70)
print("PHASE 6 SUMMARY: Block cross-check")
print("="*70)
print(f"  Reference block (0,0): chi_rms={ref_chi:.4f}  rank(GCG*)=80/186  info_med=0.167")
for k, v in cross_results.items():
    if v.get("available", False):
        b = v["block"]
        print(f"  Block ({b['s']},{b['t']}):  N_d={v['N_d']}  chi_rms={v['chi_rms_baseline']:.4f}  "
              f"rank(GCG*)={v['rank_GCG_1e8']}/{v['N_d']}  "
              f"info_med={v['info_ratio_median']:.4g}  contraction={v['contraction_vp']:.4f}")
        for row in v["var_sweep"]:
            print(f"    alpha={row['alpha']:.3g}: chi_rms={row['chi_rms']:.4f}")
print("="*70)
print("[phase6] COMPLETE")
