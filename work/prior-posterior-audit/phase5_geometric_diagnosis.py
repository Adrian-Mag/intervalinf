#!/usr/bin/env python3
"""
phase5_geometric_diagnosis.py
==============================
Phase 5 of MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT.

Geometric diagnosis:
1. Build dense matrix G (N_d x N_model) in basis representation
2. SVD of G: singular values, rank, effective rank
3. Project d onto column space of G — what fraction is explainable?
4. Unexplained signal: d_perp = d - G G^+ d  (in C_D^{-1} metric)
5. Build G C_prior G* (prior contribution to N) and C_D separately
   — show which dominates at each singular direction
6. Information spectrum: lambda_signal_j = sigma_j^2 / lambda_noise_j
   where sigma_j = G-singular-value in data-metric

This answers:
  (a) WHY is chi_rms floor ~3.9?  → rank-deficiency or model misspecification
  (b) WHY is posterior contraction always ~97-99%? → most singular directions
      have large signal-to-noise ratio, so all directions are data-constrained

Saves:
  processed/phase5_geometric.json
  figures/phase5_svd.png
  figures/phase5_information_spectrum.png
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
target_block = BlockIndex(s=params["block"]["s"], t=params["block"]["t"])

DATA_DIR   = _DEMOS / "data" / "normal-mode-data"
KERNEL_DIR = _DEMOS / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian"

print("[phase5] loading data & building forward ...")
t0 = time.perf_counter()
catalog_k = NormalModeKernelCatalog(str(KERNEL_DIR))
reg       = NormalModeDataRegistry(str(DATA_DIR), mode_filter=catalog_k.list_modes())
all_blk   = enumerate_blocks(reg, s_max=s_max)
split     = block_data_split(reg, all_blk)
parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
specs = RadialSpecs(n_basis=n_basis, parallel_cfg=parallel_cfg)
G_st, C_D_st, M_st, D_st = build_block_forward(
    target_block.s, target_block.t, split[target_block], catalog_k, specs)
print(f"[phase5] forward built {time.perf_counter()-t0:.1f}s")

d   = np.asarray(split[target_block].data_vector, float)
err = np.asarray(split[target_block].error_vector, float)
N_d = len(d)
safe_err = np.where(err > 0, err, np.inf)
chi_null = float(np.sqrt(np.mean((d / safe_err) ** 2)))

# ── Build dense matrix G (N_d × N_model) ──────────────────────────────────
# Use unit model-space elements (standard basis) to probe G
print("[phase5] building dense G matrix ...")
t0 = time.perf_counter()

# First build prior to understand structure
def _tau_fn(p, _s): return taus.get(p, 1.0)
shared = _build_custom_bessel_blocks(specs, hyper_base)
prior  = build_block_prior(
    target_block.s, target_block.t, shared, specs,
    tau_fn=_tau_fn, sigma_var=sigma_var)

# Get prior covariance in a useful form by column-by-column probe
# G_mat[j, :] = G applied to standard basis vector e_j in DATA space
# Instead: build G C G* (N_d x N_d) directly, then SVD that

C_D_diag = err ** 2
N_mat = np.empty((N_d, N_d), float)
print("[phase5] building dense N = GCG* + C_D ...")
for j in range(N_d):
    e_j = np.zeros(N_d); e_j[j] = 1.0
    N_mat[:, j] = np.asarray(G_st(prior.covariance(G_st.adjoint(e_j))), float)
N_mat = 0.5 * (N_mat + N_mat.T)
GCG = N_mat.copy()  # before adding C_D
N_mat += np.diag(C_D_diag)
print(f"[phase5] N built {time.perf_counter()-t0:.1f}s")

# Eigendecomposition of GCG*
eigvals_GCG, eigvecs_GCG = np.linalg.eigh(GCG)
eigvals_GCG = eigvals_GCG[::-1]  # descending
eigvecs_GCG = eigvecs_GCG[:, ::-1]

# C_D eigenvalues (diagonal, sorted)
eigvals_CD = np.sort(C_D_diag)[::-1]   # sorted descending

# Effective eigenvalues of N
eigvals_N = eigvals_GCG + eigvals_CD[::-1]   # rough: both sorted descending
# Exact: from the full matrix
eigvals_N_full, eigvecs_N = np.linalg.eigh(N_mat)
eigvals_N_full = eigvals_N_full[::-1]
eigvecs_N      = eigvecs_N[:, ::-1]

cond_N    = float(eigvals_N_full[0] / max(eigvals_N_full[-1], 1e-300))
rank_1e8  = int(np.sum(eigvals_N_full > eigvals_N_full[0] * 1e-8))
rank_1e12 = int(np.sum(eigvals_N_full > eigvals_N_full[0] * 1e-12))
print(f"[phase5] cond(N)={cond_N:.3g}  rank@1e-8={rank_1e8}  rank@1e-12={rank_1e12}")

# Information spectrum: in EACH eigendirection u_j of N,
# the data contribution is lambda_signal_j = u_j^T GCG* u_j
# the noise contribution is lambda_noise_j = u_j^T C_D u_j = sigma_j^2 (data noise)
# information content = lambda_signal_j / lambda_noise_j

lambda_signal = np.array([float(eigvecs_N[:, j] @ GCG @ eigvecs_N[:, j])
                           for j in range(N_d)])
lambda_noise  = np.array([float(eigvecs_N[:, j] @ (C_D_diag * eigvecs_N[:, j]))
                           for j in range(N_d)])
info_ratio = lambda_signal / np.maximum(lambda_noise, 1e-300)

# Posterior contraction in each direction:
# contraction_j = lambda_signal_j / (lambda_signal_j + lambda_noise_j)
contraction_per_dir = lambda_signal / (lambda_signal + lambda_noise)

print(f"\n[phase5] Information spectrum summary:")
print(f"  Eigenvalues of GCG*: max={eigvals_GCG[0]:.3g}  min={eigvals_GCG[-1]:.3g}")
print(f"  Fraction lambda_signal > lambda_noise: {np.mean(lambda_signal > lambda_noise):.3f}")
print(f"  Median info_ratio (lambda_s/lambda_n): {np.median(info_ratio):.3g}")
print(f"  Mean contraction per dir: {np.mean(contraction_per_dir):.4f}")
print(f"  Directions with contraction > 0.95: {np.sum(contraction_per_dir > 0.95)}/{N_d}")
print(f"  Directions with contraction > 0.99: {np.sum(contraction_per_dir > 0.99)}/{N_d}")

# ── Determine chi_rms floor from data projection ──────────────────────────
# Project d onto column space of GCG*: the "signal subspace" of data
# The column space of GCG* = column space of G
# For each eigenvector u_j of GCG* with nonzero eigenvalue:
#   d_parallel = sum_j (u_j^T d) u_j   (explained by model)
#   d_perp     = d - d_parallel         (cannot be explained by any model)

# Use a threshold to determine "zero" eigenvalues of GCG*
thresh_rank = eigvals_GCG[0] * 1e-8
rank_GCG    = int(np.sum(eigvals_GCG > thresh_rank))

# Using GCG* with C_D metric: project d onto the top rank_GCG directions
# In C_D^{-1} metric: u_j^T C_D^{-1} d for whitened data
d_white = d / safe_err   # standardized data
d_parallel_norm_sq = 0.0
d_perp_white = d_white.copy()
for j in range(rank_GCG):
    u_j = eigvecs_GCG[:, j] / safe_err   # in whitened space (approximately)
    u_j_norm = np.linalg.norm(u_j)
    if u_j_norm < 1e-15:
        continue
    u_j /= u_j_norm
    proj = float(u_j @ d_white)
    d_perp_white -= proj * u_j
    d_parallel_norm_sq += proj ** 2

chi_perp_rms = float(np.sqrt(np.mean(d_perp_white ** 2)))
chi_parallel_rms = float(np.sqrt(d_parallel_norm_sq / N_d))

# Alternative: minimum chi_rms from the min-norm LS solution
# GCG* u = d => u = eigvecs * diag(1/eigvals) * eigvecs^T * d  (pseudo-inv)
# Approximate: use only GCG* for the solve (ignore C_D regularization)
# alpha_ps = eigvecs * diag(1/max(eigvals, thresh)) * eigvecs^T * d
alpha_ps = np.zeros(N_d)
for j in range(N_d):
    ev = eigvals_GCG[j]
    if ev > thresh_rank:
        coeff = (eigvecs_GCG[:, j] @ d) / ev
        alpha_ps += coeff * eigvecs_GCG[:, j]
pred_ps = GCG @ alpha_ps
chi_ps = float(np.sqrt(np.mean(((pred_ps - d) / safe_err) ** 2)))
print(f"\n[phase5] chi_rms from pseudo-inverse of GCG* (no noise): {chi_ps:.4f}")
print(f"  [This is the best possible chi_rms from the prior basis]")

# Alternative 2: solve with very large regularization (near-minimum-norm)
# Uses ridge regression: (GCG* + epsilon*I)^{-1} d
eps = eigvals_GCG[-1] * 1e-6  # tiny regularization
N_ridge = GCG + np.eye(N_d) * eps + np.diag(C_D_diag) * 0.001
alpha_ridge = np.linalg.solve(N_ridge, d)
pred_ridge  = GCG @ alpha_ridge
chi_ridge   = float(np.sqrt(np.mean(((pred_ridge - d) / safe_err) ** 2)))
print(f"[phase5] chi_rms from tiny regularization ridge solve: {chi_ridge:.4f}")

# ── What is the intrinsic SNR in d? (per eigendirection of N) ─────────────
# Sort directions by SNR
snr_order = np.argsort(info_ratio)[::-1]
print(f"\n[phase5] Top-10 directions by info_ratio (signal/noise per N-eigendirection):")
print(f"  {'j':>4} | {'info_ratio':>12} | {'contraction':>12} | {'proj_d':>12}")
for k in range(min(10, N_d)):
    j = snr_order[k]
    proj_d_j = float(eigvecs_N[:, j] @ d)
    print(f"  {j:>4} | {info_ratio[j]:>12.4g} | {contraction_per_dir[j]:>12.4f} | {proj_d_j:>12.4g}")

print(f"\n[phase5] Bottom-10 directions (lowest SNR):")
for k in range(N_d - 10, N_d):
    j = snr_order[k]
    proj_d_j = float(eigvecs_N[:, j] @ d)
    print(f"  {j:>4} | {info_ratio[j]:>12.4g} | {contraction_per_dir[j]:>12.4f} | {proj_d_j:>12.4g}")

# ── Data signal in N-eigendirections ─────────────────────────────────────
# The chi_rms can be written as:
# chi_rms^2 = (1/N_d) sum_j [(G mu_post - d)^T e_j]^2 / sigma_j^2
# = (1/N_d) sum_j [(-C_D alpha)^T e_j]^2 / sigma_j^2  (since G mu - d = -C_D alpha)
# = (1/N_d) sum_j [C_D N^{-1} d]_j^2 / sigma_j^2

residual_post = -np.diag(C_D_diag) @ np.linalg.solve(N_mat, d)
chi_check = float(np.sqrt(np.mean((residual_post / safe_err) ** 2)))
print(f"\n[phase5] chi_rms check (from -C_D N^-1 d formula): {chi_check:.4f}")

# Decompose chi^2 by N-eigendirection
chi2_per_dir = np.zeros(N_d)
for j in range(N_d):
    u_j = eigvecs_N[:, j]
    r_j = float(u_j @ residual_post)
    n_j = float(u_j @ (safe_err ** 2 * u_j))  # noise contribution in direction j
    if n_j > 0:
        chi2_per_dir[j] = r_j ** 2 / n_j

cum_chi2 = np.cumsum(chi2_per_dir[snr_order]) / N_d  # in order of decreasing SNR

print(f"\n[phase5] chi^2 accumulation in decreasing SNR order:")
print(f"  After top   10 directions: cum chi2/N = {cum_chi2[9]:.4f}  -> chi_rms={np.sqrt(cum_chi2[9]):.4f}")
print(f"  After top   50 directions: cum chi2/N = {cum_chi2[49]:.4f}  -> chi_rms={np.sqrt(cum_chi2[49]):.4f}")
print(f"  After top  100 directions: cum chi2/N = {cum_chi2[99]:.4f}  -> chi_rms={np.sqrt(cum_chi2[99]):.4f}")
print(f"  After top  186 directions: cum chi2/N = {cum_chi2[185]:.4f}  -> chi_rms={np.sqrt(cum_chi2[185]):.4f}")
print(f"  [chi_rms from full sum: {float(np.sqrt(np.sum(chi2_per_dir)/N_d)):.4f}]")

# ── Figures ────────────────────────────────────────────────────────────────
out_figs = _OUTDIR / "figures"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Phase 5: Geometric Diagnosis — block({target_block.s},{target_block.t})", fontsize=11)

# 1. GCG* eigenvalue spectrum
ax = axes[0, 0]
ev_GCG = np.array(eigvals_GCG)
ax.semilogy(np.arange(1, N_d+1), ev_GCG, "b-", lw=1.5, label="GCG* eigenvalues")
ax.semilogy(np.arange(1, N_d+1), eigvals_CD[::-1], "r--", lw=1.2, alpha=0.7, label="C_D diagonal (sorted)")
ax.set_xlabel("index"); ax.set_ylabel("eigenvalue")
ax.set_title("Prior contribution (GCG*) vs noise (C_D)")
ax.legend(fontsize=8)

# 2. Information ratio per direction
ax = axes[0, 1]
ax.semilogy(np.arange(1, N_d+1), np.sort(info_ratio)[::-1], "g-", lw=1.5)
ax.axhline(1.0, color="red", ls="--", lw=1, label="signal=noise")
ax.set_xlabel("index (sorted by info_ratio)"); ax.set_ylabel("λ_signal / λ_noise")
ax.set_title("Information ratio per N-eigendirection")
ax.legend(fontsize=8)

# 3. Posterior contraction per direction
ax = axes[1, 0]
ax.plot(np.arange(1, N_d+1), np.sort(contraction_per_dir)[::-1], "o-", ms=3, lw=1.2, color="darkorange")
ax.axhline(0.99, color="red", ls="--", lw=1, label="99% contraction")
ax.axhline(0.95, color="orange", ls="--", lw=1, label="95% contraction")
ax.set_xlabel("index"); ax.set_ylabel("contraction")
ax.set_title("Posterior contraction per data-space direction")
ax.legend(fontsize=8)
ax.set_ylim([0, 1.05])

# 4. Cumulative chi2 by SNR order
ax = axes[1, 1]
ax.plot(np.arange(1, N_d+1), np.sqrt(cum_chi2), "b-", lw=1.5)
ax.axhline(5.34, color="red", ls="--", lw=1, label="baseline chi_rms=5.34")
ax.axhline(1.0,  color="green", ls="--", lw=1, label="chi_rms=1")
ax.set_xlabel("n_directions included (decreasing SNR order)")
ax.set_ylabel("cumulative chi_rms")
ax.set_title("chi_rms contribution by data direction")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(str(out_figs / "phase5_geometric.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("[phase5] phase5_geometric.png saved")

# ── Save ────────────────────────────────────────────────────────────────────
out_proc = _OUTDIR / "processed"
results = {
    "block": {"s": target_block.s, "t": target_block.t},
    "N_d": N_d,
    "n_basis": n_basis,
    "chi_null": chi_null,
    "cond_N": cond_N,
    "rank_GCG_1e8": rank_GCG,
    "rank_N_1e8": rank_1e8,
    "rank_N_1e12": rank_1e12,
    "chi_rms_pseudoinverse": chi_ps,
    "chi_rms_ridge": chi_ridge,
    "chi_rms_check": chi_check,
    "eigvals_GCG_top20": eigvals_GCG[:20].tolist(),
    "eigvals_GCG_bottom10": eigvals_GCG[-10:].tolist(),
    "info_ratio_median": float(np.median(info_ratio)),
    "info_ratio_min": float(np.min(info_ratio)),
    "info_ratio_max": float(np.max(info_ratio)),
    "frac_info_ratio_gt1": float(np.mean(info_ratio > 1.0)),
    "n_dirs_contraction_gt95": int(np.sum(contraction_per_dir > 0.95)),
    "n_dirs_contraction_gt99": int(np.sum(contraction_per_dir > 0.99)),
    "mean_contraction_per_dir": float(np.mean(contraction_per_dir)),
    "contraction_per_dir": contraction_per_dir.tolist(),
    "info_ratio": info_ratio.tolist(),
    "cum_chi_rms_at_10": float(np.sqrt(cum_chi2[9])),
    "cum_chi_rms_at_50": float(np.sqrt(cum_chi2[49])),
    "cum_chi_rms_at_100": float(np.sqrt(cum_chi2[99])),
    "cum_chi_rms_at_186": float(np.sqrt(cum_chi2[185])),
}
with open(out_proc / "phase5_geometric.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("[phase5] phase5_geometric.json saved")

print("\n" + "="*65)
print("PHASE 5 SUMMARY")
print("="*65)
print(f"  rank(GCG*, 1e-8) = {rank_GCG}/{N_d}")
print(f"  cond(N)          = {cond_N:.3g}")
print(f"  chi_rms(pseudo-inv of GCG*) = {chi_ps:.4f}  [best possible with this basis]")
print(f"  chi_rms(ridge, eps~0)       = {chi_ridge:.4f}")
print(f"  Median info_ratio (signal/noise per N-dir): {np.median(info_ratio):.3g}")
print(f"  Directions with info_ratio > 1:  {np.sum(info_ratio > 1)}/{N_d}")
print(f"  Directions with contraction > 95%: {np.sum(contraction_per_dir > 0.95)}/{N_d}")
print(f"  Directions with contraction > 99%: {np.sum(contraction_per_dir > 0.99)}/{N_d}")
print(f"  chi_rms contribution from top 10 dirs: {float(np.sqrt(cum_chi2[9])):.4f}")
print(f"  chi_rms contribution from all 186 dirs: {float(np.sqrt(cum_chi2[185])):.4f}")
print("="*65)
print("[phase5] COMPLETE")
