#!/usr/bin/env python3
"""
phase2_numerical_audit.py
=========================
Phase 2 of MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT.

Numerical accuracy audit:
1. Explicit data-space solve cross-checks (vs pygeoinf)
2. N matrix conditioning: eig spectrum, effective rank vs threshold
3. n_basis sensitivity sweep [30, 50, 80, 120]: does chi_rms / post std stabilize?
4. Data SNR analysis: what is the actual signal-to-noise in d?
5. Prior predictive check: what chi_rms does a draw from the prior produce?

Saves: processed/phase2_numerical_audit.json
       figures/phase2_nbasis_sweep.png
       figures/phase2_eigspectrum.png
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

# ── Load params ────────────────────────────────────────────────────────────
with open(_DEMOS / "tuner_params.json") as fh:
    params = json.load(fh)

n_basis_baseline = int(params["n_basis"])
s_max     = int(params.get("s_max", 6))
hyper     = params["hyper"]
taus      = params["taus"]
sigma_var = float(params["sigma_var"])
n_jobs    = int(params.get("n_jobs", 1))
target_block = BlockIndex(s=params["block"]["s"], t=params["block"]["t"])

DATA_DIR   = _DEMOS / "data" / "normal-mode-data"
KERNEL_DIR = _DEMOS / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian"

t0 = time.perf_counter()
catalog = NormalModeKernelCatalog(str(KERNEL_DIR))
reg     = NormalModeDataRegistry(str(DATA_DIR), mode_filter=catalog.list_modes())
all_blocks = enumerate_blocks(reg, s_max=s_max)
split      = block_data_split(reg, all_blocks)
print(f"[phase2] data loaded {time.perf_counter()-t0:.1f}s")

d   = np.asarray(split[target_block].data_vector, float)
err = np.asarray(split[target_block].error_vector, float)
N_d = len(d)
safe_err = np.where(err > 0, err, np.inf)

# ── SNR analysis ──────────────────────────────────────────────────────────
snr_per_datum = np.abs(d) / safe_err
snr_global    = np.linalg.norm(d) / np.linalg.norm(err)
chi_null      = float(np.sqrt(np.mean((d / safe_err) ** 2)))   # chi_rms if solution = 0
print(f"\n[phase2] SNR analysis:")
print(f"  N_d={N_d}")
print(f"  global SNR = {snr_global:.4g}")
print(f"  chi_rms if model=0 (null solution) = {chi_null:.4f}")
print(f"  median |d|/σ = {np.median(snr_per_datum):.4f}")
print(f"  fraction |d|/σ > 1 = {np.mean(snr_per_datum > 1):.3f}")
print(f"  fraction |d|/σ > 3 = {np.mean(snr_per_datum > 3):.3f}")

# ── n_basis sweep ─────────────────────────────────────────────────────────
n_basis_values = [30, 50, 80, 120]
sweep_results = []

for nb in n_basis_values:
    print(f"\n[phase2] n_basis={nb} ...")
    parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
    specs = RadialSpecs(n_basis=nb, parallel_cfg=parallel_cfg)

    t0 = time.perf_counter()
    G_st, C_D_st, M_st, D_st = build_block_forward(
        target_block.s, target_block.t, split[target_block], catalog, specs)
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    shared_bessel = _build_custom_bessel_blocks(specs, hyper)
    def _tau_fn(p, _s): return taus.get(p, 1.0)
    prior_st = build_block_prior(
        target_block.s, target_block.t, shared_bessel, specs,
        tau_fn=_tau_fn, sigma_var=sigma_var)
    t_prior = time.perf_counter() - t0

    t0 = time.perf_counter()
    posterior_st = solve_block(target_block.s, target_block.t, G_st, C_D_st, prior_st, d)
    t_solve = time.perf_counter() - t0

    mu_post = posterior_st.expectation
    pred    = np.asarray(G_st(mu_post), float)
    resid   = pred - d
    chi_rms = float(np.sqrt(np.mean((resid / safe_err) ** 2)))
    rel_fit = float(np.linalg.norm(resid) / max(np.linalg.norm(d), 1e-30))

    # Dense N for conditioning
    C_D_diag = err ** 2
    N_mat = np.empty((N_d, N_d), float)
    for j in range(N_d):
        e_j = np.zeros(N_d); e_j[j] = 1.0
        N_mat[:, j] = np.asarray(G_st(prior_st.covariance(G_st.adjoint(e_j))), float)
    N_mat += np.diag(C_D_diag)
    N_mat = 0.5 * (N_mat + N_mat.T)

    eigvals = np.linalg.eigvalsh(N_mat)
    cond_N  = float(eigvals[-1] / max(eigvals[0], 1e-300))
    rank_1e12 = int(np.sum(eigvals > eigvals[-1] * 1e-12))
    rank_1e8  = int(np.sum(eigvals > eigvals[-1] * 1e-8))

    # Explicit alpha solve
    alpha_ex = np.linalg.solve(N_mat, d)
    GCG = N_mat - np.diag(C_D_diag)
    pred_ex  = GCG @ alpha_ex
    chi_rms_ex = float(np.sqrt(np.mean(((pred_ex - d) / safe_err) ** 2)))

    # Prior predictive check: sample from prior and compute chi_rms of G*sample
    try:
        from intervalinf.sampling import KLSampler
        np.random.seed(42)
        n_prior_samples = 10
        prior_chis = []
        for _ in range(n_prior_samples):
            m_sample_fns = prior_st.sample()
            pred_sample  = np.asarray(G_st(m_sample_fns), float)
            prior_chis.append(float(np.sqrt(np.mean(((pred_sample - d) / safe_err) ** 2))))
        prior_chi_mean = float(np.mean(prior_chis))
        prior_chi_min  = float(np.min(prior_chis))
    except Exception as e:
        print(f"  [n_basis={nb}] prior predictive failed: {e}")
        prior_chi_mean = float("nan")
        prior_chi_min  = float("nan")

    result = {
        "n_basis": nb,
        "chi_rms": chi_rms,
        "chi_rms_explicit": chi_rms_ex,
        "rel_misfit": rel_fit,
        "cond_N": cond_N,
        "rank_1e12": rank_1e12,
        "rank_1e8":  rank_1e8,
        "eig_min":   float(eigvals[0]),
        "eig_max":   float(eigvals[-1]),
        "t_fwd_s":   t_fwd,
        "t_prior_s": t_prior,
        "t_solve_s": t_solve,
        "prior_chi_mean": prior_chi_mean,
        "prior_chi_min":  prior_chi_min,
        "eigvals": eigvals.tolist(),
    }
    sweep_results.append(result)

    print(f"  [n_basis={nb}] chi_rms={chi_rms:.4f} (expl={chi_rms_ex:.4f}) "
          f"cond={cond_N:.3g} rank={rank_1e12}/{N_d}  "
          f"prior_chi_mean={prior_chi_mean:.3f}")

# ── Summary table ─────────────────────────────────────────────────────────
print("\n[phase2] n_basis sweep summary:")
print(f"{'n_basis':>8} | {'chi_rms':>8} | {'chi_rms_ex':>10} | {'cond_N':>12} | {'rank':>8} | {'prior_chi':>10}")
print("-"*65)
for r in sweep_results:
    print(f"{r['n_basis']:>8} | {r['chi_rms']:>8.4f} | {r['chi_rms_explicit']:>10.4f} | "
          f"{r['cond_N']:>12.3e} | {r['rank_1e12']:>8} | {r['prior_chi_mean']:>10.3f}")

# ── Eigenvalue spectrum figure ─────────────────────────────────────────────
out_figs = _OUTDIR / "figures"
out_proc = _OUTDIR / "processed"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Phase 2: Normal Operator Eigenvalue Spectrum — block({target_block.s},{target_block.t})", fontsize=11)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_basis_values)))
for i, (r, col) in enumerate(zip(sweep_results, colors)):
    ev = np.array(r["eigvals"])[::-1]   # descending
    ax = axes[0]
    ax.semilogy(np.arange(1, len(ev)+1), ev, color=col, lw=1.5, label=f"n={r['n_basis']}")
    ax2 = axes[1]
    ev_norm = ev / ev[0]
    ax2.semilogy(np.arange(1, len(ev)+1), ev_norm, color=col, lw=1.5, label=f"n={r['n_basis']}")

axes[0].set_xlabel("eigenvalue index"); axes[0].set_ylabel("λ")
axes[0].set_title("Eigenvalues of N = GCG* + C_D"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("eigenvalue index"); axes[1].set_ylabel("λ / λ_max")
axes[1].set_title("Normalised eigenvalues"); axes[1].legend(fontsize=8)
for ax in axes:
    ax.axhline(1e-8,  color="orange", ls="--", lw=0.8, alpha=0.7, label="1e-8")
    ax.axhline(1e-12, color="red",    ls="--", lw=0.8, alpha=0.7, label="1e-12")

plt.tight_layout()
fig.savefig(str(out_figs / "phase2_eigspectrum.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[phase2] eigspectrum figure saved")

# ── n_basis vs chi_rms figure ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Phase 2: n_basis sensitivity", fontsize=11)

nbs  = [r["n_basis"] for r in sweep_results]
chis = [r["chi_rms"] for r in sweep_results]
conds= [r["cond_N"] for r in sweep_results]
ranks= [r["rank_1e12"] for r in sweep_results]

axes[0].plot(nbs, chis, "o-", color="steelblue", lw=1.5)
axes[0].axhline(chi_null, color="red", ls="--", lw=1, label=f"null chi_rms={chi_null:.2f}")
axes[0].axhline(1.0, color="green", ls="--", lw=1, label="chi_rms=1")
axes[0].set_xlabel("n_basis"); axes[0].set_ylabel("chi_rms"); axes[0].set_title("Data misfit vs n_basis")
axes[0].legend(fontsize=8)

axes[1].semilogy(nbs, conds, "o-", color="darkorange", lw=1.5)
axes[1].set_xlabel("n_basis"); axes[1].set_ylabel("cond(N)"); axes[1].set_title("Conditioning vs n_basis")

axes[2].plot(nbs, ranks, "o-", color="forestgreen", lw=1.5)
axes[2].axhline(N_d, color="gray", ls="--", lw=1, label=f"N_d={N_d}")
axes[2].set_xlabel("n_basis"); axes[2].set_ylabel("rank(1e-12)"); axes[2].set_title("Rank vs n_basis")
axes[2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(str(out_figs / "phase2_nbasis_sweep.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[phase2] n_basis sweep figure saved")

# ── Save results ───────────────────────────────────────────────────────────
audit_results = {
    "block": {"s": target_block.s, "t": target_block.t},
    "N_d": N_d,
    "snr_global": float(snr_global),
    "chi_null": float(chi_null),
    "median_snr_per_datum": float(np.median(snr_per_datum)),
    "frac_snr_gt1": float(np.mean(snr_per_datum > 1)),
    "frac_snr_gt3": float(np.mean(snr_per_datum > 3)),
    "n_basis_sweep": sweep_results,
}
# Remove large eigvals arrays from saved JSON to keep size manageable
for r in audit_results["n_basis_sweep"]:
    r.pop("eigvals", None)

with open(out_proc / "phase2_numerical_audit.json", "w") as fh:
    json.dump(audit_results, fh, indent=2)
print(f"[phase2] phase2_numerical_audit.json saved")

print("\n" + "="*65)
print("PHASE 2 SUMMARY")
print("="*65)
print(f"  N_d={N_d}  global_SNR={snr_global:.3g}")
print(f"  chi_rms(null model)={chi_null:.4f}   [if solution=0]")
print(f"  median |d|/σ = {np.median(snr_per_datum):.4f}")
print(f"\n  n_basis | chi_rms | cond(N)  | rank(1e-12) | prior_chi_mean")
for r in sweep_results:
    print(f"  {r['n_basis']:>6}  | {r['chi_rms']:>7.4f} | {r['cond_N']:>8.3e} | {r['rank_1e12']:>11} | {r['prior_chi_mean']:>14.3f}")
print("="*65)
print("[phase2] COMPLETE")
