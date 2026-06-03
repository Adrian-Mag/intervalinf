#!/usr/bin/env python3
"""
phase1_baseline.py
==================
Phase 1 of MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT.

Reproduces real-data posterior for block (s=0,t=0) from tuner_params.json.
Saves baseline_metrics.json, current_prior_snapshot.json, and figures.

Run:
    cd /home/adrian/PhD/Inferences/intervalinf/demos/old_demos/paper_demos
    conda activate inferences
    python /home/adrian/PhD/Inferences/intervalinf/work/prior-posterior-audit/phase1_baseline.py
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
from intervalinf.core.functions import Function as _IFunction

# ── Load params ────────────────────────────────────────────────────────────
with open(_DEMOS / "tuner_params.json") as fh:
    params = json.load(fh)

n_basis   = int(params["n_basis"])
s_max     = int(params.get("s_max", 6))
hyper     = params["hyper"]
taus      = params["taus"]
sigma_var = float(params["sigma_var"])
n_jobs    = int(params.get("n_jobs", 1))
n_grid    = int(params.get("n_grid", 300))
n_probes  = int(params.get("n_probes", 15))
target_block = BlockIndex(s=params["block"]["s"], t=params["block"]["t"])

print(f"[phase1] block={target_block}, n_basis={n_basis}, n_jobs={n_jobs}")

# ── Load data ─────────────────────────────────────────────────────────────
DATA_DIR   = _DEMOS / "data" / "normal-mode-data"
KERNEL_DIR = _DEMOS / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian"

t0 = time.perf_counter()
catalog = NormalModeKernelCatalog(str(KERNEL_DIR))
reg     = NormalModeDataRegistry(str(DATA_DIR), mode_filter=catalog.list_modes())
all_blocks = enumerate_blocks(reg, s_max=s_max)
split      = block_data_split(reg, all_blocks)
print(f"[phase1] data loaded {time.perf_counter()-t0:.1f}s, {len(all_blocks)} blocks")

# ── Build forward ─────────────────────────────────────────────────────────
parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
specs = RadialSpecs(n_basis=n_basis, parallel_cfg=parallel_cfg)

t0 = time.perf_counter()
G_st, C_D_st, M_st, D_st = build_block_forward(
    target_block.s, target_block.t, split[target_block], catalog, specs)
print(f"[phase1] forward built {time.perf_counter()-t0:.1f}s")

d   = np.asarray(split[target_block].data_vector, float)
err = np.asarray(split[target_block].error_vector, float)
N_d = len(d)
print(f"[phase1] N_d={N_d}")

# ── Build prior ────────────────────────────────────────────────────────────
t0 = time.perf_counter()
shared_bessel = _build_custom_bessel_blocks(specs, hyper)
def _tau_fn(p, _s): return taus.get(p, 1.0)
prior_st = build_block_prior(
    target_block.s, target_block.t, shared_bessel, specs,
    tau_fn=_tau_fn, sigma_var=sigma_var)
print(f"[phase1] prior built {time.perf_counter()-t0:.1f}s")

# ── Solve posterior ────────────────────────────────────────────────────────
t0 = time.perf_counter()
posterior_st = solve_block(target_block.s, target_block.t, G_st, C_D_st, prior_st, d)
print(f"[phase1] posterior solved {time.perf_counter()-t0:.1f}s")

# ── Dense normal operator for conditioning ─────────────────────────────────
print("[phase1] building dense N for conditioning ...")
t0 = time.perf_counter()
C_D_diag = err ** 2
N_mat = np.empty((N_d, N_d), float)
for j in range(N_d):
    e_j = np.zeros(N_d); e_j[j] = 1.0
    N_mat[:, j] = np.asarray(G_st(prior_st.covariance(G_st.adjoint(e_j))), float)
N_mat += np.diag(C_D_diag)
N_mat = 0.5 * (N_mat + N_mat.T)
print(f"[phase1] N built {time.perf_counter()-t0:.1f}s")

eigvals  = np.linalg.eigvalsh(N_mat)
cond_N   = float(eigvals[-1] / max(eigvals[0], 1e-300))
rank_1e12 = int(np.sum(eigvals > eigvals[-1] * 1e-12))
rank_1e8  = int(np.sum(eigvals > eigvals[-1] * 1e-8))
print(f"[phase1] cond(N)={cond_N:.4g}, rank(1e-12)={rank_1e12}/{N_d}")

# Explicit data-space solve
alpha_ex = np.linalg.solve(N_mat, d)
GCG_mat  = N_mat - np.diag(C_D_diag)
pred_ex  = GCG_mat @ alpha_ex

# pygeoinf posterior prediction
mu_post  = posterior_st.expectation
pred_pg  = np.asarray(G_st(mu_post), float)

safe_err = np.where(err > 0, err, np.inf)
chi_rms_ex = float(np.sqrt(np.mean(((pred_ex - d) / safe_err) ** 2)))
chi_rms_pg = float(np.sqrt(np.mean(((pred_pg - d) / safe_err) ** 2)))
solver_agree = abs(chi_rms_ex - chi_rms_pg) < 0.05
print(f"[phase1] chi_rms explicit={chi_rms_ex:.4f}  pygeoinf={chi_rms_pg:.4f}  agree={'YES' if solver_agree else 'NO WARNING'}")

# ── Posterior display data (covariance probing) ────────────────────────────
forward_dict = {target_block: (G_st, C_D_st, M_st, D_st)}
print(f"[phase1] probing posterior ({n_probes} probes) ...")
t0 = time.perf_counter()
dd = _compute_posterior_display_data(
    target_block, posterior_st, forward_dict, specs,
    n_grid=n_grid, n_probes=n_probes)
print(f"[phase1] posterior probing done {time.perf_counter()-t0:.1f}s")

# ── Prior covariance probing (for contraction ratios) ─────────────────────
R   = specs.earth_radius_km
ICB = specs.icb_radius_km
CMB = specs.cmb_radius_km

M_functions   = M_st.subspaces[0]
M_vp_space    = M_functions.subspaces[0]
M_vs_space    = M_functions.subspaces[1]
M_rho_space   = M_functions.subspaces[2]
M_vs_IC_space = M_vs_space.subspaces[0]
M_vs_M_space  = M_vs_space.subspaces[1]

def _make_zero(M_sp):
    return _IFunction(M_sp, evaluate_callable=lambda r: np.zeros_like(np.asarray(r, float)))

def _probe_prior_std(cov_op, M_comp, comp_name, probe_centres):
    domain  = M_comp.function_domain
    dom_len = float(domain.b - domain.a)
    sigma_b = dom_len / n_probes
    stds = np.zeros(len(probe_centres))
    for i, r_i in enumerate(probe_centres):
        r_quad    = np.linspace(float(domain.a), float(domain.b), 512)
        bv        = np.exp(-0.5 * ((r_quad - r_i) / sigma_b) ** 2)
        norm_sq   = np.trapezoid(bv ** 2, r_quad)
        if norm_sq < 1e-30: continue
        norm = np.sqrt(norm_sq)
        def _ub(r, _ri=r_i, _sb=sigma_b, _n=norm):
            return np.exp(-0.5 * ((np.asarray(r, float) - _ri) / _sb) ** 2) / _n
        bump_fn = _IFunction(M_comp, evaluate_callable=_ub)
        z1 = np.zeros(1)
        zv = _make_zero(M_vp_space); zic = _make_zero(M_vs_IC_space)
        zm = _make_zero(M_vs_M_space); zr = _make_zero(M_rho_space)
        if   comp_name == "vp":    fin = [[bump_fn, [zic, zm], zr],  [z1, z1]]
        elif comp_name == "vs_IC": fin = [[zv,      [bump_fn, zm], zr], [z1, z1]]
        elif comp_name == "vs_M":  fin = [[zv,      [zic, bump_fn], zr], [z1, z1]]
        elif comp_name == "rho":   fin = [[zv,      [zic, zm], bump_fn], [z1, z1]]
        cout = cov_op(fin)
        if   comp_name == "vp":    ofn = cout[0][0]
        elif comp_name == "vs_IC": ofn = cout[0][1][0]
        elif comp_name == "vs_M":  ofn = cout[0][1][1]
        elif comp_name == "rho":   ofn = cout[0][2]
        ov = np.asarray([ofn(r) for r in r_quad], float)
        iv = np.asarray([_ub(r) for r in r_quad], float)
        inner = np.trapezoid(iv * ov, r_quad)
        stds[i] = np.sqrt(max(0.0, inner))
    return stds

print("[phase1] probing prior ...")
t0 = time.perf_counter()
prio_std_vp    = _probe_prior_std(prior_st.covariance, M_vp_space,    "vp",    dd.probe_r_vp)
prio_std_vs_IC = _probe_prior_std(prior_st.covariance, M_vs_IC_space, "vs_IC", dd.probe_r_vs_IC)
prio_std_vs_M  = _probe_prior_std(prior_st.covariance, M_vs_M_space,  "vs_M",  dd.probe_r_vs_M)
prio_std_rho   = _probe_prior_std(prior_st.covariance, M_rho_space,   "rho",   dd.probe_r_rho)
print(f"[phase1] prior probing done {time.perf_counter()-t0:.1f}s")

def _cont(post_s, prior_s):
    v = prior_s > 1e-30
    if not np.any(v): return float("nan"), float("nan")
    r = post_s[v] / prior_s[v]
    return float(1 - np.mean(r)), float(1 - np.min(r))

cont_vp    = _cont(dd.std_vp,    prio_std_vp)
cont_vs_IC = _cont(dd.std_vs_IC, prio_std_vs_IC)
cont_vs_M  = _cont(dd.std_vs_M,  prio_std_vs_M)
cont_rho   = _cont(dd.std_rho,   prio_std_rho)

print(f"[phase1] Contraction vp={cont_vp}, vs_IC={cont_vs_IC}, vs_M={cont_vs_M}, rho={cont_rho}")

def _rough(r_grid, mu):
    dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 1.0
    return (float(np.sqrt(np.sum(np.diff(mu)**2/dr))),
            float(np.sqrt(np.sum(np.diff(np.diff(mu))**2/dr**3))))

rg_vp, rl_vp     = _rough(dd.r_vp,    dd.mean_vp)
rg_vsIC, rl_vsIC = _rough(dd.r_vs_IC, dd.mean_vs_IC)
rg_vsM, rl_vsM   = _rough(dd.r_vs_M,  dd.mean_vs_M)
rg_rho, rl_rho   = _rough(dd.r_rho,   dd.mean_rho)

# ── Save metrics ───────────────────────────────────────────────────────────
out_proc = _OUTDIR / "processed"
out_figs = _OUTDIR / "figures"

metrics = {
    "block": {"s": target_block.s, "t": target_block.t},
    "n_basis": n_basis, "N_d": N_d,
    "cond_N": cond_N, "eig_N_min": float(eigvals[0]), "eig_N_max": float(eigvals[-1]),
    "rank_1e12": rank_1e12, "rank_1e8": rank_1e8,
    "chi_rms_explicit": chi_rms_ex, "chi_rms_posterior": chi_rms_pg,
    "chi2_explicit": chi_rms_ex**2, "chi2_posterior": chi_rms_pg**2,
    "rel_misfit": float(np.linalg.norm(pred_pg - d) / max(np.linalg.norm(d), 1e-30)),
    "solver_agreement_ok": solver_agree,
    "prior_std_vp_mean": float(prio_std_vp.mean()), "prior_std_vp_max": float(prio_std_vp.max()),
    "post_std_vp_mean":  float(dd.std_vp.mean()),   "post_std_vp_max":  float(dd.std_vp.max()),
    "prior_std_vsIC_mean": float(prio_std_vs_IC.mean()), "prior_std_vsIC_max": float(prio_std_vs_IC.max()),
    "post_std_vsIC_mean":  float(dd.std_vs_IC.mean()),   "post_std_vsIC_max":  float(dd.std_vs_IC.max()),
    "prior_std_vsM_mean":  float(prio_std_vs_M.mean()),  "prior_std_vsM_max":  float(prio_std_vs_M.max()),
    "post_std_vsM_mean":   float(dd.std_vs_M.mean()),    "post_std_vsM_max":   float(dd.std_vs_M.max()),
    "prior_std_rho_mean":  float(prio_std_rho.mean()),   "prior_std_rho_max":  float(prio_std_rho.max()),
    "post_std_rho_mean":   float(dd.std_rho.mean()),     "post_std_rho_max":   float(dd.std_rho.max()),
    "contraction_mean_vp": cont_vp[0],    "contraction_max_vp": cont_vp[1],
    "contraction_mean_vsIC": cont_vs_IC[0], "contraction_max_vsIC": cont_vs_IC[1],
    "contraction_mean_vsM": cont_vs_M[0],  "contraction_max_vsM": cont_vs_M[1],
    "contraction_mean_rho": cont_rho[0],   "contraction_max_rho": cont_rho[1],
    "roughness_grad_vp": rg_vp,   "roughness_lap_vp": rl_vp,
    "roughness_grad_vsIC": rg_vsIC,"roughness_lap_vsIC": rl_vsIC,
    "roughness_grad_vsM":  rg_vsM, "roughness_lap_vsM":  rl_vsM,
    "roughness_grad_rho":  rg_rho, "roughness_lap_rho":  rl_rho,
    "mean_vp_range": [float(dd.mean_vp.min()), float(dd.mean_vp.max())],
    "mean_vsIC_range": [float(dd.mean_vs_IC.min()), float(dd.mean_vs_IC.max())],
    "mean_vsM_range":  [float(dd.mean_vs_M.min()),  float(dd.mean_vs_M.max())],
    "mean_rho_range":  [float(dd.mean_rho.min()),   float(dd.mean_rho.max())],
    "sigma_1_mean": float(dd.sigma_1_mean),
    # Arrays for subsequent phases
    "probe_r_vp":    dd.probe_r_vp.tolist(),
    "probe_std_vp_post":  dd.std_vp.tolist(),
    "probe_std_vp_prior": prio_std_vp.tolist(),
    "probe_r_vsM":         dd.probe_r_vs_M.tolist(),
    "probe_std_vsM_post":  dd.std_vs_M.tolist(),
    "probe_std_vsM_prior": prio_std_vs_M.tolist(),
    "r_vp": dd.r_vp.tolist(), "mean_vp": dd.mean_vp.tolist(),
    "r_vsM": dd.r_vs_M.tolist(), "mean_vsM": dd.mean_vs_M.tolist(),
    "r_rho": dd.r_rho.tolist(), "mean_rho": dd.mean_rho.tolist(),
}

with open(out_proc / "baseline_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)
print(f"[phase1] baseline_metrics.json saved")

with open(out_proc / "current_prior_snapshot.json", "w") as fh:
    json.dump({"block": {"s": target_block.s, "t": target_block.t},
               "n_basis": n_basis, "hyper": hyper, "taus": taus, "sigma_var": sigma_var}, fh, indent=2)
print(f"[phase1] current_prior_snapshot.json saved")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle(
    f"Baseline block({target_block.s},{target_block.t}) | n_basis={n_basis} | "
    f"chi_rms={chi_rms_pg:.3f} | cond(N)={cond_N:.2e}",
    fontsize=10)

configs = [
    ("vp",    dd.r_vp,    dd.mean_vp,    dd.probe_r_vp,    dd.std_vp,    prio_std_vp,    "steelblue",   "δvp (km/s)"),
    ("vs_IC", dd.r_vs_IC, dd.mean_vs_IC, dd.probe_r_vs_IC, dd.std_vs_IC, prio_std_vs_IC, "forestgreen", "δvs IC (km/s)"),
    ("vs_M",  dd.r_vs_M,  dd.mean_vs_M,  dd.probe_r_vs_M,  dd.std_vs_M,  prio_std_vs_M,  "seagreen",    "δvs M (km/s)"),
    ("rho",   dd.r_rho,   dd.mean_rho,   dd.probe_r_rho,   dd.std_rho,   prio_std_rho,   "goldenrod",   "δρ (g/cm³)"),
]

for ax, (pname, r_grid, mu, pr, pstd, pristd, col, ylabel) in zip(axes[:4], configs):
    ax.plot(r_grid, mu, color=col, lw=1.5, label="post mean")
    # Find closest grid index for each probe point
    idx = np.array([np.argmin(np.abs(r_grid - r)) for r in pr])
    ax.fill_between(pr, mu[idx] - pstd, mu[idx] + pstd, alpha=0.35, color=col, label="±1σ post")
    ax.fill_between(pr, mu[idx] - pristd, mu[idx] + pristd, alpha=0.12, color="orange", label="±1σ prior")
    ax.axvline(ICB, color="gray", ls="--", lw=0.7)
    ax.axvline(CMB, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("r (km)", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(pname, fontsize=9); ax.legend(fontsize=6)

ax = axes[4]
zr = (pred_pg - d) / safe_err
ax.hist(zr, bins=min(30, N_d//2+1), color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", lw=1)
for z in [-2,-1,1,2]: ax.axvline(z, color="orange", ls="--", lw=0.7, alpha=0.7)
ax.set_xlabel("(Gμ−d)/σ", fontsize=8); ax.set_ylabel("count", fontsize=8)
ax.set_title(f"Residuals\nchi_rms={chi_rms_pg:.3f}", fontsize=9)

plt.tight_layout()
fig.savefig(str(out_figs / "baseline_block_0_0.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[phase1] figure saved")

print("\n" + "="*65)
print("PHASE 1 SUMMARY")
print("="*65)
print(f"  Block ({target_block.s},{target_block.t}) | N_d={N_d} | n_basis={n_basis}")
print(f"  cond(N)={cond_N:.3e}  rank(1e-12)={rank_1e12}/{N_d}")
print(f"  chi_rms explicit={chi_rms_ex:.4f}  pygeoinf={chi_rms_pg:.4f}  agree={'YES' if solver_agree else 'WARNING NO'}")
print(f"  Contraction vp: mean={cont_vp[0]:.4f} max={cont_vp[1]:.4f}")
print(f"  Contraction vsM: mean={cont_vs_M[0]:.4f} max={cont_vs_M[1]:.4f}")
print(f"  prior std vp: mean={prio_std_vp.mean():.4g}  post std vp: mean={dd.std_vp.mean():.4g}")
print(f"  vp amplitude range: {metrics['mean_vp_range']}")
print("="*65)
print("[phase1] COMPLETE")
