#!/usr/bin/env python3
"""
analyze_full_inference.py
=========================

Load the property-posterior NPZ produced by the full all-block Bayesian
inference (``prior_posterior_tuner_remote.py --task inference``) and report
whether the results are physically sensible.

The NPZ contains:
    mu_P    : (N_p,)        property posterior mean
    sigma_P : (N_p,)        property posterior 1-sigma
    corr_P  : (N_p, N_p)    property posterior correlation matrix
    names   : (N_p,)        human-readable target labels

Targets (see target_kernel_viz._make_default_targets) — units:
    BulkTarget("vp"/"vs", ...) : relative velocity perturbation delta v / v
    CMBTarget(...)             : CMB topography (model native units)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "utils"))
sys.path.insert(0, str(_HERE / "visualization"))


def main() -> None:
    npz_path = _HERE / "tuner_inference_result.npz"
    if not npz_path.exists():
        raise SystemExit(f"Result not found: {npz_path}")

    data = np.load(str(npz_path), allow_pickle=False)
    mu = np.asarray(data["mu_P"], dtype=float)
    sigma = np.asarray(data["sigma_P"], dtype=float)
    corr = np.asarray(data["corr_P"], dtype=float)
    names = [str(n) for n in data["names"]]
    N_p = len(mu)

    print("=" * 78)
    print(f"FULL INFERENCE — property posterior  ({N_p} targets)")
    print("=" * 78)

    # ── 1. Per-target mean ± sigma, with SNR ────────────────────────────────
    print("\n[1] Posterior mean ± 1σ per target")
    print("-" * 78)
    for i in range(N_p):
        snr = abs(mu[i]) / sigma[i] if sigma[i] > 0 else np.inf
        flag = "  <-- |mean| > 1σ (resolved)" if snr >= 1.0 else ""
        print(f"  [{i}] {names[i]}")
        print(f"        mean = {mu[i]:+.4e}   σ = {sigma[i]:.4e}   "
              f"|mean|/σ = {snr:5.2f}{flag}")

    # ── 2. Sanity checks ────────────────────────────────────────────────────
    print("\n[2] Sanity checks")
    print("-" * 78)
    ok = True

    if np.all(sigma > 0):
        print("  ✓ all posterior σ strictly positive")
    else:
        ok = False
        print(f"  ✗ non-positive σ at indices {np.where(sigma <= 0)[0].tolist()}")

    if np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)):
        print("  ✓ all means and σ finite")
    else:
        ok = False
        print("  ✗ non-finite entries present")

    diag = np.diag(corr)
    if np.allclose(diag, 1.0, atol=1e-6):
        print("  ✓ correlation diagonal == 1")
    else:
        ok = False
        print(f"  ✗ correlation diagonal off: {diag}")

    if np.all(np.abs(corr) <= 1.0 + 1e-8):
        print("  ✓ all |correlations| ≤ 1")
    else:
        ok = False
        print("  ✗ correlation entries exceed 1 in magnitude")

    sym_err = np.max(np.abs(corr - corr.T))
    print(f"  • correlation symmetry max|C-Cᵀ| = {sym_err:.2e}")

    # PSD check on reconstructed covariance
    C = (sigma[:, None] * sigma[None, :]) * corr
    eigs = np.linalg.eigvalsh(0.5 * (C + C.T))
    min_eig = float(eigs.min())
    if min_eig >= -1e-10 * max(1.0, abs(eigs.max())):
        print(f"  ✓ covariance PSD (min eig = {min_eig:.2e})")
    else:
        ok = False
        print(f"  ✗ covariance not PSD (min eig = {min_eig:.2e})")

    # ── 3. Magnitude plausibility ───────────────────────────────────────────
    print("\n[3] Physical-magnitude plausibility")
    print("-" * 78)
    is_cmb = np.array(["CMB" in n for n in names])
    bulk = ~is_cmb
    if bulk.any():
        print(f"  bulk velocity targets (δv/v, dimensionless):")
        print(f"    |mean| range : {np.abs(mu[bulk]).min():.2e} … "
              f"{np.abs(mu[bulk]).max():.2e}")
        print(f"    σ range      : {sigma[bulk].min():.2e} … "
              f"{sigma[bulk].max():.2e}")
        big = np.where(bulk & (np.abs(mu) > 0.1))[0]
        if big.size:
            print(f"    ⚠ |δv/v| > 10% at indices {big.tolist()} "
                  f"(unusually large for global modes)")
        else:
            print(f"    ✓ all |δv/v| ≤ 10% (consistent with seismic perturbations)")
    if is_cmb.any():
        print(f"  CMB topography targets (model units):")
        print(f"    mean range : {mu[is_cmb].min():+.2e} … {mu[is_cmb].max():+.2e}")
        print(f"    σ range    : {sigma[is_cmb].min():.2e} … {sigma[is_cmb].max():.2e}")

    # ── 4. Notable correlations ─────────────────────────────────────────────
    print("\n[4] Strongest off-diagonal correlations")
    print("-" * 78)
    pairs = []
    for i in range(N_p):
        for j in range(i + 1, N_p):
            pairs.append((abs(corr[i, j]), corr[i, j], i, j))
    pairs.sort(reverse=True)
    for absc, c, i, j in pairs[:8]:
        print(f"  r = {c:+.3f}   [{i}] vs [{j}]")
        print(f"             {names[i]}")
        print(f"             {names[j]}")

    # ── 5. Summary figure ───────────────────────────────────────────────────
    print("\n[5] Writing summary figure")
    print("-" * 78)
    import matplotlib
    matplotlib.use("Agg")
    from full_spectrum_viz import plot_property_posterior_summary
    fig = plot_property_posterior_summary(mu, sigma, corr, names)
    out_png = _HERE / "full_inference_property_posterior.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"  saved → {out_png}")

    print("\n" + "=" * 78)
    print("OVERALL: " + ("✓ results pass basic sanity checks"
                         if ok else "✗ SOME CHECKS FAILED — inspect above"))
    print("=" * 78)


if __name__ == "__main__":
    main()
