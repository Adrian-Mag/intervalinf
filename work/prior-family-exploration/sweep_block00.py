#!/usr/bin/env python3
"""
sweep_block00.py
================

Prior-family sweep for block (s=0, t=0).

Runs the Bayesian inversion with:
  - baseline Bessel prior (from tuner_params.json)
  - 3 Bessel-family candidates (loose, moderate)
  - 1 power-law candidate
  - 1 mixture candidate (0.7 smooth + 0.3 rough)

Prints a comparison table of data-space diagnostics:
  chi_rms, rank_eff_GCGt, trace_GCGt, median_information_ratio,
  n_directions_gt_95, max_contraction.

Usage
-----
    conda activate inferences
    cd intervalinf/work/prior-family-exploration
    python sweep_block00.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path for the utils packages
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DEMOS = _HERE.parent.parent / "demos" / "old_demos" / "paper_demos"
sys.path.insert(0, str(_DEMOS / "utils"))

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from full_spectrum_utils import (  # noqa: E402
    RadialSpecs,
    block_data_split,
    build_block_forward,
    build_block_prior,
    enumerate_blocks,
    solve_block,
)
from normal_mode_kernel_utils import (  # noqa: E402
    NormalModeDataRegistry,
    NormalModeKernelCatalog,
)
from prior_family_exploration import (  # noqa: E402
    DataSpaceCoverage,
    build_block_prior_from_component_covariances,
    build_radial_component_spaces,
    data_space_coverage_from_matrices,
    make_bessel_covariance,
)
from prior_posterior_tuner import _build_custom_bessel_blocks  # noqa: E402

from intervalinf import ParallelConfig  # noqa: E402
from pygeoinf import LinearForwardProblem, LinearBayesianInversion  # noqa: E402
from pygeoinf.linear_solvers import LUSolver  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = _DEMOS / "data" / "normal-mode-data"
_KERNEL_DIR = _DEMOS / "data" / "normal-mode-kernels" / "kernels-all_PREM-layers_Adrian"
_PARAMS_JSON = _DEMOS / "tuner_params.json"
_CANDIDATES_JSON = _HERE / "configs" / "function_space_prior_candidates.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chi_rms(G_st, posterior, d_st, C_D_diag: np.ndarray) -> float:
    """Normalized data residual for the posterior mean."""
    d_pred = np.asarray(G_st(posterior.expectation), dtype=float)
    residual = np.asarray(d_st, dtype=float) - d_pred
    N_d = residual.size
    return float(np.sqrt(np.sum(residual ** 2 / C_D_diag) / N_d))


def _data_space_diagnostics(
    bayes: LinearBayesianInversion,
    C_D_diag: np.ndarray,
) -> tuple[DataSpaceCoverage, float]:
    """Materialize N = G C G* + C_D once; return (coverage, trace_GCGt)."""
    print("    materializing normal matrix…", flush=True)
    N_dense = bayes.normal_operator.matrix(dense=True)
    C_D_matrix = np.diag(C_D_diag)
    GCGt = N_dense - C_D_matrix
    coverage = data_space_coverage_from_matrices(GCGt, C_D_matrix)
    trace_val = float(np.trace(GCGt))
    return coverage, trace_val


# ---------------------------------------------------------------------------
# Prior builders
# ---------------------------------------------------------------------------

def _build_bessel_prior(specs: RadialSpecs, hyper: dict, taus: dict, sigma_var: float):
    """Build a Bessel-Sobolev block prior from hyperparameter dicts."""
    shared_bessel = _build_custom_bessel_blocks(specs, hyper)
    tau_fn = lambda p, _s: taus.get(p, 1.0)
    return build_block_prior(0, 0, shared_bessel, specs, tau_fn=tau_fn, sigma_var=sigma_var)


def _build_power_law_prior(
    specs: RadialSpecs,
    components_cfg: dict,
    sigma_var: float,
) -> "GaussianMeasure":
    """Build a power-law block prior via BesselSobolevInverse with fractional s."""
    spaces = build_radial_component_spaces(specs)
    component_covariances = {}
    for p, space in spaces.items():
        cfg = components_cfg[p]
        component_covariances[p] = make_bessel_covariance(
            space,
            s_order=float(cfg["decay_order"]),
            length=float(cfg["length"]),
            variance=float(cfg["variance"]),
            bc=cfg["bc"],
            dofs=specs.n_basis,
        )
    return build_block_prior_from_component_covariances(
        0, component_covariances, specs, sigma_var=sigma_var
    )


def _build_mixture_prior(
    specs: RadialSpecs,
    smooth_hyper: dict,
    rough_hyper: dict,
    weights: dict,
    sigma_var: float,
) -> "GaussianMeasure":
    """Build a 0.7-smooth + 0.3-rough mixture Bessel prior."""
    smooth_ops = _build_custom_bessel_blocks(specs, smooth_hyper)
    rough_ops = _build_custom_bessel_blocks(specs, rough_hyper)
    spaces = build_radial_component_spaces(specs)
    w_s = float(weights["smooth"])
    w_r = float(weights["rough"])
    component_covariances = {}
    for p, space in spaces.items():
        mix = w_s * smooth_ops[p] + w_r * rough_ops[p]
        component_covariances[p] = mix
    return build_block_prior_from_component_covariances(
        0, component_covariances, specs, sigma_var=sigma_var
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load params ──────────────────────────────────────────────────────────
    with open(_PARAMS_JSON) as fh:
        params = json.load(fh)

    with open(_CANDIDATES_JSON) as fh:
        candidates_json = json.load(fh)

    baseline_hyper = params["hyper"]
    baseline_taus = params["taus"]
    sigma_var = float(params["sigma_var"])
    n_basis = int(params["n_basis"])
    n_jobs = 4  # reasonable local parallelism

    print(f"n_basis={n_basis}, n_jobs={n_jobs}", flush=True)

    # ── Load catalog + registry ──────────────────────────────────────────────
    print("Loading catalog…", flush=True)
    catalog = NormalModeKernelCatalog(str(_KERNEL_DIR))
    print("Loading registry…", flush=True)
    reg = NormalModeDataRegistry(str(_DATA_DIR), mode_filter=catalog.list_modes())

    # ── Build specs + forward for block (0,0) ────────────────────────────────
    parallel_cfg = ParallelConfig(enabled=n_jobs > 1, n_jobs=n_jobs)
    specs = RadialSpecs(n_basis=n_basis, parallel_cfg=parallel_cfg)
    all_blocks = enumerate_blocks(reg, s_max=int(params["s_max"]))
    split = block_data_split(reg, all_blocks)

    target = next(b for b in all_blocks if b.s == 0 and b.t == 0)
    print(f"Target block: {target}", flush=True)

    print("Building forward operator…", flush=True)
    t0 = time.time()
    G_st, C_D_st, M_st, D_st = build_block_forward(0, 0, split[target], catalog, specs)
    d_st = split[target].data_vector
    C_D_diag = split[target].covariance_diagonal
    N_d = len(d_st)
    print(f"  forward built in {time.time()-t0:.1f}s, N_d={N_d}", flush=True)

    # ── Define candidate list ─────────────────────────────────────────────────
    candidates = [
        ("baseline",      "bessel", baseline_hyper, baseline_taus),
        ("loose_bessel",  "bessel", {
            "vp":    {"s_order": 2.0, "length": 80.0,  "var": 1000.0, "bc": "mixed_neumann_dirichlet"},
            "vs_IC": {"s_order": 2.0, "length": 45.0,  "var": 1000.0, "bc": "neumann"},
            "vs_M":  {"s_order": 2.0, "length": 60.0,  "var": 50.0,   "bc": "mixed_neumann_dirichlet"},
            "rho":   {"s_order": 2.0, "length": 80.0,  "var": 50.0,   "bc": "mixed_neumann_dirichlet"},
        }, {p: 1.0 for p in ["vp", "vs_IC", "vs_M", "rho"]}),
        ("moderate_bessel", "bessel", {
            "vp":    {"s_order": 3.0, "length": 105.0, "var": 1000.0, "bc": "mixed_neumann_dirichlet"},
            "vs_IC": {"s_order": 3.0, "length": 58.0,  "var": 1000.0, "bc": "neumann"},
            "vs_M":  {"s_order": 3.0, "length": 70.0,  "var": 50.0,   "bc": "mixed_neumann_dirichlet"},
            "rho":   {"s_order": 3.0, "length": 100.0, "var": 50.0,   "bc": "mixed_neumann_dirichlet"},
        }, {p: 1.0 for p in ["vp", "vs_IC", "vs_M", "rho"]}),
        ("power_law_rough", "power_law", {
            "vp":    {"decay_order": 1.2, "length": 80.0,  "variance": 1000.0, "bc": "mixed_neumann_dirichlet"},
            "vs_IC": {"decay_order": 1.2, "length": 45.0,  "variance": 1000.0, "bc": "neumann"},
            "vs_M":  {"decay_order": 1.2, "length": 60.0,  "variance": 50.0,   "bc": "mixed_neumann_dirichlet"},
            "rho":   {"decay_order": 1.2, "length": 80.0,  "variance": 50.0,   "bc": "mixed_neumann_dirichlet"},
        }, None),
        ("mixture_0.7s+0.3r", "mixture", None, None),
    ]

    loose_hyper = {
        "vp":    {"s_order": 2.0, "length": 80.0,  "var": 1000.0, "bc": "mixed_neumann_dirichlet"},
        "vs_IC": {"s_order": 2.0, "length": 45.0,  "var": 1000.0, "bc": "neumann"},
        "vs_M":  {"s_order": 2.0, "length": 60.0,  "var": 50.0,   "bc": "mixed_neumann_dirichlet"},
        "rho":   {"s_order": 2.0, "length": 80.0,  "var": 50.0,   "bc": "mixed_neumann_dirichlet"},
    }

    rows = []

    for name, family, cfg, taus in candidates:
        print(f"\n── {name} ({family}) ──", flush=True)

        # ── Build prior ──────────────────────────────────────────────────────
        t0 = time.time()
        if family == "bessel":
            prior_st = _build_bessel_prior(specs, cfg, taus, sigma_var)
        elif family == "power_law":
            prior_st = _build_power_law_prior(specs, cfg, sigma_var)
        elif family == "mixture":
            prior_st = _build_mixture_prior(
                specs,
                smooth_hyper=baseline_hyper,
                rough_hyper=loose_hyper,
                weights={"smooth": 0.7, "rough": 0.3},
                sigma_var=sigma_var,
            )
        else:
            raise ValueError(f"unknown family: {family}")
        print(f"  prior built in {time.time()-t0:.1f}s", flush=True)

        # ── Build Bayesian inversion ─────────────────────────────────────────
        problem = LinearForwardProblem(G_st, data_error_measure=C_D_st)
        bayes = LinearBayesianInversion(problem, prior_st)

        # ── G C G* diagnostics (before solving) ─────────────────────────────
        t0 = time.time()
        coverage, trace_val = _data_space_diagnostics(bayes, C_D_diag)
        print(f"  diagnostics in {time.time()-t0:.1f}s", flush=True)

        # ── Solve posterior ──────────────────────────────────────────────────
        t0 = time.time()
        posterior_st = bayes.model_posterior_measure(d_st, LUSolver())
        print(f"  posterior solved in {time.time()-t0:.1f}s", flush=True)

        # ── chi_rms ──────────────────────────────────────────────────────────
        chi = _chi_rms(G_st, posterior_st, d_st, C_D_diag)

        rows.append({
            "name": name,
            "family": family,
            "N_d": N_d,
            "chi_rms": chi,
            "rank_eff_GCGt": coverage.effective_rank,
            "trace_GCGt": trace_val,
            "median_info": coverage.median_information_ratio,
            "n_gt_95": coverage.n_directions_contraction_gt_95,
            "n_gt_99": coverage.n_directions_contraction_gt_99,
            "max_contraction": coverage.max_contraction,
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"Block (0,0)  |  N_d = {N_d}  |  n_basis = {n_basis}")
    print("=" * 100)
    header = (
        f"{'Prior':<25} {'chi_rms':>8} {'rank_eff':>9} {'trace_GCGt':>12} "
        f"{'med_info':>9} {'n>95%':>6} {'n>99%':>6} {'max_contr':>10}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{r['name']:<25} {r['chi_rms']:8.3f} {r['rank_eff_GCGt']:9d} "
            f"{r['trace_GCGt']:12.1f} {r['median_info']:9.3f} "
            f"{r['n_gt_95']:6d} {r['n_gt_99']:6d} {r['max_contraction']:10.4f}"
        )
    print("=" * 100)

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = _HERE / "results" / "block00_sweep.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
