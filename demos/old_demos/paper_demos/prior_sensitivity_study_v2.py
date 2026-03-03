#!/usr/bin/env python
"""
Prior Sensitivity Study (v2 scaffold)
=====================================
Phase 1 implementation:
- USER CONFIG block with FAST_MODE and case controls
- Fast and realistic problem builders
- Shared figure generation

Inference loop, per-case priors, and persistence are implemented in later phases.
"""

# ruff: noqa: E221
# noqa: E221

import csv
import datetime
import os
import sys
import time
import warnings
from collections import namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None

# ensure kernel_utils is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# intervalinf imports
from intervalinf import (
    BoundaryConditions,
    Function,
    IntegrationConfig,
    IntervalDomain,
    KnownRegion,
    Lebesgue,
    LebesgueIntegrationConfig,
    LebesgueSpaceDirectSum,
    ParallelConfig,
    PartitionedLebesgueSpace,
)
from intervalinf.operators import BesselSobolevInverse, Laplacian, SOLAOperator
from intervalinf.providers import BumpFunctionProvider, NullFunctionProvider
from intervalinf.sampling import KLSampler
from kernel_utils import EARTH_RADIUS_KM, SensitivityKernelCatalog, SensitivityKernelProvider

# pygeoinf imports
from pygeoinf import (
    CholeskySolver,
    EuclideanSpace,
    GaussianMeasure,
    HilbertSpaceDirectSum,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
    RowLinearOperator,
)


ProblemResult = namedtuple(
    "ProblemResult",
    [
        "M_model",
        "D",
        "P",
        "G",
        "T",
        "m_bar",
        "d_tilde",
        "d_bar",
        "true_props",
        "noise_variance",
        "function_domain",
        "catalog",
        "extras",
    ],
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIGURATION                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

FAST_MODE = True                    # True = vp-only (~30-60 s/case)
                                    # False = full 5-component (~3-4 min/case)

CASES_TO_RUN = ["all"]              # ["all"] or subset e.g. ["Reference", "R+", "M-"]

PRIOR_PREDICTIVE_CHI2_THRESHOLD = 3.0

NORMALISE_R_VARIANCE = False
CALIBRATE_PRIOR_CASES = False      # True = map offsets to property-space σ and calibrate runtime fields
                                  # False = use raw case settings (offset_sigma interpreted as model offset)

PREVIEW_PRIORS_ONLY = False         # True = generate prior previews and exit
PREVIEW_CASES = ["all"]            # ["all"] or subset e.g. ["Reference", "R+"]
PREVIEW_N_SAMPLES = 10              # set >0 to draw prior samples on quick plots

N = 100
N_d = 140
N_p = 20
noise_level_fraction = 0.1
OVERALL_VARIANCE_BASE = 10 ** 1.5

# ── Integration configs ──
Lebesgue_integration_cfg = LebesgueIntegrationConfig(
    inner_product=IntegrationConfig(method="trapz", n_points=1024),
    dual=IntegrationConfig(method="trapz", n_points=1024),
    general=IntegrationConfig(method="trapz", n_points=1024),
)
laplacian_integration_cfg = IntegrationConfig(method="trapz", n_points=1024)
sola_integration_cfg = IntegrationConfig(method="trapz", n_points=2048)
bessel_sobolev_integration_cfg = IntegrationConfig(method="trapz", n_points=2048)

# ── Parallelisation ──
parallel_cfg = ParallelConfig(enabled=True, n_jobs=12)

# ── Prior case definitions (expanded in later phases) ──
PRIOR_CASES = {
    "Reference": {"s": 6.0, "ls": 30, "ov_mult": 1.0, "offset_sigma": 0.0},
    "R+":        {"s": 12.0, "ls": 30, "ov_mult": 2.0, "offset_sigma": 0.0},
    "R-":        {"s": 3.0, "ls": 10, "ov_mult": 0.02, "offset_sigma": 0.0},
    "V+":        {"s": 6.0, "ls": 30, "ov_mult": 8.0, "offset_sigma": 0.0},
    "V-":        {"s": 6.0, "ls": 30, "ov_mult": 0.05, "offset_sigma": 0.0},
    "M+":        {"s": 6.0, "ls": 30, "ov_mult": 1.0, "offset_sigma": +0.1},
    "M-":        {"s": 6.0, "ls": 30, "ov_mult": 1.0, "offset_sigma": -0.1},
}

CASE_COLORS = {
    "Reference": "tab:blue",
    "R+": "tab:green",
    "R-": "tab:orange",
    "V+": "tab:purple",
    "V-": "tab:pink",
    "M+": "tab:red",
    "M-": "tab:brown",
}


FIGURES_FOLDER = "prior_sensitivity_fast" if FAST_MODE else "prior_sensitivity"
OUTPUT_CSV = os.path.join(FIGURES_FOLDER, "prior_sensitivity_summary.csv")
OUTPUT_MD = os.path.join(FIGURES_FOLDER, "prior_sensitivity_summary.md")
OUTPUT_TXT = os.path.join(FIGURES_FOLDER, "prior_sensitivity_detailed.txt")
os.makedirs(FIGURES_FOLDER, exist_ok=True)


RESULT_FIELDNAMES = [
    "case",
    "s",
    "length_scale",
    "overall_variance",
    "mean_offset",
    "rmse",
    "mae",
    "coverage_2sigma",
    "mean_post_std",
    "mean_prior_std",
    "unc_reduction",
    "data_misfit",
    "min_eig_cov",
    "chi2_per_dof",
    "elapsed_s",
]


def _save_fig(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, name)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _set_plot_theme() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", palette="muted", color_codes=True)


def _despine() -> None:
    if sns is not None:
        sns.despine()


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_float(value, fmt):
    if value is None:
        return "-"
    try:
        if np.isnan(value):
            return "-"
    except TypeError:
        pass
    return format(value, fmt)


def load_existing_results(path):
    """Load cached summary rows from CSV if present."""
    results = {}
    if Path(path).exists():
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = row.get("case")
                if case:
                    results[case] = row
        print(f"Loaded {len(results)} existing results from {path}")
    return results


def write_summary_outputs(results, cases_to_run, log_lines):
    """Write merged summary outputs: CSV, Markdown, and append TXT log."""
    ordered_results = [results[k] for k in PRIOR_CASES if k in results]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in ordered_results:
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDNAMES})
    print(f"Saved CSV  → {OUTPUT_CSV}")

    md_headers = [
        "Case",
        "s",
        "ls",
        "ov",
        "offset",
        "RMSE",
        "MAE",
        "Coverage(2σ)",
        "Mean σ_post",
        "Mean σ_prior",
        "Unc. Reduction",
        "Data Misfit",
        "Min Eig.",
        "chi2/dof",
        "Time (s)",
    ]
    md_keys = RESULT_FIELDNAMES
    with open(OUTPUT_MD, "w") as f:
        f.write("# Prior Sensitivity Study — Summary\n\n")
        f.write("| " + " | ".join(md_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(md_headers)) + " |\n")
        for row in ordered_results:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in md_keys) + " |\n")
    print(f"Saved MD   → {OUTPUT_MD}")

    with open(OUTPUT_TXT, "a") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"Run at {datetime.datetime.now().isoformat()}\n")
        f.write(f"Cases: {cases_to_run}\n")
        f.write(f"{'=' * 70}\n\n")
        for line in log_lines:
            f.write(line + "\n")
    print(f"Appended TXT → {OUTPUT_TXT}")


def prior_predictive_check(M_prior, G, d_tilde, C_D_matrix, parallel_cfg):
    """Compute normalised prior-predictive Mahalanobis distance chi2/dof."""
    mu_0 = M_prior.expectation
    d_prior = np.asarray(G(mu_0)).ravel()
    d_obs = np.asarray(d_tilde).ravel()
    n_d = len(d_obs)

    prior_D = M_prior.affine_mapping(operator=G)
    S = prior_D.covariance.matrix(
        dense=True,
        parallel=True,
        n_jobs=parallel_cfg.n_jobs,
    )
    S = S + C_D_matrix

    residual = d_obs - d_prior
    chi2 = float(residual @ np.linalg.solve(S, residual))
    return chi2 / n_d


def _default_laplace_kwargs() -> dict:
    return dict(
        method="spectral",
        dofs=100,
        integration_config=laplacian_integration_cfg,
        n_samples=2048,
    )


def _default_bessel_kwargs() -> dict:
    return dict(
        dofs=512,
        n_samples=2048,
        use_fast_transforms=True,
        integration_config=bessel_sobolev_integration_cfg,
    )


def build_vp_prior(
    M_vp,
    s,
    length_scale,
    overall_variance,
    mean_offset_model=0.0,
    n_kl=100,
    bcs_vp=None,
    lap_kwargs=None,
    bs_kwargs=None,
):
    """Build vp prior measure with explicit variance and mean offset."""
    if bcs_vp is None:
        bcs_vp = BoundaryConditions(bc_type="mixed_neumann_dirichlet")
    if lap_kwargs is None:
        lap_kwargs = _default_laplace_kwargs()
    if bs_kwargs is None:
        bs_kwargs = _default_bessel_kwargs()

    k = overall_variance ** (-0.5 / s)
    alpha = (length_scale ** 2) * (k ** 2)
    L = Laplacian(M_vp, bcs_vp, alpha, **lap_kwargs)
    C = BesselSobolevInverse(M_vp, M_vp, k, s, L, **bs_kwargs)

    m_0 = Function(
        M_vp,
        evaluate_callable=lambda x, _off=mean_offset_model: _off * np.ones_like(x),
    )
    sampler = KLSampler(C, mean=m_0, n_modes=n_kl)
    prior = GaussianMeasure(covariance=C, expectation=m_0, sample=sampler.sample)
    return prior, sampler


def build_fixed_realistic_priors(problem: ProblemResult) -> dict:
    """Build fixed vs/rho/sigma priors used in realistic mode."""
    M_vs_IC = problem.extras["M_vs_IC"]
    M_vs_M = problem.extras["M_vs_M"]
    M_rho = problem.extras["M_rho"]

    bcs_vs_IC = BoundaryConditions(bc_type="neumann")
    bcs_vs_M = BoundaryConditions(bc_type="mixed_neumann_dirichlet")
    bcs_rho = BoundaryConditions(bc_type="mixed_neumann_dirichlet")
    lap_kwargs = _default_laplace_kwargs()
    bs_kwargs = _default_bessel_kwargs()

    s_vs, ls_vs, ov_vs = 6.0, 30, 10 ** 3
    k_vs = ov_vs ** (-0.5 / s_vs)
    alpha_vs = (ls_vs ** 2) * (k_vs ** 2)
    L_vs_IC = Laplacian(M_vs_IC, bcs_vs_IC, alpha_vs, **lap_kwargs)
    L_vs_M = Laplacian(M_vs_M, bcs_vs_M, alpha_vs, **lap_kwargs)
    C_0_vs_IC = BesselSobolevInverse(M_vs_IC, M_vs_IC, k_vs, s_vs, L_vs_IC, **bs_kwargs)
    C_0_vs_M = BesselSobolevInverse(M_vs_M, M_vs_M, k_vs, s_vs, L_vs_M, **bs_kwargs)

    m_0_vs_IC = Function(M_vs_IC, evaluate_callable=lambda x: np.zeros_like(x))
    m_0_vs_M = Function(M_vs_M, evaluate_callable=lambda x: np.zeros_like(x))
    sampler_vs_IC = KLSampler(C_0_vs_IC, mean=m_0_vs_IC, n_modes=100)
    sampler_vs_M = KLSampler(C_0_vs_M, mean=m_0_vs_M, n_modes=100)
    M_prior_vs_IC = GaussianMeasure(
        covariance=C_0_vs_IC,
        expectation=m_0_vs_IC,
        sample=sampler_vs_IC.sample,
    )
    M_prior_vs_M = GaussianMeasure(
        covariance=C_0_vs_M,
        expectation=m_0_vs_M,
        sample=sampler_vs_M.sample,
    )
    M_prior_vs = GaussianMeasure.from_direct_sum([M_prior_vs_IC, M_prior_vs_M])

    s_rho, ls_rho, ov_rho = 6.0, 30, 10 ** 2
    k_rho = ov_rho ** (-0.5 / s_rho)
    alpha_rho = (ls_rho ** 2) * (k_rho ** 2)
    L_rho = Laplacian(M_rho, bcs_rho, alpha_rho, **lap_kwargs)
    C_0_rho = BesselSobolevInverse(M_rho, M_rho, k_rho, s_rho, L_rho, **bs_kwargs)
    m_0_rho = Function(M_rho, evaluate_callable=lambda x: np.zeros_like(x))
    sampler_rho = KLSampler(C_0_rho, mean=m_0_rho, n_modes=100)
    M_prior_rho = GaussianMeasure(
        covariance=C_0_rho,
        expectation=m_0_rho,
        sample=sampler_rho.sample,
    )

    M_prior_sigma_0 = GaussianMeasure.from_covariance_matrix(
        problem.extras["M_sigma_0"],
        np.array([[10]]),
        expectation=np.array([0.0]),
    )
    M_prior_sigma_1 = GaussianMeasure.from_covariance_matrix(
        problem.extras["M_sigma_1"],
        np.array([[10]]),
        expectation=np.array([0.0]),
    )

    return {
        "M_prior_vs": M_prior_vs,
        "M_prior_rho": M_prior_rho,
        "M_prior_sigma_0": M_prior_sigma_0,
        "M_prior_sigma_1": M_prior_sigma_1,
        "sampler_vs_IC": sampler_vs_IC,
        "sampler_vs_M": sampler_vs_M,
        "sampler_rho": sampler_rho,
    }


def build_full_prior(M_prior_vp, fixed_priors, fast_mode):
    """Return full prior for current mode."""
    if fast_mode:
        return M_prior_vp

    M_prior_functions = GaussianMeasure.from_direct_sum(
        [
            M_prior_vp,
            fixed_priors["M_prior_vs"],
            fixed_priors["M_prior_rho"],
        ]
    )
    M_prior_euclidean = GaussianMeasure.from_direct_sum(
        [
            fixed_priors["M_prior_sigma_0"],
            fixed_priors["M_prior_sigma_1"],
        ]
    )
    return GaussianMeasure.from_direct_sum([M_prior_functions, M_prior_euclidean])


def normalise_ov_for_property_std(
    target_std,
    s,
    ls,
    M_vp,
    T,
    bcs_vp,
    lap_kwargs,
    bs_kwargs,
    fixed_priors,
    fast_mode,
    tol=0.01,
    max_iter=20,
):
    """Bisect overall_variance so mean property-space std matches target."""
    ov_lo, ov_hi = 1e-2, 1e6
    ov_mid = OVERALL_VARIANCE_BASE

    for _ in range(max_iter):
        ov_mid = np.sqrt(ov_lo * ov_hi)
        prior_vp, _ = build_vp_prior(
            M_vp,
            s,
            ls,
            ov_mid,
            0.0,
            bcs_vp=bcs_vp,
            lap_kwargs=lap_kwargs,
            bs_kwargs=bs_kwargs,
        )
        full_prior = build_full_prior(prior_vp, fixed_priors, fast_mode)
        prior_P = full_prior.affine_mapping(operator=T)
        cov = prior_P.covariance.matrix(
            dense=True,
            parallel=True,
            n_jobs=parallel_cfg.n_jobs,
        )
        current_std = float(np.mean(np.sqrt(np.diag(cov))))

        rel_err = abs(current_std - target_std) / target_std
        if rel_err < tol:
            return ov_mid

        if current_std > target_std:
            ov_hi = ov_mid
        else:
            ov_lo = ov_mid

    warnings.warn(f"Bisection did not converge; using ov={ov_mid:.4f}")
    return ov_mid


def calibrate_prior_cases(problem: ProblemResult, fixed_priors, fast_mode, calibrate=True):
    """Compute runtime fields for PRIOR_CASES and return calibration stats."""
    M_vp = problem.extras["M_vp"]
    T = problem.T
    bcs_vp = BoundaryConditions(bc_type="mixed_neumann_dirichlet")
    lap_kwargs = _default_laplace_kwargs()
    bs_kwargs = _default_bessel_kwargs()

    if not calibrate:
        for _, cfg in PRIOR_CASES.items():
            cfg["overall_variance"] = OVERALL_VARIANCE_BASE * cfg["ov_mult"]
            cfg["mean_offset_model"] = float(cfg["offset_sigma"])

        return {
            "REF_PROP_STD": None,
            "T_mean_norm": None,
            "bcs_vp": bcs_vp,
            "lap_kwargs": lap_kwargs,
            "bs_kwargs": bs_kwargs,
            "calibration_enabled": False,
        }

    prior_ref_vp, _ = build_vp_prior(
        M_vp,
        s=6.0,
        length_scale=30,
        overall_variance=OVERALL_VARIANCE_BASE,
        mean_offset_model=0.0,
        bcs_vp=bcs_vp,
        lap_kwargs=lap_kwargs,
        bs_kwargs=bs_kwargs,
    )
    full_prior_ref = build_full_prior(prior_ref_vp, fixed_priors, fast_mode)

    prior_P_ref = full_prior_ref.affine_mapping(operator=T)
    cov_P_ref = prior_P_ref.covariance.matrix(
        dense=True,
        parallel=True,
        n_jobs=parallel_cfg.n_jobs,
    )
    REF_PROP_STD = float(np.mean(np.sqrt(np.diag(cov_P_ref))))

    T_mat = T.matrix(dense=True, parallel=True, n_jobs=parallel_cfg.n_jobs)
    T_row_norms = np.sqrt(np.sum(T_mat ** 2, axis=1))
    T_mean_norm = float(np.mean(T_row_norms))

    for _, cfg in PRIOR_CASES.items():
        cfg["overall_variance"] = OVERALL_VARIANCE_BASE * cfg["ov_mult"]
        if cfg["offset_sigma"] != 0.0:
            cfg["mean_offset_model"] = (
                cfg["offset_sigma"] * REF_PROP_STD / T_mean_norm
            )
        else:
            cfg["mean_offset_model"] = 0.0

    if NORMALISE_R_VARIANCE:
        for label in ("R+", "R-"):
            cfg = PRIOR_CASES[label]
            cfg["overall_variance"] = normalise_ov_for_property_std(
                target_std=REF_PROP_STD,
                s=cfg["s"],
                ls=cfg["ls"],
                M_vp=M_vp,
                T=T,
                bcs_vp=bcs_vp,
                lap_kwargs=lap_kwargs,
                bs_kwargs=bs_kwargs,
                fixed_priors=fixed_priors,
                fast_mode=fast_mode,
            )

    return {
        "REF_PROP_STD": REF_PROP_STD,
        "T_mean_norm": T_mean_norm,
        "bcs_vp": bcs_vp,
        "lap_kwargs": lap_kwargs,
        "bs_kwargs": bs_kwargs,
        "calibration_enabled": True,
    }


def build_fast_problem() -> ProblemResult:
    """Build vp-only inversion problem using real kernel catalog."""
    function_domain = IntervalDomain(0, EARTH_RADIUS_KM)
    M_vp = Lebesgue(
        N,
        function_domain,
        basis="cosine",
        integration_config=Lebesgue_integration_cfg.inner_product,
        parallel_config=parallel_cfg,
    )

    D = EuclideanSpace(N_d)
    P = EuclideanSpace(N_p)

    data_dir = Path(_SCRIPT_DIR / "../kernels_modeplotaat_Adrian")
    catalog = SensitivityKernelCatalog(data_dir)

    vp_kernel_provider = SensitivityKernelProvider(
        M_vp,
        catalog,
        interpolation_method="cubic",
        include_discontinuities=True,
        kernel_type="vp",
    )

    G = SOLAOperator(
        M_vp,
        D,
        vp_kernel_provider,
        integration_config=sola_integration_cfg,
    )

    width = 0.2 * EARTH_RADIUS_KM
    centers = np.linspace(
        function_domain.a + width / 2,
        function_domain.b - width / 2,
        N_p,
    )
    target_provider = BumpFunctionProvider(
        M_vp,
        centers=centers,
        default_width=width,
    )
    T = SOLAOperator(
        M_vp,
        P,
        target_provider,
        integration_config=sola_integration_cfg,
    )

    ran_array_vp = np.zeros(N)
    ran_array_vp[:10] = np.random.RandomState(42).uniform(-1, 1, 10)
    m_bar = M_vp.from_components(ran_array_vp)

    d_bar = np.asarray(G(m_bar)).ravel()
    noise_std = noise_level_fraction * float(np.max(np.abs(d_bar)))
    noise = np.random.RandomState(42).normal(0.0, noise_std, size=d_bar.shape)
    d_tilde = d_bar + noise

    true_props = np.asarray(T(m_bar)).ravel()
    noise_variance = noise_std ** 2

    extras = {
        "mode": "fast",
        "M_vp": M_vp,
        "G_vp": G,
        "T_vp": T,
        "centers": centers,
        "width": width,
    }

    return ProblemResult(
        M_model=M_vp,
        D=D,
        P=P,
        G=G,
        T=T,
        m_bar=m_bar,
        d_tilde=d_tilde,
        d_bar=d_bar,
        true_props=true_props,
        noise_variance=noise_variance,
        function_domain=function_domain,
        catalog=catalog,
        extras=extras,
    )


def build_realistic_problem() -> ProblemResult:
    """Build full 5-component inversion problem (vp, vs, rho, sigma0, sigma1)."""
    function_domain = IntervalDomain(0, EARTH_RADIUS_KM)
    ICB_RADIUS = 1217.5
    CMB_RADIUS = 3480.0

    M_vp = Lebesgue(
        N,
        function_domain,
        basis="ND",
        integration_config=Lebesgue_integration_cfg.inner_product,
        parallel_config=parallel_cfg,
    )

    outer_core_interval = IntervalDomain(ICB_RADIUS, CMB_RADIUS)
    outer_core = KnownRegion.zero(outer_core_interval)
    partitioned_vs = PartitionedLebesgueSpace(
        full_domain=function_domain,
        known_regions=[outer_core],
        dims=[N, N],
        bases=["cosine", "ND"],
        integration_config=Lebesgue_integration_cfg.inner_product,
        parallel_config=parallel_cfg,
    )
    M_vs = partitioned_vs.model_space
    M_vs_IC = partitioned_vs.unknown_spaces[0]
    M_vs_M = partitioned_vs.unknown_spaces[1]

    M_rho = Lebesgue(
        N,
        function_domain,
        basis="ND",
        integration_config=Lebesgue_integration_cfg.inner_product,
        parallel_config=parallel_cfg,
    )

    M_sigma_0 = EuclideanSpace(1)
    M_sigma_1 = EuclideanSpace(1)

    M_functions = LebesgueSpaceDirectSum([M_vp, M_vs, M_rho])
    M_euclidean = HilbertSpaceDirectSum([M_sigma_0, M_sigma_1])
    M_model = HilbertSpaceDirectSum([M_functions, M_euclidean])

    D = EuclideanSpace(N_d)
    P = EuclideanSpace(N_p)

    width = 0.2 * EARTH_RADIUS_KM
    centers = np.linspace(
        function_domain.a + width / 2,
        function_domain.b - width / 2,
        N_p,
    )

    data_dir = Path(_SCRIPT_DIR / "../kernels_modeplotaat_Adrian")
    catalog = SensitivityKernelCatalog(data_dir)

    vp_kernel_provider = SensitivityKernelProvider(
        M_vp,
        catalog,
        interpolation_method="cubic",
        include_discontinuities=True,
        kernel_type="vp",
    )
    rho_kernel_provider = SensitivityKernelProvider(
        M_rho,
        catalog,
        interpolation_method="cubic",
        include_discontinuities=True,
        kernel_type="rho",
    )

    vs_full_space = Lebesgue(N, function_domain, basis="none")
    vs_kernel_full_provider = SensitivityKernelProvider(
        vs_full_space,
        catalog,
        interpolation_method="cubic",
        include_discontinuities=True,
        kernel_type="vs",
    )
    vs_kernel_IC_provider = vs_kernel_full_provider.restrict(M_vs_IC)
    vs_kernel_M_provider = vs_kernel_full_provider.restrict(M_vs_M)

    kernel_provider = vp_kernel_provider
    icb_depth, cmb_depth = 5153.5, 2891.0
    mode_ids = catalog.list_modes()[:N_d]
    icb_vals, cmb_vals = [], []
    for mid in mode_ids:
        topo = kernel_provider.get_topo_kernel(mid)
        if topo is not None:
            icb_v = topo.get_value_at_depth(icb_depth, tolerance=100.0)
            cmb_v = topo.get_value_at_depth(cmb_depth, tolerance=100.0)
        else:
            icb_v = None
            cmb_v = None
        icb_vals.append(icb_v if icb_v is not None else 0.0)
        cmb_vals.append(cmb_v if cmb_v is not None else 0.0)

    K_sigma_0 = np.array(icb_vals)
    K_sigma_1 = np.array(cmb_vals)

    G_vp = SOLAOperator(M_vp, D, vp_kernel_provider, integration_config=sola_integration_cfg)
    G_rho = SOLAOperator(M_rho, D, rho_kernel_provider, integration_config=sola_integration_cfg)
    G_vs_IC = SOLAOperator(M_vs_IC, D, vs_kernel_IC_provider, integration_config=sola_integration_cfg)
    G_vs_M = SOLAOperator(M_vs_M, D, vs_kernel_M_provider, integration_config=sola_integration_cfg)
    G_vs = RowLinearOperator([G_vs_IC, G_vs_M])
    G_sigma_0 = LinearOperator(M_sigma_0, D, lambda x: K_sigma_0 * x)
    G_sigma_1 = LinearOperator(M_sigma_1, D, lambda x: K_sigma_1 * x)
    G_functions = RowLinearOperator([G_vp, G_vs, G_rho])
    G_euclidean = RowLinearOperator([G_sigma_0, G_sigma_1])
    G = RowLinearOperator([G_functions, G_euclidean])

    target_provider_vp = BumpFunctionProvider(M_vp, centers=centers, default_width=width)
    target_provider_vs_IC = NullFunctionProvider(M_vs_IC)
    target_provider_vs_M = NullFunctionProvider(M_vs_M)
    target_provider_rho = NullFunctionProvider(M_rho)

    T_vp = SOLAOperator(M_vp, P, target_provider_vp, integration_config=sola_integration_cfg)
    T_vs_IC = SOLAOperator(M_vs_IC, P, target_provider_vs_IC, integration_config=sola_integration_cfg)
    T_vs_M = SOLAOperator(M_vs_M, P, target_provider_vs_M, integration_config=sola_integration_cfg)
    T_rho = SOLAOperator(M_rho, P, target_provider_rho, integration_config=sola_integration_cfg)
    T_sigma_0 = LinearOperator(M_sigma_0, P, lambda x: np.zeros(N_p))
    T_sigma_1 = LinearOperator(M_sigma_1, P, lambda x: np.zeros(N_p))
    T_vs = RowLinearOperator([T_vs_IC, T_vs_M])
    T_functions = RowLinearOperator([T_vp, T_vs, T_rho])
    T_euclidean = RowLinearOperator([T_sigma_0, T_sigma_1])
    T = RowLinearOperator([T_functions, T_euclidean])

    ran_array_vp = np.zeros(N)
    ran_array_vp[:10] = np.random.RandomState(42).uniform(-1, 1, 10)
    ran_array_vs_IC = np.zeros(N)
    ran_array_vs_IC[:10] = np.random.RandomState(24).uniform(-1, 1, 10)
    ran_array_vs_M = np.zeros(N)
    ran_array_vs_M[:10] = np.random.RandomState(84).uniform(-1, 1, 10)
    ran_array_rho = np.zeros(N)
    ran_array_rho[:10] = np.random.RandomState(12).uniform(-1, 1, 10)

    m_bar_vp = M_vp.from_components(ran_array_vp)
    m_bar_vs_IC = M_vs_IC.from_components(ran_array_vs_IC)
    m_bar_vs_M = M_vs_M.from_components(ran_array_vs_M)
    m_bar_vs = [m_bar_vs_IC, m_bar_vs_M]
    m_bar_rho = M_rho.from_components(ran_array_rho)
    m_bar_sigma_0 = [1]
    m_bar_sigma_1 = [2]
    m_bar = [[m_bar_vp, m_bar_vs, m_bar_rho], [m_bar_sigma_0, m_bar_sigma_1]]

    d_bar = np.asarray(G(m_bar)).ravel()
    noise_std = noise_level_fraction * float(np.max(np.abs(d_bar)))
    noise = np.random.RandomState(42).normal(0.0, noise_std, size=d_bar.shape)
    d_tilde = d_bar + noise

    true_props = np.asarray(T(m_bar)).ravel()
    noise_variance = noise_std ** 2

    extras = {
        "mode": "realistic",
        "M_vp": M_vp,
        "M_sigma_0": M_sigma_0,
        "M_sigma_1": M_sigma_1,
        "M_vs_IC": M_vs_IC,
        "M_vs_M": M_vs_M,
        "M_rho": M_rho,
        "m_bar_vp": m_bar_vp,
        "m_bar_vs_IC": m_bar_vs_IC,
        "m_bar_vs_M": m_bar_vs_M,
        "m_bar_rho": m_bar_rho,
        "m_bar_sigma_0": m_bar_sigma_0,
        "m_bar_sigma_1": m_bar_sigma_1,
        "ICB_RADIUS": ICB_RADIUS,
        "CMB_RADIUS": CMB_RADIUS,
        "vp_kernel_provider": vp_kernel_provider,
        "vs_kernel_IC_provider": vs_kernel_IC_provider,
        "vs_kernel_M_provider": vs_kernel_M_provider,
        "rho_kernel_provider": rho_kernel_provider,
        "icb_vals": icb_vals,
        "cmb_vals": cmb_vals,
        "T_vp": T_vp,
        "T_vs_IC": T_vs_IC,
        "T_vs_M": T_vs_M,
        "T_rho": T_rho,
        "centers": centers,
        "width": width,
    }

    return ProblemResult(
        M_model=M_model,
        D=D,
        P=P,
        G=G,
        T=T,
        m_bar=m_bar,
        d_tilde=d_tilde,
        d_bar=d_bar,
        true_props=true_props,
        noise_variance=noise_variance,
        function_domain=function_domain,
        catalog=catalog,
        extras=extras,
    )


def generate_shared_figures(problem: ProblemResult, figures_folder: str) -> None:
    """Generate shared, case-independent figures."""
    _set_plot_theme()

    function_domain = problem.function_domain
    d_bar = np.asarray(problem.d_bar).ravel()
    d_tilde = np.asarray(problem.d_tilde).ravel()

    x = np.linspace(function_domain.a, function_domain.b, 1000)

    if problem.extras["mode"] == "fast":
        M_vp = problem.extras["M_vp"]
        G_vp = problem.extras["G_vp"]
        T_vp = problem.extras["T_vp"]
        m_bar_vp = problem.m_bar

        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.plot(x, m_bar_vp.evaluate(x), color="tab:red", linewidth=2.5)
        plt.title(r"True Model: $\delta v_p(r)$", fontsize=18)
        plt.xlabel("Radius [km]", fontsize=14)
        plt.ylabel("Model Value", fontsize=14)
        plt.grid(True, linestyle=":", alpha=0.4)
        _despine()
        plt.tight_layout()
        _save_fig(fig, figures_folder, "true_model_(vp)")

        fig = plt.figure(figsize=(12, 6), dpi=200)
        for i in range(N_d):
            plt.plot(x, G_vp.get_kernel(i).evaluate(x), color="tab:blue", alpha=0.35, linewidth=1.1)
        plt.title(r"Sensitivity Kernels: $K_{vp}$", fontsize=18)
        plt.xlabel("Radius [km]", fontsize=14)
        plt.ylabel("Kernel Value", fontsize=14)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        _despine()
        plt.tight_layout()
        _save_fig(fig, figures_folder, "sensitivity_kernels_(vp)")

        fig = plt.figure(figsize=(12, 6), dpi=200)
        for i in range(N_p):
            plt.plot(x, T_vp.get_kernel(i).evaluate(x), color="tab:blue", alpha=0.6, linewidth=1.3)
        plt.title(r"Target Kernels: $T_{vp}$", fontsize=18)
        plt.xlabel("Radius [km]", fontsize=14)
        plt.ylabel("Kernel Value", fontsize=14)
        plt.grid(True, linestyle=":", alpha=0.4)
        _despine()
        plt.tight_layout()
        _save_fig(fig, figures_folder, "target_kernels_(vp)")

    else:
        M_vs_IC = problem.extras["M_vs_IC"]
        M_vs_M = problem.extras["M_vs_M"]
        ICB_RADIUS = problem.extras["ICB_RADIUS"]
        CMB_RADIUS = problem.extras["CMB_RADIUS"]
        m_bar_vp = problem.extras["m_bar_vp"]
        m_bar_vs_IC = problem.extras["m_bar_vs_IC"]
        m_bar_vs_M = problem.extras["m_bar_vs_M"]
        m_bar_rho = problem.extras["m_bar_rho"]

        vp_kernel_provider = problem.extras["vp_kernel_provider"]
        vs_kernel_IC_provider = problem.extras["vs_kernel_IC_provider"]
        vs_kernel_M_provider = problem.extras["vs_kernel_M_provider"]
        rho_kernel_provider = problem.extras["rho_kernel_provider"]

        icb_vals = problem.extras["icb_vals"]
        cmb_vals = problem.extras["cmb_vals"]

        T_vp = problem.extras["T_vp"]
        T_vs_IC = problem.extras["T_vs_IC"]
        T_vs_M = problem.extras["T_vs_M"]
        T_rho = problem.extras["T_rho"]

        x_IC = np.linspace(M_vs_IC.function_domain.a, M_vs_IC.function_domain.b, 500)
        x_M = np.linspace(M_vs_M.function_domain.a, M_vs_M.function_domain.b, 500)
        oc_x = np.linspace(ICB_RADIUS, CMB_RADIUS, 100)

        fig, axs = plt.subplots(3, 1, figsize=(12, 9), dpi=200, sharex=True)
        axs[0].plot(x, m_bar_vp.evaluate(x), color="tab:red", linewidth=2.5, label=r"$\bar{m}_{vp}(r)$")
        axs[0].set_title(r"True Model: $\delta v_p(r)$", fontsize=16)
        axs[0].set_ylabel("Model Value", fontsize=12)
        axs[0].legend(fontsize=11)
        axs[0].grid(True, linestyle=":", alpha=0.4)

        axs[1].plot(x_IC, m_bar_vs_IC.evaluate(x_IC), color="tab:red", linewidth=2.5, label=r"$\bar{m}_{vs}$ (inner core)")
        axs[1].plot(x_M, m_bar_vs_M.evaluate(x_M), color="tab:red", linewidth=2.5, label=r"$\bar{m}_{vs}$ (mantle)")
        axs[1].plot(oc_x, np.zeros_like(oc_x), color="tab:gray", linewidth=2, linestyle="--", label="Outer core (Vs=0)")
        axs[1].axvspan(ICB_RADIUS, CMB_RADIUS, alpha=0.15, color="blue", label="Outer Core")
        axs[1].set_title(r"True Model: $\delta v_s(r)$", fontsize=16)
        axs[1].set_ylabel("Model Value", fontsize=12)
        axs[1].legend(fontsize=11)
        axs[1].grid(True, linestyle=":", alpha=0.4)

        axs[2].plot(x, m_bar_rho.evaluate(x), color="tab:red", linewidth=2.5, label=r"$\bar{m}_{\rho}(r)$")
        axs[2].set_title(r"True Model: $\delta \rho(r)$", fontsize=16)
        axs[2].set_xlabel("Radius [km]", fontsize=12)
        axs[2].set_ylabel("Model Value", fontsize=12)
        axs[2].legend(fontsize=11)
        axs[2].grid(True, linestyle=":", alpha=0.4)
        _despine()
        plt.tight_layout()
        _save_fig(fig, figures_folder, "true_models_(vp_&_vs_&_rho)")

        fig = plt.figure(figsize=(12, 16), dpi=200)
        ax1 = plt.subplot(5, 1, 1)
        ax2 = plt.subplot(5, 1, 2, sharex=ax1)
        ax3 = plt.subplot(5, 1, 3, sharex=ax1)
        ax4 = plt.subplot(5, 1, 4)
        ax5 = plt.subplot(5, 1, 5, sharex=ax4)

        ax1.set_title(r"Sensitivity Kernels: $K_{vp}$", fontsize=16)
        for i in range(N_d):
            ax1.plot(x, vp_kernel_provider.get_function_by_index(i).evaluate(x), color="tab:blue", alpha=0.35, linewidth=1.2)
        ax1.set_ylabel("Kernel Value", fontsize=12)
        ax1.grid(True, linestyle=":", alpha=0.4)
        ax1.axhline(0, color="k", linewidth=0.5, alpha=0.3)

        ax2.set_title(r"Sensitivity Kernels: $K_{vs}$ (IC + Mantle, outer core excluded)", fontsize=16)
        for i in range(N_d):
            kIC = vs_kernel_IC_provider.get_function_by_index(i)
            kM = vs_kernel_M_provider.get_function_by_index(i)
            ax2.plot(x_IC, kIC.evaluate(x_IC), color="tab:blue", alpha=0.35, linewidth=1.2)
            ax2.plot(x_M, kM.evaluate(x_M), color="tab:blue", alpha=0.35, linewidth=1.2)
        ax2.axvspan(ICB_RADIUS, CMB_RADIUS, alpha=0.2, color="gray", label="Outer Core (Vs=0)")
        ax2.set_ylabel("Kernel Value", fontsize=12)
        ax2.grid(True, linestyle=":", alpha=0.4)
        ax2.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax2.legend(loc="upper right")

        ax3.set_title(r"Sensitivity Kernels: $K_{\rho}$", fontsize=16)
        for i in range(N_d):
            ax3.plot(x, rho_kernel_provider.get_function_by_index(i).evaluate(x), color="tab:blue", alpha=0.35, linewidth=1.2)
        ax3.set_xlabel("Radius [km]", fontsize=12)
        ax3.set_ylabel("Kernel Value", fontsize=12)
        ax3.grid(True, linestyle=":", alpha=0.4)
        ax3.axhline(0, color="k", linewidth=0.5, alpha=0.3)

        ax4.set_title(r"Sensitivity Kernels: $K_{\Sigma^0}$ (ICB)", fontsize=16)
        ax4.scatter(np.arange(N_d), icb_vals, color="tab:blue", alpha=0.7, s=30)
        ax4.set_ylabel("Kernel Value", fontsize=12)
        ax4.grid(True, linestyle=":", alpha=0.4)
        ax4.axhline(0, color="k", linewidth=0.5, alpha=0.3)

        ax5.set_title(r"Sensitivity Kernels: $K_{\Sigma^1}$ (CMB)", fontsize=16)
        ax5.scatter(np.arange(N_d), cmb_vals, color="tab:blue", alpha=0.7, s=30)
        ax5.set_xlabel("Data Index", fontsize=12)
        ax5.set_ylabel("Kernel Value", fontsize=12)
        ax5.grid(True, linestyle=":", alpha=0.4)
        ax5.axhline(0, color="k", linewidth=0.5, alpha=0.3)

        _despine()
        plt.tight_layout()
        _save_fig(fig, figures_folder, "sensitivity_kernels_(vp_&_vs_&_rho)")

        fig, axs_tk = plt.subplots(3, 1, figsize=(12, 8), dpi=200, sharex=True)
        axs_tk[0].set_title(r"Target Kernels: $T_{vp}$", fontsize=16)
        for i in range(N_p):
            axs_tk[0].plot(x, T_vp.get_kernel(i).evaluate(x), color="tab:blue", alpha=0.6, linewidth=1.5)
        axs_tk[0].set_ylabel("Kernel Value", fontsize=12)
        axs_tk[0].grid(True, linestyle=":", alpha=0.4)

        axs_tk[1].set_title(r"Target Kernels: $T_{vs}$ (IC + Mantle, targets = 0)", fontsize=16)
        for i in range(N_p):
            axs_tk[1].plot(x_IC, T_vs_IC.get_kernel(i).evaluate(x_IC), color="tab:blue", alpha=0.6, linewidth=1.5)
            axs_tk[1].plot(x_M, T_vs_M.get_kernel(i).evaluate(x_M), color="tab:blue", alpha=0.6, linewidth=1.5)
        axs_tk[1].axvspan(ICB_RADIUS, CMB_RADIUS, alpha=0.2, color="gray", label="Outer Core (Vs=0)")
        axs_tk[1].set_ylabel("Kernel Value", fontsize=12)
        axs_tk[1].grid(True, linestyle=":", alpha=0.4)
        axs_tk[1].legend()

        axs_tk[2].set_title(r"Target Kernels: $T_{\rho}$", fontsize=16)
        for i in range(N_p):
            axs_tk[2].plot(x, T_rho.get_kernel(i).evaluate(x), color="tab:blue", alpha=0.6, linewidth=1.5)
        axs_tk[2].set_xlabel("Radius [km]", fontsize=12)
        axs_tk[2].set_ylabel("Kernel Value", fontsize=12)
        axs_tk[2].grid(True, linestyle=":", alpha=0.4)

        _despine()
        plt.tight_layout()
        _save_fig(fig, figures_folder, "target_kernels_(vp_&_vs_&_rho)")

    idx = np.arange(len(d_bar))
    fig = plt.figure(figsize=(12, 4), dpi=200)
    for i in range(len(d_bar)):
        plt.plot([i, i], [d_bar[i], d_tilde[i]], color="gray", alpha=0.3, linewidth=0.8)
    plt.scatter(idx, d_tilde, label="Noisy Observations", color="tab:blue", alpha=0.7, marker="o", s=25, edgecolors="white", linewidths=0.5)
    plt.scatter(idx, d_bar, label="True Data", color="tab:red", alpha=0.8, marker="x", s=30, linewidths=1.5)
    plt.xlabel("Observation Index", fontsize=16)
    plt.ylabel("Data Value", fontsize=16)
    plt.title("Synthetic Observations: Truth vs. Noisy Measurements", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.4)
    _despine()
    plt.tight_layout()
    _save_fig(fig, figures_folder, "synthetic_observations")

    std_d = float(np.sqrt(problem.noise_variance))
    fig = plt.figure(figsize=(12, 4), dpi=200)
    plt.scatter(idx, d_tilde, label="Observed Data", color="tab:blue", alpha=0.8, s=30)
    plt.errorbar(idx, d_tilde, yerr=std_d, fmt="none", color="tab:blue", alpha=0.5, capsize=2, capthick=1, label="±1σ Uncertainty")
    plt.title("Data Likelihood: Observations with Uncertainty", fontsize=18)
    plt.xlabel("Observation Index", fontsize=16)
    plt.ylabel("Data Value", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.4)
    _despine()
    plt.tight_layout()
    _save_fig(fig, figures_folder, "data_likelihood_distribution")


def _case_slug(label: str) -> str:
    return label.lower().replace("+", "_plus").replace("-", "_minus")


def generate_case_figures(
    label,
    cfg,
    color,
    problem,
    M_prior,
    posterior_model,
    sampler_vp,
    std_P_prior,
    p_tilde,
    std_P_post,
    fixed_priors,
):
    """Generate per-case figures for fast/realistic mode."""
    _set_plot_theme()
    case_dir = os.path.join(FIGURES_FOLDER, _case_slug(label))
    os.makedirs(case_dir, exist_ok=True)

    function_domain = problem.function_domain
    centers = problem.extras["centers"]
    x = np.linspace(function_domain.a, function_domain.b, 1000)

    if problem.extras["mode"] == "fast":
        m_bar_vp = problem.m_bar
        prior_mean = M_prior.expectation
        post_mean = posterior_model.expectation
        std_vp = np.sqrt(sampler_vp.variance_function().evaluate(x))

        fig = plt.figure(figsize=(12, 5), dpi=200)
        prior_mean_vals = prior_mean.evaluate(x)
        plt.fill_between(
            x,
            prior_mean_vals - 2 * std_vp,
            prior_mean_vals + 2 * std_vp,
            color=color,
            alpha=0.15,
            label="±2σ Band",
        )
        for _ in range(5):
            sample = sampler_vp.sample()
            plt.plot(x, sample.evaluate(x), color=color, alpha=0.25, linewidth=0.9)
        plt.plot(x, prior_mean_vals, color=color, linewidth=2, linestyle=":", label="Prior Mean")
        plt.plot(x, m_bar_vp.evaluate(x), color="tab:red", linewidth=2.3, linestyle="--", label="True Model")
        plt.title(f"Prior Measure on Model Space (vp) — {label}", fontsize=16)
        plt.xlabel("Radius [km]", fontsize=13)
        plt.ylabel("Model Value", fontsize=13)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=11)
        _despine()
        plt.tight_layout()
        _save_fig(fig, case_dir, "prior_measure_on_model_space_(vp)")

        mean_pp = np.asarray(problem.T(prior_mean)).ravel()
        fig = plt.figure(figsize=(12, 5), dpi=200)
        plt.errorbar(
            centers,
            mean_pp,
            yerr=2 * std_P_prior,
            fmt="o",
            color=color,
            alpha=0.7,
            capsize=4,
            capthick=2,
            markersize=6,
            label="Property Prior (mean ±2σ)",
        )
        plt.fill_between(
            centers,
            mean_pp - 2 * std_P_prior,
            mean_pp + 2 * std_P_prior,
            color=color,
            alpha=0.15,
        )
        plt.scatter(centers, problem.true_props, color="tab:red", marker="x", s=90, linewidths=2.5, label="True Properties")
        plt.title(f"Property Prior Distribution — {label}", fontsize=16)
        plt.xlabel("Target Location [km]", fontsize=13)
        plt.ylabel("Property Value", fontsize=13)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=11)
        _despine()
        plt.tight_layout()
        _save_fig(fig, case_dir, "property_prior_distribution")

        fig = plt.figure(figsize=(12, 5), dpi=200)
        plt.plot(x, post_mean.evaluate(x), color=color, linewidth=3.0, label="Posterior Mean")
        plt.plot(x, m_bar_vp.evaluate(x), color="tab:red", linewidth=2.3, linestyle="--", label="True Model")
        plt.plot(x, prior_mean_vals, color=color, linewidth=2, linestyle=":", alpha=0.8, label="Prior Mean")
        plt.title(f"Model Posterior (vp) — {label}", fontsize=16)
        plt.xlabel("Radius [km]", fontsize=13)
        plt.ylabel("Model Value", fontsize=13)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=11)
        _despine()
        plt.tight_layout()
        _save_fig(fig, case_dir, "model_posterior_distribution_(vp)")

        fig = plt.figure(figsize=(12, 5), dpi=200)
        plt.errorbar(
            centers,
            p_tilde,
            yerr=2 * std_P_post,
            fmt="o",
            color=color,
            alpha=0.8,
            capsize=4,
            capthick=2,
            markersize=7,
            label="Posterior Properties (±2σ)",
        )
        plt.fill_between(centers, p_tilde - 2 * std_P_post, p_tilde + 2 * std_P_post, color=color, alpha=0.2)
        plt.scatter(centers, problem.true_props, color="tab:red", marker="x", s=100, linewidths=3, label="True Properties")
        plt.plot(centers, mean_pp, "o--", color=color, alpha=0.55, label="Prior Properties")
        plt.title(f"Property Inference Results — {label}", fontsize=16)
        plt.xlabel("Target Location [km]", fontsize=13)
        plt.ylabel("Property Value", fontsize=13)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=11)
        _despine()
        plt.tight_layout()
        _save_fig(fig, case_dir, "property_inference_results")

        return

    m_bar_vp = problem.extras["m_bar_vp"]
    m_bar_vs_IC = problem.extras["m_bar_vs_IC"]
    m_bar_vs_M = problem.extras["m_bar_vs_M"]
    m_bar_rho = problem.extras["m_bar_rho"]
    ICB_RADIUS = problem.extras["ICB_RADIUS"]
    CMB_RADIUS = problem.extras["CMB_RADIUS"]
    x_IC = np.linspace(problem.extras["M_vs_IC"].function_domain.a, problem.extras["M_vs_IC"].function_domain.b, 500)
    x_M = np.linspace(problem.extras["M_vs_M"].function_domain.a, problem.extras["M_vs_M"].function_domain.b, 500)
    oc_x = np.linspace(ICB_RADIUS, CMB_RADIUS, 100)

    ((_m0_vp, _m0_vs, _m0_rho), _) = M_prior.expectation
    _m0_vsIC, _m0_vsM = _m0_vs
    _std_vp = np.sqrt(sampler_vp.variance_function().evaluate(x))
    _std_vsIC = np.sqrt(fixed_priors["sampler_vs_IC"].variance_function().evaluate(x_IC))
    _std_vsM = np.sqrt(fixed_priors["sampler_vs_M"].variance_function().evaluate(x_M))
    _std_rho = np.sqrt(fixed_priors["sampler_rho"].variance_function().evaluate(x))

    fig, axs = plt.subplots(5, 1, figsize=(12, 20), dpi=200)
    axs[0].fill_between(x, _m0_vp.evaluate(x) - 2 * _std_vp, _m0_vp.evaluate(x) + 2 * _std_vp, color=color, alpha=0.15)
    for _ in range(5):
        axs[0].plot(x, sampler_vp.sample().evaluate(x), color=color, alpha=0.25, linewidth=0.9)
    axs[0].plot(x, _m0_vp.evaluate(x), color=color, linestyle=":", linewidth=2, label="Prior Mean")
    axs[0].plot(x, m_bar_vp.evaluate(x), color="tab:red", linestyle="--", linewidth=2.2, label="True Model")
    axs[0].set_title(r"Prior: $\delta v_p(r)$", fontsize=15)
    axs[0].grid(True, linestyle=":", alpha=0.4)
    axs[0].legend(fontsize=10)

    axs[1].fill_between(x_IC, _m0_vsIC.evaluate(x_IC) - 2 * _std_vsIC, _m0_vsIC.evaluate(x_IC) + 2 * _std_vsIC, color=color, alpha=0.15)
    axs[1].fill_between(x_M, _m0_vsM.evaluate(x_M) - 2 * _std_vsM, _m0_vsM.evaluate(x_M) + 2 * _std_vsM, color=color, alpha=0.15)
    axs[1].plot(x_IC, _m0_vsIC.evaluate(x_IC), color=color, linestyle=":", linewidth=2)
    axs[1].plot(x_M, _m0_vsM.evaluate(x_M), color=color, linestyle=":", linewidth=2)
    axs[1].plot(x_IC, m_bar_vs_IC.evaluate(x_IC), color="tab:red", linestyle="--", linewidth=2.2)
    axs[1].plot(x_M, m_bar_vs_M.evaluate(x_M), color="tab:red", linestyle="--", linewidth=2.2)
    axs[1].axvspan(ICB_RADIUS, CMB_RADIUS, alpha=0.15, color="gray")
    axs[1].plot(oc_x, np.zeros_like(oc_x), color="tab:red", linestyle="--", linewidth=2.2)
    axs[1].set_title(r"Prior: $\delta v_s$ (partitioned)", fontsize=15)
    axs[1].grid(True, linestyle=":", alpha=0.4)

    axs[2].fill_between(x, _m0_rho.evaluate(x) - 2 * _std_rho, _m0_rho.evaluate(x) + 2 * _std_rho, color=color, alpha=0.15)
    axs[2].plot(x, _m0_rho.evaluate(x), color=color, linestyle=":", linewidth=2)
    axs[2].plot(x, m_bar_rho.evaluate(x), color="tab:red", linestyle="--", linewidth=2.2)
    axs[2].set_title(r"Prior: $\delta \rho(r)$", fontsize=15)
    axs[2].grid(True, linestyle=":", alpha=0.4)

    s0_mean = float(fixed_priors["M_prior_sigma_0"].expectation[0])
    s1_mean = float(fixed_priors["M_prior_sigma_1"].expectation[0])
    s0_std = float(np.sqrt(fixed_priors["M_prior_sigma_0"].covariance.matrix(dense=True)[0, 0]))
    s1_std = float(np.sqrt(fixed_priors["M_prior_sigma_1"].covariance.matrix(dense=True)[0, 0]))
    s0_x = np.linspace(s0_mean - 4 * s0_std, s0_mean + 4 * s0_std, 200)
    s1_x = np.linspace(s1_mean - 4 * s1_std, s1_mean + 4 * s1_std, 200)
    s0_pdf = (1.0 / (s0_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((s0_x - s0_mean) / s0_std) ** 2)
    s1_pdf = (1.0 / (s1_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((s1_x - s1_mean) / s1_std) ** 2)
    axs[3].plot(s0_x, s0_pdf, color=color, linewidth=2.2)
    axs[3].axvline(problem.extras["m_bar_sigma_0"][0], color="tab:red", linestyle="--", linewidth=2)
    axs[3].set_title(r"Prior: $\delta\Sigma^0$", fontsize=15)
    axs[3].grid(True, linestyle=":", alpha=0.4)
    axs[4].plot(s1_x, s1_pdf, color=color, linewidth=2.2)
    axs[4].axvline(problem.extras["m_bar_sigma_1"][0], color="tab:red", linestyle="--", linewidth=2)
    axs[4].set_title(r"Prior: $\delta\Sigma^1$", fontsize=15)
    axs[4].grid(True, linestyle=":", alpha=0.4)
    _despine()
    plt.tight_layout()
    _save_fig(fig, case_dir, "prior_measure_on_model_space_(vp_&_vs_&_rho_&_sigmas)")

    ((_vp_post, _vs_post, _rho_post), (_s0_post, _s1_post)) = posterior_model.expectation
    _vs_IC_post, _vs_M_post = _vs_post
    fig, axs = plt.subplots(5, 1, figsize=(12, 20), dpi=200)
    axs[0].plot(x, _vp_post.evaluate(x), color=color, linewidth=2.8)
    axs[0].plot(x, m_bar_vp.evaluate(x), color="tab:red", linestyle="--", linewidth=2.2)
    axs[0].plot(x, _m0_vp.evaluate(x), color=color, linestyle=":", linewidth=2)
    axs[1].plot(x_IC, _vs_IC_post.evaluate(x_IC), color=color, linewidth=2.8)
    axs[1].plot(x_M, _vs_M_post.evaluate(x_M), color=color, linewidth=2.8)
    axs[1].plot(x_IC, m_bar_vs_IC.evaluate(x_IC), color="tab:red", linestyle="--", linewidth=2.2)
    axs[1].plot(x_M, m_bar_vs_M.evaluate(x_M), color="tab:red", linestyle="--", linewidth=2.2)
    axs[1].axvspan(ICB_RADIUS, CMB_RADIUS, alpha=0.15, color="gray")
    axs[2].plot(x, _rho_post.evaluate(x), color=color, linewidth=2.8)
    axs[2].plot(x, m_bar_rho.evaluate(x), color="tab:red", linestyle="--", linewidth=2.2)
    axs[3].axvline(float(_s0_post[0]), color=color, linewidth=2.4)
    axs[3].axvline(problem.extras["m_bar_sigma_0"][0], color="tab:red", linestyle="--", linewidth=2.2)
    axs[4].axvline(float(_s1_post[0]), color=color, linewidth=2.4)
    axs[4].axvline(problem.extras["m_bar_sigma_1"][0], color="tab:red", linestyle="--", linewidth=2.2)
    for ax in axs:
        ax.grid(True, linestyle=":", alpha=0.4)
    axs[0].set_title("Model Posterior: vp", fontsize=14)
    axs[1].set_title("Model Posterior: vs", fontsize=14)
    axs[2].set_title("Model Posterior: rho", fontsize=14)
    axs[3].set_title("Model Posterior: sigma0", fontsize=14)
    axs[4].set_title("Model Posterior: sigma1", fontsize=14)
    _despine()
    plt.tight_layout()
    _save_fig(fig, case_dir, "model_posterior_distribution_(vp_&_vs_&_rho_&_sigmas)")

    mean_pp = np.asarray(problem.T(M_prior.expectation)).ravel()
    fig = plt.figure(figsize=(12, 5), dpi=200)
    plt.errorbar(centers, mean_pp, yerr=2 * std_P_prior, fmt="o", color=color, alpha=0.7, capsize=4, capthick=2, markersize=6)
    plt.fill_between(centers, mean_pp - 2 * std_P_prior, mean_pp + 2 * std_P_prior, color=color, alpha=0.15)
    plt.scatter(centers, problem.true_props, color="tab:red", marker="x", s=90, linewidths=2.5)
    plt.title(f"Property Prior Distribution — {label}", fontsize=16)
    plt.xlabel("Target Location [km]", fontsize=13)
    plt.ylabel("Property Value", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.4)
    _despine()
    plt.tight_layout()
    _save_fig(fig, case_dir, "property_prior_distribution")

    fig = plt.figure(figsize=(12, 5), dpi=200)
    plt.errorbar(centers, p_tilde, yerr=2 * std_P_post, fmt="o", color=color, alpha=0.8, capsize=4, capthick=2, markersize=7)
    plt.fill_between(centers, p_tilde - 2 * std_P_post, p_tilde + 2 * std_P_post, color=color, alpha=0.2)
    plt.scatter(centers, problem.true_props, color="tab:red", marker="x", s=100, linewidths=3)
    plt.plot(centers, mean_pp, "o--", color=color, alpha=0.55)
    plt.title(f"Property Inference Results — {label}", fontsize=16)
    plt.xlabel("Target Location [km]", fontsize=13)
    plt.ylabel("Property Value", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.4)
    _despine()
    plt.tight_layout()
    _save_fig(fig, case_dir, "property_inference_results")


def generate_prior_previews(problem, fixed_priors, calib):
    """Generate quick per-case prior plots without running inference."""
    preview_cases = (
        list(PRIOR_CASES.keys()) if "all" in PREVIEW_CASES else PREVIEW_CASES
    )
    unknown = [case for case in preview_cases if case not in PRIOR_CASES]
    if unknown:
        raise ValueError(f"Unknown PREVIEW_CASES entries: {unknown}")

    preview_dir = os.path.join(FIGURES_FOLDER, "prior_preview")
    os.makedirs(preview_dir, exist_ok=True)

    x = np.linspace(problem.function_domain.a, problem.function_domain.b, 1000)
    centers = problem.extras["centers"]
    if problem.extras["mode"] == "fast":
        true_vp = problem.m_bar
    else:
        true_vp = problem.extras["m_bar_vp"]

    summary = []
    for label in preview_cases:
        cfg = PRIOR_CASES[label]
        color = CASE_COLORS[label]
        prior_vp, sampler_vp = build_vp_prior(
            problem.extras["M_vp"],
            cfg["s"],
            cfg["ls"],
            cfg["overall_variance"],
            cfg["mean_offset_model"],
            bcs_vp=calib["bcs_vp"],
            lap_kwargs=calib["lap_kwargs"],
            bs_kwargs=calib["bs_kwargs"],
        )
        M_prior = build_full_prior(prior_vp, fixed_priors, FAST_MODE)

        if problem.extras["mode"] == "fast":
            m0_vp = M_prior.expectation
        else:
            ((m0_vp, _, _), _) = M_prior.expectation

        std_vp = np.sqrt(sampler_vp.variance_function().evaluate(x))
        m0_vals = m0_vp.evaluate(x)

        fig = plt.figure(figsize=(12, 5), dpi=200)
        plt.fill_between(
            x,
            m0_vals - 2 * std_vp,
            m0_vals + 2 * std_vp,
            color=color,
            alpha=0.18,
            label="Prior ±2σ",
        )
        if PREVIEW_N_SAMPLES > 0:
            for _ in range(PREVIEW_N_SAMPLES):
                sample = sampler_vp.sample()
                plt.plot(x, sample.evaluate(x), color=color, alpha=0.25, linewidth=0.8)
        plt.plot(x, m0_vals, color=color, linewidth=2.2, linestyle=":", label="Prior Mean")
        plt.plot(
            x,
            true_vp.evaluate(x),
            color="tab:red",
            linewidth=2.2,
            linestyle="--",
            label="True vp",
        )
        plt.title(f"Quick Prior Preview (vp) — {label}", fontsize=16)
        plt.xlabel("Radius [km]", fontsize=13)
        plt.ylabel("Model Value", fontsize=13)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=11)
        _despine()
        plt.tight_layout()
        _save_fig(fig, preview_dir, f"quick_prior_model_{_case_slug(label)}")

        prior_P = M_prior.affine_mapping(operator=problem.T)
        cov_P_prior = prior_P.covariance.matrix(
            dense=True,
            parallel=True,
            n_jobs=parallel_cfg.n_jobs,
        )
        std_P_prior = np.sqrt(np.diag(cov_P_prior))
        mean_pp = np.asarray(problem.T(M_prior.expectation)).ravel()

        fig = plt.figure(figsize=(12, 5), dpi=200)
        plt.errorbar(
            centers,
            mean_pp,
            yerr=2 * std_P_prior,
            fmt="o",
            color=color,
            alpha=0.75,
            capsize=4,
            capthick=2,
            markersize=6,
            label="Property Prior (mean ±2σ)",
        )
        plt.fill_between(
            centers,
            mean_pp - 2 * std_P_prior,
            mean_pp + 2 * std_P_prior,
            color=color,
            alpha=0.15,
        )
        plt.scatter(
            centers,
            problem.true_props,
            color="tab:red",
            marker="x",
            s=90,
            linewidths=2.5,
            label="True Properties",
        )
        plt.title(f"Quick Prior Preview (properties) — {label}", fontsize=16)
        plt.xlabel("Target Location [km]", fontsize=13)
        plt.ylabel("Property Value", fontsize=13)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend(fontsize=11)
        _despine()
        plt.tight_layout()
        _save_fig(fig, preview_dir, f"quick_prior_properties_{_case_slug(label)}")

        summary.append((label, float(np.mean(std_P_prior)), cfg["mean_offset_model"]))

    print(f"Quick prior previews saved in: {preview_dir}")
    print("Preview summary (mean property σ_prior, model offset):")
    for label, mean_std, model_offset in summary:
        print(f"  {label:<10} σ_prior={mean_std:.5f}  offset={model_offset:.5f}")


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Prior Sensitivity Study v2 — Phases 1-2 scaffold")
    print(f"FAST_MODE = {FAST_MODE}")
    print(f"CALIBRATE_PRIOR_CASES = {CALIBRATE_PRIOR_CASES}")
    print(f"Figures folder: {FIGURES_FOLDER}")
    print("=" * 70)

    print("Building problem ...")
    problem = build_fast_problem() if FAST_MODE else build_realistic_problem()

    print("Generating shared figures ...")
    generate_shared_figures(problem, FIGURES_FOLDER)

    print("Calibrating prior cases ...")
    fixed_priors = {} if FAST_MODE else build_fixed_realistic_priors(problem)
    calib = calibrate_prior_cases(
        problem,
        fixed_priors,
        FAST_MODE,
        calibrate=CALIBRATE_PRIOR_CASES,
    )

    if calib["calibration_enabled"]:
        print(f"REF_PROP_STD = {calib['REF_PROP_STD']:.6f}")
        print(f"T_mean_norm  = {calib['T_mean_norm']:.6f}")
        for label, cfg in PRIOR_CASES.items():
            if cfg["offset_sigma"] != 0.0:
                print(
                    f"  {label}: offset_model = {cfg['mean_offset_model']:.6f} "
                    f"(= {cfg['offset_sigma']:.1f}σ in property space)"
                )
    else:
        print("Calibration disabled: using raw PRIOR_CASES settings.")
        for label, cfg in PRIOR_CASES.items():
            if cfg["offset_sigma"] != 0.0:
                print(
                    f"  {label}: offset_model = {cfg['mean_offset_model']:.6f} "
                    "(direct model-space offset)"
                )

    if PREVIEW_PRIORS_ONLY:
        print("PREVIEW_PRIORS_ONLY=True → generating quick prior previews and exiting")
        generate_prior_previews(problem, fixed_priors, calib)
        print(
            "Note: preview outputs are written to "
            f"{os.path.join(FIGURES_FOLDER, 'prior_preview')}."
        )
        print(
            "Per-case figures in case folders are only refreshed when "
            "PREVIEW_PRIORS_ONLY=False."
        )
        elapsed = time.time() - t0
        print(f"Preview-only run completed in {elapsed:.1f}s")
        return

    solver = CholeskySolver(parallel=True, n_jobs=parallel_cfg.n_jobs)
    C_D_matrix = problem.noise_variance * np.eye(N_d)
    gaussian_D_noise = GaussianMeasure.from_covariance_matrix(
        problem.D,
        C_D_matrix,
        expectation=np.zeros(N_d),
    )

    cases_to_run = list(PRIOR_CASES.keys()) if "all" in CASES_TO_RUN else CASES_TO_RUN
    print(f"Will run: {cases_to_run}")

    results = load_existing_results(OUTPUT_CSV)
    computed_labels = set()
    log_lines = []
    for label, cfg in PRIOR_CASES.items():
        if label not in cases_to_run:
            continue

        case_t0 = time.time()
        print(f"\n{'=' * 70}")
        print(f"Running case: {label}")
        print(
            f"  s={cfg['s']}, ls={cfg['ls']}, ov={cfg['overall_variance']:.3f}, "
            f"offset_model={cfg['mean_offset_model']:.6f}"
        )
        print(f"{'=' * 70}")

        prior_vp, sampler_vp = build_vp_prior(
            problem.extras["M_vp"],
            cfg["s"],
            cfg["ls"],
            cfg["overall_variance"],
            cfg["mean_offset_model"],
            bcs_vp=calib["bcs_vp"],
            lap_kwargs=calib["lap_kwargs"],
            bs_kwargs=calib["bs_kwargs"],
        )
        M_prior = build_full_prior(prior_vp, fixed_priors, FAST_MODE)

        prior_P = M_prior.affine_mapping(operator=problem.T)
        cov_P_prior = prior_P.covariance.matrix(dense=True, parallel=True, n_jobs=parallel_cfg.n_jobs)
        std_P_prior = np.sqrt(np.diag(cov_P_prior))

        chi2_per_dof = prior_predictive_check(
            M_prior,
            problem.G,
            problem.d_tilde,
            C_D_matrix,
            parallel_cfg,
        )
        print(f"  Prior predictive check: chi2/dof = {chi2_per_dof:.4f}")
        if chi2_per_dof > PRIOR_PREDICTIVE_CHI2_THRESHOLD:
            print(
                "  WARNING: PRIOR PREDICTIVELY INCOMPATIBLE "
                f"(chi2/dof = {chi2_per_dof:.1f} > {PRIOR_PREDICTIVE_CHI2_THRESHOLD})"
            )

        forward_problem = LinearForwardProblem(problem.G, data_error_measure=gaussian_D_noise)
        bayes = LinearBayesianInversion(forward_problem, M_prior)
        posterior_model = bayes.model_posterior_measure(problem.d_tilde, solver)

        prop_post = posterior_model.affine_mapping(operator=problem.T)
        p_tilde = np.asarray(prop_post.expectation).ravel()
        cov_P = prop_post.covariance.matrix(dense=True, parallel=True, n_jobs=parallel_cfg.n_jobs)
        std_P_post = np.sqrt(np.diag(cov_P))

        elapsed_case = time.time() - case_t0
        errors = p_tilde - problem.true_props
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        cover = float(np.sum(np.abs(errors) <= 2 * std_P_post) / N_p)
        mean_post_std = float(np.mean(std_P_post))
        mean_prior_std = float(np.mean(std_P_prior))
        unc_reduction = float(1.0 - mean_post_std / mean_prior_std)
        data_misfit = float(np.linalg.norm(np.asarray(problem.G(posterior_model.expectation)).ravel() - np.asarray(problem.d_tilde).ravel()))
        min_eig = float(np.linalg.eigvalsh(cov_P).min())

        results[label] = dict(
            case=label,
            s=cfg["s"],
            length_scale=cfg["ls"],
            overall_variance=round(cfg["overall_variance"], 6),
            mean_offset=round(cfg["mean_offset_model"], 6),
            rmse=round(rmse, 6),
            mae=round(mae, 6),
            coverage_2sigma=round(cover, 4),
            mean_post_std=round(mean_post_std, 6),
            mean_prior_std=round(mean_prior_std, 6),
            unc_reduction=round(unc_reduction, 6),
            data_misfit=round(data_misfit, 6),
            min_eig_cov=round(min_eig, 8),
            chi2_per_dof=round(chi2_per_dof, 4),
            elapsed_s=round(elapsed_case, 2),
        )
        computed_labels.add(label)

        log_lines.append(
            f"Case: {label} | DONE in {elapsed_case:.1f}s | "
            f"RMSE={rmse:.4f} | MAE={mae:.4f} | Coverage={cover:.0%} | "
            f"UncRed={unc_reduction:.1%} | Misfit={data_misfit:.4f} | "
            f"MinEig={min_eig:.2e} | chi2/dof={chi2_per_dof:.4f}"
        )

        if min_eig < -1e-6:
            warnings.warn(
                f"[{label}] Negative min eigenvalue ({min_eig:.2e}) in property covariance",
                RuntimeWarning,
            )

        print(
            f"  DONE in {elapsed_case:.1f}s | RMSE={rmse:.4f} | "
            f"Coverage={cover:.0%} | UncRed={unc_reduction:.1%}"
        )

        generate_case_figures(
            label=label,
            cfg=cfg,
            color=CASE_COLORS[label],
            problem=problem,
            M_prior=M_prior,
            posterior_model=posterior_model,
            sampler_vp=sampler_vp,
            std_P_prior=std_P_prior,
            p_tilde=p_tilde,
            std_P_post=std_P_post,
            fixed_priors=fixed_priors,
        )

    write_summary_outputs(results, cases_to_run, log_lines)

    print("\n" + "=" * 94)
    print(
        f"{'Case':<14} {'RMSE':>8} {'MAE':>8} {'Cover':>8} {'UncRed':>8} "
        f"{'Misfit':>10} {'MinEig':>11} {'chi2/dof':>9} {'Time':>8}"
    )
    print("-" * 94)
    for label in PRIOR_CASES:
        if label not in results:
            continue
        row = results[label]
        case_name = row.get("case", label)
        if label not in computed_labels:
            case_name = f"{case_name} (cached)"

        rmse_v = _safe_float(row.get("rmse"))
        mae_v = _safe_float(row.get("mae"))
        cover_v = _safe_float(row.get("coverage_2sigma"))
        unc_v = _safe_float(row.get("unc_reduction"))
        misfit_v = _safe_float(row.get("data_misfit"))
        min_eig_v = _safe_float(row.get("min_eig_cov"))
        chi2_v = _safe_float(row.get("chi2_per_dof"))
        elapsed_v = _safe_float(row.get("elapsed_s"))

        print(
            f"{case_name:<14} {_fmt_float(rmse_v, '8.4f')} {_fmt_float(mae_v, '8.4f')} "
            f"{_fmt_float(cover_v * 100.0, '7.1f')}% {_fmt_float(unc_v * 100.0, '7.1f')}% "
            f"{_fmt_float(misfit_v, '10.4f')} {_fmt_float(min_eig_v, '11.2e')} "
            f"{_fmt_float(chi2_v, '9.2f')} {_fmt_float(elapsed_v, '7.1f')}s"
        )
    print("=" * 94)

    elapsed = time.time() - t0
    print(f"Total run completed in {elapsed:.1f}s")
    if FAST_MODE and elapsed > 60:
        warnings.warn(
            f"FAST_MODE total run took {elapsed:.1f}s (>60s target for single case)",
            RuntimeWarning,
        )

    print("Phases 1-5 complete.")


if __name__ == "__main__":
    main()
