#!/usr/bin/env python
"""
Prior Sensitivity Study
=======================
Runs Bayesian property inference under four prior configurations and saves
scalar summaries to CSV and Markdown files.

Prior cases
-----------
  Reference    : s=6, ls=30,  ov=10^1.5, mean_offset=0
    Over-smooth  : s=6, ls=150, ov=10^1.5, mean_offset=0
                                     (long correlation length)
    Under-smooth : s=3, ls=10,  ov=10^1.5, mean_offset=0
                                     (rough spectrum)
    Shifted-mean : s=6, ls=30,  ov=10^1.5, mean_offset=+3
                                     (prior mean far from truth)

Metrics saved per case
----------------------
  rmse             RMSE of property posterior mean vs true properties
  mae              Mean absolute error
  coverage_2sigma  Fraction of properties within ±2σ _posterior_
  mean_post_std    Mean posterior standard deviation
  mean_prior_std   Mean prior standard deviation
    unc_reduction    1 - mean_post_std / mean_prior_std
                                     (uncertainty reduction)
    data_misfit      ||G(m̃) - d̃||
                                     (how well posterior mean fits data)
    min_eig_cov      Min eigenvalue of property covariance matrix
                                     (stability check)
  elapsed_s        Wall-clock time for this case (seconds)
"""

# Disable 'multiple spaces before operator' linting (E221) for this file.
# Tools: ruff/flake8/pycodestyle recognize these module-level noqa directives.
# ruff: noqa: E221
# noqa: E221

import os
import sys
import time
import csv
import warnings
from pathlib import Path

import numpy as np

# ensure kernel_utils is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# intervalinf imports
from intervalinf import (
    Lebesgue, IntervalDomain, LebesgueIntegrationConfig,
    IntegrationConfig, ParallelConfig, LebesgueSpaceDirectSum,
    Function, BoundaryConditions,
    KnownRegion, PartitionedLebesgueSpace,
)
from intervalinf.operators import SOLAOperator, Laplacian, BesselSobolevInverse
from intervalinf.sampling import KLSampler
from intervalinf.providers import (
    BumpFunctionProvider,
    NullFunctionProvider,
)
from kernel_utils import (
    SensitivityKernelCatalog,
    SensitivityKernelProvider,
    EARTH_RADIUS_KM,
)

# pygeoinf imports
from pygeoinf import (
    EuclideanSpace,
    RowLinearOperator,
    GaussianMeasure,
    LinearForwardProblem,
    LinearBayesianInversion,
    CholeskySolver,
    HilbertSpaceDirectSum,
    LinearOperator,
)

# output directory
FIGURES_FOLDER = "wrong_prior"
os.makedirs(FIGURES_FOLDER, exist_ok=True)
OUTPUT_CSV = os.path.join(FIGURES_FOLDER, "prior_sensitivity_summary.csv")
OUTPUT_MD  = os.path.join(FIGURES_FOLDER, "prior_sensitivity_summary.md")
OUTPUT_TXT = os.path.join(FIGURES_FOLDER, "prior_sensitivity_detailed.txt")

# =============================================================================
# INTEGRATION / PARALLELISATION CONFIG
# =============================================================================
Lebesgue_integration_cfg = LebesgueIntegrationConfig(
    inner_product=IntegrationConfig(method='trapz', n_points=1024),
    dual=IntegrationConfig(method='trapz', n_points=1024),
    general=IntegrationConfig(method='trapz', n_points=1024),
)
laplacian_integration_cfg   = IntegrationConfig(method='trapz', n_points=1024)
sola_integration_cfg        = IntegrationConfig(method='trapz', n_points=2048)
bessel_sobolev_integration_cfg = IntegrationConfig(
    method='trapz',
    n_points=2048,
)

parallel_cfg = ParallelConfig(enabled=True, n_jobs=12)

# =============================================================================
# MODEL SPACES
# =============================================================================
print("Building model spaces …")
function_domain = IntervalDomain(0, EARTH_RADIUS_KM)
ICB_RADIUS = 1217.5   # km from centre (depth 5153.5 km)
CMB_RADIUS = 3480.0   # km from centre (depth 2891 km)

N = 100  # basis dimension

M_vp = Lebesgue(
    N,
    function_domain,
    basis='ND',
    integration_config=Lebesgue_integration_cfg.inner_product,
    parallel_config=parallel_cfg,
)

outer_core_interval = IntervalDomain(ICB_RADIUS, CMB_RADIUS)
outer_core = KnownRegion.zero(outer_core_interval)
partitioned_vs = PartitionedLebesgueSpace(
    full_domain=function_domain,
    known_regions=[outer_core],
    dims=[N, N],
    bases=['cosine', 'ND'],
    integration_config=Lebesgue_integration_cfg.inner_product,
    parallel_config=parallel_cfg,
)
M_vs    = partitioned_vs.model_space
M_vs_IC = partitioned_vs.unknown_spaces[0]
M_vs_M  = partitioned_vs.unknown_spaces[1]

M_rho = Lebesgue(
    N,
    function_domain,
    basis='ND',
    integration_config=Lebesgue_integration_cfg.inner_product,
    parallel_config=parallel_cfg,
)

M_sigma_0 = EuclideanSpace(1)
M_sigma_1 = EuclideanSpace(1)

M_functions = LebesgueSpaceDirectSum([M_vp, M_vs, M_rho])
M_euclidean = HilbertSpaceDirectSum([M_sigma_0, M_sigma_1])
M_model     = HilbertSpaceDirectSum([M_functions, M_euclidean])

N_d = 140
D   = EuclideanSpace(N_d)
N_p = 20
P   = EuclideanSpace(N_p)

# =============================================================================
# FORWARD AND PROPERTY OPERATORS
# =============================================================================
print("Building forward operators …")
width   = 0.2 * EARTH_RADIUS_KM
centers = np.linspace(function_domain.a + width / 2,
                      function_domain.b - width / 2, N_p)

data_dir = Path(_SCRIPT_DIR / '../kernels_modeplotaat_Adrian')
catalog = SensitivityKernelCatalog(data_dir)

vp_kernel_function_provider  = SensitivityKernelProvider(
    M_vp, catalog, interpolation_method='cubic',
    include_discontinuities=True, kernel_type='vp')
rho_kernel_function_provider = SensitivityKernelProvider(
    M_rho, catalog, interpolation_method='cubic',
    include_discontinuities=True, kernel_type='rho')

vs_full_space = Lebesgue(N, function_domain, basis='none')
vs_kernel_full_provider = SensitivityKernelProvider(
    vs_full_space, catalog, interpolation_method='cubic',
    include_discontinuities=True, kernel_type='vs')
vs_kernel_IC_provider = vs_kernel_full_provider.restrict(M_vs_IC)
vs_kernel_M_provider  = vs_kernel_full_provider.restrict(M_vs_M)

# Topo (discontinuity) sensitivity values
kernel_provider = vp_kernel_function_provider
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

G_vp = SOLAOperator(
    M_vp,
    D,
    vp_kernel_function_provider,
    integration_config=sola_integration_cfg,
)
G_rho = SOLAOperator(
    M_rho,
    D,
    rho_kernel_function_provider,
    integration_config=sola_integration_cfg,
)
G_vs_IC = SOLAOperator(
    M_vs_IC,
    D,
    vs_kernel_IC_provider,
    integration_config=sola_integration_cfg,
)
G_vs_M = SOLAOperator(
    M_vs_M,
    D,
    vs_kernel_M_provider,
    integration_config=sola_integration_cfg,
)
G_vs        = RowLinearOperator([G_vs_IC, G_vs_M])
G_sigma_0 = LinearOperator(
    M_sigma_0,
    D,
    lambda x: K_sigma_0 * x,
)
G_sigma_1 = LinearOperator(
    M_sigma_1,
    D,
    lambda x: K_sigma_1 * x,
)
G_functions = RowLinearOperator([G_vp, G_vs, G_rho])
G_euclidean = RowLinearOperator([G_sigma_0, G_sigma_1])
G           = RowLinearOperator([G_functions, G_euclidean])

target_provider_vp = BumpFunctionProvider(
    M_vp,
    centers=centers,
    default_width=width,
)
target_provider_vs_IC = NullFunctionProvider(M_vs_IC)
target_provider_vs_M  = NullFunctionProvider(M_vs_M)
target_provider_rho   = NullFunctionProvider(M_rho)

T_vp = SOLAOperator(
    M_vp,
    P,
    target_provider_vp,
    integration_config=sola_integration_cfg,
)
T_vs_IC = SOLAOperator(
    M_vs_IC,
    P,
    target_provider_vs_IC,
    integration_config=sola_integration_cfg,
)
T_vs_M = SOLAOperator(
    M_vs_M,
    P,
    target_provider_vs_M,
    integration_config=sola_integration_cfg,
)
T_rho = SOLAOperator(
    M_rho,
    P,
    target_provider_rho,
    integration_config=sola_integration_cfg,
)
T_sigma_0 = LinearOperator(
    M_sigma_0,
    P,
    lambda x: np.zeros(N_p),
)
T_sigma_1 = LinearOperator(
    M_sigma_1,
    P,
    lambda x: np.zeros(N_p),
)
T_vs        = RowLinearOperator([T_vs_IC, T_vs_M])
T_functions = RowLinearOperator([T_vp, T_vs, T_rho])
T_euclidean = RowLinearOperator([T_sigma_0, T_sigma_1])
T           = RowLinearOperator([T_functions, T_euclidean])

# =============================================================================
# TRUE MODEL & SYNTHETIC DATA
# =============================================================================
print("Building true model and synthetic data …")
ran_array_vp = np.zeros(N)
ran_array_vp[:10] = np.random.RandomState(42).uniform(-1, 1, 10)

ran_array_vs_IC = np.zeros(N)
ran_array_vs_IC[:10] = np.random.RandomState(24).uniform(-1, 1, 10)

ran_array_vs_M = np.zeros(N)
ran_array_vs_M[:10] = np.random.RandomState(84).uniform(-1, 1, 10)

ran_array_rho = np.zeros(N)
ran_array_rho[:10] = np.random.RandomState(12).uniform(-1, 1, 10)

m_bar_vp    = M_vp.from_components(ran_array_vp)
m_bar_vs_IC = M_vs_IC.from_components(ran_array_vs_IC)
m_bar_vs_M  = M_vs_M.from_components(ran_array_vs_M)
m_bar_vs    = [m_bar_vs_IC, m_bar_vs_M]
m_bar_rho   = M_rho.from_components(ran_array_rho)
m_bar_sigma_0 = [1]
m_bar_sigma_1 = [2]
m_bar = [[m_bar_vp, m_bar_vs, m_bar_rho], [m_bar_sigma_0, m_bar_sigma_1]]

d_bar       = G(m_bar)
noise_level = 0.1 * np.max(np.abs(d_bar))
np.random.seed(42)
d_tilde = [
    sub + np.random.normal(0, noise_level, sub.shape)
    for sub in d_bar
]

noise_variance    = (0.1 * np.max(np.abs(d_tilde))) ** 2
C_D_matrix        = noise_variance * np.eye(N_d)
gaussian_D_noise  = GaussianMeasure.from_covariance_matrix(
    D, C_D_matrix, expectation=np.zeros(N_d))

true_props = T(m_bar)

# =============================================================================
# FIXED PRIOR COMPONENTS (vs, rho, sigma — identical across all cases)
# =============================================================================
print("Building fixed prior components (vs, rho, sigma) …")
bcs_vs_IC = BoundaryConditions(bc_type='neumann')
bcs_vs_M  = BoundaryConditions(bc_type='mixed_neumann_dirichlet')
bcs_rho   = BoundaryConditions(bc_type='mixed_neumann_dirichlet')

s_vs, length_scale_vs, overall_variance_vs = 6.0, 30, 10**3
k_vs = overall_variance_vs ** (-0.5 / s_vs)
alpha_vs = (length_scale_vs ** 2) * (k_vs ** 2)
lap_kwargs = dict(
    method='spectral',
    dofs=100,
    integration_config=laplacian_integration_cfg,
    n_samples=2048,
)
L_vs_IC = Laplacian(
    M_vs_IC,
    bcs_vs_IC,
    alpha_vs,
    **lap_kwargs,
)
L_vs_M = Laplacian(
    M_vs_M,
    bcs_vs_M,
    alpha_vs,
    **lap_kwargs,
)
bs_kwargs = dict(
    dofs=512,
    n_samples=2048,
    use_fast_transforms=True,
    integration_config=bessel_sobolev_integration_cfg,
)
C_0_vs_IC = BesselSobolevInverse(
    M_vs_IC,
    M_vs_IC,
    k_vs,
    s_vs,
    L_vs_IC,
    **bs_kwargs,
)
C_0_vs_M = BesselSobolevInverse(
    M_vs_M,
    M_vs_M,
    k_vs,
    s_vs,
    L_vs_M,
    **bs_kwargs,
)
s_rho, length_scale_rho, overall_variance_rho = 6.0, 30, 10**2
k_rho    = overall_variance_rho ** (-0.5 / s_rho)
alpha_rho = (length_scale_rho ** 2) * (k_rho ** 2)
L_rho = Laplacian(
    M_rho,
    bcs_rho,
    alpha_rho,
    **lap_kwargs,
)
C_0_rho = BesselSobolevInverse(
    M_rho,
    M_rho,
    k_rho,
    s_rho,
    L_rho,
    **bs_kwargs,
)

m_0_vs_IC = Function(M_vs_IC, evaluate_callable=lambda x: np.zeros_like(x))
m_0_vs_M  = Function(M_vs_M,  evaluate_callable=lambda x: np.zeros_like(x))
m_0_rho   = Function(M_rho,   evaluate_callable=lambda x: np.zeros_like(x))

sampler_vs_IC = KLSampler(C_0_vs_IC, mean=m_0_vs_IC, n_modes=100)
M_prior_vs_IC = GaussianMeasure(
    covariance=C_0_vs_IC,
    expectation=m_0_vs_IC,
    sample=sampler_vs_IC.sample,
)
sampler_vs_M = KLSampler(C_0_vs_M, mean=m_0_vs_M, n_modes=100)
M_prior_vs_M = GaussianMeasure(
    covariance=C_0_vs_M,
    expectation=m_0_vs_M,
    sample=sampler_vs_M.sample,
)
M_prior_vs = GaussianMeasure.from_direct_sum([M_prior_vs_IC, M_prior_vs_M])

sampler_rho = KLSampler(C_0_rho, mean=m_0_rho, n_modes=100)
M_prior_rho = GaussianMeasure(
    covariance=C_0_rho,
    expectation=m_0_rho,
    sample=sampler_rho.sample,
)

M_prior_sigma_0 = GaussianMeasure.from_covariance_matrix(
    M_sigma_0,
    np.array([[10]]),
    expectation=np.array([0.0]),
)
M_prior_sigma_1 = GaussianMeasure.from_covariance_matrix(
    M_sigma_1,
    np.array([[10]]),
    expectation=np.array([0.0]),
)

# =============================================================================
# PRIOR BUILDER  (only varies vp component)
# =============================================================================
bcs_vp = BoundaryConditions(bc_type='mixed_neumann_dirichlet')

def build_vp_prior(
    s,
    length_scale,
    overall_variance,
    mean_offset=0.0,
    n_kl=100,
):
    """Build a GaussianMeasure on M_vp given BesselSobolev hyperparameters.

    Parameters
    ----------
    s, length_scale, overall_variance : float
        BesselSobolev hyperparameters.
    mean_offset : float
        Constant offset added to the prior mean function.
    n_kl : int
        Number of Karhunen-Loève terms for sampling.

    Returns
    -------
    GaussianMeasure on M_vp.
    """
    k     = overall_variance ** (-0.5 / s)
    alpha = (length_scale ** 2) * (k ** 2)

    L = Laplacian(
        M_vp,
        bcs_vp,
        alpha,
        **lap_kwargs,
    )
    C = BesselSobolevInverse(
        M_vp,
        M_vp,
        k,
        s,
        L,
        **bs_kwargs,
    )

    def _m0_callable(x, _off=mean_offset):
        return _off * np.ones_like(x)

    m_0 = Function(M_vp, evaluate_callable=_m0_callable)
    sampler = KLSampler(C, mean=m_0, n_modes=n_kl)
    return GaussianMeasure(
        covariance=C,
        expectation=m_0,
        sample=sampler.sample,
    )


def build_full_prior(M_prior_vp):
    """Combine a vp prior with the fixed vs/rho/sigma priors."""
    M_prior_functions = GaussianMeasure.from_direct_sum(
        [M_prior_vp, M_prior_vs, M_prior_rho]
    )
    M_prior_euclidean = GaussianMeasure.from_direct_sum(
        [M_prior_sigma_0, M_prior_sigma_1]
    )
    return GaussianMeasure.from_direct_sum([
        M_prior_functions,
        M_prior_euclidean,
    ])


# =============================================================================
# PRIOR CASES
# =============================================================================
# Key: (s, length_scale, overall_variance, mean_offset)
# overall_variance governs the amplitude; we keep it fixed at 10^1.5.
# The only driver of change between cases is the _shape_ of the prior, not
# its overall scale.
#
# Over-smooth uses s=12 (doubled Sobolev order) rather than a longer
# length_scale. Increasing length_scale also increases alpha = ls^2 * k^2,
# collapsing the BesselSobolev eigenvalues and making the prior
# over-constrained (roughly 100x tighter). Increasing s instead
# leaves alpha similar (~675 vs ~605 for reference) and changes only
# the rate at which high-frequency components are penalised.
#
# For 'Shifted-mean' the offset of +3 is chosen so that the prior mean is
# noticeably far from the truth to demonstrate mean-bias, while not being so
# extreme that the Cholesky solver fails to converge.

PRIOR_CASES = {
    "Reference": {
        "s": 6.0,
        "length_scale": 30,
        "overall_variance": 10**1.5,
        "mean_offset": 0.0,
    },
    "Over-smooth": {
        "s": 12.0,
        "length_scale": 30,
        "overall_variance": 10**1.5,
        "mean_offset": 0.0,
    },
    "Under-smooth": {
        "s": 3.0,
        "length_scale": 10,
        "overall_variance": 10**1.5,
        "mean_offset": 0.0,
    },
    "Shifted-mean": {
        "s": 6.0,
        "length_scale": 30,
        "overall_variance": 10**1.5,
        "mean_offset": 3.0,
    },
}

CASE_COLORS = {
    "Reference":    "tab:blue",
    "Over-smooth":  "tab:green",
    "Under-smooth": "tab:orange",
    "Shifted-mean": "tab:red",
}

# =============================================================================
# INFERENCE LOOP
# =============================================================================
solver = CholeskySolver(parallel=True, n_jobs=parallel_cfg.n_jobs)

results   = {}
log_lines = []

for label, cfg in PRIOR_CASES.items():
    print(f"\n{'='*70}")
    print(f"Running case: {label}")
    print(f"  s={cfg['s']}, length_scale={cfg['length_scale']}, "
          f"overall_variance={cfg['overall_variance']:.2f}, "
          f"mean_offset={cfg['mean_offset']}")
    print(f"{'='*70}")

    t0 = time.time()

    # Build vp prior for this case
    print("  Building vp prior …")
    M_prior_vp = build_vp_prior(
        s=cfg['s'], length_scale=cfg['length_scale'],
        overall_variance=cfg['overall_variance'],
        mean_offset=cfg['mean_offset'],
    )

    # Combine into full joint prior
    M_prior = build_full_prior(M_prior_vp)

    # Compute PRIOR property uncertainty (before seeing data)
    print("  Computing property prior uncertainty …")
    prior_P    = M_prior.affine_mapping(operator=T)
    cov_P_prior = prior_P.covariance.matrix(
        dense=True, parallel=True, n_jobs=parallel_cfg.n_jobs)
    std_P_prior = np.sqrt(np.diag(cov_P_prior))

    # Set up and run inference (Workflow 2: skip dense model posterior)
    print("  Running Bayesian inference (Workflow 2) …")
    forward_problem = LinearForwardProblem(
        G,
        data_error_measure=gaussian_D_noise,
    )
    bayesian_inference = LinearBayesianInversion(
        forward_problem,
        M_prior,
    )

    posterior_model = bayesian_inference.model_posterior_measure(
        d_tilde,
        solver,
    )
    m_tilde_post     = posterior_model.expectation

    property_posterior = posterior_model.affine_mapping(operator=T)
    p_tilde     = property_posterior.expectation
    cov_P_matrix = property_posterior.covariance.matrix(
        dense=True, parallel=True, n_jobs=parallel_cfg.n_jobs)

    elapsed = time.time() - t0

    # ── diagnostics ─────────────────────────────────────────────────────────
    std_P_post  = np.sqrt(np.diag(cov_P_matrix))
    errors      = p_tilde - true_props
    rmse        = float(np.sqrt(np.mean(errors ** 2)))
    mae         = float(np.mean(np.abs(errors)))
    cover       = float(np.sum(np.abs(errors) <= 2 * std_P_post) / N_p)
    mean_post_std  = float(np.mean(std_P_post))
    mean_prior_std = float(np.mean(std_P_prior))
    unc_reduction  = float(1.0 - mean_post_std / mean_prior_std)
    data_misfit    = float(np.linalg.norm(G(m_tilde_post) - d_tilde))
    min_eig        = float(np.linalg.eigvalsh(cov_P_matrix).min())

    row = dict(
        case           = label,
        s              = cfg['s'],
        length_scale   = cfg['length_scale'],
        overall_variance = cfg['overall_variance'],
        mean_offset    = cfg['mean_offset'],
        rmse           = round(rmse,  6),
        mae            = round(mae,   6),
        coverage_2sigma= round(cover, 4),
        mean_post_std  = round(mean_post_std,  6),
        mean_prior_std = round(mean_prior_std, 6),
        unc_reduction  = round(unc_reduction,  6),
        data_misfit    = round(data_misfit,    6),
        min_eig_cov    = round(min_eig,        8),
        elapsed_s      = round(elapsed,        2),
    )
    results[label] = row

    msg = (
        f"  DONE in {elapsed:.1f}s\n"
        f"  RMSE={rmse:.4f}  MAE={mae:.4f}\n"
        f"  Coverage(2σ)={cover:.0%}\n"
        f"  Mean post σ={mean_post_std:.4f}\n"
        f"  Mean prior σ={mean_prior_std:.4f}\n"
        f"  Uncertainty reduction={unc_reduction:.1%}\n"
        f"  Data misfit={data_misfit:.4f}\n"
        f"  min eigenvalue={min_eig:.2e}\n"
    )
    print(msg)
    log_lines.append(f"Case: {label}\n" + msg)

    if min_eig < -1e-6:
        warnings.warn(
            f"[{label}] Negative min eigenvalue ({min_eig:.2e}) — "
            "covariance matrix is not positive semi-definite; "
            "consider increasing regularisation or reducing length_scale.",
            RuntimeWarning,
        )

# =============================================================================
# SAVE RESULTS
# =============================================================================
fieldnames = [
    "case", "s", "length_scale", "overall_variance", "mean_offset",
    "rmse", "mae", "coverage_2sigma",
    "mean_post_std", "mean_prior_std", "unc_reduction",
    "data_misfit", "min_eig_cov", "elapsed_s",
]

# CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results.values():
        writer.writerow(row)
print(f"\nSaved CSV  → {OUTPUT_CSV}")

# Markdown
md_col_headers = [
    "Case", "s", "ls", "ov", "offset",
    "RMSE", "MAE", "Coverage(2σ)",
    "Mean σ_post", "Mean σ_prior", "Unc. Reduction",
    "Data Misfit", "Min Eig.", "Time (s)",
]
md_col_keys = fieldnames

with open(OUTPUT_MD, "w") as f:
    f.write("# Prior Sensitivity Study — Summary\n\n")
    f.write("All cases use identical vs, rho, σ₀, σ₁ priors. "
            "Only the vp prior is varied.\n\n")
    # Header
    f.write("| " + " | ".join(md_col_headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(md_col_headers)) + " |\n")
    for row in results.values():
        f.write("| " + " | ".join(str(row[k]) for k in md_col_keys) + " |\n")
    f.write("\n\n## Metric Definitions\n\n")
    f.write("| Metric | Definition |\n| --- | --- |\n")
    defs = [
        ("RMSE",            r"√( (1/Nₚ) Σᵢ (p̃ᵢ − pᵢ*)² )"),
        ("MAE",             "(1/Nₚ) Σᵢ |p̃ᵢ − pᵢ*|"),
        ("Coverage(2σ)",    "Fraction of i with |p̃ᵢ − pᵢ*| ≤ 2σᵢ"),
        ("Mean σ_post",     "(1/Nₚ) Σᵢ σᵢ  (posterior std)"),
        ("Mean σ_prior",    "(1/Nₚ) Σᵢ σᵢ⁰ (prior std pushed through T)"),
        ("Unc. Reduction",  "1 − Mean σ_post / Mean σ_prior"),
        ("Data Misfit",     "‖G(m̃) − d̃‖₂"),
        (
            "Min Eig.",
            "λ_min of property posterior covariance (stability)",
        ),
    ]
    for name, defn in defs:
        f.write(f"| {name} | {defn} |\n")
print(f"Saved MD   → {OUTPUT_MD}")

# Detailed text log
with open(OUTPUT_TXT, "w") as f:
    f.write("Prior Sensitivity Study — Detailed Log\n")
    f.write("=" * 70 + "\n\n")
    f.write("\n\n".join(log_lines))
print(f"Saved TXT  → {OUTPUT_TXT}")

# =============================================================================
# CONSOLE SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 90)
print(f"{'Case':<16} {'RMSE':>8} {'MAE':>8} {'Cover':>8} "
      f"{'UncRed':>8} {'Misfit':>9} {'MinEig':>10} {'Time':>7}")
print("-" * 90)
for row in results.values():
    print(
        f"{row['case']:<16} "
        f"{row['rmse']:>8.4f} "
        f"{row['mae']:>8.4f} "
        f"{row['coverage_2sigma']:>8.1%} "
        f"{row['unc_reduction']:>8.1%} "
        f"{row['data_misfit']:>9.4f} "
        f"{row['min_eig_cov']:>10.2e} "
        f"{row['elapsed_s']:>7.1f}s"
    )
print("=" * 90)
print("\nAll done.")
