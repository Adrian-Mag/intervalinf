"""
benchmark_dli_instrumentation.py
=================================
Phase 1 proof-of-concept: validates that lightweight DLI instrumentation is
populated after running a small intervalinf-backed proximal-bundle DLI solve.

This script is intentionally minimal — it exercises the instrumentation hooks
added in Phase 1 rather than running a full parameter sweep.

Usage:
    cd /home/adrian/PhD/Inferences
    conda run -n inferences3 python intervalinf/rough_work/benchmark_dli_instrumentation.py
"""

from __future__ import annotations

import time
import textwrap

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Intervalinf / pygeoinf imports
# ---------------------------------------------------------------------------
from intervalinf import IntervalDomain, Lebesgue, Function
from intervalinf import (
    IntegrationConfig,
    ParallelConfig,
    LebesgueIntegrationConfig,
)
from intervalinf.providers import NormalModesProvider, BumpFunctionProvider
from intervalinf.operators import SOLAOperator

from pygeoinf import CholeskySolver, EuclideanSpace, LinearOperator
from pygeoinf.convex_analysis import (
    BallSupportFunction,
    EllipsoidSupportFunction,
)
from pygeoinf.backus_gilbert import DualMasterCostFunction, DualMasterStats
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    ProximalBundleStats,
    solve_support_values,
    best_available_qp_solver,
)

# ---------------------------------------------------------------------------
# Problem configuration (kept tiny so the script runs in a few seconds)
# ---------------------------------------------------------------------------
N_D = 10                    # data dimension
FORWARD_SEED = 7
DATA_SEED = 42
NOISE_FRACTION = 0.10
CONFIDENCE_LEVEL = 0.95
PRIOR_RADIUS_FACTOR = 1.05
LEBESGUE_N_POINTS = 200
SOLA_N_POINTS = 400
N_JOBS_SOLA = 4

DLI_TOLERANCE = 1e-4
DLI_MAX_ITERATIONS = 300
DLI_BUNDLE_SIZE = 30
DLI_RHO0 = 1.0
DLI_RHO_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Build problem
# ---------------------------------------------------------------------------

def build_small_problem() -> dict:
    """Build a minimal 1D DLI test problem."""
    lebesgue_cfg = LebesgueIntegrationConfig(
        inner_product=IntegrationConfig(method="simpson", n_points=LEBESGUE_N_POINTS),
        dual=IntegrationConfig(method="simpson", n_points=LEBESGUE_N_POINTS),
        general=IntegrationConfig(method="simpson", n_points=LEBESGUE_N_POINTS),
    )
    sola_cfg = IntegrationConfig(method="simpson", n_points=SOLA_N_POINTS)
    par_cfg = ParallelConfig(enabled=True, n_jobs=N_JOBS_SOLA)

    function_domain = IntervalDomain(0, 1)
    M = Lebesgue(0, function_domain, integration_config=lebesgue_cfg, parallel_config=par_cfg)
    D = EuclideanSpace(N_D)
    P = EuclideanSpace(1)

    normal_modes = NormalModesProvider(
        M,
        n_modes_range=(1, 30),
        coeff_range=(-5, 5),
        gaussian_width_percent_range=(1, 5),
        freq_range=(0.1, 20),
        random_state=FORWARD_SEED,
    )
    G = SOLAOperator(M, D, kernels=normal_modes, cache_kernels=True, integration_config=sola_cfg)

    bump_provider = BumpFunctionProvider(M, centers=np.array([0.5]), default_width=0.3)
    T = SOLAOperator(M, P, kernels=bump_provider, cache_kernels=True, integration_config=sola_cfg)

    # True model and data
    m_bar = Function(M, evaluate_callable=lambda x: np.exp(-((x - 0.5) / 0.5) ** 2) * np.sin(5 * np.pi * x) + x)
    d_bar = G(m_bar)
    signal_rms = np.linalg.norm(d_bar) / np.sqrt(N_D)
    sigma_d = NOISE_FRACTION * signal_rms

    rng = np.random.default_rng(DATA_SEED)
    d_tilde = d_bar + rng.normal(0.0, sigma_d, N_D)

    # Prior ball
    m_0 = Function(M, evaluate_callable=lambda x: x)
    model_radius = PRIOR_RADIUS_FACTOR * M.norm(M.subtract(m_bar, m_0))
    model_prior_support = BallSupportFunction(M, m_0, model_radius)

    # Data confidence ellipsoid
    chi2_q = chi2.ppf(CONFIDENCE_LEVEL, df=N_D)
    s_conf = np.sqrt(0.5 * chi2_q)
    A_op = LinearOperator.self_adjoint(D, lambda v: v / (2.0 * sigma_d ** 2))
    A_inv_op = LinearOperator.self_adjoint(D, lambda v: 2.0 * sigma_d ** 2 * v)
    A_inv_sqrt = LinearOperator.self_adjoint(D, lambda v: np.sqrt(2.0) * sigma_d * v)
    data_conf_ellipsoid = EllipsoidSupportFunction(
        D,
        center=D.zero,
        radius=float(s_conf),
        shape_operator=A_op,
        inverse_operator=A_inv_op,
        inverse_sqrt_operator=A_inv_sqrt,
    )

    return dict(
        M=M, D=D, P=P, G=G, T=T,
        d_tilde=d_tilde, sigma_d=sigma_d,
        model_prior_support=model_prior_support,
        data_conf_ellipsoid=data_conf_ellipsoid,
    )


# ---------------------------------------------------------------------------
# Run instrumented DLI
# ---------------------------------------------------------------------------

def run_instrumented_dli(prob: dict) -> dict:
    """Run DLI for +e1 and -e1 and return instrumentation summaries."""
    M, D, P, G, T = prob["M"], prob["D"], prob["P"], prob["G"], prob["T"]
    d_tilde = prob["d_tilde"]
    model_prior_support = prob["model_prior_support"]
    data_conf_ellipsoid = prob["data_conf_ellipsoid"]

    e1 = P.basis_vector(0)
    neg_e1 = P.negative(e1)
    lambda0 = D.zero

    cost = DualMasterCostFunction(
        D, P, M, G, T,
        model_prior_support, data_conf_ellipsoid,
        d_tilde, e1,
    )

    solver = ProximalBundleMethod(
        cost,
        rho0=DLI_RHO0,
        rho_factor=DLI_RHO_FACTOR,
        tolerance=DLI_TOLERANCE,
        max_iterations=DLI_MAX_ITERATIONS,
        bundle_size=DLI_BUNDLE_SIZE,
        qp_solver=best_available_qp_solver(),
    )

    t0 = time.perf_counter()

    # Solve +e1
    cost.reset_instrumentation()
    vals_pos, _, diags_pos = solve_support_values(cost, [e1], solver, lambda0, warm_start=False)
    dm_stats_pos: DualMasterStats = cost.instrumentation_stats
    pb_stats_pos: ProximalBundleStats = solver.instrumentation_stats
    h_pos = float(vals_pos[0])

    # Solve -e1 (reuse cost/solver; reset instrumentation)
    cost.reset_instrumentation()
    vals_neg, _, diags_neg = solve_support_values(cost, [neg_e1], solver, lambda0, warm_start=False)
    dm_stats_neg: DualMasterStats = cost.instrumentation_stats
    pb_stats_neg: ProximalBundleStats = solver.instrumentation_stats
    h_neg = float(vals_neg[0])

    elapsed = time.perf_counter() - t0

    return dict(
        h_pos=h_pos,
        h_neg=h_neg,
        upper=h_pos,
        lower=-h_neg,
        width=h_pos + h_neg,
        diag_pos=diags_pos[0],
        diag_neg=diags_neg[0],
        dm_stats_pos=dm_stats_pos,
        dm_stats_neg=dm_stats_neg,
        pb_stats_pos=pb_stats_pos,
        pb_stats_neg=pb_stats_neg,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _fmt_stats_table(label: str, dm: DualMasterStats, pb: ProximalBundleStats) -> str:
    lines = [f"\n=== {label} ==="]
    lines.append("  DualMasterCostFunction instrumentation:")
    lines.append(f"    num_value_and_subgradient_calls : {dm.num_value_and_subgradient_calls}")
    lines.append(f"    time_value_and_subgradient_s    : {dm.time_value_and_subgradient_s:.4f} s")
    lines.append(f"    time_gstar_apply_s              : {dm.time_gstar_apply_s:.4f} s")
    lines.append(f"    time_support_point_model_s      : {dm.time_support_point_model_s:.4f} s")
    lines.append(f"    time_support_point_data_s       : {dm.time_support_point_data_s:.4f} s")
    lines.append(f"    time_support_value_model_s      : {dm.time_support_value_model_s:.4f} s")
    lines.append(f"    time_support_value_data_s       : {dm.time_support_value_data_s:.4f} s")
    lines.append(f"    num_support_point_failures      : {dm.num_support_point_failures}")
    lines.append(f"    num_finite_difference_fallbacks : {dm.num_finite_difference_fallbacks}")
    lines.append(f"    time_finite_difference_s        : {dm.time_finite_difference_s:.4f} s")
    lines.append("  ProximalBundleMethod instrumentation:")
    lines.append(f"    num_master_solves               : {pb.num_master_solves}")
    lines.append(f"    time_master_solve_s             : {pb.time_master_solve_s:.4f} s")
    lines.append(f"    num_serious_steps               : {pb.num_serious_steps}")
    lines.append(f"    num_null_steps                  : {pb.num_null_steps}")
    lines.append(f"    time_oracle_total_s             : {pb.time_oracle_total_s:.4f} s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_instrumentation_populated(
    label: str,
    dm: DualMasterStats,
    pb: ProximalBundleStats,
    diag,
) -> None:
    assert dm.num_value_and_subgradient_calls >= 1, (
        f"{label}: expected >= 1 value_and_subgradient calls, got {dm.num_value_and_subgradient_calls}"
    )
    assert dm.time_value_and_subgradient_s > 0, (
        f"{label}: time_value_and_subgradient_s should be > 0"
    )
    assert dm.time_gstar_apply_s > 0, (
        f"{label}: time_gstar_apply_s should be > 0"
    )
    # num_value_and_subgradient_calls should match oracle evaluations in BundleResult
    # (+1 for the initial evaluation in solve())
    assert dm.num_value_and_subgradient_calls == diag.num_iterations + 1, (
        f"{label}: dm calls ({dm.num_value_and_subgradient_calls}) != "
        f"bundle iterations+1 ({diag.num_iterations + 1})"
    )
    assert pb.num_master_solves >= 1, (
        f"{label}: expected >= 1 master solves, got {pb.num_master_solves}"
    )
    assert pb.time_master_solve_s > 0, (
        f"{label}: time_master_solve_s should be > 0"
    )
    assert pb.num_master_solves == diag.num_iterations, (
        f"{label}: pb.num_master_solves ({pb.num_master_solves}) != "
        f"bundle iterations ({diag.num_iterations})"
    )
    assert pb.num_serious_steps + pb.num_null_steps == diag.num_iterations, (
        f"{label}: serious+null ({pb.num_serious_steps}+{pb.num_null_steps}) "
        f"!= num_iterations ({diag.num_iterations})"
    )
    assert pb.num_serious_steps == diag.num_serious_steps, (
        f"{label}: pb.num_serious_steps ({pb.num_serious_steps}) != "
        f"BundleResult.num_serious_steps ({diag.num_serious_steps})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building small intervalinf DLI problem ...")
    t0 = time.perf_counter()
    prob = build_small_problem()
    print(f"  Problem built in {time.perf_counter() - t0:.2f} s")

    print("\nRunning instrumented DLI solve (proximal bundle, +e1 and -e1) ...")
    result = run_instrumented_dli(prob)

    # Print interval result
    print(f"\nDLI interval: [{result['lower']:.6f}, {result['upper']:.6f}]")
    print(f"Width        : {result['width']:.6f}")
    print(f"Total elapsed: {result['elapsed_s']:.3f} s")
    print(f"Converged (+e1 / -e1): "
          f"{result['diag_pos'].converged} / {result['diag_neg'].converged}")
    print(f"Iterations  (+e1 / -e1): "
          f"{result['diag_pos'].num_iterations} / {result['diag_neg'].num_iterations}")

    # Print instrumentation summary
    print(_fmt_stats_table("+e1 direction", result["dm_stats_pos"], result["pb_stats_pos"]))
    print(_fmt_stats_table("-e1 direction", result["dm_stats_neg"], result["pb_stats_neg"]))

    # Assertions
    print("\nRunning assertions ...")
    assert_instrumentation_populated("+e1", result["dm_stats_pos"], result["pb_stats_pos"], result["diag_pos"])
    assert_instrumentation_populated("-e1", result["dm_stats_neg"], result["pb_stats_neg"], result["diag_neg"])
    print("  All assertions PASSED.")


if __name__ == "__main__":
    main()
