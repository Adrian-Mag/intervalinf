"""
Benchmark: Continuous DLI convex optimisation methods.

Tests ProximalBundleMethod, LevelBundleMethod, ChambollePockSolver, and
SmoothedLBFGSSolver (via SmoothedDualMaster) on the continuous DLI problem
built from intervalinf Lebesgue + SOLAOperator, WITHOUT Galerkin matrix
conversion.

Usage:
    conda run -n inferences3 python intervalinf/rough_work/benchmark_dli_solvers.py

Problem sizes:
    tiny  : Nd=1, Np=1
    small : Nd=5, Np=2
    medium: Nd=50, Np=20
"""

from __future__ import annotations

import time
import textwrap
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from intervalinf import IntervalDomain, Lebesgue, Function
from intervalinf import IntegrationConfig, LebesgueIntegrationConfig
from intervalinf.providers import NormalModesProvider, BumpFunctionProvider
from intervalinf.operators import SOLAOperator

from pygeoinf import EuclideanSpace
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    LevelBundleMethod,
    ChambollePockSolver,
    SmoothedLBFGSSolver,
    SmoothedDualMaster,
    OSQPQPSolver,
    solve_support_values,
    solve_primal_feasibility,
    best_available_qp_solver,
)

# ---------------------------------------------------------------------------
# Process-based timeout
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _worker(fn, args, result_queue):
    """Target for the worker process: put (result,) or (exception,) on the queue."""
    try:
        result_queue.put(("ok", fn(*args)))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("err", exc))


def run_with_timeout(fn, args, seconds: float):
    """Run fn(*args) in a child process, kill it if it exceeds *seconds*.

    Returns the function's return value, or raises _TimeoutError / the
    original exception as appropriate.

    Unlike signal.SIGALRM, this reliably interrupts C extensions (numpy,
    scipy, OSQP, etc.) because it terminates the whole child process.
    """
    ctx = mp.get_context("fork")   # fork is fastest on Linux; avoids re-importing
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, args, q), daemon=True)
    p.start()
    p.join(seconds)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
            p.join()
        raise _TimeoutError(f"Timed out after {seconds}s")
    status, payload = q.get_nowait()
    if status == "err":
        raise payload
    return payload


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def build_dli_problem(Nd: int, Np: int, *, dim_basis: int = 30, seed: int = 42):
    """Build a continuous DLI problem instance.

    Returns a dict with all needed objects; no Galerkin matrix is built.
    """
    domain = IntervalDomain(0, 1)

    int_cfg = LebesgueIntegrationConfig(
        inner_product=IntegrationConfig(method="simpson", n_points=500),
        dual=IntegrationConfig(method="simpson", n_points=500),
        general=IntegrationConfig(method="simpson", n_points=500),
    )
    sola_cfg = IntegrationConfig(method="simpson", n_points=500)

    M = Lebesgue(dim_basis, domain, basis="cosine", integration_config=int_cfg)
    D = EuclideanSpace(Nd)
    P = EuclideanSpace(Np)

    normal_modes = NormalModesProvider(
        M,
        n_modes_range=(1, min(20, dim_basis)),
        coeff_range=(-3, 3),
        gaussian_width_percent_range=(2, 8),
        freq_range=(0.1, 10),
        random_state=seed,
    )
    G = SOLAOperator(M, D, kernels=normal_modes, integration_config=sola_cfg, cache_kernels=True)

    width = min(0.3, 0.8 / max(Np, 1))
    centers = np.linspace(domain.a + width / 2, domain.b - width / 2, Np)
    bump_provider = BumpFunctionProvider(M, centers=centers, default_width=width)
    T = SOLAOperator(M, P, kernels=bump_provider, integration_config=sola_cfg, cache_kernels=True)

    # True model and synthetic data
    rng = np.random.default_rng(seed)
    m_bar = Function(M, evaluate_callable=lambda x: np.sin(3 * np.pi * x) + 0.5 * x)
    d_bar = G(m_bar)
    noise_level = 0.1 * max(float(np.max(np.abs(d_bar))), 1e-6)
    d_tilde = d_bar + rng.normal(0, noise_level, d_bar.shape)

    # Prior sets
    m_0 = Function(M, evaluate_callable=lambda x: np.zeros_like(x) if not np.isscalar(x) else 0.0)
    model_radius = max(1.05 * M.norm(M.subtract(m_bar, m_0)), 1e-3)
    data_radius = max(1.05 * float(np.linalg.norm(d_tilde - d_bar)), 1e-6)

    model_prior_support = BallSupportFunction(M, m_0, model_radius)
    data_error_support = BallSupportFunction(D, D.zero, data_radius)

    return dict(
        M=M, D=D, P=P,
        G=G, T=T,
        m_bar=m_bar, d_tilde=d_tilde,
        model_prior_support=model_prior_support,
        data_error_support=data_error_support,
        centers=centers,
    )


# ---------------------------------------------------------------------------
# Operator-call profiling
# ---------------------------------------------------------------------------

def profile_operator_calls(prob: dict, n_repeats: int = 3) -> dict[str, float]:
    """Time individual operator evaluations on a given problem."""
    M, D, G, T = prob["M"], prob["D"], prob["G"], prob["T"]
    d_tilde = prob["d_tilde"]
    model_prior_support = prob["model_prior_support"]
    data_error_support = prob["data_error_support"]

    # A random lambda (dual variable in data space D)
    lam_np = np.random.default_rng(0).standard_normal(D.dim)
    lam = D.from_components(lam_np)

    q = prob["P"].basis_vector(0)
    Tstar_q = T.adjoint(q)

    # Build cost just for profiling
    cost = DualMasterCostFunction(
        D, prob["P"], M, G, T,
        model_prior_support, data_error_support,
        d_tilde, q,
    )

    def _time_fn(fn, *args) -> float:
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            fn(*args)
        return (time.perf_counter() - t0) / n_repeats

    t_Gm = _time_fn(G, M.basis_vector(0))
    t_Gadj = _time_fn(G.adjoint, lam)
    t_cost = _time_fn(cost.value_and_subgradient, lam)

    return {
        "G(m)_ms": t_Gm * 1e3,
        "G.adjoint(lam)_ms": t_Gadj * 1e3,
        "cost.value_and_subgradient(lam)_ms": t_cost * 1e3,
    }


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    method: str
    size: str
    Nd: int
    Np: int
    status: str          # "ok" | "failed" | "timeout"
    wall_s: float
    iterations: Optional[int]
    upper_bounds: Optional[np.ndarray]
    lower_bounds: Optional[np.ndarray]
    gap: Optional[float]
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Individual method runners
# ---------------------------------------------------------------------------

TIMEOUT_S = 1800.0   # 30 minutes per method/size combination


def _run_proximal_bundle(prob: dict, qs_pos, qs_neg, lambda0) -> tuple:
    qp_solver = best_available_qp_solver()
    solver = ProximalBundleMethod(
        None,  # oracle set per-direction inside solve_support_values
        rho0=1.0,
        rho_factor=2.0,
        tolerance=1e-3,
        max_iterations=1000,
        bundle_size=100,
        qp_solver=OSQPQPSolver(),  # Use OSQP directly for the proximal step, to avoid overhead of generic QP solver selection in ProximalBundleMethod
    )
    D, P, M, G, T = prob["D"], prob["P"], prob["M"], prob["G"], prob["T"]
    cost = DualMasterCostFunction(
        D, P, M, G, T,
        prob["model_prior_support"], prob["data_error_support"],
        prob["d_tilde"], qs_pos[0],
    )
    # Re-bind oracle
    solver._oracle = cost

    vals_pos, _, diags_pos = solve_support_values(cost, qs_pos, solver, lambda0)
    vals_neg, _, diags_neg = solve_support_values(cost, qs_neg, solver, lambda0)
    upper = vals_pos
    lower = -vals_neg
    total_iters = sum(d.num_iterations for d in diags_pos + diags_neg)
    return upper, lower, total_iters


def _run_level_bundle(prob: dict, qs_pos, qs_neg, lambda0) -> tuple:
    qp_solver = best_available_qp_solver()
    solver = LevelBundleMethod(
        None,
        alpha=0.1,
        tolerance=1e-4,
        max_iterations=200,
        bundle_size=30,
        qp_solver=qp_solver,
    )
    D, P, M, G, T = prob["D"], prob["P"], prob["M"], prob["G"], prob["T"]
    cost = DualMasterCostFunction(
        D, P, M, G, T,
        prob["model_prior_support"], prob["data_error_support"],
        prob["d_tilde"], qs_pos[0],
    )
    solver._oracle = cost

    vals_pos, _, diags_pos = solve_support_values(cost, qs_pos, solver, lambda0)
    vals_neg, _, diags_neg = solve_support_values(cost, qs_neg, solver, lambda0)
    upper = vals_pos
    lower = -vals_neg
    total_iters = sum(d.num_iterations for d in diags_pos + diags_neg)
    return upper, lower, total_iters


def _run_chambolle_pock(prob: dict, qs_pos, qs_neg) -> tuple:
    D, P, M, G, T = prob["D"], prob["P"], prob["M"], prob["G"], prob["T"]
    cost = DualMasterCostFunction(
        D, P, M, G, T,
        prob["model_prior_support"], prob["data_error_support"],
        prob["d_tilde"], qs_pos[0],
    )
    cp_solver = ChambollePockSolver(
        prob["model_prior_support"],
        prob["data_error_support"],
        G,
        prob["d_tilde"],
        max_iterations=1000,
        tolerance=1e-4,
    )
    upper = solve_primal_feasibility(cost, qs_pos, cp_solver)
    lower = -solve_primal_feasibility(cost, qs_neg, cp_solver)
    # No per-direction iteration count from solve_primal_feasibility
    return upper, lower, None


def _run_smoothed_lbfgsb(prob: dict, qs_pos, qs_neg, lambda0) -> tuple:
    D, P, M, G, T = prob["D"], prob["P"], prob["M"], prob["G"], prob["T"]
    cost = DualMasterCostFunction(
        D, P, M, G, T,
        prob["model_prior_support"], prob["data_error_support"],
        prob["d_tilde"], qs_pos[0],
    )
    solver = SmoothedLBFGSSolver(
        cost,
        epsilon0=1e-2,
        n_levels=4,
        tolerance=1e-6,
        max_iter_per_level=200,
    )

    vals_pos = []
    vals_neg = []
    total_iters = 0
    lam_current = lambda0

    for q in qs_pos:
        cost.set_direction(q)
        r = solver.solve(lam_current)
        vals_pos.append(r.f_best)
        lam_current = r.x_best
        total_iters += r.num_iterations

    lam_current = lambda0
    for q in qs_neg:
        cost.set_direction(q)
        r = solver.solve(lam_current)
        vals_neg.append(r.f_best)
        lam_current = r.x_best
        total_iters += r.num_iterations

    upper = np.array(vals_pos)
    lower = -np.array(vals_neg)
    return upper, lower, total_iters


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

SIZES = {
    "tiny":   (1, 1),
    "small":  (5, 2),
    "medium - small": (50, 20),
    "medium": (100, 20),
    "medium - large": (200, 20),
    "large":  (500, 20)
}

METHODS = [
    "ProximalBundle",
]


def run_benchmark(timeout: float = TIMEOUT_S) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for size_name, (Nd, Np) in SIZES.items():
        print(f"\n{'='*60}")
        print(f"SIZE: {size_name}  (Nd={Nd}, Np={Np})")
        print(f"{'='*60}")

        print(f"  Building problem...", end=" ", flush=True)
        t0 = time.perf_counter()
        prob = build_dli_problem(Nd, Np)
        t_build = time.perf_counter() - t0
        print(f"done ({t_build:.2f}s)")

        P = prob["P"]
        D = prob["D"]
        qs_pos = [P.basis_vector(i) for i in range(P.dim)]
        qs_neg = [P.multiply(-1.0, P.basis_vector(i)) for i in range(P.dim)]
        lambda0 = D.zero

        for method in METHODS:
            print(f"  [{method}]", end=" ", flush=True)
            t_start = time.perf_counter()
            status = "ok"
            upper = lower = None
            iters = None
            err_msg = ""

            try:
                if method == "ProximalBundle":
                    upper, lower, iters = run_with_timeout(
                        _run_proximal_bundle,
                        (prob, qs_pos, qs_neg, lambda0),
                        timeout,
                    )
                elif method == "LevelBundle":
                    upper, lower, iters = run_with_timeout(
                        _run_level_bundle,
                        (prob, qs_pos, qs_neg, lambda0),
                        timeout,
                    )
                elif method == "ChambollePock":
                    upper, lower, iters = run_with_timeout(
                        _run_chambolle_pock,
                        (prob, qs_pos, qs_neg),
                        timeout,
                    )
                elif method == "SmoothedLBFGSB":
                    upper, lower, iters = run_with_timeout(
                        _run_smoothed_lbfgsb,
                        (prob, qs_pos, qs_neg, lambda0),
                        timeout,
                    )

            except _TimeoutError:
                status = "timeout"
                err_msg = f">{timeout:.0f}s"
            except Exception as exc:
                status = "failed"
                err_msg = str(exc)[:80]

            wall = time.perf_counter() - t_start

            # Compute gap (half-width) if available
            gap_summary = None
            if upper is not None and lower is not None:
                gap_summary = float(np.mean(upper - lower))

            results.append(
                BenchmarkResult(
                    method=method,
                    size=size_name,
                    Nd=Nd, Np=Np,
                    status=status,
                    wall_s=wall,
                    iterations=iters,
                    upper_bounds=upper,
                    lower_bounds=lower,
                    gap=gap_summary,
                    error_msg=err_msg,
                )
            )

            # Inline summary
            if status == "ok":
                bounds_str = ""
                if upper is not None:
                    bounds_str = (
                        f"upper={np.round(upper, 4)}, lower={np.round(lower, 4)}"
                    )
                iters_str = f"iters={iters}" if iters is not None else "iters=n/a"
                print(
                    f"{status} | {wall:.2f}s | {iters_str} | gap≈{gap_summary:.4f} | {bounds_str}"
                )
            else:
                print(f"{status} | {wall:.2f}s | {err_msg}")

    return results


# ---------------------------------------------------------------------------
# Profiling section
# ---------------------------------------------------------------------------

def run_profiling():
    print(f"\n{'='*60}")
    print("OPERATOR-CALL PROFILING  (avg of 3 repeats)")
    print(f"{'='*60}")

    header = f"{'size':<8} | {'Nd':>4} {'Np':>4} | {'G(m) ms':>12} | {'G.adj ms':>12} | {'cost_vs ms':>14}"
    print(header)
    print("-" * len(header))

    for size_name, (Nd, Np) in SIZES.items():
        prob = build_dli_problem(Nd, Np)
        timings = profile_operator_calls(prob)
        print(
            f"{size_name:<8} | {Nd:>4} {Np:>4} | "
            f"{timings['G(m)_ms']:>11.2f}ms | "
            f"{timings['G.adjoint(lam)_ms']:>11.2f}ms | "
            f"{timings['cost.value_and_subgradient(lam)_ms']:>13.2f}ms"
        )


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(results: list[BenchmarkResult]):
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY TABLE")
    print(f"{'='*80}")

    col_w = [16, 8, 6, 6, 10, 8, 10, 14, 30]
    header = (
        f"{'Method':<16} | {'Size':<8} | {'Nd':>4} | {'Np':>4} | "
        f"{'Status':<10} | {'Wall(s)':>7} | {'Iters':>7} | {'AvgGap':>8} | {'Bounds (first prop)'}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in results:
        bounds_str = "n/a"
        if r.upper_bounds is not None and len(r.upper_bounds) > 0:
            bounds_str = f"[{r.lower_bounds[0]:.3f}, {r.upper_bounds[0]:.3f}]"
        iters_str = str(r.iterations) if r.iterations is not None else "n/a"
        gap_str = f"{r.gap:.4f}" if r.gap is not None else "n/a"

        print(
            f"{r.method:<16} | {r.size:<8} | {r.Nd:>4} | {r.Np:>4} | "
            f"{r.status:<10} | {r.wall_s:>7.2f} | {iters_str:>7} | "
            f"{gap_str:>8} | {bounds_str}"
        )

        if r.error_msg:
            print(f"  Error: {r.error_msg}")

    print(sep)

    # Bottleneck analysis
    print("\nBOTTLENECK ANALYSIS:")
    ok_results = [r for r in results if r.status == "ok"]
    if ok_results:
        slowest = max(ok_results, key=lambda r: r.wall_s)
        fastest = min(ok_results, key=lambda r: r.wall_s)
        print(f"  Slowest: {slowest.method} on {slowest.size} → {slowest.wall_s:.2f}s")
        print(f"  Fastest: {fastest.method} on {fastest.size} → {fastest.wall_s:.2f}s")

        # Group by method across sizes
        from collections import defaultdict
        by_method: dict = defaultdict(list)
        for r in ok_results:
            by_method[r.method].append(r.wall_s)
        print("\n  Mean wall-clock per method (over successful runs):")
        for method in METHODS:
            times = by_method.get(method, [])
            if times:
                print(f"    {method:<18}: mean={np.mean(times):.2f}s, max={np.max(times):.2f}s")
            else:
                print(f"    {method:<18}: no successful runs")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print(f"\n  FAILURES / TIMEOUTS ({len(failed)}):")
        for r in failed:
            print(f"    {r.method} on {r.size}: {r.status} — {r.error_msg}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DLI CONTINUOUS SOLVER BENCHMARK")
    print("=" * 60)
    print(f"Timeout per method/size: {TIMEOUT_S}s")
    print(f"Problem sizes: {list(SIZES.keys())}")
    print(f"Methods: {METHODS}")

    # Profiling
    run_profiling()

    # Main benchmark
    results = run_benchmark(timeout=TIMEOUT_S)

    # Summary table
    print_summary_table(results)

    print("\nDone.")
