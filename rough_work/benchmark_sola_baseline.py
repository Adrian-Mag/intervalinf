"""
Baseline benchmark harness for SOLAOperator.

Measures the current (un-optimised) runtime of the hottest SOLAOperator
code paths across a parameter matrix covering:

  - integration method: simpson, trapz
  - n_points: 200, 500, 1000, 2000
  - data dimension N_d: 1, 5, 20, 50, 200
  - property dimension N_p: 1, 2, 5, 20 (for DualMasterCostFunction)
  - kernel source: provider-backed (SineFunctionProvider), direct-callable,
                   compact-support (BumpFunctionProvider)
  - forward G(f), adjoint G*(y), DualMasterCostFunction.value_and_subgradient

Benchmark taxonomy
------------------
This harness intentionally separates:

- pure SOLA microbenchmarks: forward `G(f)` and adjoint `G*(y)` timings
- downstream workflow timings: `DualMasterCostFunction.value_and_subgradient`

The current implementation requires placing SOLA on a `Lebesgue` space object, so
an ambient basis dimension still exists in the setup. For this baseline harness we
hold that basis dimension fixed; it is treated as an implementation detail of the
surrounding space, not as a primary SOLA benchmark axis.

This harness is intentionally BASELINE-ONLY; it does not make any correctness
assertions.  Run it to capture numbers before Phase 3 optimization work starts,
and again afterward to measure speedup.

Usage
-----
From the workspace root, with the inferences3 conda environment:

    conda run -n inferences3 python intervalinf/rough_work/benchmark_sola_baseline.py

Options are driven by the SCENARIOS list at the bottom of this file.

Output
------
A CSV-style table printed to stdout, suitable for redirect to a .csv file:

    conda run -n inferences3 python intervalinf/rough_work/benchmark_sola_baseline.py \
        > intervalinf/rough_work/benchmark_sola_baseline_results.csv

Notes
-----
- Timings are wall-clock medians over N_REPEATS repeats (warm calls after the
  first).  First-call kernel-allocation overhead is averaged into the median
  for small N_REPEATS but is not isolated; treat these as steady-state medians.
- "adjoint_ms" measures the cost of one G.adjoint(y) call, including the full
  dual-to-primal reconstruction.
- "cost_vs_ms" measures one DualMasterCostFunction.value_and_subgradient(lam)
  call (only for N_p > 0 scenarios).
"""

from __future__ import annotations

import csv
import sys
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from intervalinf import IntervalDomain, Lebesgue, Function
from intervalinf import IntegrationConfig
from intervalinf import LebesgueIntegrationConfig
from intervalinf.providers import SineFunctionProvider, BumpFunctionProvider
from intervalinf.operators import SOLAOperator

from pygeoinf import EuclideanSpace
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_REPEATS = 5          # repeats for each timing
RNG_SEED = 42
DEFAULT_BASIS_DIM = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median_ms(fn, *args, n: int = N_REPEATS) -> float:
    """Return median wall-clock time in milliseconds for `fn(*args)`."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _make_test_function(space: Lebesgue, seed: int = 0) -> Function:
    """Return a simple analytic test function attached to `space`."""
    rng = np.random.default_rng(seed)
    a_coeff = rng.standard_normal()
    b_coeff = rng.standard_normal()
    return Function(
        space,
        evaluate_callable=lambda x: a_coeff * np.sin(np.pi * x) + b_coeff * x
    )


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    scenario: str          # human-readable label
    kernel_source: str     # "sine_provider" | "callable" | "bump_provider"
    method: str            # integration method
    n_points: int
    N_d: int
    N_p: int               # 0 means DualMasterCostFunction not measured
    basis_dim: int
    forward_ms: float
    adjoint_ms: float
    cost_vs_ms: Optional[float]   # None if N_p == 0


HEADER = (
    "scenario,kernel_source,method,n_points,N_d,N_p,basis_dim,"
    "forward_ms,adjoint_ms,cost_vs_ms"
)


def row_to_csv(r: BenchmarkRow) -> str:
    cost_str = f"{r.cost_vs_ms:.4f}" if r.cost_vs_ms is not None else ""
    return (
        f"{r.scenario},{r.kernel_source},{r.method},{r.n_points},"
        f"{r.N_d},{r.N_p},{r.basis_dim},"
        f"{r.forward_ms:.4f},{r.adjoint_ms:.4f},{cost_str}"
    )


# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------

def benchmark_forward(G: SOLAOperator, f: Function) -> float:
    """Time one G(f) call in ms."""
    return _median_ms(G, f)


def benchmark_adjoint(G: SOLAOperator, y: np.ndarray) -> float:
    """Time one G.adjoint(y) call in ms."""
    return _median_ms(G.adjoint, y)


def build_cost_function(
    M: Lebesgue,
    D: EuclideanSpace,
    P: EuclideanSpace,
    G: SOLAOperator,
    T: SOLAOperator,
    m_bar: Function,
    d_tilde: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> DualMasterCostFunction:
    """Build a DualMasterCostFunction from the given problem components."""
    m_0 = Function(M, evaluate_callable=lambda x: np.zeros_like(x) if not np.isscalar(x) else 0.0)
    model_radius = max(1.05 * M.norm(M.subtract(m_bar, m_0)), 1e-3)
    d_bar = G(m_bar)
    data_radius = max(1.05 * float(np.linalg.norm(d_tilde - d_bar)), 1e-6)
    model_prior_support = BallSupportFunction(M, m_0, model_radius)
    data_error_support = BallSupportFunction(D, D.zero, data_radius)
    return DualMasterCostFunction(
        D, P, M, G, T,
        model_prior_support, data_error_support,
        d_tilde, q,
    )


def benchmark_cost(cost: DualMasterCostFunction, lam: np.ndarray) -> float:
    """Time one cost.value_and_subgradient(lam) call in ms."""
    return _median_ms(cost.value_and_subgradient, lam)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_name: str,
    kernel_source: str,
    method: str,
    n_points: int,
    N_d: int,
    N_p: int,
    basis_dim: int,
) -> BenchmarkRow:
    """
    Build a full problem and benchmark forward, adjoint, and cost paths.

    Parameters
    ----------
    kernel_source : "sine_provider" | "callable" | "bump_provider"
    method        : "simpson" | "trapz"
    n_points      : quadrature points for SOLAOperator
    N_d           : data dimension
    N_p           : property dimension (0 → skip cost benchmark)
    basis_dim     : Lebesgue basis dimension
    """
    rng = np.random.default_rng(RNG_SEED)
    domain = IntervalDomain(0.0, 1.0)
    int_cfg = IntegrationConfig(method=method, n_points=n_points)
    leb_cfg = LebesgueIntegrationConfig.from_single(int_cfg)
    M = Lebesgue(basis_dim, domain, basis="cosine", integration_config=leb_cfg)
    D = EuclideanSpace(N_d)

    # --- Build data kernel provider / list ---
    if kernel_source == "sine_provider":
        data_kernels: object = SineFunctionProvider(domain)
    elif kernel_source == "callable":
        data_kernels = [
            (lambda k: lambda x: np.sin((k + 1) * np.pi * x))(i)
            for i in range(N_d)
        ]
    elif kernel_source == "bump_provider":
        centers = np.linspace(0.05, 0.95, N_d) if N_d > 1 else np.array([0.5])
        width = min(0.35, 0.8 / max(N_d, 1))
        data_kernels = BumpFunctionProvider(domain, centers=centers, default_width=width)
    else:
        raise ValueError(f"Unknown kernel_source: {kernel_source!r}")

    G = SOLAOperator(M, D, kernels=data_kernels, integration_config=int_cfg)

    # --- Test function and vector ---
    f = _make_test_function(M, seed=0)
    lam_np = rng.standard_normal(N_d)
    y = D.from_components(lam_np)

    # --- Timings ---
    fwd_ms = benchmark_forward(G, f)
    adj_ms = benchmark_adjoint(G, y)

    cost_ms: Optional[float] = None
    if N_p > 0:
        P = EuclideanSpace(N_p)
        prop_centers = np.linspace(0.1, 0.9, N_p) if N_p > 1 else np.array([0.5])
        prop_width = min(0.35, 0.8 / max(N_p, 1))
        prop_kernels = BumpFunctionProvider(domain, centers=prop_centers, default_width=prop_width)
        T = SOLAOperator(M, P, kernels=prop_kernels, integration_config=int_cfg)

        m_bar = _make_test_function(M, seed=1)
        d_tilde = G(m_bar) + rng.standard_normal(N_d) * 0.05
        q = P.basis_vector(0)
        cost = build_cost_function(M, D, P, G, T, m_bar, d_tilde, q, rng)
        cost_ms = benchmark_cost(cost, lam_np)

    return BenchmarkRow(
        scenario=scenario_name,
        kernel_source=kernel_source,
        method=method,
        n_points=n_points,
        N_d=N_d,
        N_p=N_p,
        basis_dim=basis_dim,
        forward_ms=fwd_ms,
        adjoint_ms=adj_ms,
        cost_vs_ms=cost_ms,
    )


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------

def build_scenario_list():
    """Return the list of (name, kwargs) dicts to benchmark."""
    scenarios = []

    # ---- Pure SOLA: integration method × n_points sweep (fixed ambient space) ---
    for method in ("simpson", "trapz"):
        for n_pts in (200, 500, 1000, 2000):
            name = f"method={method}_npts={n_pts}"
            scenarios.append(dict(
                scenario_name=name,
                kernel_source="sine_provider",
                method=method,
                n_points=n_pts,
                N_d=20,
                N_p=0,
                basis_dim=DEFAULT_BASIS_DIM,
            ))

    # ---- Pure SOLA: N_d sweep (fixed ambient space) ----
    for N_d in (1, 5, 20, 50, 200):
        name = f"N_d={N_d}"
        scenarios.append(dict(
            scenario_name=name,
            kernel_source="sine_provider",
            method="simpson",
            n_points=1000,
            N_d=N_d,
            N_p=0,
                basis_dim=DEFAULT_BASIS_DIM,
        ))

    # ---- Pure SOLA: kernel source sweep ----
    for ksrc in ("sine_provider", "callable", "bump_provider"):
        name = f"ksrc={ksrc}"
        scenarios.append(dict(
            scenario_name=name,
            kernel_source=ksrc,
            method="simpson",
            n_points=1000,
            N_d=20,
            N_p=0,
                basis_dim=DEFAULT_BASIS_DIM,
        ))

    # ---- Downstream workflow: DualMasterCostFunction hotspot (fixed ambient space) ----
    for N_d in (5, 20, 50):
        for N_p in (1, 2, 5, 20):
            name = f"cost_Nd={N_d}_Np={N_p}"
            scenarios.append(dict(
                scenario_name=name,
                kernel_source="sine_provider",
                method="simpson",
                n_points=1000,
                N_d=N_d,
                N_p=N_p,
                basis_dim=DEFAULT_BASIS_DIM,
            ))

    # ---- Pure SOLA: compact-support kernel performance ----
    for N_d in (5, 20, 50):
        name = f"bump_Nd={N_d}"
        scenarios.append(dict(
            scenario_name=name,
            kernel_source="bump_provider",
            method="simpson",
            n_points=1000,
            N_d=N_d,
            N_p=0,
                basis_dim=DEFAULT_BASIS_DIM,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(verbose: bool = True):
    scenarios = build_scenario_list()
    total = len(scenarios)

    print(HEADER, flush=True)
    for i, kwargs in enumerate(scenarios, 1):
        label = kwargs["scenario_name"]
        if verbose:
            print(
                f"[{i:3d}/{total}] running: {label} "
                f"(N_d={kwargs['N_d']}, N_p={kwargs['N_p']}, "
                f"basis={kwargs['basis_dim']}, method={kwargs['method']}, "
                f"n_pts={kwargs['n_points']}, ksrc={kwargs['kernel_source']})",
                file=sys.stderr,
                flush=True,
            )
        try:
            row = run_scenario(**kwargs)
            print(row_to_csv(row), flush=True)
        except Exception as exc:
            err_row = (
                f"{label},{kwargs['kernel_source']},{kwargs['method']},"
                f"{kwargs['n_points']},{kwargs['N_d']},{kwargs['N_p']},"
                f"{kwargs['basis_dim']},ERROR,ERROR,{exc!r}"
            )
            print(err_row, flush=True)
            if verbose:
                import traceback
                traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOLAOperator baseline benchmark")
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output to stderr"
    )
    args = parser.parse_args()
    main(verbose=not args.quiet)
