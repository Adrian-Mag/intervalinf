"""
Phase 6 end-to-end comparison benchmark.

Produces a final summary comparing three forward-path variants across the full
scenario matrix from Phase 2:

    generic   — forced generic path in the current codebase,
                            corresponding to Phase 2 behavior
    fast      — forced Phase 4 batched path
    cached    — forced Phase 5 cached batched path

Also validates:
    accuracy  — max absolute error vs a high-accuracy adaptive reference
    adjoint   — heuristic residual <G(f), y>_D - <f, G*(y)>_M

Run from workspace root:
    python rough_work/benchmark_phase6_comparison.py

Output: printed table + a CSV artifact written alongside this file.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from intervalinf import IntervalDomain, Lebesgue, Function, IntegrationConfig
from intervalinf.providers import SineFunctionProvider, BumpFunctionProvider
from intervalinf.operators import SOLAOperator
from pygeoinf import EuclideanSpace


MethodName = Literal["simpson", "trapz", "adaptive", "quad"]


class Scenario(TypedDict):
    label: str
    method: MethodName
    n_points: int
    N_d: int
    kernel_source: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_OUTER = 8          # outer timing repeats for single-call benchmarks
N_REPS = 30          # input functions for repeated/cached timing
RNG_SEED = 42
rng_global = np.random.default_rng(RNG_SEED)


def _median_ms(fn, *args, n: int = N_OUTER) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Accuracy validation
# ---------------------------------------------------------------------------

def _accuracy_error(
    G_test: SOLAOperator,
    f: Function,
    G_ref: SOLAOperator,
) -> float:
    """Max absolute error of G_test(f) vs G_ref(f)."""
    y_test = G_test(f)
    y_ref = G_ref(f)
    return float(np.max(np.abs(y_test - y_ref)))


def _adjoint_consistency(
    G: SOLAOperator,
    f: Function,
    y_np: np.ndarray,
) -> float:
    """|<G(f), y>_D − <f, G*(y)>_M|.

    G.adjoint(y) takes a numpy array for EuclideanSpace codomain and returns a
    Function.  We compute the rhs via direct integration of f * G*(y).
    """
    Gf = G(f)
    lhs = float(np.dot(Gf, y_np))
    Gstar_y = G.adjoint(y_np)  # returns a Function
    domain = f.function_domain
    product_fn = Function(
        domain,
        evaluate_callable=lambda x, _f=f, _g=Gstar_y: (
            _f.evaluate(x, check_domain=False)
            * _g.evaluate(x, check_domain=False)
        ),
    )
    rhs = float(product_fn.integrate(method="simpson", n_points=2000))
    return abs(lhs - rhs)


# ---------------------------------------------------------------------------
# Per-scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    *,
    label: str,
    method: MethodName,
    n_points: int,
    N_d: int,
    kernel_source: str,
) -> dict:
    domain = IntervalDomain(0.0, 1.0)
    M = Lebesgue(30, domain, basis="cosine")
    D = EuclideanSpace(N_d)

    cfg_test = IntegrationConfig(method=method, n_points=n_points)
    # High-accuracy reference via adaptive quad
    cfg_ref = IntegrationConfig(method="adaptive", n_points=n_points)

    # --- kernels ---
    if kernel_source == "sine_provider":
        kernels_obj = SineFunctionProvider(domain)
    elif kernel_source == "callable":
        kernels_obj = [
            (lambda k: lambda x: np.sin((k + 1) * np.pi * x))(i)
            for i in range(N_d)
        ]
    elif kernel_source == "bump_provider":
        centers = np.linspace(0.05, 0.95, N_d) if N_d > 1 else np.array([0.5])
        width = min(0.35, 0.8 / max(N_d, 1))
        kernels_obj = BumpFunctionProvider(
            domain,
            centers=centers,
            default_width=width,
        )
    else:
        raise ValueError(kernel_source)

    # Three operator variants
    G_generic = SOLAOperator(M, D, kernels=kernels_obj, cache_kernels=False,
                             integration_config=cfg_test)
    G_fast = SOLAOperator(M, D, kernels=kernels_obj, cache_kernels=False,
                          integration_config=cfg_test)
    G_cached = SOLAOperator(M, D, kernels=kernels_obj, cache_kernels=True,
                            integration_config=cfg_test)
    G_ref = SOLAOperator(M, D, kernels=kernels_obj, cache_kernels=False,
                         integration_config=cfg_ref)

    # Test function
    f = Function(M, evaluate_callable=lambda x: np.sin(np.pi * x) + 0.5 * x)
    y_np = rng_global.standard_normal(N_d)

    # --- single-call timings ---
    t_generic = _median_ms(lambda: G_generic._apply_kernels_generic(f))
    t_fast = _median_ms(lambda: G_fast._apply_kernels_fixed_grid(f))
    _ = G_cached._apply_kernels_fixed_grid(f)  # warm up eval cache
    t_cached = _median_ms(lambda: G_cached._apply_kernels_fixed_grid(f))

    # --- repeated-workload timings (N_REPS distinct inputs) ---
    inputs = [
        Function(M, evaluate_callable=(
            lambda a: (lambda x: np.sin((a + 1) * np.pi * x)))(a))
        for a in range(N_REPS)
    ]
    # cached: pre-warm
    _ = G_cached._apply_kernels_fixed_grid(inputs[0])

    def _timed_batch(op):
        t0 = time.perf_counter()
        for fi in inputs:
            op(fi)
        return (time.perf_counter() - t0) * 1000

    batch_generic = statistics.median(
        [_timed_batch(G_generic._apply_kernels_generic) for _ in range(4)]
    )
    batch_fast = statistics.median(
        [_timed_batch(G_fast._apply_kernels_fixed_grid) for _ in range(4)]
    )
    batch_cached = statistics.median(
        [_timed_batch(G_cached._apply_kernels_fixed_grid) for _ in range(4)]
    )

    # --- accuracy ---
    acc_generic = (
        _accuracy_error(G_generic, f, G_ref)
        if method != "adaptive"
        else 0.0
    )
    acc_fast = (
        _accuracy_error(G_fast, f, G_ref)
        if method != "adaptive"
        else 0.0
    )

    # --- adjoint consistency ---
    adj_err = _adjoint_consistency(G_fast, f, y_np)

    return {
        "label": label,
        "method": method,
        "n_points": n_points,
        "N_d": N_d,
        "kernel_source": kernel_source,
        # single-call ms
        "generic_ms": t_generic,
        "fast_ms": t_fast,
        "cached_ms": t_cached,
        "speedup_fast_vs_generic": t_generic / max(t_fast, 1e-9),
        "speedup_cached_vs_generic": t_generic / max(t_cached, 1e-9),
        # batch timings (N_REPS=30 total)
        "batch_generic_ms": batch_generic,
        "batch_fast_ms": batch_fast,
        "batch_cached_ms": batch_cached,
        "batch_speedup_cached_vs_generic": (
            batch_generic / max(batch_cached, 1e-9)
        ),
        # accuracy vs adaptive reference
        "acc_generic_vs_ref": acc_generic,
        "acc_fast_vs_ref": acc_fast,
        # adjoint consistency
        "adjoint_residual": adj_err,
    }


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    # N_d scaling (main axis)
    *[
        Scenario(
            label=f"N_d={N}",
            method="simpson",
            n_points=1000,
            N_d=N,
            kernel_source="sine_provider",
        )
        for N in [5, 10, 20, 50, 100, 200]
    ],
    # n_points sweep
    *[
        Scenario(
            label=f"npts={p}",
            method="simpson",
            n_points=p,
            N_d=20,
            kernel_source="sine_provider",
        )
        for p in [200, 500, 1000, 2000]
    ],
    # trapz variant
    *[
        Scenario(
            label=f"trapz_N_d={N}",
            method="trapz",
            n_points=1000,
            N_d=N,
            kernel_source="sine_provider",
        )
        for N in [20, 100]
    ],
    # kernel source comparison
    Scenario(
        label="ksrc=callable",
        method="simpson",
        n_points=1000,
        N_d=20,
        kernel_source="callable",
    ),
    Scenario(
        label="ksrc=bump_Nd=20",
        method="simpson",
        n_points=1000,
        N_d=20,
        kernel_source="bump_provider",
    ),
    Scenario(
        label="ksrc=bump_Nd=50",
        method="simpson",
        n_points=1000,
        N_d=50,
        kernel_source="bump_provider",
    ),
]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_COL_W = {
    "label": 22, "method": 8, "n_points": 8, "N_d": 5,
    "generic_ms": 12, "fast_ms": 10, "cached_ms": 10,
    "sp_4": 8, "sp_5": 8,
    "batch_sp": 10,
    "acc_fast": 12, "adj": 12,
}


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:<22}  "
        f"{r['method']:<7}  "
        f"{r['n_points']:>5}  "
        f"{r['N_d']:>5}  "
        f"{r['generic_ms']:>10.3f}  "
        f"{r['fast_ms']:>8.3f}  "
        f"{r['cached_ms']:>8.3f}  "
        f"{r['speedup_fast_vs_generic']:>6.2f}x  "
        f"{r['speedup_cached_vs_generic']:>6.2f}x  "
        f"{r['batch_speedup_cached_vs_generic']:>7.2f}x  "
        f"{r['acc_fast_vs_ref']:>10.2e}  "
        f"{r['adjoint_residual']:>10.2e}"
    )


HEADER = (
    f"{'label':<22}  {'method':<7}  {'npts':>5}  {'N_d':>5}  "
    f"{'generic_ms':>10}  {'fast_ms':>8}  {'cached_ms':>8}  "
    f"{'P4_spdup':>6}  {'P5_spdup':>6}  "
    f"{'batchSp':>7}  "
    f"{'acc_fast':>10}  {'adjoint':>10}"
)

LEGEND = """
Columns:
    generic_ms   — per-call time using the forced generic path
                                 (_apply_kernels_generic), corresponding to Phase 2 behavior
    fast_ms      — per-call time using the forced Phase 4 batched path
    cached_ms    — per-call time using the forced Phase 5 cached batched path
  P4_spdup     — single-call speedup of fast over generic
  P5_spdup     — single-call speedup of cached over generic
  batchSp      — batch speedup of cached over generic (30 distinct inputs)
  acc_fast     — max |fast - adaptive_ref| (quality of batched quadrature)
  adjoint      — |<G(f),y>_D - <f,G*(y)>_M| (adjoint consistency residual)
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []
    print(LEGEND)
    print(HEADER)
    print("─" * len(HEADER))
    for sc in SCENARIOS:
        name = sc["label"]
        print(f"  running: {name} ...", end="\r", flush=True)
        r = run_scenario(**sc)
        rows.append(r)
        print(_fmt_row(r))

    # --- aggregate accuracy / adjoint summary ---
    print()
    max_acc = max(r["acc_fast_vs_ref"] for r in rows)
    max_adj = max(r["adjoint_residual"] for r in rows)
    median_p4 = statistics.median(r["speedup_fast_vs_generic"] for r in rows)
    median_p5 = statistics.median(r["speedup_cached_vs_generic"] for r in rows)
    median_batch = statistics.median(
        r["batch_speedup_cached_vs_generic"] for r in rows)

    print(f"Max accuracy error  (fast vs adaptive-ref): {max_acc:.2e}")
    print(f"Max adjoint residual:                       {max_adj:.2e}")
    print(f"Median Phase-4 single-call speedup:         {median_p4:.2f}x")
    print(f"Median Phase-5 single-call speedup:         {median_p5:.2f}x")
    print(f"Median Phase-5 batch speedup (N=30):        {median_batch:.2f}x")

    # --- CSV artifact ---
    out_path = Path(__file__).parent / "benchmark_phase6_results.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
