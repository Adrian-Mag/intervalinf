"""
Phase 2 benchmark: SOLA compact-support forward-path cost measurement.

Objective
---------
Quantify the timing penalty of losing the batched fixed-grid cache path when
overlapping compact-support metadata forces per-kernel fallback integration
inside ``SOLAOperator._apply_kernels_fixed_grid``.

Scenarios covered
-----------------
1. ``full_domain``       — no support metadata on func or kernels; all N_d
                           kernels go through the fast batched fixed-grid path.
2. ``disjoint_support``  — func has compact support disjoint from all kernels;
                           every kernel is skipped without evaluating the
                           integrand (disjoint_skips = N_d).
3. ``overlapping_fallback`` — func and all kernels carry overlapping compact
                           support; every kernel falls back to per-kernel
                           domain.integrate (compact_support_fallbacks = N_d).
4. ``mixed_paths``       — half of the kernels have no support metadata
                           (batched path) and half have overlapping compact
                           support (fallback).  Measures the split cost.

Sweep axes
----------
* ``N_d``      : [5, 10, 20, 40]
* ``n_points`` : [500, 1000, 2000]

Accuracy validation
-------------------
For each scenario, G(f) is compared against a high-accuracy adaptive
reference built from the SAME kernels (including compact-support metadata)
and the SAME input function f.  This ensures the fixed-grid integration
(batched or fallback) is tested on the ACTUAL computation path, not a
support-stripped surrogate that sidesteps the fallback.  In the disjoint
case the reference is exactly zero.  An assertion enforces
max absolute error < MAX_ACCURACY_TOL.

Built-in sanity checks
----------------------
* For scenario ``full_domain``     : assert batched_fixed_grid_kernels == N_d
* For scenario ``disjoint_support``: assert disjoint_skips == N_d and result == 0
* For scenario ``overlapping_fallback``: assert compact_support_fallbacks == N_d
* For scenario ``mixed_paths``     : assert batched == N_d//2 and
                                     fallbacks == N_d - N_d//2

Usage
-----
From workspace root, with inferences3 conda env active:

    conda run -n inferences3 python intervalinf/rough_work/benchmark_phase2_compact_support.py

Output
------
Prints a CSV-style table to stdout.  Also writes a CSV artifact alongside this
script: ``benchmark_phase2_compact_support_results.csv``.
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from intervalinf import IntervalDomain, Lebesgue, Function, IntegrationConfig
from intervalinf.operators import SOLAOperator
from intervalinf.providers import SineFunctionProvider
from pygeoinf import EuclideanSpace

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_REPEATS = 8         # wall-clock timing repeats per cell
RNG_SEED = 42
DOMAIN_A, DOMAIN_B = 0.0, 1.0
BASIS_DIM = 30        # Lebesgue basis dim; irrelevant to SOLA perf

# Sweep axes
ND_VALUES = [5, 10, 20, 40]
NPOINT_VALUES = [500, 1000, 2000]

# Compact-support widths used in scenarios
FUNC_SUPPORT_FULL = None                     # no metadata (full domain)
FUNC_SUPPORT_LEFT = [(0.0, 0.45)]            # left half
KERNEL_SUPPORT_DISJOINT = (0.55, 1.0)       # right half – disjoint from FUNC_SUPPORT_LEFT
KERNEL_SUPPORT_OVERLAP = (0.1, 0.4)         # overlaps with FUNC_SUPPORT_LEFT

MAX_ACCURACY_TOL = 1e-4   # max abs error vs adaptive reference (per element)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_domain():
    return IntervalDomain(DOMAIN_A, DOMAIN_B)


def _make_space(domain: IntervalDomain) -> Lebesgue:
    return Lebesgue(BASIS_DIM, domain, basis="cosine")


def _make_test_func(space: Lebesgue, domain: IntervalDomain,
                    support=None) -> Function:
    """Smooth test function, optionally carrying compact-support metadata."""
    rng = np.random.default_rng(RNG_SEED)
    a_coeff, b_coeff = rng.standard_normal(2)

    def _eval(x):
        arr = np.asarray(x)
        return a_coeff * np.sin(np.pi * arr) + b_coeff * arr

    return Function(space, evaluate_callable=_eval, support=support)


def _make_sine_kernels(N_d: int, domain: IntervalDomain) -> list:
    """Full-domain sine kernels without support metadata."""
    return [
        Function(domain, evaluate_callable=lambda x, _i=i: np.sin((_i + 1) * np.pi * np.asarray(x)))
        for i in range(N_d)
    ]


def _kernel_with_support(domain: IntervalDomain, support_interval: tuple) -> Function:
    """Flat (constant-1) kernel on a compact support interval."""
    a_s, b_s = support_interval

    def _eval(x):
        arr = np.asarray(x)
        return np.where((arr >= a_s) & (arr <= b_s), 1.0, 0.0)

    return Function(domain, evaluate_callable=_eval, support=[support_interval])


def _make_operator(
    space: Lebesgue,
    kernels: list,
    n_points: int,
    method: str = "simpson",
) -> SOLAOperator:
    D = EuclideanSpace(len(kernels))
    cfg = IntegrationConfig(method=method, n_points=n_points)
    return SOLAOperator(space, D, kernels=kernels, integration_config=cfg)


def _make_reference_operator(
    space: Lebesgue,
    kernels: list,
) -> SOLAOperator:
    """High-accuracy adaptive reference operator (no instrumentation needed)."""
    D = EuclideanSpace(len(kernels))
    cfg = IntegrationConfig(method="adaptive", n_points=2000)
    return SOLAOperator(space, D, kernels=kernels, integration_config=cfg)


def _median_ms(G: SOLAOperator, f: Function, n: int = N_REPEATS) -> float:
    """Return median wall-clock time in ms for G(f) over n calls."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        G(f)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _max_abs_error(y_test: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.max(np.abs(y_test - y_ref)))


# ---------------------------------------------------------------------------
# Scenario builders — return (G, f, G_ref, f_ref, scenario_label)
# ---------------------------------------------------------------------------

def build_full_domain(N_d: int, n_points: int, domain: IntervalDomain, space: Lebesgue):
    """Scenario 1: no support metadata — all N_d kernels go through batched path."""
    kernels = _make_sine_kernels(N_d, domain)
    f = _make_test_func(space, domain, support=None)
    G = _make_operator(space, kernels, n_points)
    G_ref = _make_reference_operator(space, kernels)
    return G, f, G_ref


def build_disjoint_support(N_d: int, n_points: int, domain: IntervalDomain, space: Lebesgue):
    """Scenario 2: func support and all kernel supports are disjoint.
    
    All N_d kernels should be skipped (disjoint_skips == N_d).
    Result is exactly zero.  Reference is np.zeros(N_d) (no G_ref needed).
    """
    kernels = [_kernel_with_support(domain, KERNEL_SUPPORT_DISJOINT)
               for _ in range(N_d)]
    f = _make_test_func(space, domain, support=FUNC_SUPPORT_LEFT)
    G = _make_operator(space, kernels, n_points)
    return G, f, None


def build_overlapping_fallback(N_d: int, n_points: int, domain: IntervalDomain, space: Lebesgue):
    """Scenario 3: func and all kernels have overlapping compact support.
    
    All N_d kernels fall back to per-kernel domain.integrate inside the
    fixed-grid path (compact_support_fallbacks == N_d).  G_ref uses the SAME
    kernels (with support metadata) and adaptive integration so the reference
    also narrows to the support intersection — testing whether the fixed-grid
    fallback matches adaptive on ∫_{support-intersection} f·k dx.
    """
    kernels = [_kernel_with_support(domain, KERNEL_SUPPORT_OVERLAP)
               for _ in range(N_d)]
    f = _make_test_func(space, domain, support=FUNC_SUPPORT_LEFT)
    G = _make_operator(space, kernels, n_points)
    G_ref = _make_reference_operator(space, kernels)  # same kernels, adaptive
    return G, f, G_ref


def build_mixed_paths(N_d: int, n_points: int, domain: IntervalDomain, space: Lebesgue):
    """Scenario 4: half kernels batched (no metadata), half fallback (overlap).
    
    n_batched = N_d // 2 (sine kernels, no support metadata)
    n_fallback = N_d - N_d // 2 (flat kernels with overlapping support)

    G_ref uses the SAME kernels (including support metadata on fallback
    kernels) with adaptive integration so both the batched and the fallback
    integrals are compared against their respective high-accuracy references.
    """
    n_batched = N_d // 2
    n_fallback = N_d - n_batched
    kernels_batched = _make_sine_kernels(n_batched, domain)
    kernels_fallback = [_kernel_with_support(domain, KERNEL_SUPPORT_OVERLAP)
                        for _ in range(n_fallback)]
    kernels = kernels_batched + kernels_fallback
    f = _make_test_func(space, domain, support=FUNC_SUPPORT_LEFT)
    G = _make_operator(space, kernels, n_points)
    G_ref = _make_reference_operator(space, kernels)  # same kernels, adaptive
    return G, f, G_ref


# ---------------------------------------------------------------------------
# Sanity assertions
# ---------------------------------------------------------------------------

def _assert_scenario_counters(
    scenario: str, stats: dict, N_d: int,
    result: np.ndarray,
):
    """Raise AssertionError if the scenario's instrumentation counters are wrong."""
    n_batched = stats["batched_fixed_grid_kernels"]
    n_fallback = stats["compact_support_fallbacks"]
    n_skip = stats["disjoint_skips"]

    if scenario == "full_domain":
        assert n_batched == N_d, (
            f"full_domain: expected batched={N_d}, got {n_batched}"
        )
        assert n_fallback == 0, f"full_domain: unexpected fallbacks {n_fallback}"
        assert n_skip == 0, f"full_domain: unexpected skips {n_skip}"

    elif scenario == "disjoint_support":
        assert n_skip == N_d, (
            f"disjoint_support: expected skips={N_d}, got {n_skip}"
        )
        assert n_batched == 0, f"disjoint_support: unexpected batched {n_batched}"
        assert n_fallback == 0, f"disjoint_support: unexpected fallbacks {n_fallback}"
        assert np.allclose(result, 0.0, atol=1e-12), (
            "disjoint_support: result should be exactly zero"
        )

    elif scenario == "overlapping_fallback":
        assert n_fallback == N_d, (
            f"overlapping_fallback: expected fallbacks={N_d}, got {n_fallback}"
        )
        assert n_batched == 0, f"overlapping_fallback: unexpected batched {n_batched}"
        assert n_skip == 0, f"overlapping_fallback: unexpected skips {n_skip}"

    elif scenario == "mixed_paths":
        n_expected_batched = N_d // 2
        n_expected_fallback = N_d - N_d // 2
        assert n_batched == n_expected_batched, (
            f"mixed_paths: expected batched={n_expected_batched}, got {n_batched}"
        )
        assert n_fallback == n_expected_fallback, (
            f"mixed_paths: expected fallbacks={n_expected_fallback}, got {n_fallback}"
        )
        assert n_skip == 0, f"mixed_paths: unexpected skips {n_skip}"


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

SCENARIO_BUILDERS = {
    "full_domain": build_full_domain,
    "disjoint_support": build_disjoint_support,
    "overlapping_fallback": build_overlapping_fallback,
    "mixed_paths": build_mixed_paths,
}

HEADER_FIELDS = [
    "scenario", "N_d", "n_points",
    "wall_ms_median",
    "batched_kernels", "fallback_kernels", "disjoint_skips",
    "fallback_time_ms",
    "max_abs_error",
    "sanity_ok",
]


def run_benchmark() -> list[dict]:
    rows = []

    for scenario_name, builder in SCENARIO_BUILDERS.items():
        for N_d in ND_VALUES:
            for n_points in NPOINT_VALUES:
                domain = _make_domain()
                space = _make_space(domain)

                G, f, G_ref = builder(N_d, n_points, domain, space)

                # --- Accuracy: compare G(f) against the same integral computed
                # with high-accuracy adaptive integration (G_ref uses identical
                # kernels and identical f, so both operators evaluate exactly the
                # same mathematical integral via different numerical methods).
                # disjoint_support: G(f) == 0 exactly; reference is the zero vector.
                y_test = G(f)
                if scenario_name == "disjoint_support":
                    y_ref = np.zeros(N_d)
                else:
                    y_ref = G_ref(f)
                max_err = _max_abs_error(y_test, y_ref)

                # --- Sanity-check counters after a clean single call ---
                G.reset_stats()
                y_single = G(f)
                single_stats = G.stats
                sanity_ok = True
                try:
                    _assert_scenario_counters(scenario_name, single_stats, N_d, y_single)
                except AssertionError as exc:
                    print(f"  [SANITY FAIL] {scenario_name} N_d={N_d} n_points={n_points}: {exc}",
                          file=sys.stderr)
                    sanity_ok = False
                if max_err > MAX_ACCURACY_TOL:
                    print(
                        f"  [ACCURACY FAIL] {scenario_name} N_d={N_d} n_points={n_points}: "
                        f"max_abs_error={max_err:.3e} > MAX_ACCURACY_TOL={MAX_ACCURACY_TOL:.3e}",
                        file=sys.stderr,
                    )
                    sanity_ok = False

                # --- Timing: reset then time N_REPEATS calls ---
                G.reset_stats()
                wall_ms = _median_ms(G, f, n=N_REPEATS)
                final_stats = G.stats

                row = {
                    "scenario": scenario_name,
                    "N_d": N_d,
                    "n_points": n_points,
                    "wall_ms_median": round(wall_ms, 4),
                    "batched_kernels": final_stats["batched_fixed_grid_kernels"],
                    "fallback_kernels": final_stats["compact_support_fallbacks"],
                    "disjoint_skips": final_stats["disjoint_skips"],
                    "fallback_time_ms": round(
                        final_stats["compact_support_fallback_time_total_s"] * 1000, 4
                    ),
                    "max_abs_error": f"{max_err:.3e}",
                    "sanity_ok": sanity_ok,
                }
                rows.append(row)

                # Live progress
                print(
                    f"  {scenario_name:22s}  N_d={N_d:>3}  n_pts={n_points:>5}"
                    f"  wall={wall_ms:8.3f}ms"
                    f"  batched={final_stats['batched_fixed_grid_kernels']:>4}"
                    f"  fallback={final_stats['compact_support_fallbacks']:>4}"
                    f"  skip={final_stats['disjoint_skips']:>4}"
                    f"  err={max_err:.2e}"
                    f"  sane={'Y' if sanity_ok else 'N'}",
                    flush=True,
                )

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("Phase 2 Benchmark: SOLA Compact-Support Forward-Path Cost")
    print("=" * 90)
    print(f"N_REPEATS={N_REPEATS}, ND_VALUES={ND_VALUES}, NPOINT_VALUES={NPOINT_VALUES}")
    print()

    print("Running scenarios...")
    rows = run_benchmark()

    # --- CSV output ---
    out_path = Path(__file__).with_name("benchmark_phase2_compact_support_results.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 90)
    print(f"Results written to: {out_path}")
    print()

    # --- Summary table to stdout ---
    col_w = [22, 5, 8, 12, 10, 10, 8, 14, 14, 8]
    hdr = [f.upper()[:col_w[i]] for i, f in enumerate(HEADER_FIELDS)]
    print("  ".join(h.ljust(col_w[i]) for i, h in enumerate(hdr)))
    print("-" * 120)
    for row in rows:
        vals = [str(row[f]) for f in HEADER_FIELDS]
        print("  ".join(v.ljust(col_w[i]) for i, v in enumerate(vals)))

    n_fail = sum(1 for r in rows if not r["sanity_ok"])
    if n_fail:
        print(f"\n  [{n_fail} sanity check(s) FAILED — see stderr for details]")
        sys.exit(1)
    else:
        print(f"\n  All {len(rows)} benchmark cells passed sanity checks.")


if __name__ == "__main__":
    main()
