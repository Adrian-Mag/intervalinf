"""
benchmark_dli_oracle_bundle.py
==============================
Phase 3 benchmark: DLI oracle-cost and proximal-bundle end-to-end cost.

Three measurement modes (all run by default):

1. **oracle**     -- Isolated cost of repeated ``DualMasterCostFunction
                    .value_and_subgradient`` calls at random dual points;
                    reports timing breakdowns from ``DualMasterStats``.

2. **bundle**     -- Full proximal-bundle solve for +q and -q directions;
                    reports ``DualMasterStats`` and ``ProximalBundleStats``
                    breakdowns (oracle fraction, master-QP fraction, etc.).

3. **warmstart**  -- Nested-N_d experiment.  Solve the +q support value for
                    a small N_d, pad the optimal lambda with zeros, and use
                    it as the initial point for the next larger N_d problem.
                    Compares cold-start vs padded-warm-start iterations and
                    wall time.

Constraints
-----------
- Proximal bundle only (no solver comparisons).
- No behaviour changes to intervalinf or pygeoinf.
- Uses chi-square calibrated data-confidence ellipsoid (same as
  bg_dli_1d_sweep.py Phase 2).

Reuse
-----
Problem builders are imported directly from bg_dli_1d_sweep
(which lives in the same directory).  Sys.path is patched at import time
so this script can be run from *any* current working directory.

Usage
-----
    cd /home/adrian/PhD/Inferences
    conda run -n inferences3 python intervalinf/rough_work/benchmark_dli_oracle_bundle.py

Output
------
    intervalinf/rough_work/benchmark_dli_oracle_bundle_oracle.csv
    intervalinf/rough_work/benchmark_dli_oracle_bundle_bundle.csv
    intervalinf/rough_work/benchmark_dli_oracle_bundle_warmstart.csv
    (plus human-readable summary on stdout)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure rough_work directory is importable so we can reuse bg_dli_1d_sweep.
# ---------------------------------------------------------------------------

_ROUGH_WORK = Path(__file__).resolve().parent
if str(_ROUGH_WORK) not in sys.path:
    sys.path.insert(0, str(_ROUGH_WORK))

# Import builders from the existing Phase 1/2 sweep script.
from bg_dli_1d_sweep import (  # noqa: E402
    build_problem,
    FORWARD_SEEDS,
    DATA_SEEDS,
    N_D_VALUES,
    DLI_TOLERANCE,
    DLI_MAX_ITERATIONS,
    DLI_BUNDLE_SIZE,
    DLI_RHO0,
    DLI_RHO_FACTOR,
)

from pygeoinf import EuclideanSpace
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    ProximalBundleStats,
    solve_support_values,
    best_available_qp_solver,
)

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# N_d values to cover in oracle and bundle modes.
# Use the full sweep list; override for quick smoke tests with a subset.
BENCH_N_D: list[int] = N_D_VALUES   # e.g. [5, 10, 20, 40, 100, 200, 400, 1000]

# For the warm-start experiment, only sweep the nested sequence (ascending).
WARMSTART_N_D: list[int] = sorted(BENCH_N_D)

# Number of random dual-variable evaluations per N_d in oracle mode.
N_ORACLE_REPS: int = 10

# Forward / data seed combination used for all benchmarks.
# Keep it single to focus on N_d scaling; use index 0 from global lists.
FORWARD_SEED: int = FORWARD_SEEDS[0]
DATA_SEED: int = DATA_SEEDS[0]

# Output: CSV files written next to this script.
_OUT_DIR = _ROUGH_WORK
ORACLE_CSV = _OUT_DIR / "benchmark_dli_oracle_bundle_oracle.csv"
BUNDLE_CSV = _OUT_DIR / "benchmark_dli_oracle_bundle_bundle.csv"
WARMSTART_CSV = _OUT_DIR / "benchmark_dli_oracle_bundle_warmstart.csv"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_cost_and_solver(prob: dict) -> tuple:
    """Return (cost, solver) for a pre-built problem dict.

    The cost is set to +e1 direction by default; call ``cost.set_direction``
    or pass to ``solve_support_values`` to change it.

    Args:
        prob: Dict from ``build_problem``.

    Returns:
        (cost, solver) — a fresh DualMasterCostFunction and
        ProximalBundleMethod, each with reset instrumentation.
    """
    D, P, M = prob["D"], prob["P"], prob["M"]
    G, T = prob["G"], prob["T"]
    e1 = P.basis_vector(0)

    cost = DualMasterCostFunction(
        D, P, M, G, T,
        prob["model_prior_support"],
        prob["data_conf_ellipsoid"],
        prob["d_tilde"],
        e1,
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

    return cost, solver


def _pad_lambda(
    lam_small: np.ndarray,
    D_large: EuclideanSpace,
) -> np.ndarray:
    """Pad a dual vector from a smaller problem into a larger D space.

    The first ``len(lam_small)`` components of the returned vector are taken
    from *lam_small*; remaining components are zero.  This exploits the
    nested-kernel property: for a fixed forward seed, the first N_d_small rows
    of the N_d_large Gram matrix coincide with those of the smaller problem,
    so the small-problem optimum is a natural warm start.

    Args:
        lam_small: Component array of length N_d_small.
        D_large: Target EuclideanSpace of dimension N_d_large >= N_d_small.

    Returns:
        Component array of length D_large.dim, zero-padded.
    """
    N_small = len(lam_small)
    N_large = D_large.dim
    assert N_large >= N_small, (
        f"D_large.dim={N_large} < len(lam_small)={N_small}"
    )
    padded = np.zeros(N_large, dtype=float)
    padded[:N_small] = lam_small
    return padded


def _flatten_dual_stats(s) -> dict:
    """Convert a DualMasterStats object to a flat dict (no units in keys).

    All timing fields are in seconds (trailing _s suffix preserved).

    Args:
        s: A ``DualMasterStats`` instance.

    Returns:
        Dict with string keys and scalar values.
    """
    return {
        "oracle_calls": s.num_value_and_subgradient_calls,
        "total_oracle_s": s.time_value_and_subgradient_s,
        "gstar_s": s.time_gstar_apply_s,
        "support_pt_model_s": s.time_support_point_model_s,
        "support_pt_data_s": s.time_support_point_data_s,
        "support_val_model_s": s.time_support_value_model_s,
        "support_val_data_s": s.time_support_value_data_s,
        "sp_failures": s.num_support_point_failures,
        "fd_fallbacks": s.num_finite_difference_fallbacks,
        "fd_time_s": s.time_finite_difference_s,
    }


def _flatten_bundle_stats(s: Optional[ProximalBundleStats]) -> dict:
    """Convert a ProximalBundleStats object to a flat dict.

    Returns all-zero dict when *s* is None (solver not yet run).

    Args:
        s: A ``ProximalBundleStats`` instance or None.

    Returns:
        Dict with string keys and scalar values.
    """
    if s is None:
        return {
            "master_solves": 0,
            "master_time_s": 0.0,
            "null_steps": 0,
            "serious_steps": 0,
            "pb_oracle_time_s": 0.0,
        }
    return {
        "master_solves": s.num_master_solves,
        "master_time_s": s.time_master_solve_s,
        "null_steps": s.num_null_steps,
        "serious_steps": s.num_serious_steps,
        "pb_oracle_time_s": s.time_oracle_total_s,
    }


# ---------------------------------------------------------------------------
# Mode 1: Oracle-only benchmark
# ---------------------------------------------------------------------------


def bench_oracle(
    n_d_values: list[int] = BENCH_N_D,
    n_reps: int = N_ORACLE_REPS,
    forward_seed: int = FORWARD_SEED,
    data_seed: int = DATA_SEED,
) -> list[dict]:
    """Time isolated DualMasterCostFunction.value_and_subgradient calls.

    For each N_d in *n_d_values*, builds the full problem, evaluates the
    oracle at ``n_reps`` random dual points, and records DualMasterStats
    timing breakdowns (G* apply, support-point model, support-point data,
    support-value model, support-value data, totals).

    Args:
        n_d_values: Data dimensions to benchmark.
        n_reps: Number of oracle evaluations per N_d.
        forward_seed: Kernel seed (NormalModesProvider).
        data_seed: Noise seed.

    Returns:
        List of dicts, one per N_d; suitable for CSV output.
    """
    print("\n=== Mode 1: Oracle-only benchmark ===")
    print(
        f"  N_d values : {n_d_values}\n"
        f"  Reps / N_d : {n_reps}\n"
        f"  Seeds      : forward={forward_seed}, data={data_seed}\n"
    )

    rows = []
    rng = np.random.default_rng(99)   # fixed for reproducibility across runs

    for N_d in n_d_values:
        print(f"  Building problem N_d={N_d} ...", end=" ", flush=True)
        t_build = time.perf_counter()
        prob = build_problem(N_d, forward_seed, data_seed)
        print(f"done ({time.perf_counter() - t_build:.1f}s)")

        cost, _ = _build_cost_and_solver(prob)
        D = prob["D"]
        e1 = prob["P"].basis_vector(0)

        # Use +e1 direction throughout oracle timing.
        cost.set_direction(e1)
        cost.reset_instrumentation()

        # Generate random dual points in D (unit normal, not normalised).
        lam_np_batch = rng.standard_normal((n_reps, N_d))

        t_wall_start = time.perf_counter()
        for i in range(n_reps):
            lam = D.from_components(lam_np_batch[i])
            cost.value_and_subgradient(lam)
        wall_s = time.perf_counter() - t_wall_start

        ds = cost.instrumentation_stats

        # Consistency assertion: cumulative oracle time should be at most
        # the total wall time (it has no concurrent overhead).
        assert ds.time_value_and_subgradient_s <= wall_s + 1e-6, (
            f"N_d={N_d}: oracle time ({ds.time_value_and_subgradient_s:.4f}s) "
            f"exceeds wall time ({wall_s:.4f}s)"
        )
        # Call count should match exactly.
        assert ds.num_value_and_subgradient_calls == n_reps, (
            f"N_d={N_d}: expected {n_reps} calls, got "
            f"{ds.num_value_and_subgradient_calls}"
        )

        per_call_ms = (ds.time_value_and_subgradient_s / n_reps) * 1e3

        flat = _flatten_dual_stats(ds)
        row = {
            "N_d": N_d,
            "forward_seed": forward_seed,
            "data_seed": data_seed,
            "n_reps": n_reps,
            "wall_s": round(wall_s, 6),
            "per_call_ms": round(per_call_ms, 4),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in flat.items()},
        }
        rows.append(row)

        tot_s = max(flat["total_oracle_s"], 1e-12)
        gstar_pct = 100 * flat["gstar_s"] / tot_s
        sp_model_pct = 100 * flat["support_pt_model_s"] / tot_s
        sp_data_pct = 100 * flat["support_pt_data_s"] / tot_s

        print(
            f"  N_d={N_d:4d}  per_call={per_call_ms:.2f}ms  "
            f"Gstar={gstar_pct:.1f}%  "
            f"sp_model={sp_model_pct:.1f}%  "
            f"sp_data={sp_data_pct:.1f}%  "
            f"fd_fallbacks={flat['fd_fallbacks']}"
        )

    return rows


# ---------------------------------------------------------------------------
# Mode 2: Full proximal-bundle benchmark
# ---------------------------------------------------------------------------


def bench_bundle(
    n_d_values: list[int] = BENCH_N_D,
    forward_seed: int = FORWARD_SEED,
    data_seed: int = DATA_SEED,
) -> list[dict]:
    """Benchmark full proximal-bundle solves for +q and -q directions.

    For each N_d, runs two separate proximal-bundle solves (one per direction)
    from a zero start, recording:
      - total elapsed wall time
      - DualMasterStats breakdowns (G* apply, support-points, etc.)
      - ProximalBundleStats breakdowns (master QP time, oracle time,
        serious/null steps, etc.)
      - convergence and duality gap from BundleResult

    Args:
        n_d_values: Data dimensions to benchmark.
        forward_seed: Kernel seed.
        data_seed: Noise seed.

    Returns:
        List of dicts (two rows per N_d: one for +q, one for -q).
    """
    print("\n=== Mode 2: Full proximal-bundle benchmark ===")
    print(
        f"  N_d values : {n_d_values}\n"
        f"  Seeds      : forward={forward_seed}, data={data_seed}\n"
    )

    rows = []

    for N_d in n_d_values:
        print(f"  Building problem N_d={N_d} ...", end=" ", flush=True)
        t_build = time.perf_counter()
        prob = build_problem(N_d, forward_seed, data_seed)
        print(f"done ({time.perf_counter() - t_build:.1f}s)")

        D, P = prob["D"], prob["P"]
        e1 = P.basis_vector(0)
        neg_e1 = P.negative(e1)
        lambda0 = D.zero

        for direction_label, q_vec in [("+q", e1), ("-q", neg_e1)]:
            cost, solver = _build_cost_and_solver(prob)
            cost.set_direction(q_vec)
            cost.reset_instrumentation()

            t0 = time.perf_counter()
            vals, lams, diags = solve_support_values(
                cost, [q_vec], solver, lambda0, warm_start=False
            )
            elapsed_s = time.perf_counter() - t0

            diag = diags[0]
            ds = cost.instrumentation_stats
            bs = solver.instrumentation_stats

            # Consistency assertion: ProximalBundleStats oracle time should be
            # close to DualMasterStats total oracle time (same calls, just
            # measured at different layers).  Allow 2x slop for clock overhead.
            if ds.num_value_and_subgradient_calls > 0 and bs is not None:
                flat_ds_check = _flatten_dual_stats(ds)
                flat_bs_check = _flatten_bundle_stats(bs)
                pb_oracle = flat_bs_check["pb_oracle_time_s"]
                dm_oracle = flat_ds_check["total_oracle_s"]
                ratio = pb_oracle / max(dm_oracle, 1e-12)
                assert ratio <= 2.0, (
                    f"N_d={N_d} {direction_label}: ProximalBundleStats oracle time "
                    f"({pb_oracle:.4f}s) is >2x DualMasterStats total "
                    f"({dm_oracle:.4f}s)"
                )

            flat_ds = _flatten_dual_stats(ds)
            flat_bs = _flatten_bundle_stats(bs)

            oracle_frac = flat_ds["total_oracle_s"] / max(elapsed_s, 1e-12)
            master_frac = flat_bs["master_time_s"] / max(elapsed_s, 1e-12)

            row = {
                "N_d": N_d,
                "direction": direction_label,
                "forward_seed": forward_seed,
                "data_seed": data_seed,
                "converged": diag.converged,
                "num_iterations": diag.num_iterations,
                "gap": round(float(diag.gap), 8),
                "support_value": round(float(vals[0]), 8),
                "elapsed_s": round(elapsed_s, 6),
                "oracle_frac": round(oracle_frac, 4),
                "master_frac": round(master_frac, 4),
                **{f"ds_{k}": round(v, 6) if isinstance(v, float) else v
                   for k, v in flat_ds.items()},
                **{f"bs_{k}": round(v, 6) if isinstance(v, float) else v
                   for k, v in flat_bs.items()},
            }
            rows.append(row)

            print(
                f"    N_d={N_d:4d} {direction_label}  "
                f"iters={diag.num_iterations:4d}  "
                f"converged={diag.converged}  "
                f"gap={diag.gap:.2e}  "
                f"t={elapsed_s:.2f}s  "
                f"oracle={oracle_frac*100:.1f}%  "
                f"master={master_frac*100:.1f}%  "
                f"serious={flat_bs['serious_steps']}  "
                f"null={flat_bs['null_steps']}"
            )

    return rows


# ---------------------------------------------------------------------------
# Mode 3: Nested N_d warm-start experiment
# ---------------------------------------------------------------------------


def bench_warmstart(
    n_d_values: list[int] = WARMSTART_N_D,
    forward_seed: int = FORWARD_SEED,
    data_seed: int = DATA_SEED,
) -> list[dict]:
    """Compare cold start vs padded warm start across nested N_d pairs.

    For each consecutive (N_small, N_large) pair in the sorted *n_d_values*
    list, this function:

    1. Solves the +q support value for N_small (cold start from zero); records
       its optimal lambda.
    2. Solves the +q support value for N_large from a zero initial point
       (cold start).
    3. Pads the N_small optimal lambda with zeros and uses it as the initial
       point for a second N_large solve (warm start).
    4. Compares iterations, wall time, and final gap between modes 2 and 3.

    Rationale: NormalModesProvider uses nested semantics so the first N_small
    kernels are identical across all N_d values sharing the same forward seed.
    The N_small optimal dual variable should therefore be a useful warm start
    for the N_large problem, reducing the number of bundle iterations needed.

    Args:
        n_d_values: Sorted list of data dimensions; pairs are taken
                    consecutively.
        forward_seed: Kernel seed (fixed across all sizes for nesting).
        data_seed: Noise seed.

    Returns:
        List of dicts, one per (N_small, N_large) pair.
    """
    print("\n=== Mode 3: Nested N_d warm-start experiment ===")
    sorted_vals = sorted(n_d_values)
    pairs = list(zip(sorted_vals, sorted_vals[1:]))
    print(
        f"  Pairs      : {pairs}\n"
        f"  Seeds      : forward={forward_seed}, data={data_seed}\n"
    )

    rows = []

    for N_small, N_large in pairs:
        print(
            f"  Pair ({N_small}, {N_large}) ...",
            end=" ", flush=True,
        )

        # ------------------------------------------------------------------
        # Build both problems (same forward seed => nested kernels).
        # ------------------------------------------------------------------
        prob_small = build_problem(N_small, forward_seed, data_seed)
        prob_large = build_problem(N_large, forward_seed, data_seed)

        D_small = prob_small["D"]
        D_large = prob_large["D"]
        P_large = prob_large["P"]
        e1_large = P_large.basis_vector(0)

        assert D_small.dim == N_small, f"D_small.dim={D_small.dim} != {N_small}"
        assert D_large.dim == N_large, f"D_large.dim={D_large.dim} != {N_large}"

        # ------------------------------------------------------------------
        # Step 1: Solve small problem (cold start at zero) to get anchor lambda.
        # ------------------------------------------------------------------
        e1_small = prob_small["P"].basis_vector(0)
        cost_small, solver_small = _build_cost_and_solver(prob_small)
        cost_small.set_direction(e1_small)
        vals_s, lams_s, _ = solve_support_values(
            cost_small, [e1_small], solver_small,
            D_small.zero, warm_start=False,
        )
        lam_small_opt_comps = D_small.to_components(lams_s[0])

        # ------------------------------------------------------------------
        # Step 2: Cold start for N_large problem.
        # ------------------------------------------------------------------
        cost_cold, solver_cold = _build_cost_and_solver(prob_large)
        cost_cold.set_direction(e1_large)
        t_cold_0 = time.perf_counter()
        vals_cold, _, diags_cold = solve_support_values(
            cost_cold, [e1_large], solver_cold,
            D_large.zero, warm_start=False,
        )
        cold_s = time.perf_counter() - t_cold_0
        diag_cold = diags_cold[0]

        # ------------------------------------------------------------------
        # Step 3: Warm start from padded small-problem lambda.
        # ------------------------------------------------------------------
        lam_padded_comps = _pad_lambda(lam_small_opt_comps, D_large)
        lam_padded = D_large.from_components(lam_padded_comps)

        # Sanity check: dimension of padded vector equals N_large.
        assert len(lam_padded_comps) == N_large, (
            f"Padded lambda has {len(lam_padded_comps)} components, "
            f"expected {N_large}"
        )
        # Sanity check: first N_small entries are unchanged.
        np.testing.assert_allclose(
            lam_padded_comps[:N_small], lam_small_opt_comps,
            rtol=1e-12, atol=0.0,
            err_msg="Padding altered the first N_small components",
        )
        # Sanity check: tail is zero.
        np.testing.assert_array_equal(
            lam_padded_comps[N_small:],
            np.zeros(N_large - N_small),
        )

        cost_warm, solver_warm = _build_cost_and_solver(prob_large)
        cost_warm.set_direction(e1_large)
        t_warm_0 = time.perf_counter()
        vals_warm, _, diags_warm = solve_support_values(
            cost_warm, [e1_large], solver_warm,
            lam_padded, warm_start=False,
        )
        warm_s = time.perf_counter() - t_warm_0
        diag_warm = diags_warm[0]

        iter_reduction = diag_cold.num_iterations - diag_warm.num_iterations
        time_speedup = cold_s / max(warm_s, 1e-9)

        row = {
            "N_small": N_small,
            "N_large": N_large,
            "forward_seed": forward_seed,
            "data_seed": data_seed,
            # Cold start results
            "cold_iters": diag_cold.num_iterations,
            "cold_s": round(cold_s, 6),
            "cold_converged": diag_cold.converged,
            "cold_gap": round(float(diag_cold.gap), 8),
            "cold_support_value": round(float(vals_cold[0]), 8),
            # Warm start results
            "warm_iters": diag_warm.num_iterations,
            "warm_s": round(warm_s, 6),
            "warm_converged": diag_warm.converged,
            "warm_gap": round(float(diag_warm.gap), 8),
            "warm_support_value": round(float(vals_warm[0]), 8),
            # Comparison
            "iter_reduction": iter_reduction,
            "time_speedup": round(time_speedup, 4),
        }
        rows.append(row)

        print(
            f"\n    cold: iters={diag_cold.num_iterations}  "
            f"converged={diag_cold.converged}  "
            f"t={cold_s:.2f}s  gap={diag_cold.gap:.2e}"
        )
        print(
            f"    warm: iters={diag_warm.num_iterations}  "
            f"converged={diag_warm.converged}  "
            f"t={warm_s:.2f}s  gap={diag_warm.gap:.2e}"
        )
        print(
            f"    reduction: iters={iter_reduction:+d}  "
            f"speedup={time_speedup:.2f}x  "
            f"sv_diff={abs(float(vals_cold[0]) - float(vals_warm[0])):.2e}"
        )

    return rows


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts as a CSV to *path*.

    Args:
        rows: Non-empty list of dicts with uniform keys.
        path: Output file path (parent directories must exist).
    """
    if not rows:
        print(f"  [skip] No rows to write for {path.name}")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV written: {path}")


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------


def _print_oracle_summary(rows: list[dict]) -> None:
    print("\n--- Oracle summary (per-call cost in ms) ---")
    print(f"  {'N_d':>6}  {'per_call_ms':>12}  {'Gstar%':>7}  "
          f"{'sp_mod%':>7}  {'sp_dat%':>7}  {'fd_falls':>8}")
    for r in rows:
        tot = max(r["total_oracle_s"], 1e-12)
        gs_pct = 100 * r["gstar_s"] / tot
        sm_pct = 100 * r["support_pt_model_s"] / tot
        sd_pct = 100 * r["support_pt_data_s"] / tot
        print(f"  {r['N_d']:>6d}  {r['per_call_ms']:>12.4f}  "
              f"{gs_pct:>7.1f}  {sm_pct:>7.1f}  {sd_pct:>7.1f}  "
              f"{r['fd_fallbacks']:>8d}")


def _print_bundle_summary(rows: list[dict]) -> None:
    print("\n--- Bundle summary ---")
    print(f"  {'N_d':>6}  {'dir':>3}  {'iters':>6}  {'t(s)':>8}  "
          f"{'oracle%':>8}  {'master%':>8}  {'serious':>7}  {'null':>6}")
    for r in rows:
        print(
            f"  {r['N_d']:>6d}  {r['direction']:>3}  "
            f"{r['num_iterations']:>6d}  {r['elapsed_s']:>8.2f}  "
            f"{r['oracle_frac']*100:>8.1f}  {r['master_frac']*100:>8.1f}  "
            f"{r['bs_serious_steps']:>7d}  {r['bs_null_steps']:>6d}"
        )


def _print_warmstart_summary(rows: list[dict]) -> None:
    print("\n--- Warm-start summary ---")
    print(f"  {'N_s':>5} -> {'N_l':>5}  {'cold_i':>7}  {'warm_i':>7}  "
          f"{'delta_i':>8}  {'cold_s':>8}  {'warm_s':>8}  {'speedup':>8}")
    for r in rows:
        print(
            f"  {r['N_small']:>5d} -> {r['N_large']:>5d}  "
            f"{r['cold_iters']:>7d}  {r['warm_iters']:>7d}  "
            f"{r['iter_reduction']:>+8d}  "
            f"{r['cold_s']:>8.2f}  {r['warm_s']:>8.2f}  "
            f"{r['time_speedup']:>8.2f}x"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all three benchmark modes and emit CSV + stdout summaries."""
    print("=" * 72)
    print(
        "benchmark_dli_oracle_bundle.py  —  Phase 3 DLI instrumentation"
    )
    print(f"  N_d_values : {BENCH_N_D}")
    print(f"  Forward seed  : {FORWARD_SEED}")
    print(f"  Data seed     : {DATA_SEED}")
    print("=" * 72)

    t_global_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Mode 1: Oracle
    # ------------------------------------------------------------------
    oracle_rows = bench_oracle()
    _write_csv(oracle_rows, ORACLE_CSV)
    _print_oracle_summary(oracle_rows)

    # ------------------------------------------------------------------
    # Mode 2: Bundle
    # ------------------------------------------------------------------
    bundle_rows = bench_bundle()
    _write_csv(bundle_rows, BUNDLE_CSV)
    _print_bundle_summary(bundle_rows)

    # ------------------------------------------------------------------
    # Mode 3: Warm start
    # ------------------------------------------------------------------
    warmstart_rows = bench_warmstart()
    _write_csv(warmstart_rows, WARMSTART_CSV)
    _print_warmstart_summary(warmstart_rows)

    total_s = time.perf_counter() - t_global_start
    print(f"\n{'='*72}")
    print(f"Benchmark complete in {total_s:.1f}s")
    print(f"  {ORACLE_CSV}")
    print(f"  {BUNDLE_CSV}")
    print(f"  {WARMSTART_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
