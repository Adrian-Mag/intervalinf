"""Interval DLI — Solver comparison timing study.

Sweeps over data-space dimension {10, 20, 50, 100} with model space fixed at
a Lebesgue L^2([0,1]) space, comparing wall-clock time of four optimisation
methods on the 1-D bump-target inference problem from
``intervalinf/demos/convex_analysis/dli.ipynb``:

    ProximalBundle      — proximal bundle method (OSQP QP master)
    SmoothedLBFGS       — Moreau-Yosida smoothing + L-BFGS-B continuation
    PrimalKKT           — Woodbury KKT in abstract space (no model-space matrix)

Note: ``ChambollePockSolver`` is also omitted: its step-size estimation uses
power iteration which calls ``model_space.from_components()``, requiring a
finite-dimensional basis — incompatible with the Lebesgue space used here.

The figure is rebuilt and saved after every completed (solver, dim) data point
so it can be inspected live.

Outputs (written to ``intervalinf/work/figures/``):
    interval_dli_solver_comparison.png   — timing + bound-width figure
    interval_dli_partial.npz             — incremental results (overwritten)
    interval_dli_results.npz             — final complete results

Configuration via environment variables::

    DLI_NUM_THREADS     — BLAS/OpenMP threads (default 4)
    DLI_NUM_REPEATS     — timed repeats per (solver, dim) (default 3)
    DLI_SUPPORT_N_JOBS  — parallel directions for dual methods (default 1)
    DLI_DATA_DIMS       — comma-separated list of N_d values (default 10,20,50,100)

Run::

    conda activate inferences3
    cd <workspace-root>
    python -u intervalinf/work/interval_dli_solver_comparison.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

# Keep BLAS/OpenMP thread usage explicit for reproducible timing.
DEFAULT_NUM_THREADS = int(os.environ.get("DLI_NUM_THREADS", "4"))
for _env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_name, str(DEFAULT_NUM_THREADS))

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# intervalinf / pygeoinf imports
# ---------------------------------------------------------------------------
from intervalinf import IntervalDomain, Lebesgue, Function
from intervalinf import (
    IntegrationConfig,
    ParallelConfig,
    LebesgueIntegrationConfig,
    LebesgueParallelConfig,
)
from intervalinf.providers import NormalModesProvider, BumpFunctionProvider
from intervalinf.operators import SOLAOperator

from pygeoinf import EuclideanSpace
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    SmoothedLBFGSSolver,
    PrimalKKTSolver,
    best_available_qp_solver,
    solve_support_values,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIMS = [10, 20, 50, 100, 200, 500, 1000]
DATA_DIMS: list[int] = [
    int(x) for x in os.environ.get("DLI_DATA_DIMS", ",".join(map(str, _DEFAULT_DATA_DIMS))).split(",")
]

# ChambollePock omitted: requires model_space.from_components() in power-iteration
# step-size estimation, which fails for infinite-dimensional Lebesgue space.
SOLVER_NAMES = ["ProximalBundle", "SmoothedLBFGS", "PrimalKKT"]

SOLVER_COLORS = {
    "ProximalBundle":    "tab:blue",
    "SmoothedLBFGS":     "tab:green",
    "PrimalKKT":         "tab:orange",
}
SOLVER_MARKERS = {
    "ProximalBundle":    "o",
    "SmoothedLBFGS":     "^",
    "PrimalKKT":         "s",
}

# Problem constants (match dli.ipynb)
SEED            = 42
MAX_ITER        = 300
TOL             = 1e-4
N_P             = 8     # number of property targets (bump functions)
BUMP_WIDTH      = 0.2
PRIOR_MULTIPLIER = 1.05  # how much to inflate data/model radii around truth

N_REPEATS      = int(os.environ.get("DLI_NUM_REPEATS", "3"))
SUPPORT_N_JOBS = int(os.environ.get("DLI_SUPPORT_N_JOBS", "1"))

_WORK_DIR = Path(__file__).parent
FIG_DIR   = _WORK_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PARTIAL_SAVE = FIG_DIR / "interval_dli_partial.npz"
FIGURE_PATH  = FIG_DIR / "interval_dli_solver_comparison.png"


# ---------------------------------------------------------------------------
# Problem setup helpers
# ---------------------------------------------------------------------------

def build_spaces_and_operators(n_data: int, *, seed: int = SEED):
    """Build model, data, property spaces and operators for given N_d.

    Returns:
        M, D, P, G, T, m_bar, d_tilde, model_prior_support, data_error_support
    """
    rng = np.random.default_rng(seed)

    function_domain = IntervalDomain(0, 1)

    lebesgue_integration_cfg = LebesgueIntegrationConfig(
        inner_product=IntegrationConfig(method="simpson", n_points=500),
        dual=IntegrationConfig(method="simpson", n_points=500),
        general=IntegrationConfig(method="simpson", n_points=500),
    )
    parallel_cfg = ParallelConfig(enabled=False, n_jobs=1)
    sola_integration_cfg = IntegrationConfig(method="simpson", n_points=1000)

    M = Lebesgue(
        0,
        function_domain,
        basis=None,
        integration_config=lebesgue_integration_cfg,
        parallel_config=parallel_cfg,
    )
    D = EuclideanSpace(n_data)
    P = EuclideanSpace(N_P)

    # Forward operator: random normal-modes kernels
    normal_modes_provider = NormalModesProvider(
        M,
        n_modes_range=(1, 50),
        coeff_range=(-5, 5),
        gaussian_width_percent_range=(1, 5),
        freq_range=(0.1, 20),
        random_state=rng.integers(0, 2**31),
    )
    G = SOLAOperator(
        M,
        D,
        kernels=normal_modes_provider,
        cache_kernels=True,
        integration_config=sola_integration_cfg,
    )

    # Property operator: bump functions at evenly-spaced centres
    centers = np.linspace(
        function_domain.a + BUMP_WIDTH / 2,
        function_domain.b - BUMP_WIDTH / 2,
        N_P,
    )
    target_provider = BumpFunctionProvider(M, centers=centers, default_width=BUMP_WIDTH)
    T = SOLAOperator(
        M,
        P,
        kernels=target_provider,
        cache_kernels=True,
        integration_config=sola_integration_cfg,
    )

    # True model and noisy data (match dli.ipynb)
    m_bar = Function(
        M,
        evaluate_callable=lambda x: (
            np.exp(-((x - function_domain.center) / 0.5) ** 2) * np.sin(5 * np.pi * x) + x
        ),
    )

    np.random.seed(seed)
    d_bar = G(m_bar)
    noise_level = 0.1 * np.max(np.abs(d_bar))
    noise_vec = np.random.default_rng(seed).normal(0, noise_level, n_data)
    d_tilde = d_bar + noise_vec

    # Deterministic confidence sets
    noise_actual  = d_tilde - d_bar
    data_radius   = PRIOR_MULTIPLIER * np.linalg.norm(noise_actual)

    m_0 = Function(M, evaluate_callable=lambda x: x)
    model_radius = PRIOR_MULTIPLIER * M.norm(M.subtract(m_bar, m_0))

    model_prior_support = BallSupportFunction(M, m_0, model_radius)
    data_error_support  = BallSupportFunction(D, D.zero, data_radius)

    return M, D, P, G, T, m_bar, d_tilde, model_prior_support, data_error_support


def _build_dli_components(M, D, P, G, T, m_bar, d_tilde, model_prior_support, data_error_support):
    """Build shared DLI components: cost function, basis directions, true values."""
    basis_dirs  = [P.basis_vector(i) for i in range(P.dim)]
    neg_basis   = [P.multiply(-1.0, q) for q in basis_dirs]

    observed = np.asarray(d_tilde, dtype=float)

    cost = DualMasterCostFunction(
        D, P, M, G, T,
        model_prior_support,
        data_error_support,
        observed,
        basis_dirs[0],
    )

    true_values  = np.asarray(T(m_bar), dtype=float)
    prior_bounds = np.array([model_prior_support(T.adjoint(q)) for q in basis_dirs])

    return cost, basis_dirs, neg_basis, true_values, prior_bounds


# ---------------------------------------------------------------------------
# Per-solver solve functions
# ---------------------------------------------------------------------------

def solve_with_proximal_bundle(cost, basis_dirs, neg_basis, D):
    qp = best_available_qp_solver()
    solver = ProximalBundleMethod(
        cost, tolerance=TOL, max_iterations=MAX_ITER, qp_solver=qp,
    )
    lambda0 = D.zero
    upper_vals, _, _ = solve_support_values(
        cost, basis_dirs, solver, lambda0, n_jobs=SUPPORT_N_JOBS
    )
    lower_neg, _, _ = solve_support_values(
        cost, neg_basis,   solver, lambda0, n_jobs=SUPPORT_N_JOBS
    )
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def solve_with_smoothed_lbfgs(cost, basis_dirs, neg_basis, D):
    solver = SmoothedLBFGSSolver(
        cost, epsilon0=1e-2, n_levels=5, tolerance=TOL, max_iter_per_level=MAX_ITER,
    )
    lambda0 = D.zero
    upper_vals, _, _ = solve_support_values(
        cost, basis_dirs, solver, lambda0, n_jobs=SUPPORT_N_JOBS
    )
    lower_neg, _, _ = solve_support_values(
        cost, neg_basis,   solver, lambda0, n_jobs=SUPPORT_N_JOBS
    )
    return np.asarray(upper_vals), -np.asarray(lower_neg)


def solve_with_primal_kkt(
    basis_dirs, neg_basis, M, T,
    model_prior_support, data_error_support, G, d_tilde,
    n_jobs: int = 1,
):
    """Run PrimalKKTSolver — model space never discretised.

    Args:
        n_jobs: Number of parallel jobs for direction solves (default 1).
            >1 uses joblib; each direction gets its own solver instance.
    """
    d_obs = np.asarray(d_tilde, dtype=float)

    def _make_solver():
        return PrimalKKTSolver(
            model_prior_support,
            data_error_support,
            G,
            d_obs,
        )

    def _solve_one_fresh(q):
        """Create a fresh solver per call — safe for parallel workers."""
        c = T.adjoint(q)
        result = _make_solver().solve(c)
        return float(M.inner_product(c, result.m))

    def _solve_one_reuse(solver, q):
        """Reuse a single solver — for sequential execution."""
        c = T.adjoint(q)
        result = solver.solve(c)
        return float(M.inner_product(c, result.m))

    _n_jobs = n_jobs
    if _n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            all_qs = list(basis_dirs) + list(neg_basis)
            vals = Parallel(n_jobs=_n_jobs)(
                delayed(_solve_one_fresh)(q) for q in all_qs
            )
            n = len(basis_dirs)
            upper_vals = np.array(vals[:n])
            lower_neg  = np.array(vals[n:])
        except ImportError:
            print("  Warning: joblib not available — falling back to sequential.")
            sys.stdout.flush()
            _n_jobs = 1

    if _n_jobs <= 1:
        solver = _make_solver()
        upper_vals = np.array([_solve_one_reuse(solver, q) for q in basis_dirs])
        lower_neg  = np.array([_solve_one_reuse(solver, q) for q in neg_basis])

    return np.asarray(upper_vals), -np.asarray(lower_neg)


# ---------------------------------------------------------------------------
# run_one / run_one_repeated
# ---------------------------------------------------------------------------

def run_one(
    solver_name: str,
    M, D, P, G, T,
    m_bar, d_tilde,
    model_prior_support, data_error_support,
) -> dict[str, Any]:
    """Run a single (solver, N_d) timing measurement."""
    n_data = D.dim
    print(f"    [{solver_name}]  N_d={n_data}")
    sys.stdout.flush()

    cost, basis_dirs, neg_basis, true_values, prior_bounds = _build_dli_components(
        M, D, P, G, T, m_bar, d_tilde, model_prior_support, data_error_support
    )

    t0 = time.time()

    if solver_name == "ProximalBundle":
        upper, lower = solve_with_proximal_bundle(cost, basis_dirs, neg_basis, D)
    elif solver_name == "SmoothedLBFGS":
        upper, lower = solve_with_smoothed_lbfgs(cost, basis_dirs, neg_basis, D)
    elif solver_name == "PrimalKKT":
        upper, lower = solve_with_primal_kkt(
            basis_dirs, neg_basis, M, T,
            model_prior_support, data_error_support, G, d_tilde,
            n_jobs=SUPPORT_N_JOBS,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.time() - t0
    print(f"      → {elapsed:.2f}s   bounds: [{lower[0]:.4f}, {upper[0]:.4f}]")
    sys.stdout.flush()

    return {
        "solver":      solver_name,
        "data_dim":    n_data,
        "solve_time":  elapsed,
        "upper":       upper,
        "lower":       lower,
        "true_values": true_values,
        "prior_lower": -prior_bounds,
        "prior_upper": prior_bounds,
    }


def run_one_repeated(
    solver_name: str,
    M, D, P, G, T,
    m_bar, d_tilde,
    model_prior_support, data_error_support,
    *,
    n_repeats: int,
    progress_callback=None,
) -> dict[str, Any]:
    """Run repeated timing for one (solver, N_d) and aggregate statistics."""

    def _aggregate(per_run_local: list[dict[str, Any]]) -> dict[str, Any]:
        times  = np.array([r["solve_time"] for r in per_run_local], dtype=float)
        uppers = np.array([r["upper"] for r in per_run_local], dtype=float)
        lowers = np.array([r["lower"] for r in per_run_local], dtype=float)
        n_done = len(per_run_local)

        return {
            "solver":          solver_name,
            "data_dim":        int(D.dim),
            "n_repeats":       int(n_done),
            "solve_time_mean": float(np.mean(times)),
            "solve_time_std":  float(np.std(times, ddof=1)) if n_done > 1 else 0.0,
            "solve_times":     times,
            "upper":           np.mean(uppers, axis=0),
            "lower":           np.mean(lowers, axis=0),
            "upper_std":       np.std(uppers, axis=0, ddof=1) if n_done > 1 else np.zeros_like(uppers[0]),
            "lower_std":       np.std(lowers, axis=0, ddof=1) if n_done > 1 else np.zeros_like(lowers[0]),
            "true_values":     per_run_local[0]["true_values"],
            "prior_lower":     per_run_local[0]["prior_lower"],
            "prior_upper":     per_run_local[0]["prior_upper"],
        }

    per_run: list[dict[str, Any]] = []
    for rep in range(n_repeats):
        print(f"      repeat {rep + 1}/{n_repeats}")
        sys.stdout.flush()
        per_run.append(
            run_one(solver_name, M, D, P, G, T,
                    m_bar, d_tilde, model_prior_support, data_error_support)
        )
        if progress_callback is not None:
            progress_callback(_aggregate(per_run), rep + 1, n_repeats)

    return _aggregate(per_run)


# ---------------------------------------------------------------------------
# Progressive figure
# ---------------------------------------------------------------------------

def plot_progress(results: list[dict]) -> None:
    """Rebuild and save the comparison figure from all completed results."""
    if not results:
        return

    all_dims    = sorted({r["data_dim"] for r in results})
    all_solvers = SOLVER_NAMES

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel 1: solve time vs N_d ----------------------------------------
    ax = axes[0]
    for sname in all_solvers:
        rows = sorted(
            [r for r in results if r["solver"] == sname],
            key=lambda r: r["data_dim"],
        )
        if not rows:
            continue

        xs   = np.array([r["data_dim"]        for r in rows], dtype=float)
        ys   = np.array([r["solve_time_mean"]  for r in rows], dtype=float)
        yerr = np.array([r["solve_time_std"]   for r in rows], dtype=float)
        if xs.size > 0:
            yerr = np.minimum(yerr, np.maximum(ys * 0.99, 1e-12))
            ax.errorbar(
                xs, ys, yerr=yerr,
                color=SOLVER_COLORS[sname],
                marker=SOLVER_MARKERS[sname],
                linestyle="-",
                markersize=8,
                linewidth=1.5,
                label=sname,
                capsize=3,
            )
            ax.annotate(
                f"{ys[-1]:.2f}s",
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(6, 2),
                fontsize=8,
                color=SOLVER_COLORS[sname],
            )
        else:
            ax.plot([], [],
                    color=SOLVER_COLORS[sname],
                    marker=SOLVER_MARKERS[sname],
                    linestyle="-", markersize=8, linewidth=1.5, label=sname)

    # Reference slope lines anchored at (dim=10, time=1s)
    _xlim = np.array([7.0, max(120.0, max(all_dims) * 1.2)])
    _t0   = 1.0
    for exp_ref, ls, lbl in [
        (1,   ":",  r"$n^1$"),
        (1.5, "--", r"$n^{1.5}$"),
        (2,   "-.", r"$n^2$"),
        (3,   "-",  r"$n^3$"),
    ]:
        _y = _t0 * (_xlim / 10) ** exp_ref
        ax.loglog(_xlim, _y, color="lightgrey", linestyle=ls, linewidth=1)
        ax.text(_xlim[-1] * 0.6, _y[-1] * 0.7, lbl, fontsize=8, color="grey")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Data-space dimension  $N_d$", fontsize=11)
    ax.set_ylabel("Solve time (s)", fontsize=11)
    ax.set_title(r"Solver timing vs $N_d$  (1-D interval DLI)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ---- Panel 2: mean bound width vs N_d, per solver ----------------------
    ax2 = axes[1]
    for sname in all_solvers:
        rows = sorted(
            [r for r in results if r["solver"] == sname],
            key=lambda r: r["data_dim"],
        )
        if not rows:
            continue
        xs     = np.array([r["data_dim"] for r in rows])
        widths = np.array([r["upper"] - r["lower"] for r in rows])  # (n_dims, N_p)
        mean_w = widths.mean(axis=1)
        ax2.semilogx(
            xs, mean_w,
            color=SOLVER_COLORS[sname],
            marker=SOLVER_MARKERS[sname],
            linestyle="-",
            markersize=8,
            linewidth=1.5,
            label=f"{sname} (mean)",
        )

    # Per-target widths for ProximalBundle as reference
    pb_rows = sorted(
        [r for r in results if r["solver"] == "ProximalBundle"],
        key=lambda r: r["data_dim"],
    )
    if pb_rows:
        xs_pb = np.array([r["data_dim"] for r in pb_rows])
        cap_colors = plt.cm.tab10(np.linspace(0, 1, N_P))
        for tgt_idx in range(N_P):
            cap_widths = np.array([r["upper"][tgt_idx] - r["lower"][tgt_idx] for r in pb_rows])
            ax2.semilogx(
                xs_pb, cap_widths,
                color=cap_colors[tgt_idx],
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label=f"Target {tgt_idx + 1}" if tgt_idx < 3 else "",
            )

    ax2.set_xlabel(r"Data-space dimension  $N_d$", fontsize=11)
    ax2.set_ylabel("Mean bound width", fontsize=11)
    ax2.set_title("DLI bound width vs $N_d$", fontsize=11)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    completed = sum(int(r.get("n_repeats", 0)) >= N_REPEATS for r in results)
    total     = len(DATA_DIMS) * len(SOLVER_NAMES)
    fig.suptitle(
        f"1-D Interval DLI — solver comparison  "
        f"(N_p={N_P}, seed={SEED})  —  "
        f"{completed}/{total} cases complete,  repeats={N_REPEATS}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      [figure saved → {FIGURE_PATH}]")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Partial save / load
# ---------------------------------------------------------------------------

def save_partial(results: list[dict]) -> None:
    if not results:
        return
    np.savez(
        PARTIAL_SAVE,
        solvers          = np.array([r["solver"]          for r in results]),
        data_dims        = np.array([r["data_dim"]        for r in results]),
        n_repeats        = np.array([r["n_repeats"]       for r in results]),
        solve_time_mean  = np.array([r["solve_time_mean"] for r in results]),
        solve_time_std   = np.array([r["solve_time_std"]  for r in results]),
        solve_times      = np.array([r["solve_times"]     for r in results], dtype=object),
        upper            = np.array([r["upper"]           for r in results]),
        lower            = np.array([r["lower"]           for r in results]),
        upper_std        = np.array([r["upper_std"]       for r in results]),
        lower_std        = np.array([r["lower_std"]       for r in results]),
        true_values      = np.array([r["true_values"]     for r in results]),
        prior_lower      = np.array([r["prior_lower"]     for r in results]),
        prior_upper      = np.array([r["prior_upper"]     for r in results]),
    )


def load_partial() -> tuple[list[dict], set[tuple[str, int]]]:
    if not PARTIAL_SAVE.exists():
        return [], set()
    try:
        d = np.load(PARTIAL_SAVE, allow_pickle=True)
        if "solve_time_mean" not in d:
            print("  Warning: partial save has legacy format — ignoring.")
            return [], set()
        results = []
        completed_keys: set[tuple[str, int]] = set()
        partial_count = 0
        for (solver_name, dim, nrep, t_mean, t_std, ts,
             up, lo, up_std, lo_std, tv, pl, pu) in zip(
            d["solvers"], d["data_dims"], d["n_repeats"],
            d["solve_time_mean"], d["solve_time_std"], d["solve_times"],
            d["upper"], d["lower"], d["upper_std"], d["lower_std"],
            d["true_values"], d["prior_lower"], d["prior_upper"],
        ):
            entry = {
                "solver":          str(solver_name),
                "data_dim":        int(dim),
                "n_repeats":       int(nrep),
                "solve_time_mean": float(t_mean),
                "solve_time_std":  float(t_std),
                "solve_times":     np.asarray(ts),
                "upper":           np.asarray(up),
                "lower":           np.asarray(lo),
                "upper_std":       np.asarray(up_std),
                "lower_std":       np.asarray(lo_std),
                "true_values":     np.asarray(tv),
                "prior_lower":     np.asarray(pl),
                "prior_upper":     np.asarray(pu),
            }
            results.append(entry)
            n_done = int(nrep)
            if n_done >= N_REPEATS:
                completed_keys.add((str(solver_name), int(dim)))
            else:
                partial_count += 1
        print(
            f"  Resuming: loaded {len(results)} runs from {PARTIAL_SAVE} "
            f"({len(completed_keys)} complete, {partial_count} partial for repeats={N_REPEATS})"
        )
        return results, completed_keys
    except Exception as exc:
        print(f"  Warning: could not load partial save ({exc}) — starting fresh.")
        return [], set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Interval DLI — solver comparison timing study")
    print("=" * 65)
    print(f"Data dims (N_d): {DATA_DIMS}")
    print(f"Solvers:         {SOLVER_NAMES}")
    print(f"N_p={N_P}, seed={SEED}, max_iter={MAX_ITER}, tol={TOL}")
    print(f"repeats={N_REPEATS},  support_n_jobs={SUPPORT_N_JOBS},  threads={DEFAULT_NUM_THREADS}")
    sys.stdout.flush()

    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(DEFAULT_NUM_THREADS)
    except Exception:
        print("  Warning: threadpoolctl not available.")

    # ---- Resume from partial save ------------------------------------------
    results, completed_keys = load_partial()

    # ---- Pre-build problem for each N_d ------------------------------------
    print("\nPre-building operators for all N_d values...")
    sys.stdout.flush()
    problems: dict[int, tuple] = {}
    t0_build = time.time()
    for n_d in DATA_DIMS:
        (M, D, P, G, T,
         m_bar, d_tilde,
         model_prior_support, data_error_support) = build_spaces_and_operators(n_d, seed=SEED)
        problems[n_d] = (M, D, P, G, T, m_bar, d_tilde, model_prior_support, data_error_support)
        print(f"  N_d={n_d}: built  ({time.time() - t0_build:.1f}s)")
        sys.stdout.flush()

    # ---- Main sweep --------------------------------------------------------
    print("\nStarting solver sweep...")
    sys.stdout.flush()

    def _upsert_result(results_list: list[dict[str, Any]], new_entry: dict[str, Any]) -> None:
        for i, old in enumerate(results_list):
            if old["solver"] == new_entry["solver"] and old["data_dim"] == new_entry["data_dim"]:
                results_list[i] = new_entry
                return
        results_list.append(new_entry)

    for n_d in DATA_DIMS:
        (M, D, P, G, T,
         m_bar, d_tilde,
         model_prior_support, data_error_support) = problems[n_d]

        for solver_name in SOLVER_NAMES:
            key = (solver_name, n_d)
            if key in completed_keys:
                print(f"  [skip] {solver_name}  N_d={n_d}  (already done)")
                sys.stdout.flush()
                continue

            print(f"\n  {solver_name}  N_d={n_d}")
            sys.stdout.flush()
            try:
                def _on_repeat_checkpoint(intermediate_result, rep_done, rep_total):
                    _upsert_result(results, intermediate_result)
                    save_partial(results)
                    plot_progress(results)
                    print(f"      [checkpoint saved after repeat {rep_done}/{rep_total}]")
                    sys.stdout.flush()

                result = run_one_repeated(
                    solver_name,
                    M, D, P, G, T,
                    m_bar, d_tilde,
                    model_prior_support, data_error_support,
                    n_repeats=N_REPEATS,
                    progress_callback=_on_repeat_checkpoint,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")
                sys.stdout.flush()
                continue

            _upsert_result(results, result)
            completed_keys.add(key)
            save_partial(results)
            plot_progress(results)

    # ---- Final save --------------------------------------------------------
    final_path = FIG_DIR / "interval_dli_results.npz"
    np.savez(
        final_path,
        solvers          = np.array([r["solver"]          for r in results]),
        data_dims        = np.array([r["data_dim"]        for r in results]),
        n_repeats        = np.array([r["n_repeats"]       for r in results]),
        solve_time_mean  = np.array([r["solve_time_mean"] for r in results]),
        solve_time_std   = np.array([r["solve_time_std"]  for r in results]),
        upper            = np.array([r["upper"]           for r in results]),
        lower            = np.array([r["lower"]           for r in results]),
        true_values      = np.array([r["true_values"]     for r in results]),
        prior_lower      = np.array([r["prior_lower"]     for r in results]),
        prior_upper      = np.array([r["prior_upper"]     for r in results]),
    )
    print(f"\nFinal results saved → {final_path}")
    print(f"Figure saved        → {FIGURE_PATH}")
    print("Done.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
