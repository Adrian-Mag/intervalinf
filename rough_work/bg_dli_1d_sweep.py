"""
bg_dli_1d_sweep.py
==================
Script scaffold and shared builders for a 1D BG-vs-DLI sweep.

Phase 1: Configuration, space/operator builders, synthetic data,
         prior/confidence-set builders, and validation.
Phase 2: DLI and BG 1D interval solvers and validation.
Phases 3-4: Sweep loop, CSV/PNG outputs (to be added).

Usage:
    conda run -n inferences3 python intervalinf/rough_work/bg_dli_1d_sweep.py

Design decisions:
- N_p = 1 throughout; property space is R^1.
- Forward (kernel) seed is separate from data (noise) seed.
  Same forward seed across all N_d values => NormalModesProvider kernels
  form a strict nested prefix as N_d grows.
- Confidence set is chi-square calibrated (fair, not fitted to realised noise);
  the same EllipsoidSupportFunction is shared by both DLI and BG in Phase 2.
- Both methods use identical model prior and data-confidence sets so results
  are directly comparable.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional, cast

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
    SupportFunction,
)
from pygeoinf.backus_gilbert import DualMasterCostFunction
from pygeoinf.convex_optimisation import (
    ProximalBundleMethod,
    solve_support_values,
    best_available_qp_solver,
)

# Matplotlib: use non-interactive backend before importing pyplot.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

# Data dimensions to sweep.  NormalModesProvider uses nested semantics:
# N_d=10 inherits the first 10 kernels generated for N_d=5, 10, 20, 40.
N_D_VALUES: list[int] = [5, 10, 20, 40, 100, 200, 400, 1000]

# Seeds for NormalModesProvider (forward operator / kernel generation).
# Same forward seed across all N_d values gives nested kernels.
FORWARD_SEEDS: list[int] = [2, 7, 13]

# Seeds for noise draw (independent of forward seed).
DATA_SEEDS: list[int] = [42, 123, 456]

# Per-datum noise level as a fraction of the RMS clean-data amplitude.
NOISE_FRACTION: float = 0.10

# Chi-square confidence level for the data-confidence ellipsoid.
# df = N_d, so the ellipsoid reflects the geometry of the Gaussian model.
CONFIDENCE_LEVEL: float = 0.95

# Multiplier on ||m_bar - m_0||_M for the model prior radius.
PRIOR_RADIUS_FACTOR: float = 1.05

# Output directories (created on first use).
RESULTS_FOLDER: str = "bg_dli_1d_sweep_results"
FIGURES_FOLDER: str = "bg_dli_1d_sweep_figures"

# Tolerance for reporting containment violations (machine-precision guard).
CONTAINMENT_TOL: float = 1e-6

# Integration resolution.
LEBESGUE_N_POINTS: int = 500
SOLA_N_POINTS: int = 1000
N_JOBS: int = 8

# Property target settings: single bump at x=0.5, width 0.3.
PROPERTY_CENTER: float = 0.5
PROPERTY_BUMP_WIDTH: float = 0.3

# ---------------------------------------------------------------------------
# Phase 2 solver configuration
# ---------------------------------------------------------------------------

# DLI proximal bundle method settings (small-to-moderate tolerances for
# validation; the sweep will accept these defaults).
DLI_TOLERANCE: float = 1e-4
DLI_MAX_ITERATIONS: int = 500
DLI_BUNDLE_SIZE: int = 50
DLI_RHO0: float = 1.0
DLI_RHO_FACTOR: float = 2.0


# ---------------------------------------------------------------------------
# Integration and parallel configuration
# ---------------------------------------------------------------------------

def build_integration_configs(
    lebesgue_n_points: int = LEBESGUE_N_POINTS,
    sola_n_points: int = SOLA_N_POINTS,
    n_jobs: int = N_JOBS,
):
    """Return shared Lebesgue, SOLA, and parallel configs."""
    lebesgue_cfg = LebesgueIntegrationConfig(
        inner_product=IntegrationConfig(
            method="simpson", n_points=lebesgue_n_points
        ),
        dual=IntegrationConfig(method="simpson", n_points=lebesgue_n_points),
        general=IntegrationConfig(
            method="simpson", n_points=lebesgue_n_points
        ),
    )
    sola_cfg = IntegrationConfig(method="simpson", n_points=sola_n_points)
    par_cfg = ParallelConfig(enabled=True, n_jobs=n_jobs)
    return lebesgue_cfg, sola_cfg, par_cfg


# ---------------------------------------------------------------------------
# Helper 1 -- Spaces and operators (N_p = 1 always)
# ---------------------------------------------------------------------------

def build_spaces_and_operators(
    N_d: int,
    forward_seed: int,
    lebesgue_cfg: Optional[LebesgueIntegrationConfig] = None,
    sola_cfg: Optional[IntegrationConfig] = None,
    par_cfg: Optional[ParallelConfig] = None,
    property_center: float = PROPERTY_CENTER,
    property_bump_width: float = PROPERTY_BUMP_WIDTH,
) -> dict:
    """Build M, D, P, G, T for a 1D sweep case (N_p=1).

    The NormalModesProvider is seeded with *forward_seed* so that the first
    N_d kernels are a nested prefix for all N_d values sharing the same seed.

    Args:
        N_d: Dimension of data space D = R^{N_d}.
        forward_seed: Seed for NormalModesProvider (kept fixed across N_d).
        lebesgue_cfg: LebesgueIntegrationConfig; built from defaults if None.
        sola_cfg: Integration config for SOLA operators;
            built from defaults if None.
        par_cfg: ParallelConfig; built from defaults if None.
        property_center: Centre of the single bump target kernel in [0, 1].
        property_bump_width: Width of the bump target kernel.

    Returns:
        Dict with keys: M, D, P, G, T, function_domain.
    """
    if lebesgue_cfg is None or sola_cfg is None or par_cfg is None:
        lebesgue_cfg, sola_cfg, par_cfg = build_integration_configs()

    function_domain = IntervalDomain(0, 1)

    M = Lebesgue(
        0,
        function_domain,
        basis=None,
        integration_config=lebesgue_cfg,
        parallel_config=par_cfg,
    )
    D = EuclideanSpace(N_d)
    P = EuclideanSpace(1)

    normal_modes = NormalModesProvider(
        M,
        n_modes_range=(1, 50),
        coeff_range=(-5, 5),
        gaussian_width_percent_range=(1, 5),
        freq_range=(0.1, 20),
        random_state=forward_seed,
    )

    G = SOLAOperator(
        M, D,
        kernels=normal_modes,
        cache_kernels=True,
        integration_config=sola_cfg,
    )

    target_provider = BumpFunctionProvider(
        M,
        centers=np.array([property_center]),
        default_width=property_bump_width,
    )
    T = SOLAOperator(
        M, P,
        kernels=target_provider,
        cache_kernels=True,
        integration_config=sola_cfg,
    )

    return {
        "M": M,
        "D": D,
        "P": P,
        "G": G,
        "T": T,
        "function_domain": function_domain,
    }


# ---------------------------------------------------------------------------
# Helper 2 -- Synthetic data generation
# ---------------------------------------------------------------------------

def build_true_model_and_data(
    M: Lebesgue,
    G: SOLAOperator,
    T: SOLAOperator,
    N_d: int,
    noise_fraction: float = NOISE_FRACTION,
    data_seed: int = 42,
) -> dict:
    """Construct the true model, clean data, and a noisy realisation.

    The per-datum noise level is calibrated from the RMS clean-data amplitude
    so that sigma_d is independent of N_d in expectation:

        sigma_d = noise_fraction * ||d_bar||_2 / sqrt(N_d).

    The data seed controls only the noise draw; the forward operator (and thus
    d_bar) is determined entirely by the model and the kernel seed.

    Args:
        M: Model Hilbert space L^2([0,1]).
        G: Forward SOLA operator M -> D.
        T: Property SOLA operator M -> P.
        N_d: Data dimension (used for RMS normalisation and shape check).
        noise_fraction: Per-datum noise as a fraction of RMS signal amplitude.
        data_seed: RNG seed for noise draw.

    Returns:
        Dict with keys: m_bar, d_bar, d_tilde, noise_vector,
                        sigma_d, signal_rms, p_bar.
    """
    m_bar = Function(
        M,
        evaluate_callable=lambda x: (
            np.exp(-((x - 0.5) / 0.5) ** 2) * np.sin(5 * np.pi * x) + x
        ),
    )

    d_bar = G(m_bar)
    signal_rms = np.linalg.norm(d_bar) / np.sqrt(N_d)
    sigma_d = noise_fraction * signal_rms

    rng = np.random.default_rng(data_seed)
    noise_vector = rng.normal(0.0, sigma_d, N_d)
    d_tilde = d_bar + noise_vector

    p_bar = T(m_bar)

    return {
        "m_bar": m_bar,
        "d_bar": d_bar,
        "d_tilde": d_tilde,
        "noise_vector": noise_vector,
        "sigma_d": sigma_d,
        "signal_rms": signal_rms,
        "p_bar": p_bar,
    }


# ---------------------------------------------------------------------------
# Helper 3 -- Prior and confidence-set builders
# ---------------------------------------------------------------------------

def _scalar_op(space, scale: float):
    """Return a self-adjoint scale * I LinearOperator on *space*."""
    return LinearOperator.self_adjoint(space, lambda v: scale * v)


def build_prior_and_confidence(
    M: Lebesgue,
    D: EuclideanSpace,
    m_bar: Function,
    sigma_d: float,
    N_d: int,
    confidence_level: float = CONFIDENCE_LEVEL,
    prior_radius_factor: float = PRIOR_RADIUS_FACTOR,
) -> dict:
    """Build the model prior ball and the chi-square calibrated data ellipsoid.

    The model ball is centred at m_0(x) = x with radius:

        r = prior_radius_factor * ||m_bar - m_0||_M.

    The data-confidence ellipsoid is calibrated from the Gaussian noise model
    with covariance C_d = sigma_d^2 * I and chi-square quantile:

        s^2 = (1/2) * chi2(N_d, confidence_level),
        r_V = sigma_d * sqrt(chi2(N_d, confidence_level)).

    This is a fair, fixed-confidence construction that does NOT fit the radius
    to the realised noise vector. The ellipsoid is shared by DLI and BG.

    Args:
        M: Model Hilbert space.
        D: Data Euclidean space R^{N_d}.
        m_bar: True model Function (for prior radius calibration only).
        sigma_d: Per-datum noise standard deviation.
        N_d: Data dimension (chi-square df).
        confidence_level: Probability level 1 - beta.
        prior_radius_factor: Multiplier on ||m_bar - m_0||.

    Returns:
        Dict with keys:
            model_prior_support  -- BallSupportFunction (M),
            data_conf_ellipsoid  -- EllipsoidSupportFunction (D),
            s_confidence         -- chi-square scale s (float),
            r_data_conf          -- Euclidean radius r_V of ellipsoid (float),
            model_radius         -- model ball radius r (float),
            m_0                  -- centre Function of model ball.
    """
    m_0 = Function(M, evaluate_callable=lambda x: x)
    model_radius = prior_radius_factor * M.norm(M.subtract(m_bar, m_0))
    model_prior_support = BallSupportFunction(M, m_0, model_radius)

    chi2_quantile = chi2.ppf(confidence_level, df=N_d)
    s_confidence = np.sqrt(0.5 * chi2_quantile)
    r_data_conf = sigma_d * np.sqrt(chi2_quantile)

    # Ellipsoid {v : v^T A v <= s^2}, A = C_d^{-1}/2 = I/(2*sigma_d^2).
    A_op = _scalar_op(D, 1.0 / (2.0 * sigma_d ** 2))
    A_inv_op = _scalar_op(D, 2.0 * sigma_d ** 2)
    A_inv_sqrt = _scalar_op(D, np.sqrt(2.0) * sigma_d)

    data_conf_ellipsoid = EllipsoidSupportFunction(
        D,
        center=D.zero,
        radius=float(s_confidence),
        shape_operator=A_op,
        inverse_operator=A_inv_op,
        inverse_sqrt_operator=A_inv_sqrt,
    )

    return {
        "model_prior_support": model_prior_support,
        "data_conf_ellipsoid": data_conf_ellipsoid,
        "s_confidence": s_confidence,
        "r_data_conf": r_data_conf,
        "model_radius": model_radius,
        "m_0": m_0,
    }


# ---------------------------------------------------------------------------
# Helper 4 -- Convenience: build a full problem instance in one call
# ---------------------------------------------------------------------------

def build_problem(
    N_d: int,
    forward_seed: int,
    data_seed: int,
    noise_fraction: float = NOISE_FRACTION,
    confidence_level: float = CONFIDENCE_LEVEL,
    prior_radius_factor: float = PRIOR_RADIUS_FACTOR,
    lebesgue_n_points: int = LEBESGUE_N_POINTS,
    sola_n_points: int = SOLA_N_POINTS,
    n_jobs: int = N_JOBS,
) -> dict:
    """Build a complete Phase-1 problem instance.

    Args:
        N_d: Data dimension.
        forward_seed: Seed for NormalModesProvider kernel generation.
        data_seed: Seed for noise draw.
        noise_fraction: Per-datum noise fraction.
        confidence_level: Chi-square confidence level.
        prior_radius_factor: Multiplier for model ball radius.
        lebesgue_n_points: Integration grid for Lebesgue space.
        sola_n_points: Integration grid for SOLA operators.
        n_jobs: Parallel workers.

    Returns:
        Flat dict merging spaces, data, and confidence objects.
        Keys from build_spaces_and_operators, build_true_model_and_data,
        and build_prior_and_confidence, plus 'N_d', 'forward_seed',
        'data_seed'.
    """
    lebesgue_cfg, sola_cfg, par_cfg = build_integration_configs(
        lebesgue_n_points=lebesgue_n_points,
        sola_n_points=sola_n_points,
        n_jobs=n_jobs,
    )
    spaces = build_spaces_and_operators(
        N_d=N_d,
        forward_seed=forward_seed,
        lebesgue_cfg=lebesgue_cfg,
        sola_cfg=sola_cfg,
        par_cfg=par_cfg,
    )
    M, D, G, T = spaces["M"], spaces["D"], spaces["G"], spaces["T"]

    data = build_true_model_and_data(
        M=M, G=G, T=T, N_d=N_d,
        noise_fraction=noise_fraction,
        data_seed=data_seed,
    )

    conf = build_prior_and_confidence(
        M=M, D=D,
        m_bar=data["m_bar"],
        sigma_d=data["sigma_d"],
        N_d=N_d,
        confidence_level=confidence_level,
        prior_radius_factor=prior_radius_factor,
    )

    result = {"N_d": N_d, "forward_seed": forward_seed, "data_seed": data_seed}
    result.update(spaces)
    result.update(data)
    result.update(conf)
    return result


# ---------------------------------------------------------------------------
# Phase 1 validation
# ---------------------------------------------------------------------------

def run_phase1_validation(verbose: bool = True) -> None:
    """Run lightweight assertions that Phase 1 builders are correct.

    Checks:
    1. P.dim == 1 for all N_d values.
    2. D.dim matches the requested N_d.
    3. Generated data d_bar and d_tilde have shape (N_d,).
    4. p_bar has shape (1,).
    5. sigma_d, model_radius, r_data_conf, s_confidence are all positive
       and finite.
    6. Nested-kernel property: the first N_d1 components of G(m_bar) for
       N_d=N_d2 > N_d1 (same forward_seed) match G(m_bar) for N_d=N_d1.
    7. Independent data seeds produce different d_tilde vectors.
    """
    _vprint = print if verbose else (lambda *a, **k: None)

    _vprint("--- Phase 1 Validation ---\n")

    forward_seed = FORWARD_SEEDS[0]
    data_seed_a = DATA_SEEDS[0]
    data_seed_b = DATA_SEEDS[1]
    N_d_small, N_d_large = N_D_VALUES[0], N_D_VALUES[1]   # e.g. 5 and 10

    # ------------------------------------------------------------------
    # Check 1-5: Dimensions, shapes, and scalar finiteness
    # ------------------------------------------------------------------
    for N_d in N_D_VALUES:
        prob = build_problem(
            N_d,
            forward_seed=forward_seed,
            data_seed=data_seed_a,
        )

        # 1. Property space is 1D.
        assert prob["P"].dim == 1, (
            f"N_d={N_d}: P.dim={prob['P'].dim}, expected 1"
        )

        # 2. Data space dimension matches.
        assert prob["D"].dim == N_d, f"N_d={N_d}: D.dim={prob['D'].dim}"

        # 3. Data vectors have correct shape.
        assert prob["d_bar"].shape == (N_d,), (
            f"N_d={N_d}: d_bar shape {prob['d_bar'].shape}"
        )
        assert prob["d_tilde"].shape == (N_d,), (
            f"N_d={N_d}: d_tilde shape {prob['d_tilde'].shape}"
        )

        # 4. Property vector has shape (1,).
        assert prob["p_bar"].shape == (1,), (
            f"N_d={N_d}: p_bar shape {prob['p_bar'].shape}"
        )

        # 5. Scalars are positive and finite.
        for key in ("sigma_d", "model_radius", "r_data_conf", "s_confidence"):
            val = prob[key]
            assert np.isfinite(val), f"N_d={N_d}: {key}={val} is not finite"
            assert val > 0, f"N_d={N_d}: {key}={val} is not positive"

        _vprint(
            f"  N_d={N_d:3d}  P.dim={prob['P'].dim}"
            f"  sigma_d={prob['sigma_d']:.5f}"
            f"  r_V={prob['r_data_conf']:.4f}  r_B={prob['model_radius']:.4f}"
            f"  p_bar={float(prob['p_bar'][0]):.4f}"
        )

    _vprint()

    # ------------------------------------------------------------------
    # Check 6: Nested kernel property
    # ------------------------------------------------------------------
    _vprint(
        f"  Nested kernel check: forward_seed={forward_seed},"
        f" N_d={N_d_small} vs N_d={N_d_large}"
    )

    lebesgue_cfg, sola_cfg, par_cfg = build_integration_configs()

    spaces_small = build_spaces_and_operators(
        N_d_small, forward_seed,
        lebesgue_cfg=lebesgue_cfg, sola_cfg=sola_cfg, par_cfg=par_cfg,
    )
    spaces_large = build_spaces_and_operators(
        N_d_large, forward_seed,
        lebesgue_cfg=lebesgue_cfg, sola_cfg=sola_cfg, par_cfg=par_cfg,
    )

    M_s = spaces_small["M"]
    G_s = spaces_small["G"]
    G_l = spaces_large["G"]

    # Use the same true model - re-evaluate directly.
    m_bar_s = Function(
        M_s,
        evaluate_callable=lambda x: (
            np.exp(-((x - 0.5) / 0.5) ** 2) * np.sin(5 * np.pi * x) + x
        ),
    )
    d_small = G_s(m_bar_s)
    d_large = G_l(m_bar_s)

    # First N_d_small components of d_large must equal d_small.
    np.testing.assert_allclose(
        d_large[:N_d_small], d_small,
        rtol=1e-8, atol=1e-10,
        err_msg=(
            f"Nested kernel check FAILED: first {N_d_small} entries of "
            f"G(N_d={N_d_large})(m) do not match G(N_d={N_d_small})(m)"
        ),
    )
    _vprint(
        f"  Nested kernel check PASSED: "
        f"max |d_large[:N_d_small] - d_small| = "
        f"{float(np.max(np.abs(d_large[:N_d_small] - d_small))):.2e}"
    )

    _vprint()

    # ------------------------------------------------------------------
    # Check 7: Different data seeds produce different d_tilde
    # ------------------------------------------------------------------
    lebesgue_cfg, sola_cfg, par_cfg = build_integration_configs()
    spaces_a = build_spaces_and_operators(
        N_d_small, forward_seed,
        lebesgue_cfg=lebesgue_cfg, sola_cfg=sola_cfg, par_cfg=par_cfg,
    )
    data_a = build_true_model_and_data(
        M=spaces_a["M"], G=spaces_a["G"], T=spaces_a["T"],
        N_d=N_d_small, data_seed=data_seed_a,
    )
    data_b = build_true_model_and_data(
        M=spaces_a["M"], G=spaces_a["G"], T=spaces_a["T"],
        N_d=N_d_small, data_seed=data_seed_b,
    )

    # d_bar must be identical (same forward operator, same model).
    np.testing.assert_allclose(data_a["d_bar"], data_b["d_bar"], rtol=1e-12)

    # d_tilde must differ (different noise seeds).
    assert not np.allclose(data_a["d_tilde"], data_b["d_tilde"]), (
        "data_seed_a and data_seed_b produced identical d_tilde; "
        "seeds may be equal"
    )
    _vprint(
        f"  Independent seeds check PASSED: d_bar identical,"
        " d_tilde differ by L2 norm "
        f"{np.linalg.norm(data_a['d_tilde'] - data_b['d_tilde']):.4f}"
    )

    _vprint()
    _vprint("--- Phase 1 Validation PASSED ---")


# ---------------------------------------------------------------------------
# Phase 2: DLI interval solver
# ---------------------------------------------------------------------------

def compute_dli_interval(
    M: Lebesgue,
    D: EuclideanSpace,
    P: EuclideanSpace,
    G: SOLAOperator,
    T: SOLAOperator,
    d_tilde: np.ndarray,
    model_prior_support: BallSupportFunction,
    data_conf_support,
    *,
    tolerance: float = DLI_TOLERANCE,
    max_iterations: int = DLI_MAX_ITERATIONS,
    bundle_size: int = DLI_BUNDLE_SIZE,
    rho0: float = DLI_RHO0,
    rho_factor: float = DLI_RHO_FACTOR,
) -> dict:
    r"""Compute the 1D DLI admissible interval via the dual master equation.

    Solves the dual master cost minimisation for directions $q = +e_1$ and
    $q = -e_1$ in the 1D property space $P = \mathbb{R}^1$:

        $h_U(q) = \min_{\lambda \in D}
            \{ \langle\lambda, \tilde{d}\rangle_D
             + \sigma_B(T^* q - G^* \lambda)
             + \sigma_V(-\lambda) \}$

    The interval $[L, U]$ is assembled as:
        $U = h_U(+e_1)$,  $L = -h_U(-e_1)$.

    Both directions are solved independently (no warm-starting across
    opposite directions since they are typically far apart in dual space).

    Args:
        M: Model Hilbert space $L^2([0,1])$.
        D: Data Euclidean space $\mathbb{R}^{N_d}$.
        P: Property Euclidean space $\mathbb{R}^1$ (must have dim=1).
        G: Forward SOLA operator $M \to D$.
        T: Property SOLA operator $M \to P$.
        d_tilde: Observed (noisy) data vector, shape (N_d,).
        model_prior_support: Support function of the model prior ball $B$.
        data_conf_support: Support function of the data confidence set $V$
            (EllipsoidSupportFunction or BallSupportFunction, centred at 0).
        tolerance: Bundle method convergence tolerance.
        max_iterations: Maximum bundle iterations per direction.
        bundle_size: Maximum number of cuts in the bundle.
        rho0: Initial proximal weight $\rho_0 > 0$.
        rho_factor: Multiplicative factor for $\rho$ on null steps.

    Returns:
        Dict with keys:
            upper        -- float, upper endpoint $h_U(+e_1)$,
            lower        -- float, lower endpoint $-h_U(-e_1)$,
            width        -- float, interval width (upper - lower),
            h_pos        -- float, raw support value $h_U(+e_1)$,
            h_neg        -- float, raw support value $h_U(-e_1)$,
            total_iters  -- int, total bundle iterations (both directions),
            converged    -- tuple[bool, bool], convergence flags (+e1, -e1),
            gap          -- tuple[float, float], final duality gaps (+e1, -e1),
            elapsed_s    -- float, wall-clock time (seconds).
    """
    assert P.dim == 1, f"P must be 1-dimensional (N_p=1), got dim={P.dim}"

    # Standard basis directions in R^1.
    e1 = P.basis_vector(0)       # np.array([1.0])
    neg_e1 = P.negative(e1)      # np.array([-1.0])
    lambda0 = D.zero             # warm-start from zero each time

    # Build the cost with an initial direction
    # (overridden by solve_support_values for each direction).
    cost = DualMasterCostFunction(
        D, P, M, G, T,
        model_prior_support, data_conf_support,
        d_tilde, e1,
    )

    solver = ProximalBundleMethod(
        cost,
        rho0=rho0,
        rho_factor=rho_factor,
        tolerance=tolerance,
        max_iterations=max_iterations,
        bundle_size=bundle_size,
        qp_solver=best_available_qp_solver(),
    )

    t0 = time.perf_counter()

    # Solve for +e1: yields upper = h_U(+e1).
    vals_pos, _, diags_pos = solve_support_values(
        cost, [e1], solver, lambda0, warm_start=False
    )
    h_pos = float(vals_pos[0])
    diag_pos = diags_pos[0]

    # Solve for -e1: yields -lower = h_U(-e1), hence lower = -h_neg.
    vals_neg, _, diags_neg = solve_support_values(
        cost, [neg_e1], solver, lambda0, warm_start=False
    )
    h_neg = float(vals_neg[0])
    diag_neg = diags_neg[0]

    elapsed = time.perf_counter() - t0

    upper = h_pos
    lower = -h_neg
    width = upper - lower

    return {
        "method": "DLI",
        "upper": upper,
        "lower": lower,
        "width": width,
        "h_pos": h_pos,
        "h_neg": h_neg,
        "total_iters": diag_pos.num_iterations + diag_neg.num_iterations,
        "converged": (diag_pos.converged, diag_neg.converged),
        "gap": (diag_pos.gap, diag_neg.gap),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Phase 2: BG interval solver
# ---------------------------------------------------------------------------

def compute_bg_interval(
    M: Lebesgue,
    D: EuclideanSpace,
    P: EuclideanSpace,
    G: SOLAOperator,
    T: SOLAOperator,
    d_tilde: np.ndarray,
    model_prior_support: BallSupportFunction,
    data_conf_support,
    model_radius: float,
    r_data_conf: float,
    *,
    n_jobs: int = N_JOBS,
) -> dict:
    r"""Compute the 1D BG admissible interval via Minkowski-sum
    support algebra.

    Builds the optimal Backus-Gilbert estimator

        $X^* = T G^* (G G^* + \alpha I_D)^{-1}$,
        $\alpha = (r_V / r_B)^2$

    where $r_V$ = ``r_data_conf`` is the Euclidean outer radius of the data
    confidence set $V$ and $r_B$ = ``model_radius`` is the model ball radius.

    The BG admissible set satisfies the Minkowski-sum inclusion

        $\mathcal{U}_\text{BG} \subseteq
         \tilde{p}_\text{BG} + H^*(B) + X^*(V)$

    with $\tilde{p}_\text{BG} = X^* \tilde{d}$ and $H = T - X^* G$.
    The support function is assembled algebraically:

        $h_{\mathcal{U}_\text{BG}}(q)
         = \langle q, \tilde{p}\rangle
           + h_B(H^* q)
           + h_V((X^*)^* q).$

    Evaluating at $q = +e_1$ and $q = -e_1$ gives the 1D interval.

    Args:
        M: Model Hilbert space.
        D: Data Euclidean space $\mathbb{R}^{N_d}$.
        P: Property Euclidean space $\mathbb{R}^1$.
        G: Forward SOLA operator $M \to D$.
        T: Property SOLA operator $M \to P$.
        d_tilde: Observed data vector, shape (N_d,).
        model_prior_support: Support function of model prior ball $B$.
        data_conf_support: Support function of data confidence set $V$
            (centred at 0; EllipsoidSupportFunction or BallSupportFunction).
        model_radius: Hilbert-space ball radius $r_B$ (used for $\alpha$).
        r_data_conf: Euclidean outer radius $r_V$ of the confidence set
            (used for $\alpha$; equals sigma_d * sqrt(chi2_quantile)).
        n_jobs: Number of parallel worker threads for CholeskySolver.

    Returns:
        Dict with keys:
            upper      -- float, upper endpoint $h_\text{BG}(+e_1)$,
            lower      -- float, lower endpoint $-h_\text{BG}(-e_1)$,
            width      -- float, interval width (upper - lower),
            h_pos      -- float, raw support value at $+e_1$,
            h_neg      -- float, raw support value at $-e_1$,
            p_t        -- np.ndarray shape (1,), BG point estimate,
            alpha_bg   -- float, regularisation parameter $\alpha$.
    """
    assert P.dim == 1, f"P must be 1-dimensional (N_p=1), got dim={P.dim}"

    # Regularisation balancing data confidence and model prior radii.
    alpha_bg = (r_data_conf / model_radius) ** 2

    I_D_scaled = LinearOperator.self_adjoint(D, lambda v: alpha_bg * v)
    regularized_data_op = cast(
        LinearOperator, G @ G.adjoint + I_D_scaled
    )

    solver_bg = CholeskySolver(parallel=True, n_jobs=n_jobs)
    regularized_inv = solver_bg(regularized_data_op)

    X_star = cast(LinearOperator, T @ G.adjoint @ regularized_inv)  # D -> P
    p_t = X_star(d_tilde)        # BG point estimate, shape (1,)
    H = cast(LinearOperator, T - X_star @ G)      # residual operator M -> P

    # Minkowski-sum support function assembled algebraically.
    bg_support = (
        SupportFunction.point(P, p_t)
        + model_prior_support.image(H)
        + data_conf_support.image(X_star)
    )

    e1 = P.basis_vector(0)
    neg_e1 = P.negative(e1)

    h_pos = float(bg_support(e1))
    h_neg = float(bg_support(neg_e1))

    upper = h_pos
    lower = -h_neg
    width = upper - lower

    return {
        "method": "BG",
        "upper": upper,
        "lower": lower,
        "width": width,
        "h_pos": h_pos,
        "h_neg": h_neg,
        "p_t": p_t,
        "alpha_bg": alpha_bg,
    }


# ---------------------------------------------------------------------------
# Phase 2 validation
# ---------------------------------------------------------------------------

def run_phase2_validation(verbose: bool = True) -> None:
    """Validate DLI and BG interval solvers on a small problem instance.

    Checks:
    1. Both DLI and BG intervals have finite endpoints and non-negative width.
    2. The lower-bound formula lower = -h(-e1) holds exactly by construction.
    3. The true property p_bar is contained in both intervals (containment).
    4. BG interval width >= DLI width
       (informational; not asserted, just printed).

    Uses N_d = N_D_VALUES[0] (smallest problem) for speed.
    """
    _vprint = print if verbose else (lambda *a, **k: None)

    _vprint("--- Phase 2 Validation ---\n")

    N_d = N_D_VALUES[0]   # smallest problem for validation speed
    forward_seed = FORWARD_SEEDS[0]
    data_seed = DATA_SEEDS[0]

    prob = build_problem(N_d, forward_seed, data_seed)
    M, D, P = prob["M"], prob["D"], prob["P"]
    G, T = prob["G"], prob["T"]
    d_tilde = prob["d_tilde"]
    model_prior_support = prob["model_prior_support"]
    data_conf_ellipsoid = prob["data_conf_ellipsoid"]
    p_bar = prob["p_bar"]
    model_radius = prob["model_radius"]
    r_data_conf = prob["r_data_conf"]

    _vprint(
        f"  Problem: N_d={N_d}, forward_seed={forward_seed}, "
        f"data_seed={data_seed}"
    )
    _vprint(
        f"  True property p_bar = {float(p_bar[0]):.6f}  "
        f"sigma_d={prob['sigma_d']:.5f}"
    )
    _vprint()

    # ------------------------------------------------------------------
    # DLI interval
    # ------------------------------------------------------------------
    _vprint("  Computing DLI interval (ProximalBundleMethod, 2 directions)...")
    dli = compute_dli_interval(
        M, D, P, G, T, d_tilde,
        model_prior_support, data_conf_ellipsoid,
    )
    _vprint(
        f"  DLI: [{dli['lower']:+.6f}, {dli['upper']:+.6f}]"
        f"  width={dli['width']:.6f}"
        f"  iters={dli['total_iters']}"
        f"  t={dli['elapsed_s']:.2f}s"
        f"  converged=({dli['converged'][0]}, {dli['converged'][1]})"
        f"  gap=({dli['gap'][0]:.2e}, {dli['gap'][1]:.2e})"
    )

    # ------------------------------------------------------------------
    # BG interval
    # ------------------------------------------------------------------
    _vprint("  Computing BG  interval (Minkowski-sum support algebra)...")
    t_bg = time.perf_counter()
    bg = compute_bg_interval(
        M, D, P, G, T, d_tilde,
        model_prior_support, data_conf_ellipsoid,
        model_radius, r_data_conf,
    )
    bg_elapsed = time.perf_counter() - t_bg
    _vprint(
        f"  BG:  [{bg['lower']:+.6f}, {bg['upper']:+.6f}]"
        f"  width={bg['width']:.6f}"
        f"  t={bg_elapsed:.3f}s"
        f"  alpha_bg={bg['alpha_bg']:.4e}"
        f"  p_t={float(bg['p_t'][0]):.6f}"
    )

    _vprint()

    # ------------------------------------------------------------------
    # Assertion 1: finite endpoints and non-negative widths
    # ------------------------------------------------------------------
    for label, result in [("DLI", dli), ("BG", bg)]:
        assert np.isfinite(result["upper"]), (
            f"{label}: upper endpoint is not finite: {result['upper']}"
        )
        assert np.isfinite(result["lower"]), (
            f"{label}: lower endpoint is not finite: {result['lower']}"
        )
        assert np.isfinite(result["width"]), (
            f"{label}: width is not finite: {result['width']}"
        )
        assert result["width"] >= -1e-9, (
            f"{label}: negative width {result['width']:.6f}"
        )

    # ------------------------------------------------------------------
    # Assertion 2: lower = -h(-e1) by construction (documents formula)
    # ------------------------------------------------------------------
    np.testing.assert_allclose(
        dli["lower"], -dli["h_neg"],
        atol=1e-12,
        err_msg="DLI: lower endpoint does not equal -h(-e1)",
    )
    np.testing.assert_allclose(
        bg["lower"], -bg["h_neg"],
        atol=1e-12,
        err_msg="BG: lower endpoint does not equal -h(-e1)",
    )

    # ------------------------------------------------------------------
    # Assertion 3: p_bar ∈ [lower, upper] for both methods
    # ------------------------------------------------------------------
    p_scalar = float(p_bar[0])
    for label, result in [("DLI", dli), ("BG", bg)]:
        assert result["lower"] <= p_scalar + 1e-9, (
            f"{label} containment FAIL: "
            f"lower={result['lower']:.6f} > p_bar={p_scalar:.6f}"
        )
        assert result["upper"] >= p_scalar - 1e-9, (
            f"{label} containment FAIL: "
            f"upper={result['upper']:.6f} < p_bar={p_scalar:.6f}"
        )

    _vprint("  PASS: all Phase 2 assertions satisfied")
    _vprint()

    # Informational: width comparison.
    ratio = bg["width"] / max(dli["width"], 1e-12)
    _vprint(
        f"  Width ratio BG/DLI = {ratio:.4f}"
        f"  (BG={'wider' if ratio > 1 else 'narrower'} than DLI)"
    )

    _vprint()
    _vprint("--- Phase 2 Validation PASSED ---")


# ---------------------------------------------------------------------------
# Phase 3: gap metrics, sweep, CSV writers, and validation
# ---------------------------------------------------------------------------

def _results_dir() -> Path:
    """Return the results directory, creating it if needed."""
    base = Path(__file__).parent / RESULTS_FOLDER
    base.mkdir(parents=True, exist_ok=True)
    return base


def compute_gap_metrics(dli: dict, bg: dict) -> dict:
    r"""Compute BG-vs-DLI gap metrics from two interval dicts.

    Both dicts must have keys ``lower``, ``upper``, ``width``.

    Metrics:
    - ``width_ratio``  = width_BG / width_DLI  (guarded against zero width).
    - ``excess_width`` = width_BG - width_DLI.
    - ``upper_slack``  = bg_upper - dli_upper  (>0 means BG upper is higher).
    - ``lower_slack``  = dli_lower - bg_lower  (>0 means BG lower is lower).
    - ``hausdorff_gap``  = max(upper_slack, lower_slack).
    - ``containment_violation`` = max(0, bg_lower - dli_lower,
      dli_upper - bg_upper). Theoretically zero; guards against
      rare machine-precision disagreements.

    Returns:
        Dict with all six metrics plus copies of the raw interval endpoints.
    """
    dli_lower = float(dli["lower"])
    dli_upper = float(dli["upper"])
    dli_width = float(dli["width"])
    bg_lower = float(bg["lower"])
    bg_upper = float(bg["upper"])
    bg_width = float(bg["width"])

    width_ratio = bg_width / max(dli_width, 1e-15)
    excess_width = bg_width - dli_width
    upper_slack = bg_upper - dli_upper
    lower_slack = dli_lower - bg_lower
    hausdorff_gap = max(upper_slack, lower_slack)
    containment_violation = max(
        0.0, bg_lower - dli_lower, dli_upper - bg_upper
    )

    return {
        "dli_lower": dli_lower,
        "dli_upper": dli_upper,
        "dli_width": dli_width,
        "bg_lower": bg_lower,
        "bg_upper": bg_upper,
        "bg_width": bg_width,
        "width_ratio": width_ratio,
        "excess_width": excess_width,
        "upper_slack": upper_slack,
        "lower_slack": lower_slack,
        "hausdorff_gap": hausdorff_gap,
        "containment_violation": containment_violation,
    }


def evaluate_case(
    N_d: int,
    forward_seed: int,
    data_seed: int,
) -> dict:
    """Build one problem instance and evaluate both DLI and BG intervals.

    Args:
        N_d: Data dimension.
        forward_seed: Seed for NormalModesProvider.
        data_seed: Seed for noise draw.

    Returns:
        Flat dict with N_d/seed keys, all gap metrics, and DLI diagnostics.
    """
    prob = build_problem(N_d, forward_seed, data_seed)
    M, D, P = prob["M"], prob["D"], prob["P"]
    G, T = prob["G"], prob["T"]
    d_tilde = prob["d_tilde"]
    model_prior_support = prob["model_prior_support"]
    data_conf_ellipsoid = prob["data_conf_ellipsoid"]
    model_radius = prob["model_radius"]
    r_data_conf = prob["r_data_conf"]
    p_bar = float(prob["p_bar"][0])

    dli = compute_dli_interval(
        M, D, P, G, T, d_tilde,
        model_prior_support, data_conf_ellipsoid,
    )
    bg = compute_bg_interval(
        M, D, P, G, T, d_tilde,
        model_prior_support, data_conf_ellipsoid,
        model_radius, r_data_conf,
    )

    metrics = compute_gap_metrics(dli, bg)

    row: dict = {
        "N_d": N_d,
        "forward_seed": forward_seed,
        "data_seed": data_seed,
        "p_bar": p_bar,
    }
    row.update(metrics)
    row["dli_iters"] = dli["total_iters"]
    row["dli_converged_pos"] = int(dli["converged"][0])
    row["dli_converged_neg"] = int(dli["converged"][1])
    row["dli_elapsed_s"] = dli["elapsed_s"]
    return row


def run_sweep(verbose: bool = True) -> list:
    """Run the full multi-seed sweep and return a list of per-case dicts.

    Iterates over all combinations of N_d in N_D_VALUES, forward seeds in
    FORWARD_SEEDS, and data seeds in DATA_SEEDS, calling evaluate_case for
    each.

    Args:
        verbose: If True, print a progress line per case.

    Returns:
        List of row dicts (one per case) in N_d / forward_seed / data_seed
        order.
    """
    _vprint = print if verbose else (lambda *a, **k: None)

    total = len(N_D_VALUES) * len(FORWARD_SEEDS) * len(DATA_SEEDS)
    _vprint(
        f"  Sweep: {len(N_D_VALUES)} N_d values × "
        f"{len(FORWARD_SEEDS)} forward seeds × "
        f"{len(DATA_SEEDS)} data seeds = {total} cases"
    )

    rows: list = []
    case_idx = 0
    t_sweep = time.perf_counter()

    for N_d in N_D_VALUES:
        for forward_seed in FORWARD_SEEDS:
            for data_seed in DATA_SEEDS:
                case_idx += 1
                t_case = time.perf_counter()
                row = evaluate_case(N_d, forward_seed, data_seed)
                t_elapsed = time.perf_counter() - t_case
                _vprint(
                    f"  [{case_idx:2d}/{total}]"
                    f" N_d={N_d:3d} fseed={forward_seed:3d}"
                    f" dseed={data_seed:3d}"
                    f"  ratio={row['width_ratio']:6.3f}"
                    f"  cv={row['containment_violation']:.2e}"
                    f"  t={t_elapsed:.1f}s"
                )
                rows.append(row)

    _vprint(f"  Sweep done in {time.perf_counter() - t_sweep:.1f}s")
    return rows


def _raw_csv_columns() -> list:
    """Column order for the raw per-case CSV."""
    return [
        "N_d", "forward_seed", "data_seed", "p_bar",
        "dli_lower", "dli_upper", "dli_width",
        "bg_lower", "bg_upper", "bg_width",
        "width_ratio", "excess_width",
        "upper_slack", "lower_slack", "hausdorff_gap",
        "containment_violation",
        "dli_iters", "dli_converged_pos", "dli_converged_neg",
        "dli_elapsed_s",
    ]


def _summary_csv_columns() -> list:
    """Column order for the aggregated summary CSV."""
    return [
        "N_d", "n_cases",
        "width_ratio_mean", "width_ratio_std", "width_ratio_median",
        "excess_width_mean", "excess_width_std",
        "dli_width_mean", "bg_width_mean",
        "hausdorff_gap_mean", "hausdorff_gap_max",
        "containment_violation_max",
    ]


def write_raw_csv(rows: list, path: Path) -> None:
    """Write per-case rows to *path* as a CSV file (stdlib csv module)."""
    columns = _raw_csv_columns()
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_results_by_nd(rows: list) -> list:
    """Aggregate per-case rows by N_d and return a list of summary dicts.

    For each N_d value, computes mean / std / median of key metrics across
    all (forward_seed, data_seed) combinations.

    Returns:
        List of summary dicts, one per unique N_d, sorted ascending.
    """
    nd_groups: dict = {}
    for row in rows:
        nd_groups.setdefault(row["N_d"], []).append(row)

    summary_rows = []
    for N_d in sorted(nd_groups):
        grp = nd_groups[N_d]
        n = len(grp)

        def _mean(key: str) -> float:
            return float(np.mean([r[key] for r in grp]))

        def _std(key: str) -> float:
            vals = [r[key] for r in grp]
            return float(np.std(vals, ddof=1)) if n > 1 else 0.0

        def _median(key: str) -> float:
            vals = sorted(r[key] for r in grp)
            mid = n // 2
            return float(
                vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])
            )

        def _max(key: str) -> float:
            return float(max(r[key] for r in grp))

        summary_rows.append({
            "N_d": N_d,
            "n_cases": n,
            "width_ratio_mean": _mean("width_ratio"),
            "width_ratio_std": _std("width_ratio"),
            "width_ratio_median": _median("width_ratio"),
            "excess_width_mean": _mean("excess_width"),
            "excess_width_std": _std("excess_width"),
            "dli_width_mean": _mean("dli_width"),
            "bg_width_mean": _mean("bg_width"),
            "hausdorff_gap_mean": _mean("hausdorff_gap"),
            "hausdorff_gap_max": _max("hausdorff_gap"),
            "containment_violation_max": _max("containment_violation"),
        })

    return summary_rows


def write_summary_csv(summary_rows: list, path: Path) -> None:
    """Write aggregated summary rows to *path* as a CSV file."""
    columns = _summary_csv_columns()
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


def run_phase3_validation(verbose: bool = True):
    """Run the full multi-seed sweep, validate results, and save CSV outputs.

    Assertions:
    1. Row count equals len(N_D_VALUES) * len(FORWARD_SEEDS) * len(DATA_SEEDS).
    2. All key gap metrics are finite for every row.
    3. Max containment_violation is at or below CONTAINMENT_TOL.
    4. Both CSV files are written to RESULTS_FOLDER.

    Returns:
        Tuple (rows, summary_rows).
    """
    _vprint = print if verbose else (lambda *a, **k: None)
    _vprint("--- Phase 3: Multi-Seed Sweep ---\n")

    rows = run_sweep(verbose=verbose)

    # Assertion 1: row count.
    expected = len(N_D_VALUES) * len(FORWARD_SEEDS) * len(DATA_SEEDS)
    assert len(rows) == expected, (
        f"Expected {expected} rows, got {len(rows)}"
    )
    _vprint(f"\n  Row count PASSED: {len(rows)} rows")

    # Assertion 2: metric finiteness.
    key_metrics = [
        "dli_lower", "dli_upper", "dli_width",
        "bg_lower", "bg_upper", "bg_width",
        "width_ratio", "excess_width",
        "upper_slack", "lower_slack", "hausdorff_gap",
        "containment_violation",
    ]
    for i, row in enumerate(rows):
        for key in key_metrics:
            val = row[key]
            assert np.isfinite(val), (
                f"Row {i} (N_d={row['N_d']}, fseed={row['forward_seed']}, "
                f"dseed={row['data_seed']}): '{key}'={val} not finite"
            )
    _vprint("  Metric finiteness PASSED")

    # Assertion 3: containment violation near zero.
    max_cv = max(r["containment_violation"] for r in rows)
    assert max_cv <= CONTAINMENT_TOL, (
        f"max containment_violation={max_cv:.3e} exceeds "
        f"tolerance {CONTAINMENT_TOL:.1e}"
    )
    _vprint(
        f"  Containment violation PASSED: max={max_cv:.3e} "
        f"<= {CONTAINMENT_TOL:.1e}"
    )

    # Write CSVs.
    results_dir = _results_dir()
    raw_path = results_dir / "bg_dli_raw.csv"
    summary_rows = summarize_results_by_nd(rows)
    summary_path = results_dir / "bg_dli_summary_by_nd.csv"

    write_raw_csv(rows, raw_path)
    write_summary_csv(summary_rows, summary_path)

    # Assertion 4: files exist.
    assert raw_path.exists(), f"Raw CSV not written: {raw_path}"
    assert summary_path.exists(), f"Summary CSV not written: {summary_path}"
    _vprint("\n  CSVs written:")
    _vprint(f"    {raw_path}")
    _vprint(f"    {summary_path}")

    # Print summary table.
    _vprint("\n  Summary by N_d:")
    _vprint(
        f"  {'N_d':>5}  {'n':>4}  "
        f"{'ratio_mean':>10}  {'ratio_med':>9}  "
        f"{'dli_w_mean':>10}  {'bg_w_mean':>9}  "
        f"{'cv_max':>8}"
    )
    _vprint("  " + "-" * 68)
    for s in summary_rows:
        _vprint(
            f"  {s['N_d']:>5}  {s['n_cases']:>4}  "
            f"{s['width_ratio_mean']:>10.4f}  "
            f"{s['width_ratio_median']:>9.4f}  "
            f"{s['dli_width_mean']:>10.5f}  {s['bg_width_mean']:>9.5f}  "
            f"{s['containment_violation_max']:>8.2e}"
        )

    _vprint("\n--- Phase 3 PASSED ---")
    return rows, summary_rows


# ---------------------------------------------------------------------------
# Phase 4: Figures, summary printing, and validation
# ---------------------------------------------------------------------------

def _figures_dir() -> Path:
    """Return the figures directory, creating it if needed."""
    base = Path(__file__).parent / FIGURES_FOLDER
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_width_ratio_vs_nd(summary_rows: list, figures_dir: Path):
    """Plot BG/DLI width ratio vs N_d with error bars.

    Args:
        summary_rows: List of summary dicts from summarize_results_by_nd.
        figures_dir: Directory to save figures.

    Returns:
        matplotlib.figure.Figure object.
    """
    nd_vals = [r["N_d"] for r in summary_rows]
    ratio_mean = [r["width_ratio_mean"] for r in summary_rows]
    ratio_std = [r["width_ratio_std"] for r in summary_rows]
    ratio_median = [r["width_ratio_median"] for r in summary_rows]

    fig, ax = plt.subplots()
    ax.errorbar(
        nd_vals, ratio_mean, yerr=ratio_std,
        fmt="o-", capsize=4, label="mean ± std",
        color="steelblue",
    )
    ax.plot(
        nd_vals, ratio_median,
        "s", color="darkorange", label="median", zorder=5,
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8,
               label="ratio = 1 (DLI = BG)")

    ax.set_xlabel("$N_d$")
    ax.set_ylabel(r"Width ratio $w_{\mathrm{BG}} / w_{\mathrm{DLI}}$")
    ax.set_title("BG vs DLI Width Ratio")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    png_path = figures_dir / "width_ratio_vs_nd.png"
    pdf_path = figures_dir / "width_ratio_vs_nd.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig


def plot_absolute_widths_vs_nd(summary_rows: list, figures_dir: Path):
    """Plot absolute DLI and BG mean widths vs N_d.

    Args:
        summary_rows: List of summary dicts from summarize_results_by_nd.
        figures_dir: Directory to save figures.

    Returns:
        matplotlib.figure.Figure object.
    """
    nd_vals = [r["N_d"] for r in summary_rows]
    dli_widths = [r["dli_width_mean"] for r in summary_rows]
    bg_widths = [r["bg_width_mean"] for r in summary_rows]

    fig, ax = plt.subplots()
    ax.plot(nd_vals, dli_widths, "o-", color="steelblue", label="DLI")
    ax.plot(nd_vals, bg_widths, "s-", color="darkorange", label="BG")

    ax.set_xlabel("$N_d$")
    ax.set_ylabel("Interval width")
    ax.set_title("Admissible Interval Widths vs $N_d$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    png_path = figures_dir / "widths_vs_nd.png"
    pdf_path = figures_dir / "widths_vs_nd.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig


def print_sweep_summary(summary_rows: list) -> None:
    """Print a formatted textual summary of sweep results by N_d."""
    sep = "  " + "\u2500" * 58
    print("  BG vs DLI 1D Sweep Summary")
    print(sep)
    print(
        f"  {'N_d':>5}  {'ratio_mean':>10}  {'ratio_median':>12}"
        f"  {'dli_w':>7}  {'bg_w':>7}  {'excess_w':>9}"
    )
    print(sep)
    for r in summary_rows:
        excess = r["bg_width_mean"] - r["dli_width_mean"]
        print(
            f"  {r['N_d']:>5}  {r['width_ratio_mean']:>10.4f}"
            f"  {r['width_ratio_median']:>12.4f}"
            f"  {r['dli_width_mean']:>7.4f}  {r['bg_width_mean']:>7.4f}"
            f"  {excess:>9.4f}"
        )
    print(sep)

    # Autogenerated trend line.
    ratios = [r["width_ratio_mean"] for r in summary_rows]
    if all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1)):
        trend = "width_ratio increases with N_d (DLI tightens faster than BG)."
    elif all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
        trend = "width_ratio decreases with N_d."
    else:
        trend = "width_ratio shows non-monotone behaviour with N_d."
    print(f"  Trend: {trend}")


def run_phase4_validation(verbose: bool = True) -> None:
    """Read existing summary CSV, produce plots, and print textual summary.

    Does NOT re-run the sweep. Reads bg_dli_summary_by_nd.csv from
    _results_dir(), generates two figures, and asserts PNG files exist.
    """
    _vprint = print if verbose else (lambda *a, **k: None)
    _vprint("--- Phase 4: Plots and Summary ---\n")

    # 1. Read summary CSV.
    summary_csv = _results_dir() / "bg_dli_summary_by_nd.csv"
    summary_rows: list = []
    with open(summary_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            summary_rows.append({
                "N_d": int(row["N_d"]),
                "n_cases": int(row["n_cases"]),
                "width_ratio_mean": float(row["width_ratio_mean"]),
                "width_ratio_std": float(row["width_ratio_std"]),
                "width_ratio_median": float(row["width_ratio_median"]),
                "excess_width_mean": float(row["excess_width_mean"]),
                "excess_width_std": float(row["excess_width_std"]),
                "dli_width_mean": float(row["dli_width_mean"]),
                "bg_width_mean": float(row["bg_width_mean"]),
                "hausdorff_gap_mean": float(row["hausdorff_gap_mean"]),
                "hausdorff_gap_max": float(row["hausdorff_gap_max"]),
                "containment_violation_max": float(
                    row["containment_violation_max"]
                ),
            })

    # 2. Print textual summary.
    print_sweep_summary(summary_rows)
    _vprint()

    # 3 & 4. Generate figures.
    figs_dir = _figures_dir()
    plot_width_ratio_vs_nd(summary_rows, figs_dir)
    plot_absolute_widths_vs_nd(summary_rows, figs_dir)
    plt.close("all")

    # 5. Assert figures exist.
    png_ratio = figs_dir / "width_ratio_vs_nd.png"
    png_widths = figs_dir / "widths_vs_nd.png"
    assert figs_dir.exists(), f"Figures directory not created: {figs_dir}"
    assert png_ratio.exists(), f"Figure not written: {png_ratio}"
    assert png_widths.exists(), f"Figure not written: {png_widths}"

    _vprint("  Figures written:")
    _vprint(f"    {png_ratio}")
    _vprint(f"    {png_widths}")
    _vprint()
    _vprint("--- Phase 4 PASSED ---")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Phase 1, Phase 2, Phase 3, and Phase 4 validation."""
    print("=" * 62)
    print("bg_dli_1d_sweep.py  --  Phases 1-4")
    print("=" * 62)
    print()
    print(f"N_D_VALUES      : {N_D_VALUES}")
    print(f"FORWARD_SEEDS   : {FORWARD_SEEDS}")
    print(f"DATA_SEEDS      : {DATA_SEEDS}")
    print(f"NOISE_FRACTION  : {NOISE_FRACTION}")
    print(f"CONFIDENCE_LEVEL: {CONFIDENCE_LEVEL}")
    print()

    run_phase1_validation(verbose=True)

    print()
    run_phase2_validation(verbose=True)

    print()
    run_phase3_validation(verbose=True)

    print()
    run_phase4_validation(verbose=True)


if __name__ == "__main__":
    main()
