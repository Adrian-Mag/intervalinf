"""
synthetic_data.py
=================

Synthetic data utilities for the prior/posterior tuner.

In **synthetic mode** the GUI replaces the real observation vector
``d_st = split[block].data_vector`` with

.. math::

    d_{\\rm synth} = G_{st} \\, m_{\\rm true} + s \\cdot \\eta,
    \\qquad \\eta \\sim \\mathcal{N}(0,\\ C_D)

where ``s = config.noise_scale``.  The default true model is a
**least-norm solution**

.. math::

    m_{\\rm LN} = G^*(G G^*)^{-1} \\tilde{d}

which is guaranteed to satisfy :math:`G m_{\\rm LN} = \\tilde{d}` exactly,
or a smooth Bessel-Sobolev target-fit variant

.. math::

    m_\\lambda = C_{\\rm ref}G^*(GC_{\\rm ref}G^* + \\lambda C_D)^{-1}
    \\tilde{d}

where :math:`\\lambda` is chosen to keep the RMS standardized residual below
a requested threshold.  Unlike the Bayesian posterior mean, the true-model
construction uses a fixed reference covariance and is independent of the
tunable prior amplitudes.

An optional null-space perturbation is available:

.. math::

    m_{\\rm null} = \\xi - E\\![\\xi \\mid G\\xi],
    \\qquad m_{\\rm true} = m_{\\rm LN} + \\alpha \\cdot m_{\\rm null}

so ``G m_{\\rm true} \\approx G m_{\\rm LN} = \\tilde{d}`` (the perturbation
is approximately invisible to the data).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration & state
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SynthConfig:
    """User-controllable knobs for synthetic data generation."""
    noise_scale: float = 1.0
    null_amp:    float = 0.0
    seed:        int   = 0


@dataclasses.dataclass
class SynthLeastNorm:
    """Result of the least-norm true-model solve.

    The L2 variant satisfies ``G m_LN = d_real`` up to floating-point error.
    The smooth target-fit variant uses a fixed reference covariance and stores
    the achieved standardized residual in ``fit_chi_rms``.
    """
    m_ln:         object        # abstract HilbertSpace element (opaque)
    d_clean_base: np.ndarray    # G m_LN — equals d_real up to rounding
    # Dense radial grids
    r_vp:     np.ndarray
    r_vs_IC:  np.ndarray
    r_vs_M:   np.ndarray
    r_rho:    np.ndarray
    true_vp:     np.ndarray
    true_vs_IC:  np.ndarray
    true_vs_M:   np.ndarray
    true_rho:    np.ndarray
    sigma_1_true: float
    fit_mode: str = "l2"
    fit_lambda: Optional[float] = None
    fit_chi_rms: Optional[float] = None
    fit_chi2_per_datum: Optional[float] = None
    fit_target_chi_rms: Optional[float] = None
    fit_target_met: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class _TargetFitResult:
    """Private data-space result for target-fit smooth solves."""
    alpha: np.ndarray
    lambda_value: float
    chi_rms: float
    chi2_per_datum: float
    target_met: bool


def _standardized_rms(
    prediction: np.ndarray,
    data: np.ndarray,
    data_std: np.ndarray,
) -> tuple[float, float]:
    """Return RMS standardized residual and chi^2 per datum."""
    safe_std = np.where(data_std > 0.0, data_std, np.inf)
    residual = (prediction - data) / safe_std
    chi2_per_datum = float(np.mean(residual**2))
    return float(np.sqrt(chi2_per_datum)), chi2_per_datum


def _solve_target_smooth_alpha(
    gram: np.ndarray,
    data: np.ndarray,
    data_std: np.ndarray,
    target_chi_rms: float,
    *,
    lambdas: Optional[np.ndarray] = None,
) -> _TargetFitResult:
    """Choose the smoothest regularized data-space solve meeting a fit target.

    Solves

    .. math:: (G C G^* + \\lambda C_D)\\alpha = d

    over a logarithmic grid of positive ``lambda`` values and selects the
    largest ``lambda`` whose RMS standardized residual is at most
    ``target_chi_rms``. If the target is unreachable in the finite smooth
    reference space, the best-fitting candidate is returned with
    ``target_met=False``.
    """
    if target_chi_rms <= 0.0:
        raise ValueError("target_chi_rms must be positive")

    data = np.asarray(data, dtype=float)
    data_std = np.asarray(data_std, dtype=float)
    if data.shape != data_std.shape:
        raise ValueError("data_std must have the same shape as data")

    if lambdas is None:
        lambdas = np.logspace(-14.0, 8.0, 111)
    else:
        lambdas = np.asarray(lambdas, dtype=float)
        lambdas = lambdas[lambdas > 0.0]
    if lambdas.size == 0:
        raise ValueError("lambdas must contain at least one positive value")

    positive_std = data_std[data_std > 0.0]
    fallback_var = (
        float(np.min(positive_std) ** 2) if positive_std.size else 1.0
    )
    data_cov_diag = np.where(data_std > 0.0, data_std**2, fallback_var)
    data_cov = np.diag(data_cov_diag)

    candidates: list[_TargetFitResult] = []
    for lambda_value in lambdas:
        try:
            alpha = np.linalg.solve(gram + lambda_value * data_cov, data)
        except np.linalg.LinAlgError:
            continue
        prediction = gram @ alpha
        chi_rms, chi2_per_datum = _standardized_rms(prediction, data, data_std)
        candidates.append(_TargetFitResult(
            alpha=alpha,
            lambda_value=float(lambda_value),
            chi_rms=chi_rms,
            chi2_per_datum=chi2_per_datum,
            target_met=chi_rms <= target_chi_rms,
        ))

    if not candidates:
        raise np.linalg.LinAlgError("all target-fit smooth solves failed")

    feasible = [candidate for candidate in candidates if candidate.target_met]
    if feasible:
        return max(feasible, key=lambda candidate: candidate.lambda_value)
    return min(candidates, key=lambda candidate: candidate.chi_rms)


@dataclasses.dataclass(frozen=True)
class SynthCore:
    """Expensive intermediate results: true model + clean prediction + noise draw.

    Cached by ``(block, null_amp, seed)``; changing only ``noise_scale``
    reuses this object and skips the forward-solve.
    """
    d_clean:  np.ndarray   # noiseless model prediction  G m_true
    eta:      np.ndarray   # unit noise draw from C_D (before noise_scale)
    null_amp: float
    seed:     int

    r_vp:     np.ndarray
    r_vs_IC:  np.ndarray
    r_vs_M:   np.ndarray
    r_rho:    np.ndarray

    true_vp:     np.ndarray
    true_vs_IC:  np.ndarray
    true_vs_M:   np.ndarray
    true_rho:    np.ndarray

    sigma_1_true: float


@dataclasses.dataclass(frozen=True)
class SynthState:
    """Cached synthetic state for one (s, t) block."""
    d_synth:  np.ndarray   # noisy synthetic data
    d_clean:  np.ndarray   # noiseless model prediction G m_true
    config:   SynthConfig

    r_vp:     np.ndarray
    r_vs_IC:  np.ndarray
    r_vs_M:   np.ndarray
    r_rho:    np.ndarray

    true_vp:     np.ndarray
    true_vs_IC:  np.ndarray
    true_vs_M:   np.ndarray
    true_rho:    np.ndarray

    sigma_1_true: float


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _evaluate_model_on_grids(m, specs, n_grid: int) -> dict:
    """Evaluate a model element on dense radial grids."""
    R, ICB, CMB = specs.earth_radius_km, specs.icb_radius_km, specs.cmb_radius_km

    f_vp    = m[0][0]
    f_vs_IC = m[0][1][0]
    f_vs_M  = m[0][1][1]
    f_rho   = m[0][2]
    sigma_1 = float(m[1][1][0])

    r_vp    = np.linspace(0.0, R,   n_grid)
    r_vs_IC = np.linspace(0.0, ICB, n_grid)
    r_vs_M  = np.linspace(CMB, R,   n_grid)
    r_rho   = np.linspace(0.0, R,   n_grid)

    return dict(
        r_vp=r_vp, r_vs_IC=r_vs_IC, r_vs_M=r_vs_M, r_rho=r_rho,
        true_vp=np.asarray([f_vp(r)    for r in r_vp],    dtype=float),
        true_vs_IC=np.asarray([f_vs_IC(r) for r in r_vs_IC], dtype=float),
        true_vs_M=np.asarray([f_vs_M(r)  for r in r_vs_M],  dtype=float),
        true_rho=np.asarray([f_rho(r)   for r in r_rho],   dtype=float),
        sigma_1_true=sigma_1,
    )


def compute_least_norm(
    block,
    G_st,
    d_real: np.ndarray,
    specs,
    *,
    n_grid: int = 300,
    C_ref=None,
    data_std: Optional[np.ndarray] = None,
    target_chi_rms: Optional[float] = None,
) -> SynthLeastNorm:
    """Compute a minimum-norm true model for ``d_real``.

    The default path is the L2 minimum-norm solution
    ``m_LN = G*(GG*)^{-1}d``. It is exact but can be oscillatory.

    When ``C_ref`` is supplied, the smooth data-space Gram matrix
    ``G C_ref G*`` is assembled. If ``data_std`` and ``target_chi_rms`` are
    also supplied, the method solves
    ``(G C_ref G* + lambda C_D) alpha = d`` over a logarithmic grid and picks
    the largest ``lambda`` whose RMS standardized residual is below the target.
    This gives a smooth synthetic true model without demanding an unstable
    exact fit.

    If only ``C_ref`` is supplied, the old exact smooth solve is retained for
    backwards compatibility, but it can be severely ill-conditioned when the
    finite smooth reference covariance cannot span all data-side directions.
    """
    d_real = np.asarray(d_real, dtype=float)
    n_d = len(d_real)
    fit_mode = "l2"
    fit_lambda = None
    fit_target_met = None
    fit_chi_rms = None
    fit_chi2_per_datum = None

    if C_ref is None:
        # ── L2 minimum-norm: GG* ─────────────────────────────────────────
        gram = np.empty((n_d, n_d), dtype=float)
        for j in range(n_d):
            e_j = np.zeros(n_d)
            e_j[j] = 1.0
            adj_e_j = G_st.adjoint(e_j)
            gram[:, j] = np.asarray(G_st(adj_e_j), dtype=float)

        alpha = np.linalg.solve(gram, d_real)
        m_ln = G_st.adjoint(alpha)
    else:
        # ── C_ref-weighted minimum-norm: G C_ref G* ──────────────────────
        gram = np.empty((n_d, n_d), dtype=float)
        for j in range(n_d):
            e_j = np.zeros(n_d)
            e_j[j] = 1.0
            adj_e_j = G_st.adjoint(e_j)   # model-space element
            C_adj = C_ref(adj_e_j)        # apply reference covariance
            gram[:, j] = np.asarray(G_st(C_adj), dtype=float)

        if target_chi_rms is not None:
            if data_std is None:
                raise ValueError(
                    "data_std is required when target_chi_rms is provided."
                )
            target_result = _solve_target_smooth_alpha(
                gram, d_real, data_std, target_chi_rms,
            )
            alpha = target_result.alpha
            fit_mode = "smooth_target"
            fit_lambda = target_result.lambda_value
            fit_chi_rms = target_result.chi_rms
            fit_chi2_per_datum = target_result.chi2_per_datum
            fit_target_met = target_result.target_met
        else:
            alpha = np.linalg.solve(gram, d_real)
            fit_mode = "smooth_exact"
        m_ln = C_ref(G_st.adjoint(alpha))  # smooth minimum-norm solution

    # Verify G m against d_real.
    if fit_mode == "smooth_target":
        _tag = "[compute_target_smooth_ln]"
    elif fit_mode == "smooth_exact":
        _tag = "[compute_smooth_ln]"
    else:
        _tag = "[compute_least_norm]"
    d_clean_base = np.asarray(G_st(m_ln), dtype=float)
    d_norm = np.linalg.norm(d_real)
    residual_rel = np.linalg.norm(d_clean_base - d_real) / max(d_norm, 1e-30)

    if data_std is not None:
        fit_chi_rms, fit_chi2_per_datum = _standardized_rms(
            d_clean_base, d_real, np.asarray(data_std, dtype=float)
        )
        if target_chi_rms is not None:
            fit_target_met = fit_chi_rms <= target_chi_rms

    if target_chi_rms is not None and fit_chi_rms is not None:
        if fit_chi_rms > target_chi_rms:
            import warnings
            warnings.warn(
                f"{_tag} block ({block.s},{block.t}): "
                f"RMS standardized residual = {fit_chi_rms:.2f} "
                f"> target {target_chi_rms:.2f}. "
                "Smooth reference space may be too small for this data fit."
            )
        else:
            print(
                f"{_tag} block ({block.s},{block.t}): "
                f"RMS standardized residual = {fit_chi_rms:.2f} "
                f"(target {target_chi_rms:.2f}), lambda={fit_lambda:.2e}"
            )
    elif residual_rel > 1e-6:
        import warnings
        warnings.warn(
            f"{_tag} block ({block.s},{block.t}): "
            f"||G m - d|| / ||d|| = {residual_rel:.2e}. "
            "Forward operator may be rank-deficient."
        )
    else:
        print(
            f"{_tag} block ({block.s},{block.t}): "
            f"||G m - d|| / ||d|| = {residual_rel:.2e}  ✓"
        )

    grids = _evaluate_model_on_grids(m_ln, specs, n_grid)
    return SynthLeastNorm(
        m_ln=m_ln,
        d_clean_base=d_clean_base,
        fit_mode=fit_mode,
        fit_lambda=fit_lambda,
        fit_chi_rms=fit_chi_rms,
        fit_chi2_per_datum=fit_chi2_per_datum,
        fit_target_chi_rms=target_chi_rms,
        fit_target_met=fit_target_met,
        **grids,
    )


def compute_synth_core(
    block,
    G_st,
    C_D_st,
    prior_st,
    d_real: np.ndarray,
    specs,
    *,
    null_amp: float = 0.0,
    seed: int = 0,
    n_grid: int = 300,
    posterior_mean: Optional[SynthLeastNorm] = None,
) -> SynthCore:
    """Compute the intermediate synthetic results for one ``(null_amp, seed)`` pair.

    The result is keyed by ``(block, null_amp, seed)``; pass it to
    :func:`synth_state_from_core` to vary ``noise_scale`` instantly.

    Parameters
    ----------
    posterior_mean :
        Pass a cached :class:`SynthLeastNorm` to skip the least-norm
        computation.  If *None*, it is computed internally.
    prior_st :
        Required when ``null_amp > 0`` (needed for null-space sampling).
        May be *None* when ``null_amp == 0``.
    """
    from full_spectrum_utils import solve_block

    # ── Step 1: least-norm solution (reuse cached result if available) ────
    if posterior_mean is None:
        posterior_mean = compute_least_norm(
            block, G_st, d_real, specs, n_grid=n_grid
        )

    m_post = posterior_mean.m_ln

    # ── Step 2: optional null-space perturbation ───────────────────────────
    if null_amp > 0.0:
        if prior_st is None:
            raise ValueError(
                "prior_st is required for null_amp > 0."
            )
        if not prior_st.sample_set:
            raise ValueError(
                "Null-space perturbation (null_amp > 0) requires a sampleable "
                "prior, but the current prior has no sampling method.  This "
                "happens when the prior is built with only a covariance "
                "operator (no covariance factor).  Set null_amp = 0 or "
                "rebuild the prior with a covariance factor."
            )
        np.random.seed(int(seed))
        xi = prior_st.sample()

        d_xi = np.asarray(G_st(xi), dtype=float)
        posterior_xi = solve_block(block.s, block.t, G_st, C_D_st, prior_st, d_xi)
        m_xi_post = posterior_xi.expectation

        domain = prior_st.domain
        m_null = domain.subtract(xi, m_xi_post)
        m_true = domain.add(m_post, domain.multiply(float(null_amp), m_null))

        # Grid evaluations differ from m_post when null_amp > 0.
        np.random.seed(int(seed) + 1)
        eta     = np.asarray(C_D_st.sample(), dtype=float)
        d_clean = np.asarray(G_st(m_true), dtype=float)
        grids   = _evaluate_model_on_grids(m_true, specs, n_grid)
    else:
        # null_amp = 0: m_true = m_post — reuse everything from posterior_mean.
        np.random.seed(int(seed) + 1)
        eta     = np.asarray(C_D_st.sample(), dtype=float)
        d_clean = posterior_mean.d_clean_base
        grids   = dict(
            r_vp=posterior_mean.r_vp,         r_vs_IC=posterior_mean.r_vs_IC,
            r_vs_M=posterior_mean.r_vs_M,      r_rho=posterior_mean.r_rho,
            true_vp=posterior_mean.true_vp,    true_vs_IC=posterior_mean.true_vs_IC,
            true_vs_M=posterior_mean.true_vs_M, true_rho=posterior_mean.true_rho,
            sigma_1_true=posterior_mean.sigma_1_true,
        )

    return SynthCore(
        d_clean=d_clean,
        eta=eta,
        null_amp=float(null_amp),
        seed=int(seed),
        **grids,
    )


def synth_state_from_core(core: SynthCore, config: SynthConfig) -> SynthState:
    """Build a :class:`SynthState` from a cached :class:`SynthCore`.

    This is the **cheap** step: only ``d_synth = d_clean + noise_scale * eta``
    needs to be recomputed.  Safe to call on the GUI thread.
    """
    d_synth = core.d_clean + float(config.noise_scale) * core.eta
    return SynthState(
        d_synth=d_synth,
        d_clean=core.d_clean,
        config=config,
        r_vp=core.r_vp,
        r_vs_IC=core.r_vs_IC,
        r_vs_M=core.r_vs_M,
        r_rho=core.r_rho,
        true_vp=core.true_vp,
        true_vs_IC=core.true_vs_IC,
        true_vs_M=core.true_vs_M,
        true_rho=core.true_rho,
        sigma_1_true=core.sigma_1_true,
    )


def generate_synth_state(
    block,
    G_st,
    C_D_st,
    prior_st,
    d_real: np.ndarray,
    specs,
    *,
    config: SynthConfig,
    n_grid: int = 300,
) -> SynthState:
    """Build a synthetic state in one call (convenience wrapper).

    Internally calls :func:`compute_synth_core` then
    :func:`synth_state_from_core`.  When iterating over ``noise_scale``
    values only, prefer calling those two functions separately to avoid
    redundant forward-solves.
    """
    core = compute_synth_core(
        block, G_st, C_D_st, prior_st, d_real, specs,
        null_amp=config.null_amp,
        seed=config.seed,
        n_grid=n_grid,
    )
    return synth_state_from_core(core, config)
