"""Phase 5 — function-space credible-set tests on intervalinf.

Exercises ``GaussianMeasure.credible_set`` from pygeoinf with basis-free
``Lebesgue`` spaces and the analytic Laplacian / Bessel-Sobolev spectra
supplied by intervalinf operators.

These tests target the new function-space modes:

* ``geometry="ambient_ball"`` — spectral and sampling radius paths.
* ``geometry="weakened_ellipsoid"`` — Lanczos backend for the fractional
  gauge $C^{-\theta/2}$.

Reference plan:
``pygeoinf/docs/agent-docs/active-plans/function-space-hardening-plan.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.functional_calculus import apply_operator_function

from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.domain import IntervalDomain
from intervalinf.operators.bessel import BesselSobolevInverse
from intervalinf.operators.laplacian import InverseLaplacian, Laplacian
from intervalinf.sampling.kl_sampler import KLSampler
from intervalinf.spaces.lebesgue import Lebesgue


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_inverse_laplacian_measure(
    *,
    n_modes: int = 200,
    dofs: int = 200,
    seed: int = 0,
):
    r"""Build a basis-free Lebesgue measure with covariance $(-\Delta)^{-1}$.

    Dirichlet eigenvalues of $(-\Delta)^{-1}$ on $[0,1]$ are
    $\lambda_j = 1/((j+1)\pi)^2$, the Cauchy spectrum used throughout
    the plan.
    """
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(0, domain, basis=None)
    bc = BoundaryConditions.dirichlet()
    cov = InverseLaplacian(space, bc, method="spectral", dofs=dofs)
    sampler = KLSampler(
        cov, n_modes=n_modes, rng=np.random.default_rng(seed)
    )
    expectation = sampler.sample()
    measure = GaussianMeasure(
        covariance=cov, sample=sampler.sample, expectation=expectation
    )
    return measure, cov, space


def _spectrum_callable(cov):
    """Wrap a per-index ``get_eigenvalue`` into the first-k array form."""

    def spec(k: int) -> np.ndarray:
        return np.array(
            [cov.get_eigenvalue(j) for j in range(k)], dtype=float
        )

    return spec


# ---------------------------------------------------------------------------
# Ambient ball — spectral truncation convergence
# ---------------------------------------------------------------------------


def test_ambient_ball_spectral_convergence():
    r"""The spectral radius is Cauchy in the truncation level $N$.

    For $\lambda_j = 1/((j+1)\pi)^2$ the sum converges as
    $\zeta(2)/\pi^2 = 1/6$ so the 0.9-quantile of $\sum_j \lambda_j Z_j^2$
    asymptotes quickly. Successive truncations should agree to better
    than $1\%$.
    """
    measure, cov, _ = _make_inverse_laplacian_measure()
    spec = _spectrum_callable(cov)
    probability = 0.9

    radii = {}
    for N in (50, 200, 1000):
        ball = measure.credible_set(
            probability,
            geometry="ambient_ball",
            spectrum=spec,
            spectrum_size=N,
        )
        radii[N] = float(ball.radius)
        assert radii[N] > 0.0

    # Cauchy in N: successive increments < 1 % of the largest radius.
    r_max = max(radii.values())
    assert abs(radii[200] - radii[50]) < 1.0e-2 * r_max
    assert abs(radii[1000] - radii[200]) < 1.0e-2 * r_max


# ---------------------------------------------------------------------------
# Ambient ball — sampling vs spectral
# ---------------------------------------------------------------------------


def test_ambient_ball_spectral_vs_sampling():
    """Sampling-based radius matches the spectral radius within MC tolerance."""
    measure, cov, _ = _make_inverse_laplacian_measure(n_modes=200, seed=1)
    spec = _spectrum_callable(cov)
    probability = 0.9

    ball_spectral = measure.credible_set(
        probability,
        geometry="ambient_ball",
        spectrum=spec,
        spectrum_size=200,
    )
    ball_sampling = measure.credible_set(
        probability,
        geometry="ambient_ball",
        radius_method="sampling",
        n_samples=500,
        rng=np.random.default_rng(42),
    )

    # 15 % relative tolerance: the KL sampler uses a finite-mode truncation
    # which adds a small bias on top of Monte Carlo noise.
    # n_samples=500 is sufficient (MC noise ~1.3 % for this distribution).
    assert_allclose(
        ball_sampling.radius, ball_spectral.radius, rtol=1.5e-1
    )


# ---------------------------------------------------------------------------
# Weakened ellipsoid — Lanczos backend vs analytical eigenfunction action
# ---------------------------------------------------------------------------


def test_weakened_ellipsoid_lanczos_vs_spectral():
    r"""Lanczos gauge $C^{-\theta}$ on an eigenfunction matches $\lambda^{-\theta}$.

    The radius depends only on the eigenvalue array, which is shared
    across backends, so this test instead checks that the Lanczos
    fractional-apply (used for the gauge action) reproduces the
    analytic action on an eigenpair to high accuracy.
    """
    measure, cov, space = _make_inverse_laplacian_measure(
        n_modes=200, seed=2
    )
    spec = _spectrum_callable(cov)
    theta = 0.5
    probability = 0.9
    n_lanczos = 50

    ellipsoid = measure.credible_set(
        probability,
        geometry="weakened_ellipsoid",
        theta=theta,
        spectrum=spec,
        spectrum_size=200,
        fractional_apply="lanczos",
        lanczos_size_estimate=n_lanczos,
    )
    assert ellipsoid.radius > 0.0

    # Lanczos action on the leading eigenfunction should give
    # lambda_0^{-theta} * f_0 since C f_0 = lambda_0 f_0.
    f0 = cov.get_eigenfunction(0)
    lambda0 = cov.get_eigenvalue(0)

    # C^{-theta} f_0  (matches the operator returned by the planner)
    g = apply_operator_function(
        cov, f0, lambda x: np.power(x, -theta), n_lanczos, method="fixed"
    )

    # Project onto f_0 via the L2 inner product.
    coeff = space.inner_product(g, f0) / space.inner_product(f0, f0)
    assert_allclose(coeff, lambda0 ** (-theta), rtol=5.0e-3)


# ---------------------------------------------------------------------------
# Weakened ellipsoid — Sobolev (Bessel) covariance
# ---------------------------------------------------------------------------


def test_sobolev_weakened_ellipsoid():
    """The same machinery applies to Bessel-potential covariances on $L^2$."""
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(0, domain, basis=None)
    bc = BoundaryConditions.dirichlet()
    laplacian = Laplacian(space, bc, method="spectral", dofs=200)
    # Bessel potential (k^2 I - \Delta)^{-s/2} with k=1, s=2.
    cov = BesselSobolevInverse(
        space, space, k=1.0, s=2.0, L=laplacian, dofs=200
    )
    sampler = KLSampler(cov, n_modes=200, rng=np.random.default_rng(3))
    expectation = sampler.sample()
    measure = GaussianMeasure(
        covariance=cov, sample=sampler.sample, expectation=expectation
    )

    spec = _spectrum_callable(cov)
    ellipsoid = measure.credible_set(
        0.9,
        geometry="weakened_ellipsoid",
        theta=0.5,
        spectrum=spec,
        spectrum_size=200,
        fractional_apply="lanczos",
        lanczos_size_estimate=50,
    )
    assert ellipsoid.radius > 0.0
    assert np.isfinite(ellipsoid.radius)


# ---------------------------------------------------------------------------
# Error path — no spectrum and no sampling
# ---------------------------------------------------------------------------


def test_no_spectrum_no_sampling_raises():
    """Without a spectrum and without sampling, the call must fail loudly."""
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(0, domain, basis=None)
    bc = BoundaryConditions.dirichlet()
    cov = InverseLaplacian(space, bc, method="spectral", dofs=200)
    # No ``sample`` callable -> non-sampling measure.
    measure = GaussianMeasure(covariance=cov)

    with pytest.raises(ValueError):
        measure.credible_set(0.9, geometry="ambient_ball")


# ---------------------------------------------------------------------------
# Trace-borderline warning for theta near 1
# ---------------------------------------------------------------------------


def test_cm_warning():
    r"""Near the Cameron-Martin boundary the truncation warns about trace.

    With $\lambda_j = j^{-2}$ and $\theta$ close to 1, the series
    $\sum_j \lambda_j^{1-\theta}$ is at the borderline of summability;
    the implementation emits a ``UserWarning`` for truncations that fall
    in this regime.
    """
    space = EuclideanSpace(20)
    eigenvalues = np.array(
        [1.0 / ((j + 1) ** 2) for j in range(space.dim)], dtype=float
    )
    measure = GaussianMeasure.from_standard_deviation(space, 1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measure.credible_set(
            0.9,
            geometry="weakened_ellipsoid",
            theta=0.99,
            spectrum=eigenvalues,
        )

    assert any(
        issubclass(w.category, UserWarning) for w in caught
    ), "Expected a UserWarning for near-Cameron-Martin truncation."
