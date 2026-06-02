"""Tests for weighted (r²) Bessel-Sobolev covariances on the radial Laplacian.

These verify that ``BesselSobolev``/``BesselSobolevInverse`` are correct and
self-adjoint with respect to the weighted inner product

    ⟨f, g⟩_w = ∫ f(r) g(r) w(r) dr,   w(r) = r²,

when built on a ``RadialLaplacian`` whose eigenfunctions are w-orthonormal.
This is the inner product / operator pairing required to use the operator as a
covariance in spherical coordinates.

The key implementation facts under test:

1. The slow-path spectral projection uses the *space* inner product (weighted),
   so the functional calculus A = Σ_k g(λ_k) ⟨φ_k, ·⟩_w φ_k reproduces g(L)
   exactly on the (w-orthonormal) radial eigenfunctions.
2. The flat DST/DCT fast path is disabled whenever the domain carries a weight,
   because those transforms compute *unweighted* coefficients.
"""

import numpy as np
from numpy.testing import assert_allclose

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.spaces.weighted_lebesgue import WeightedLebesgue
from intervalinf.operators import (
    BesselSobolev,
    BesselSobolevInverse,
    RadialLaplacian,
    Laplacian,
)


def _r2(r):
    return np.asarray(r) ** 2


def _make_radial_setup(n=24, k=1.5, s=2.0):
    """Weighted radial space (0,1) + RadialLaplacian + BesselSobolevInverse."""
    domain = IntervalDomain(0.0, 1.0)
    bc = BoundaryConditions.dirichlet()
    integration = IntegrationConfig(method="simpson", n_points=4000)
    space = WeightedLebesgue(
        n, domain, _r2, integration_config=integration
    )
    L = RadialLaplacian(
        space, bc, 1.0, method="spectral", dofs=n,
        integration_config=integration,
    )
    A = BesselSobolevInverse(
        space, space, k, s, L, dofs=n, n_samples=512,
        integration_config=integration,
    )
    return space, L, A, k, s


# ---------------------------------------------------------------------------
# Core correctness: functional calculus on the radial eigenbasis
# ---------------------------------------------------------------------------

def test_radial_bessel_inverse_eigenfunction_scaling_weighted():
    """A φ_j ≈ (k² + λ_j)^{-s/2} φ_j for radial eigenfunctions under w=r².

    This passes only if the spectral projection uses the *weighted* inner
    product (so the eigenfunctions are correctly isolated). With the unweighted
    projection the radial eigenfunctions are not orthonormal and the scaling
    breaks.
    """
    space, L, A, k, s = _make_radial_setup()
    r = np.linspace(0.02, 0.98, 200)

    for j in (0, 1, 2):
        phi_j = L.get_eigenfunction(j)
        lam_j = L.get_eigenvalue(j)
        g_j = (k ** 2 + lam_j) ** (-s / 2.0)

        out = A(phi_j)
        assert_allclose(out(r), g_j * phi_j(r), rtol=1e-2, atol=1e-3)


def test_radial_dirichlet_fast_path_enabled():
    """A radial Dirichlet domain enables the radial fast path (and disables the
    generic flat fast path, since the domain is mass-weighted)."""
    _space, _L, A, _k, _s = _make_radial_setup()
    assert A._radial_dirichlet_fast is True
    assert A._can_use_fast_transforms is False


def test_radial_fast_matches_slow():
    """The radial DST fast path agrees with the weight-aware slow projection."""
    domain = IntervalDomain(0.0, 1.0)
    bc = BoundaryConditions.dirichlet()
    integration = IntegrationConfig(method="simpson", n_points=4000)
    n, k, s = 24, 1.5, 2.0
    space = WeightedLebesgue(n, domain, _r2, integration_config=integration)
    L = RadialLaplacian(space, bc, 1.0, method="spectral", dofs=n,
                        integration_config=integration)
    A_fast = BesselSobolevInverse(space, space, k, s, L, dofs=n, n_samples=512,
                                  integration_config=integration,
                                  use_fast_transforms=True)
    A_slow = BesselSobolevInverse(space, space, k, s, L, dofs=n, n_samples=512,
                                  integration_config=integration,
                                  use_fast_transforms=False)
    assert A_fast._radial_dirichlet_fast is True
    assert A_slow._radial_dirichlet_fast is False

    f = Function(space.function_domain,
                 evaluate_callable=lambda r: np.sin(np.pi * np.asarray(r)) + 0.3)
    r = np.linspace(0.02, 0.98, 200)
    assert_allclose(A_fast(f)(r), A_slow(f)(r), rtol=1e-2, atol=1e-3)


def test_radial_bessel_inverse_self_adjoint_weighted():
    """⟨A f, g⟩_w ≈ ⟨f, A g⟩_w for the weighted radial covariance."""
    space, L, A, k, s = _make_radial_setup()

    # Test functions satisfying the Dirichlet BC at r=1 (and regular at 0).
    f = Function(space.function_domain, evaluate_callable=lambda r: np.sin(np.pi * np.asarray(r)))
    g = Function(space.function_domain, evaluate_callable=lambda r: np.sin(2 * np.pi * np.asarray(r)))

    lhs = space.inner_product(A(f), g)
    rhs = space.inner_product(f, A(g))
    assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-8)


def test_radial_bessel_inverse_positive_weighted():
    """⟨A f, f⟩_w > 0 — the inverse Bessel operator is a valid covariance."""
    space, L, A, k, s = _make_radial_setup()
    f = Function(space.function_domain, evaluate_callable=lambda r: np.sin(np.pi * np.asarray(r)))
    quad = space.inner_product(A(f), f)
    assert quad > 0.0


# ---------------------------------------------------------------------------
# Fast-path gating
# ---------------------------------------------------------------------------

def test_weighted_domain_disables_fast_transforms_radial():
    """A weighted (radial) domain must NOT use the flat DST/DCT fast path."""
    _space, _L, A, _k, _s = _make_radial_setup()
    assert A._can_use_fast_transforms is False


def test_weighted_domain_disables_fast_transforms_flat():
    """Even a flat Laplacian on a mass-weighted domain must fall back to the slow
    path, since the DST/DCT compute unweighted coefficients."""
    n = 20
    domain = IntervalDomain(0.1, np.pi)
    bc = BoundaryConditions.dirichlet()
    integration = IntegrationConfig(method="simpson", n_points=2000)
    space = WeightedLebesgue(n, domain, _r2, integration_config=integration)
    L = Laplacian(space.underlying_space, bc, 1.0, method="spectral",
                  integration_config=integration)
    A = BesselSobolevInverse(space, space, 1.0, 2.0, L, dofs=n,
                             integration_config=integration)
    assert A._can_use_fast_transforms is False


# ---------------------------------------------------------------------------
# Backward compatibility: unweighted flat case unchanged
# ---------------------------------------------------------------------------

def test_unweighted_flat_bessel_still_uses_fast_path():
    """Unweighted flat Bessel keeps the fast path enabled (no regression)."""
    n = 20
    domain = IntervalDomain(0.0, np.pi)
    bc = BoundaryConditions.dirichlet()
    integration = IntegrationConfig(method="simpson", n_points=2000)
    space = Lebesgue(n, domain, basis=None, integration_config=integration)
    L = Laplacian(space, bc, 1.0, method="spectral", integration_config=integration)
    A = BesselSobolev(space, space, 1.0, 2.0, L, dofs=n,
                      integration_config=integration)
    assert A._can_use_fast_transforms is True


def test_unweighted_flat_bessel_eigenfunction_scaling():
    """Sanity: unweighted flat Bessel scales sine eigenfunctions correctly."""
    n = 20
    k, s = 1.0, 2.0
    domain = IntervalDomain(0.0, np.pi)
    bc = BoundaryConditions.dirichlet()
    integration = IntegrationConfig(method="simpson", n_points=4000)
    space = Lebesgue(n, domain, basis=None, integration_config=integration)
    L = Laplacian(space, bc, 1.0, method="spectral", integration_config=integration)
    A = BesselSobolevInverse(space, space, k, s, L, dofs=n,
                             integration_config=integration)
    r = np.linspace(0.05, np.pi - 0.05, 200)
    for j in (0, 1, 2):
        phi_j = L.get_eigenfunction(j)
        lam_j = L.get_eigenvalue(j)
        g_j = (k ** 2 + lam_j) ** (-s / 2.0)
        out = A(phi_j)
        assert_allclose(out(r), g_j * phi_j(r), rtol=1e-2, atol=1e-3)
