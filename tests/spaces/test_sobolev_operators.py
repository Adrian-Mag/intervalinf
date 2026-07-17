"""
Tests for Sobolev space operator properties (mass_operator_factor,
inverse_mass_operator_factor, restrict, with_discontinuities).

These tests verify that the Sobolev space correctly uses operators
from intervalinf.operators rather than the defunct pygeoinf.interval.operators.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.config import IntegrationConfig
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.spaces.sobolev import Sobolev, SobolevSpaceDirectSum
from intervalinf.operators.bessel import BesselSobolev, BesselSobolevInverse
from intervalinf.operators.laplacian import Laplacian


@pytest.fixture
def sobolev_space():
    """A small Sobolev H^1 space on [0, π] with Dirichlet BCs."""
    domain = IntervalDomain(0, np.pi)
    bc = BoundaryConditions.dirichlet()
    integration = IntegrationConfig(method='simpson', n_points=200)
    lebesgue = Lebesgue(30, domain, basis=None, integration_config=integration)
    lap = Laplacian(lebesgue, bc, 1.0, method='spectral', integration_config=integration)
    return Sobolev(30, domain, 1.0, 1.0, lap,
                   integration_config=integration)


class TestMassOperatorFactor:
    """Test mass_operator_factor and inverse_mass_operator_factor properties."""

    def test_mass_operator_factor_returns_bessel_sobolev(self, sobolev_space):
        """mass_operator_factor must return a BesselSobolev instance."""
        factor = sobolev_space.mass_operator_factor
        assert isinstance(factor, BesselSobolev), (
            f"Expected BesselSobolev, got {type(factor)}"
        )

    def test_inverse_mass_operator_factor_returns_bessel_sobolev_inverse(
        self, sobolev_space
    ):
        """inverse_mass_operator_factor must return a BesselSobolevInverse instance."""
        inv_factor = sobolev_space.inverse_mass_operator_factor
        assert isinstance(inv_factor, BesselSobolevInverse), (
            f"Expected BesselSobolevInverse, got {type(inv_factor)}"
        )

    def test_mass_operator_factor_does_not_raise(self, sobolev_space):
        """mass_operator_factor must not raise NotImplementedError."""
        # Previously raised NotImplementedError due to stale pygeoinf.interval import
        factor = sobolev_space.mass_operator_factor  # must not raise
        assert factor is not None

    def test_inverse_mass_operator_factor_does_not_raise(self, sobolev_space):
        """inverse_mass_operator_factor must not raise NotImplementedError."""
        inv_factor = sobolev_space.inverse_mass_operator_factor  # must not raise
        assert inv_factor is not None

    def test_mass_operator_factor_inverse_roundtrip(self, sobolev_space):
        """Applying factor then inverse factor should approximately recover input."""
        from intervalinf.core.functions import Function

        # A simple sine function defined on [0, π]
        domain = sobolev_space._underlying_space._function_domain
        f = Function(domain, evaluate_callable=np.sin)

        factor = sobolev_space.mass_operator_factor
        inv_factor = sobolev_space.inverse_mass_operator_factor

        f_transformed = factor(f)
        f_roundtrip = inv_factor(f_transformed)

        # Compare at interior sample points
        xs = np.linspace(domain.a + 0.05, domain.b - 0.05, 30)
        orig_vals = np.array([f(x) for x in xs])
        roundtrip_vals = np.array([f_roundtrip(x) for x in xs])

        assert_allclose(roundtrip_vals, orig_vals, rtol=1e-4,
                        err_msg="Mass factor inverse roundtrip failed")


class TestRestrict:
    """Test Sobolev.restrict correctly updates mass operators."""

    def test_restrict_returns_sobolev(self, sobolev_space):
        """restrict must return a Sobolev instance."""
        domain_small = IntervalDomain(0.1, np.pi - 0.1)
        bc = BoundaryConditions.dirichlet()
        integration = IntegrationConfig(method='simpson', n_points=200)
        lebesgue_small = Lebesgue(
            20, domain_small, basis=None, integration_config=integration
        )
        lap_small = Laplacian(
            lebesgue_small, bc, 1.0, method='spectral',
            integration_config=integration
        )
        restricted = Sobolev(20, domain_small, 1.0, 1.0, lap_small,
                             integration_config=integration)

        result = sobolev_space.restrict(restricted)
        assert isinstance(result, Sobolev)

    def test_restrict_updates_mass_operators(self, sobolev_space):
        """After restrict, mass operators must be BesselSobolev instances."""
        domain_small = IntervalDomain(0.1, np.pi - 0.1)
        bc = BoundaryConditions.dirichlet()
        integration = IntegrationConfig(method='simpson', n_points=200)
        lebesgue_small = Lebesgue(
            20, domain_small, basis=None, integration_config=integration
        )
        lap_small = Laplacian(
            lebesgue_small, bc, 1.0, method='spectral',
            integration_config=integration
        )
        restricted = Sobolev(20, domain_small, 1.0, 1.0, lap_small,
                             integration_config=integration)

        result = sobolev_space.restrict(restricted)
        assert isinstance(result._mass_operator, BesselSobolev), (
            f"Expected BesselSobolev mass operator after restrict, "
            f"got {type(result._mass_operator)}"
        )
        assert isinstance(result._inverse_mass_operator, BesselSobolevInverse), (
            f"Expected BesselSobolevInverse inverse mass operator after restrict, "
            f"got {type(result._inverse_mass_operator)}"
        )


class TestWithDiscontinuities:
    """Test Sobolev.with_discontinuities uses correct operators."""

    def test_with_discontinuities_returns_direct_sum(self):
        """with_discontinuities must return a SobolevSpaceDirectSum."""
        domain = IntervalDomain(0, 2)
        bc = BoundaryConditions.dirichlet()

        result = Sobolev.with_discontinuities(
            20, domain, [1.0],
            s=1.0, k=1.0, bcs=bc, alpha=1.0,
            dofs=30, n_samples=64,
        )
        assert isinstance(result, SobolevSpaceDirectSum), (
            f"Expected SobolevSpaceDirectSum, got {type(result)}"
        )

    def test_with_discontinuities_does_not_raise_not_implemented(self):
        """with_discontinuities must not raise NotImplementedError."""
        domain = IntervalDomain(0, 2)
        bc = BoundaryConditions.dirichlet()

        # Must not raise NotImplementedError (stale import guard)
        result = Sobolev.with_discontinuities(
            20, domain, [1.0],
            s=1.0, k=1.0, bcs=bc, alpha=1.0,
            dofs=30, n_samples=64,
        )
        assert result is not None
