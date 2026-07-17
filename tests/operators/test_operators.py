"""Tests for the operators module."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.spaces.sobolev import Sobolev


class TestOperatorImports:
    """Test that all operators can be imported."""

    def test_import_base(self):
        """Test importing base classes."""
        from intervalinf.operators import SpectralOperator
        assert SpectralOperator is not None

    def test_import_laplacian(self):
        """Test importing Laplacian operators."""
        from intervalinf.operators import Laplacian, InverseLaplacian
        assert Laplacian is not None
        assert InverseLaplacian is not None

    def test_import_gradient(self):
        """Test importing Gradient operator."""
        from intervalinf.operators import Gradient
        assert Gradient is not None

    def test_import_bessel(self):
        """Test importing Bessel operators."""
        from intervalinf.operators import BesselSobolev, BesselSobolevInverse
        assert BesselSobolev is not None
        assert BesselSobolevInverse is not None

    def test_import_sola(self):
        """Test importing SOLA operator."""
        from intervalinf.operators import SOLAOperator
        assert SOLAOperator is not None

    def test_import_radial(self):
        """Test importing radial operators."""
        from intervalinf.operators import (
            RadialLaplacian,
            InverseRadialLaplacian,
            RadialLaplacianEigenvalueProvider,
            RadialLaplacianSpectrumProvider,
        )
        assert RadialLaplacian is not None
        assert InverseRadialLaplacian is not None
        assert RadialLaplacianEigenvalueProvider is not None
        assert RadialLaplacianSpectrumProvider is not None

    def test_import_spectral_helpers(self):
        """Test importing spectral helper functions."""
        from intervalinf.operators import (
            build_eigenfunction_expansion,
            compute_spectral_coefficients_fast,
            compute_spectral_coefficients_slow,
            validate_eigenvalue,
        )
        assert build_eigenfunction_expansion is not None
        assert compute_spectral_coefficients_fast is not None
        assert compute_spectral_coefficients_slow is not None
        assert validate_eigenvalue is not None

    def test_import_impl(self):
        """Test importing implementation utilities."""
        from intervalinf.operators import (
            GeneralFEMSolver,
            fast_spectral_coefficients,
            create_uniform_samples,
        )
        assert GeneralFEMSolver is not None
        assert fast_spectral_coefficients is not None
        assert create_uniform_samples is not None


class TestLaplacianOperator:
    """Test Laplacian operator basic functionality."""

    @pytest.fixture
    def setup_dirichlet(self):
        """Set up Lebesgue space with Dirichlet BC."""
        domain = IntervalDomain(0, np.pi)
        bc = BoundaryConditions.dirichlet()
        integration = IntegrationConfig(method='simpson', n_points=500)
        # Use basis=None (operators use their own basis for spectral method)
        space = Lebesgue(100, domain, basis=None, integration_config=integration)
        return domain, bc, space, integration

    def test_laplacian_init_spectral(self, setup_dirichlet):
        """Test Laplacian initialization with spectral method."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import Laplacian

        lap = Laplacian(
            space, bc, 1.0,
            method='spectral',
            integration_config=integration
        )
        assert lap is not None
        assert lap._method == 'spectral'

    def test_laplacian_init_fd(self, setup_dirichlet):
        """Test Laplacian initialization with FD method."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import Laplacian

        lap = Laplacian(
            space, bc, 1.0,
            method='fd',
            fd_order=2,
            integration_config=integration
        )
        assert lap is not None
        assert lap._method == 'fd'

    def test_laplacian_eigenvalue(self, setup_dirichlet):
        """Test Laplacian eigenvalues for Dirichlet BC."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import Laplacian

        lap = Laplacian(
            space, bc, 1.0,
            method='spectral',
            integration_config=integration
        )

        # For Dirichlet on [0, π]: λ_k = k²
        for k in range(1, 5):
            expected = k ** 2
            actual = lap.get_eigenvalue(k - 1)
            assert_allclose(actual, expected, rtol=1e-10)


class TestInverseLaplacianOperator:
    """Test Inverse Laplacian operator basic functionality."""

    @pytest.fixture
    def setup_dirichlet(self):
        """Set up Lebesgue space with Dirichlet BC."""
        domain = IntervalDomain(0, np.pi)
        bc = BoundaryConditions.dirichlet()
        integration = IntegrationConfig(method='simpson', n_points=500)
        # Use basis=None (operators use their own basis for spectral method)
        space = Lebesgue(100, domain, basis=None, integration_config=integration)
        return domain, bc, space, integration

    def test_inverse_laplacian_init_spectral(self, setup_dirichlet):
        """Test InverseLaplacian initialization with spectral method."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import InverseLaplacian

        inv_lap = InverseLaplacian(
            space, bc, 1.0,
            method='spectral',
            integration_config=integration
        )
        assert inv_lap is not None
        assert inv_lap._method == 'spectral'

    def test_inverse_laplacian_eigenvalue(self, setup_dirichlet):
        """Test InverseLaplacian eigenvalues."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import InverseLaplacian

        inv_lap = InverseLaplacian(
            space, bc, 1.0,
            method='spectral',
            integration_config=integration
        )

        # For Dirichlet on [0, π]: eigenvalues of L⁻¹ are 1/k²
        for k in range(1, 5):
            expected = 1.0 / (k ** 2)
            actual = inv_lap.get_eigenvalue(k - 1)
            assert_allclose(actual, expected, rtol=1e-10)


class TestGradientOperator:
    """Test Gradient operator basic functionality."""

    @pytest.fixture
    def setup_space(self):
        """Set up Lebesgue spaces for gradient."""
        domain = IntervalDomain(0, 1)
        integration = IntegrationConfig(method='simpson', n_points=200)
        # Use basis=None for gradient tests
        space = Lebesgue(
            50, domain, basis=None, integration_config=integration
        )
        return domain, space

    def test_gradient_init(self, setup_space):
        """Test Gradient initialization."""
        domain, space = setup_space
        from intervalinf.operators import Gradient

        grad = Gradient(space, fd_order=2)
        assert grad is not None

    def test_gradient_of_linear(self, setup_space):
        """Test gradient of a linear function."""
        domain, space = setup_space
        from intervalinf.operators import Gradient

        grad = Gradient(space, fd_order=4)

        # Create f(x) = x (use the domain from space)
        f = Function(space.function_domain, evaluate_callable=lambda x: x)

        # df/dx = 1
        result = grad._apply(f)

        # Check at interior points
        x_test = np.linspace(0.1, 0.9, 10)
        result_vals = result.evaluate(x_test)
        expected_vals = np.ones_like(x_test)

        assert_allclose(result_vals, expected_vals, rtol=0.1)


class TestBesselOperator:
    """Test Bessel-Sobolev operator basic functionality."""

    @pytest.fixture
    def setup_dirichlet(self):
        """Set up Lebesgue space with Dirichlet BC."""
        domain = IntervalDomain(0, np.pi)
        bc = BoundaryConditions.dirichlet()
        integration = IntegrationConfig(method='simpson', n_points=500)
        # Use basis=None (operators use their own basis)
        space = Lebesgue(100, domain, basis=None, integration_config=integration)
        return domain, bc, space, integration

    def test_bessel_init(self, setup_dirichlet):
        """Test BesselSobolev initialization."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import BesselSobolev, Laplacian

        # First need to create a Laplacian operator
        lap = Laplacian(space, bc, 1.0, method='spectral', integration_config=integration)

        bessel = BesselSobolev(
            space, space,
            k=1.0,  # Bessel parameter
            s=1.0,  # Sobolev order
            L=lap,  # The spectral operator
            integration_config=integration
        )
        assert bessel is not None

    def test_bessel_inverse_init(self, setup_dirichlet):
        """Test BesselSobolevInverse initialization."""
        domain, bc, space, integration = setup_dirichlet
        from intervalinf.operators import BesselSobolevInverse, Laplacian

        # First need to create a Laplacian operator
        lap = Laplacian(space, bc, 1.0, method='spectral', integration_config=integration)

        bessel_inv = BesselSobolevInverse(
            space, space,
            k=1.0,  # Bessel parameter
            s=1.0,  # Sobolev order
            L=lap,  # The spectral operator
            integration_config=integration
        )
        assert bessel_inv is not None


class TestRadialOperators:
    """Test radial Laplacian operators."""

    @pytest.fixture
    def setup_radial_space(self):
        """Set up Lebesgue space for radial domain (0, R)."""
        domain = IntervalDomain(0, 1)  # Domain (0, R) with R=1
        bc = BoundaryConditions.dirichlet()
        integration = IntegrationConfig(method='simpson', n_points=500)
        # Use basis=None for radial operators (they provide their own)
        space = Lebesgue(50, domain, basis=None, integration_config=integration)
        return domain, bc, space, integration

    def test_radial_laplacian_eigenvalue_provider(self, setup_radial_space):
        """Test RadialLaplacianEigenvalueProvider."""
        domain, bc, space, integration = setup_radial_space
        from intervalinf.operators import RadialLaplacianEigenvalueProvider

        provider = RadialLaplacianEigenvalueProvider(
            domain, bc, inverse=False, alpha=1.0, ell=0
        )

        # For regularity-Dirichlet on (0, R=1): λ_k = (kπ/R)²
        for k in range(1, 5):
            expected = (k * np.pi) ** 2
            actual = provider.get_eigenvalue(k - 1)
            assert_allclose(actual, expected, rtol=1e-10)


class TestSpectralHelpers:
    """Test spectral helper functions."""

    def test_validate_eigenvalue_positive(self):
        """Test validate_eigenvalue with positive value."""
        from intervalinf.operators import validate_eigenvalue
        # Should not raise for positive value
        validate_eigenvalue(1.0, 0)

    def test_validate_eigenvalue_zero(self):
        """Test validate_eigenvalue with zero value."""
        from intervalinf.operators import validate_eigenvalue
        # Should not raise for zero value
        validate_eigenvalue(0.0, 0)

    def test_validate_eigenvalue_none_raises(self):
        """Test validate_eigenvalue raises for None value."""
        from intervalinf.operators import validate_eigenvalue
        with pytest.raises(ValueError, match="Eigenvalue not available"):
            validate_eigenvalue(None, 0)

    def test_validate_eigenvalue_negative_allowed(self):
        """Test validate_eigenvalue allows negative when permitted."""
        from intervalinf.operators import validate_eigenvalue
        # Should not raise when allow_negative=True (default)
        validate_eigenvalue(-1.0, 0)

    def test_validate_eigenvalue_negative_disallowed(self):
        """Test validate_eigenvalue raises for negative when disallowed."""
        from intervalinf.operators import validate_eigenvalue
        with pytest.raises(ValueError, match="Negative eigenvalue"):
            validate_eigenvalue(-1.0, 0, allow_negative=False)


class TestLaplacianAlphaScaling:
    """Tests that Laplacian eigenvalues scale correctly with alpha.

    These tests would have caught the double-alpha bug where
    Laplacian.get_eigenvalue() multiplied alpha twice, producing
    alpha² × geo(n) instead of the correct alpha × geo(n).
    """

    @pytest.fixture
    def setup(self):
        domain = IntervalDomain(0, np.pi)
        bc = BoundaryConditions.dirichlet()
        from intervalinf.operators import Laplacian, InverseLaplacian
        return domain, bc, Laplacian, InverseLaplacian

    def test_laplacian_eigenvalue_scales_linearly_with_alpha(self, setup):
        """Laplacian eigenvalue must scale as alpha × geo(n), not alpha² × geo(n).

        For Dirichlet on [0, π] with alpha, λ_k = alpha × k².
        With alpha=4 the expected values are 4, 16, 36; not 16, 64, 144.
        """
        domain, bc, Laplacian, _ = setup
        from intervalinf.core.config import IntegrationConfig
        space = Lebesgue(100, domain, basis=None,
                         integration_config=IntegrationConfig('simpson', 500))
        for alpha in [2.0, 4.0, 10.0]:
            lap = Laplacian(space, bc, alpha, method='spectral')
            for k in range(1, 4):
                expected = alpha * k ** 2   # single factor of alpha
                actual = lap.get_eigenvalue(k - 1)
                assert_allclose(actual, expected, rtol=1e-10,
                                err_msg=f"alpha={alpha}, mode k={k}")

    def test_inverse_laplacian_eigenvalue_scales_inversely_with_alpha(self, setup):
        """InverseLaplacian eigenvalue must scale as 1/(alpha × geo(n)).

        For Dirichlet on [0, π] with alpha, λ_k(L⁻¹) = 1/(alpha × k²).
        """
        domain, bc, _, InverseLaplacian = setup
        from intervalinf.core.config import IntegrationConfig
        space = Lebesgue(100, domain, basis=None,
                         integration_config=IntegrationConfig('simpson', 500))
        for alpha in [2.0, 4.0, 10.0]:
            inv_lap = InverseLaplacian(space, bc, alpha, method='spectral')
            for k in range(1, 4):
                expected = 1.0 / (alpha * k ** 2)
                actual = inv_lap.get_eigenvalue(k - 1)
                assert_allclose(actual, expected, rtol=1e-10,
                                err_msg=f"alpha={alpha}, mode k={k}")

    def test_laplacian_inverse_laplacian_eigenvalue_product_is_unity(self, setup):
        """Product of Laplacian and InverseLaplacian eigenvalues must be 1.

        L.get_eigenvalue(n) × L⁻¹.get_eigenvalue(n) == 1 for all n and alpha.
        """
        domain, bc, Laplacian, InverseLaplacian = setup
        from intervalinf.core.config import IntegrationConfig
        space = Lebesgue(100, domain, basis=None,
                         integration_config=IntegrationConfig('simpson', 500))
        for alpha in [1.0, 2.0, 4.0, 10.0]:
            lap = Laplacian(space, bc, alpha, method='spectral')
            inv_lap = InverseLaplacian(space, bc, alpha, method='spectral')
            for n in range(4):
                product = lap.get_eigenvalue(n) * inv_lap.get_eigenvalue(n)
                assert_allclose(product, 1.0, rtol=1e-10,
                                err_msg=f"alpha={alpha}, mode n={n}")

    def test_laplacian_inverse_laplacian_roundtrip_on_eigenfunction(self, setup):
        """L(L⁻¹(φ_n)) ≈ φ_n for any alpha ≠ 1.

        This is the functional round-trip test: applying the inverse then the
        forward operator must return the original function.
        """
        domain, bc, Laplacian, InverseLaplacian = setup
        from intervalinf.core.config import IntegrationConfig
        space = Lebesgue(100, domain, basis=None,
                         integration_config=IntegrationConfig('simpson', 500))
        x = np.linspace(0.05, np.pi - 0.05, 200)
        for alpha in [2.0, 4.0]:
            lap = Laplacian(space, bc, alpha, method='spectral', dofs=60)
            inv_lap = InverseLaplacian(space, bc, alpha, method='spectral', dofs=60)
            phi0 = lap.get_eigenfunction(0)
            phi0_vals = np.array(phi0.evaluate(x))
            result_vals = np.array(lap(inv_lap(phi0)).evaluate(x))
            # ratio should be 1 everywhere; ignore near-zero points
            mask = np.abs(phi0_vals) > 0.05 * np.max(np.abs(phi0_vals))
            assert_allclose(result_vals[mask] / phi0_vals[mask],
                            np.ones(mask.sum()), rtol=1e-2,
                            err_msg=f"Round-trip failed for alpha={alpha}")


class TestProviderRadialImports:
    """Test that radial providers can be imported."""

    def test_import_radial_providers(self):
        """Test importing radial function providers."""
        from intervalinf.providers import (
            RadialLaplacianDirichletProvider,
            RadialLaplacianNeumannProvider,
            RadialLaplacianDDProvider,
            RadialLaplacianDNProvider,
            RadialLaplacianNDProvider,
            RadialLaplacianNNProvider,
        )
        assert RadialLaplacianDirichletProvider is not None
        assert RadialLaplacianNeumannProvider is not None
        assert RadialLaplacianDDProvider is not None
        assert RadialLaplacianDNProvider is not None
        assert RadialLaplacianNDProvider is not None
        assert RadialLaplacianNNProvider is not None
