"""
Tests for standalone function providers.

Tests that providers can work with just a domain (without a full space).
"""

import pytest
import numpy as np

from intervalinf.core import IntervalDomain


@pytest.fixture
def unit_domain():
    """Standard [0, 1] domain."""
    return IntervalDomain(0, 1)


@pytest.fixture
def pi_domain():
    """[0, π] domain for trigonometric functions."""
    return IntervalDomain(0, np.pi)


class TestTrigonometricProvidersStandalone:
    """Test trigonometric providers with domain-only."""

    def test_sine_provider_with_domain(self, pi_domain):
        """SineFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import SineFunctionProvider

        provider = SineFunctionProvider(pi_domain)

        # Provider should work in standalone mode
        assert provider.is_standalone
        assert provider.domain == pi_domain

        # Should be able to get functions
        f0 = provider.get_function_by_index(0)

        # Function should be standalone
        assert not f0.is_attached
        assert f0.function_domain == pi_domain

    def test_cosine_provider_with_domain(self, pi_domain):
        """CosineFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import CosineFunctionProvider

        provider = CosineFunctionProvider(pi_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached

    def test_fourier_provider_with_domain(self, pi_domain):
        """FourierFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import FourierFunctionProvider

        provider = FourierFunctionProvider(pi_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached

    def test_sine_function_evaluation(self, pi_domain):
        """Standalone sine functions evaluate correctly."""
        from intervalinf.providers.functions import SineFunctionProvider

        provider = SineFunctionProvider(pi_domain)
        # Index 0 gives k=1: sqrt(2/L) * sin(1*π*x/L)
        f1 = provider.get_function_by_index(0)

        L = np.pi  # Domain length
        normalization = np.sqrt(2 / L)

        x = np.array([0.0, np.pi/4, np.pi/2, np.pi])
        result = f1(x)

        # Expected: sqrt(2/π) * sin(π*x/π) = sqrt(2/π) * sin(x)
        expected = normalization * np.sin(np.pi * x / L)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestFEMProvidersStandalone:
    """Test FEM providers with domain-only."""

    def test_hat_provider_with_domain(self, unit_domain):
        """HatFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import HatFunctionProvider

        provider = HatFunctionProvider(unit_domain, n_nodes=6)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached

    def test_spline_provider_with_domain(self, unit_domain):
        """SplineFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import SplineFunctionProvider

        provider = SplineFunctionProvider(unit_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0, degree=2, n_knots=5)
        assert not f.is_attached


class TestSmoothProvidersStandalone:
    """Test smooth function providers with domain-only."""

    def test_bump_provider_with_domain(self, unit_domain):
        """BumpFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import BumpFunctionProvider

        provider = BumpFunctionProvider(unit_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached

    def test_bump_gradient_provider_with_domain(self, unit_domain):
        """BumpFunctionGradientProvider can create functions from domain."""
        from intervalinf.providers.functions.smooth import (
            BumpFunctionGradientProvider
        )

        provider = BumpFunctionGradientProvider(unit_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached


class TestStepProvidersStandalone:
    """Test step function providers with domain-only."""

    def test_boxcar_provider_with_domain(self, unit_domain):
        """BoxCarFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import BoxCarFunctionProvider

        provider = BoxCarFunctionProvider(unit_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached

    def test_discontinuous_provider_with_domain(self, unit_domain):
        """DiscontinuousFunctionProvider can create functions from domain."""
        from intervalinf.providers.functions.step import (
            DiscontinuousFunctionProvider
        )

        provider = DiscontinuousFunctionProvider(unit_domain, random_state=42)

        assert provider.is_standalone
        f = provider.get_random_function()
        assert not f.is_attached


class TestWaveletProvidersStandalone:
    """Test wavelet providers with domain-only."""

    def test_wavelet_provider_with_domain(self, unit_domain):
        """WaveletFunctionProvider can create functions from domain only."""
        from intervalinf.providers.functions import WaveletFunctionProvider

        provider = WaveletFunctionProvider(unit_domain)

        assert provider.is_standalone
        f = provider.get_function_by_index(0)
        assert not f.is_attached


class TestDataProvidersStandalone:
    """Test data-based providers with domain-only."""

    def test_normal_modes_provider_with_domain(self, unit_domain):
        """NormalModesProvider can create functions from domain only."""
        from intervalinf.providers.functions.data import NormalModesProvider

        provider = NormalModesProvider(unit_domain, random_state=42)

        assert provider.is_standalone
        f = provider.get_random_function()
        assert not f.is_attached

        # Indexed access should also work
        f_indexed = provider.get_function_by_index(0)
        assert not f_indexed.is_attached


class TestProviderBackwardCompatibility:
    """Test that providers still work with spaces."""

    def test_sine_provider_with_space(self, pi_domain):
        """SineFunctionProvider still works with a full space."""
        from intervalinf.providers.functions import SineFunctionProvider
        from intervalinf.spaces import Lebesgue

        space = Lebesgue(10, pi_domain)
        provider = SineFunctionProvider(space)

        # Provider should NOT be standalone
        assert not provider.is_standalone
        assert provider.space == space

        # Functions should be attached to space
        f = provider.get_function_by_index(0)
        assert f.is_attached
        assert f.space == space

    def test_hat_provider_with_space(self, unit_domain):
        """HatFunctionProvider still works with a full space."""
        from intervalinf.providers.functions import HatFunctionProvider
        from intervalinf.spaces import Lebesgue

        space = Lebesgue(5, unit_domain)
        provider = HatFunctionProvider(space)

        assert not provider.is_standalone
        f = provider.get_function_by_index(0)
        assert f.is_attached


class TestProviderDomainAccess:
    """Test domain access in both modes."""

    def test_domain_access_standalone(self, unit_domain):
        """Domain accessible in standalone mode."""
        from intervalinf.providers.functions import SineFunctionProvider

        provider = SineFunctionProvider(unit_domain)
        assert provider.domain == unit_domain

    def test_domain_access_with_space(self, unit_domain):
        """Domain accessible when initialized with space."""
        from intervalinf.providers.functions import SineFunctionProvider
        from intervalinf.spaces import Lebesgue

        space = Lebesgue(10, unit_domain)
        provider = SineFunctionProvider(space)
        assert provider.domain == unit_domain

    def test_space_access_standalone_returns_none(self, unit_domain):
        """Space returns None in standalone mode."""
        from intervalinf.providers.functions import SineFunctionProvider

        provider = SineFunctionProvider(unit_domain)
        assert provider.space is None

    def test_space_access_with_space(self, unit_domain):
        """Space accessible when initialized with space."""
        from intervalinf.providers.functions import SineFunctionProvider
        from intervalinf.spaces import Lebesgue

        space = Lebesgue(10, unit_domain)
        provider = SineFunctionProvider(space)
        assert provider.space == space


class TestStandaloneFunctionAttachment:
    """Test attaching standalone functions to spaces."""

    def test_attach_standalone_to_space(self, pi_domain):
        """Standalone function can be attached to a space."""
        from intervalinf.providers.functions import SineFunctionProvider
        from intervalinf.spaces import Lebesgue

        # Create standalone function
        provider = SineFunctionProvider(pi_domain)
        f = provider.get_function_by_index(0)
        assert not f.is_attached

        # Attach to space - returns a new function by default
        space = Lebesgue(10, pi_domain)
        f_attached = f.attach_to_space(space)

        assert f_attached.is_attached
        assert f_attached.space == space

        # Original is still unattached
        assert not f.is_attached

    def test_attach_standalone_in_place(self, pi_domain):
        """Standalone function can be attached in place."""
        from intervalinf.providers.functions import SineFunctionProvider
        from intervalinf.spaces import Lebesgue

        provider = SineFunctionProvider(pi_domain)
        f = provider.get_function_by_index(0)
        assert not f.is_attached

        space = Lebesgue(10, pi_domain)
        f.attach_to_space(space, copy=False)

        assert f.is_attached
        assert f.space == space
