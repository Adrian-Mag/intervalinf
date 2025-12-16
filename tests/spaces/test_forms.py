"""
Tests for LinearFormKernel.

Tests basic functionality of the LinearFormKernel class for
representing linear forms via kernel functions.
"""

import pytest
import numpy as np

from intervalinf.core import IntervalDomain, Function
from intervalinf.core.config import IntegrationConfig, ParallelConfig
from intervalinf.spaces.forms import LinearFormKernel


class MockHilbertSpace:
    """Mock Hilbert space for testing LinearFormKernel."""

    def __init__(self, dim: int = 10):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim


class TestLinearFormKernelInit:
    """Test LinearFormKernel initialization."""

    def test_init_with_kernel(self):
        """Test initialization with kernel function."""
        domain = IntervalDomain(0, 1)
        space = MockHilbertSpace(10)
        space._function_domain = domain  # Add for Function

        # Create a simple kernel function
        kernel_space = MockHilbertSpace(10)
        kernel_space.function_domain = domain
        kernel_fn = Function(
            kernel_space,
            evaluate_callable=lambda x: np.ones_like(x)
        )

        form = LinearFormKernel(
            space,
            kernel=kernel_fn,
            integration_config=IntegrationConfig(),
        )

        assert form.domain is space
        assert form._kernel is kernel_fn

    def test_init_with_components(self):
        """Test initialization with component array."""
        space = MockHilbertSpace(5)

        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        form = LinearFormKernel(
            space,
            components=components,
            integration_config=IntegrationConfig(),
        )

        assert form._components is components
        assert form._kernel is None

    def test_init_requires_one_specification(self):
        """Test that exactly one of mapping/kernel/components is required."""
        space = MockHilbertSpace(5)

        with pytest.raises(ValueError, match="Either mapping, kernel"):
            LinearFormKernel(
                space,
                integration_config=IntegrationConfig(),
            )

    def test_default_parallel_config(self):
        """Test default parallel config is disabled."""
        space = MockHilbertSpace(5)
        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        form = LinearFormKernel(
            space,
            components=components,
            integration_config=IntegrationConfig(),
        )

        assert form.parallel.enabled is False


class TestLinearFormKernelKernelProperty:
    """Test LinearFormKernel.kernel property."""

    def test_kernel_property_no_weight(self):
        """Test kernel property without weight function."""
        domain = IntervalDomain(0, 1)
        space = MockHilbertSpace(10)
        space._weight = None  # No weight

        kernel_space = MockHilbertSpace(10)
        kernel_space.function_domain = domain
        kernel_fn = Function(
            kernel_space,
            evaluate_callable=lambda x: np.sin(x)
        )

        form = LinearFormKernel(
            space,
            kernel=kernel_fn,
            integration_config=IntegrationConfig(),
        )

        # Without weight, should return kernel as-is
        assert form.kernel is kernel_fn

    def test_kernel_property_returns_none_when_no_kernel(self):
        """Test kernel property returns None when initialized with components."""
        space = MockHilbertSpace(5)
        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        form = LinearFormKernel(
            space,
            components=components,
            integration_config=IntegrationConfig(),
        )

        assert form.kernel is None


class TestLinearFormKernelConfigs:
    """Test configuration handling."""

    def test_integration_config_stored(self):
        """Test integration config is properly stored."""
        space = MockHilbertSpace(5)
        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        config = IntegrationConfig(method='simpson', n_points=5000)

        form = LinearFormKernel(
            space,
            components=components,
            integration_config=config,
        )

        assert form.integration is config
        assert form.integration.method == 'simpson'
        assert form.integration.n_points == 5000

    def test_parallel_config_stored(self):
        """Test parallel config is properly stored."""
        space = MockHilbertSpace(5)
        components = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        par_config = ParallelConfig(enabled=True, n_jobs=4)

        form = LinearFormKernel(
            space,
            components=components,
            integration_config=IntegrationConfig(),
            parallel_config=par_config,
        )

        assert form.parallel.enabled is True
        assert form.parallel.n_jobs == 4


class TestLinearFormKernelListKernel:
    """Test LinearFormKernel with list of kernels (for direct sums)."""

    def test_list_kernel_stored(self):
        """Test that list kernels are stored correctly."""
        domain = IntervalDomain(0, 1)
        space = MockHilbertSpace(10)
        space._weight = None

        # Create list of kernel functions
        kernel_space = MockHilbertSpace(10)
        kernel_space.function_domain = domain
        kernels = [
            Function(kernel_space, evaluate_callable=lambda x: np.sin(x)),
            Function(kernel_space, evaluate_callable=lambda x: np.cos(x)),
        ]

        form = LinearFormKernel(
            space,
            kernel=kernels,
            integration_config=IntegrationConfig(),
        )

        assert isinstance(form._kernel, list)
        assert len(form._kernel) == 2
