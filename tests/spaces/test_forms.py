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


class TestLinearFormKernelNestedDirectSum:
    """
    Tests for LinearFormKernel._mapping_impl with nested-list kernels.

    Reproduces the bug surfaced by realistic_dli.ipynb: a three-component
    LebesgueSpaceDirectSum where one component is itself a two-component
    LebesgueSpaceDirectSum produces a nested-list kernel.  The _mapping_impl
    must handle this recursively or it raises:
        TypeError: can't multiply sequence by non-int of type 'list'
    """

    def _make_fn(self, domain, f):
        """Make a standalone Function with an evaluate_callable."""
        return Function(domain, evaluate_callable=f)

    def test_nested_list_kernel_evaluates_without_error(self):
        """
        _mapping_impl must not raise when the kernel is a nested list
        (list-of-lists), as occurs when to_dual is called on a
        LebesgueSpaceDirectSum whose components include another
        LebesgueSpaceDirectSum.
        """
        from intervalinf import IntervalDomain, Lebesgue, LebesgueSpaceDirectSum, Function
        from intervalinf.core.config import IntegrationConfig, ParallelConfig
        from intervalinf.spaces.forms import LinearFormKernel

        domain = IntervalDomain(0, 1)
        cfg = IntegrationConfig(method='trapz', n_points=50)

        # L^2 spaces
        M_a = Lebesgue(0, domain, integration_config=cfg)
        M_b = Lebesgue(0, domain, integration_config=cfg)
        M_c = Lebesgue(0, domain, integration_config=cfg)

        # Inner direct sum:  M_inner = M_b ⊕ M_c
        M_inner = LebesgueSpaceDirectSum([M_b, M_c])

        # Outer direct sum:  M_outer = M_a ⊕ M_inner
        M_outer = LebesgueSpaceDirectSum([M_a, M_inner])

        # Vectors in each space
        f_a = Function(domain, evaluate_callable=lambda x: np.ones_like(x))
        f_b = Function(domain, evaluate_callable=lambda x: np.sin(np.pi * x))
        f_c = Function(domain, evaluate_callable=lambda x: np.cos(np.pi * x))

        # The nested-list vector representation
        v = [f_a, [f_b, f_c]]

        # This is what LebesgueSpaceDirectSum.to_dual produces —
        # a LinearFormKernel with a nested-list kernel.
        form = LinearFormKernel(
            M_outer,
            kernel=v,
            integration_config=cfg,
        )

        # Evaluating on the same vector must not raise
        result = form(v)  # = <v, v>_{M_outer}
        assert np.isfinite(result)
        assert result >= 0.0  # should be a norm-squared like quantity

    def test_nested_list_kernel_value_matches_sum_of_parts(self):
        """
        The value computed by _mapping_impl on a nested-list kernel must
        equal the sum of the individual inner products.
        """
        from intervalinf import IntervalDomain, Lebesgue, LebesgueSpaceDirectSum, Function
        from intervalinf.core.config import IntegrationConfig
        from intervalinf.spaces.forms import LinearFormKernel

        domain = IntervalDomain(0, 1)
        cfg = IntegrationConfig(method='trapz', n_points=200)

        M_a = Lebesgue(0, domain, integration_config=cfg)
        M_b = Lebesgue(0, domain, integration_config=cfg)
        M_c = Lebesgue(0, domain, integration_config=cfg)
        M_inner = LebesgueSpaceDirectSum([M_b, M_c])
        M_outer = LebesgueSpaceDirectSum([M_a, M_inner])

        f_a = Function(domain, evaluate_callable=lambda x: np.ones_like(x))
        f_b = Function(domain, evaluate_callable=lambda x: x)
        f_c = Function(domain, evaluate_callable=lambda x: x * (1 - x))
        g_a = Function(domain, evaluate_callable=lambda x: np.cos(np.pi * x))
        g_b = Function(domain, evaluate_callable=lambda x: np.sin(np.pi * x))
        g_c = Function(domain, evaluate_callable=lambda x: np.ones_like(x) * 2)

        form = LinearFormKernel(M_outer, kernel=[f_a, [f_b, f_c]], integration_config=cfg)
        result = form([g_a, [g_b, g_c]])

        # Expected: ∫f_a g_a + ∫f_b g_b + ∫f_c g_c
        expected = (
            (f_a * g_a).integrate(method='trapz', n_points=200)
            + (f_b * g_b).integrate(method='trapz', n_points=200)
            + (f_c * g_c).integrate(method='trapz', n_points=200)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10)
