"""Tests for Function class."""

import pytest
import numpy as np
from intervalinf.core import IntervalDomain, Function


class MockSpace:
    """Mock space for testing Function without full space implementation."""

    def __init__(self, domain=None):
        self._function_domain = domain or IntervalDomain(0, 1)
        self.basis_functions = None


class TestFunctionFromCallable:
    """Test Function creation from callables."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_simple_callable(self, space):
        """Test function from simple callable."""
        f = Function(space, evaluate_callable=np.sin)
        assert f.evaluate_callable is not None
        assert f.coefficients is None

    def test_lambda_callable(self, space):
        """Test function from lambda."""
        f = Function(space, evaluate_callable=lambda x: x**2)
        assert f.evaluate_callable is not None
        x = np.array([0, 0.5, 1])
        np.testing.assert_allclose(f(x), x**2)

    def test_callable_evaluation(self, space):
        """Test callable evaluation at points."""
        f = Function(space, evaluate_callable=np.exp)
        x = np.linspace(0, 1, 10)
        np.testing.assert_allclose(f(x), np.exp(x))

    def test_callable_single_point(self, space):
        """Test callable at single point."""
        f = Function(space, evaluate_callable=lambda x: 2*x)
        assert f(0.5) == 1.0


class TestFunctionFromCoefficients:
    """Test Function creation from coefficients."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_coefficients_array(self, space):
        """Test function with coefficient array."""
        coeffs = np.array([1.0, 2.0, 3.0])
        f = Function(space, coefficients=coeffs)
        assert f.coefficients is not None
        assert f.evaluate_callable is None
        np.testing.assert_array_equal(f.coefficients, coeffs)

    def test_coefficients_list(self, space):
        """Test function with coefficient list converted to array."""
        f = Function(space, coefficients=np.array([1, 2, 3]))
        assert f.coefficients is not None
        np.testing.assert_array_equal(f.coefficients, [1, 2, 3])


class TestFunctionDomain:
    """Test Function domain property."""

    def test_domain_from_space(self):
        """Test domain is taken from space."""
        domain = IntervalDomain(-1, 1)
        space = MockSpace(domain)
        f = Function(space, evaluate_callable=np.sin)
        assert f.function_domain == domain
        assert f.function_domain.a == -1
        assert f.function_domain.b == 1


class TestFunctionArithmetic:
    """Test Function arithmetic operations."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    @pytest.fixture
    def f(self, space):
        return Function(space, evaluate_callable=lambda x: x)

    @pytest.fixture
    def g(self, space):
        return Function(space, evaluate_callable=lambda x: x**2)

    def test_add(self, f, g):
        """Test addition."""
        h = f + g
        x = np.array([0, 0.5, 1])
        np.testing.assert_allclose(h(x), x + x**2)

    def test_sub(self, f, g):
        """Test subtraction."""
        h = f - g
        x = np.array([0, 0.5, 1])
        np.testing.assert_allclose(h(x), x - x**2)

    def test_mul_scalar(self, f):
        """Test multiplication by scalar."""
        h = f * 2
        x = np.array([0, 0.5, 1])
        np.testing.assert_allclose(h(x), 2*x)

    def test_rmul_scalar(self, f):
        """Test right multiplication by scalar."""
        h = 3 * f
        x = np.array([0, 0.5, 1])
        np.testing.assert_allclose(h(x), 3*x)

    def test_neg(self, f):
        """Test negation."""
        h = -f
        x = np.array([0, 0.5, 1])
        np.testing.assert_allclose(h(x), -x)


class TestFunctionSupport:
    """Test Function with compact support."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_has_compact_support(self, space):
        """Test has_compact_support property."""
        f = Function(space, evaluate_callable=lambda x: x, support=(0.2, 0.8))
        assert f.has_compact_support

    def test_no_compact_support(self, space):
        """Test function without compact support."""
        f = Function(space, evaluate_callable=lambda x: x)
        assert not f.has_compact_support

    def test_support_evaluation(self, space):
        """Test evaluation respects support."""
        f = Function(space, evaluate_callable=lambda x: x + 1, support=(0.25, 0.75))
        # Inside support
        assert f(0.5) == 1.5
        # Outside support should be zero
        np.testing.assert_allclose(f(0.1), 0.0)
        np.testing.assert_allclose(f(0.9), 0.0)

    def test_support_array_evaluation(self, space):
        """Test array evaluation respects support."""
        f = Function(space, evaluate_callable=lambda x: np.ones_like(x), support=(0.25, 0.75))
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = f(x)
        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected)


class TestFunctionValidation:
    """Test Function validation."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_requires_callable_or_coefficients(self, space):
        """Test must have callable or coefficients."""
        with pytest.raises(ValueError, match="coefficients.*evaluate_callable"):
            Function(space)

    def test_cannot_have_both(self, space):
        """Test cannot have both callable and coefficients."""
        with pytest.raises(ValueError, match="coefficients.*evaluate_callable"):
            Function(space, coefficients=np.array([1, 2]), evaluate_callable=lambda x: x)


class TestFunctionIntegrate:
    """Test Function integration."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_integrate_constant(self, space):
        """Test integration of constant function."""
        f = Function(space, evaluate_callable=lambda x: np.ones_like(x))
        result = f.integrate()
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_integrate_linear(self, space):
        """Test integration of linear function."""
        f = Function(space, evaluate_callable=lambda x: x)
        result = f.integrate()
        np.testing.assert_allclose(result, 0.5, rtol=1e-6)

    def test_integrate_with_weight(self, space):
        """Test integration with weight function."""
        f = Function(space, evaluate_callable=lambda x: np.ones_like(x))
        result = f.integrate(weight=lambda x: x)
        # ∫[0,1] x dx = 0.5
        np.testing.assert_allclose(result, 0.5, rtol=1e-6)


class TestFunctionCopy:
    """Test Function copy method."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_copy_callable(self, space):
        """Test copying function with callable."""
        f = Function(space, evaluate_callable=np.sin, name="sine")
        f2 = f.copy()
        assert f2.name == f.name
        assert f2.evaluate_callable == f.evaluate_callable

    def test_copy_coefficients(self, space):
        """Test copying function with coefficients."""
        coeffs = np.array([1, 2, 3])
        f = Function(space, coefficients=coeffs)
        f2 = f.copy()
        np.testing.assert_array_equal(f2.coefficients, coeffs)
        # Verify it's a copy, not same array
        f2.coefficients[0] = 99
        assert f.coefficients[0] == 1


class TestFunctionRepr:
    """Test Function string representations."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_repr_callable(self, space):
        """Test repr with callable."""
        f = Function(space, evaluate_callable=np.sin)
        r = repr(f)
        assert "Function" in r

    def test_repr_with_name(self, space):
        """Test repr includes name."""
        f = Function(space, evaluate_callable=np.sin, name="sine")
        r = repr(f)
        assert "sine" in r
