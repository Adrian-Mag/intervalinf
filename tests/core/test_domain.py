"""Tests for IntervalDomain."""

import pytest
import numpy as np
from intervalinf.core import IntervalDomain


class TestIntervalDomainBasics:
    """Test basic IntervalDomain functionality."""

    def test_init_valid(self):
        """Test valid domain creation."""
        domain = IntervalDomain(0, 1)
        assert domain.a == 0
        assert domain.b == 1
        assert domain.boundary_type == "closed"

    def test_init_invalid_order(self):
        """Test that a >= b raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            IntervalDomain(1, 0)
        with pytest.raises(ValueError, match="must be less than"):
            IntervalDomain(1, 1)

    def test_length(self):
        """Test length property."""
        domain = IntervalDomain(2, 5)
        assert domain.length == 3

    def test_center(self):
        """Test center property."""
        domain = IntervalDomain(0, 10)
        assert domain.center == 5

    def test_radius(self):
        """Test radius property."""
        domain = IntervalDomain(0, 10)
        assert domain.radius == 5


class TestIntervalDomainContains:
    """Test contains method."""

    def test_contains_closed(self):
        """Test contains for closed interval."""
        domain = IntervalDomain(0, 1, boundary_type="closed")
        assert domain.contains(0)
        assert domain.contains(1)
        assert domain.contains(0.5)
        assert not domain.contains(-0.1)
        assert not domain.contains(1.1)

    def test_contains_open(self):
        """Test contains for open interval."""
        domain = IntervalDomain(0, 1, boundary_type="open")
        assert not domain.contains(0)
        assert not domain.contains(1)
        assert domain.contains(0.5)

    def test_contains_left_open(self):
        """Test contains for left-open interval."""
        domain = IntervalDomain(0, 1, boundary_type="left_open")
        assert not domain.contains(0)
        assert domain.contains(1)
        assert domain.contains(0.5)

    def test_contains_right_open(self):
        """Test contains for right-open interval."""
        domain = IntervalDomain(0, 1, boundary_type="right_open")
        assert domain.contains(0)
        assert not domain.contains(1)
        assert domain.contains(0.5)

    def test_contains_array(self):
        """Test contains with array input."""
        domain = IntervalDomain(0, 1)
        x = np.array([-0.5, 0, 0.5, 1, 1.5])
        result = domain.contains(x)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(result, expected)


class TestIntervalDomainMesh:
    """Test mesh generation."""

    def test_uniform_mesh_closed(self):
        """Test uniform mesh for closed interval."""
        domain = IntervalDomain(0, 1, boundary_type="closed")
        mesh = domain.uniform_mesh(5)
        expected = np.array([0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_array_almost_equal(mesh, expected)

    def test_uniform_mesh_open(self):
        """Test uniform mesh for open interval stays inside."""
        domain = IntervalDomain(0, 1, boundary_type="open")
        mesh = domain.uniform_mesh(5)
        # Should not include exact endpoints
        assert mesh[0] > 0
        assert mesh[-1] < 1

    def test_uniform_mesh_count(self):
        """Test that mesh has correct number of points."""
        domain = IntervalDomain(0, 1)
        for n in [3, 10, 100]:
            mesh = domain.uniform_mesh(n)
            assert len(mesh) == n


class TestIntervalDomainIntegration:
    """Test numerical integration."""

    def test_integrate_constant(self):
        """Test integration of constant function."""
        domain = IntervalDomain(0, 1)
        result = domain.integrate(lambda x: 2.0, n_points=10)
        assert result == pytest.approx(2.0, rel=1e-11)

    def test_integrate_linear(self):
        """Test integration of linear function: ∫₀¹ x dx = 0.5."""
        domain = IntervalDomain(0, 1)
        result = domain.integrate(lambda x: x, n_points=100)
        assert result == pytest.approx(0.5, rel=1e-10)

    def test_integrate_quadratic(self):
        """Test integration of quadratic: ∫₀¹ x² dx = 1/3."""
        domain = IntervalDomain(0, 1)
        result = domain.integrate(lambda x: x**2, n_points=100)
        assert result == pytest.approx(1/3, rel=1e-10)

    def test_integrate_sine(self):
        """Test integration of sin: ∫₀^π sin(x) dx = 2."""
        domain = IntervalDomain(0, np.pi)
        result = domain.integrate(np.sin, n_points=1000)
        assert result == pytest.approx(2.0, rel=1e-10)

    def test_integrate_with_support(self):
        """Test integration over subdomain."""
        domain = IntervalDomain(0, 1)
        # ∫₀^0.5 2 dx = 1
        result = domain.integrate(
            lambda x: 2.0, support=(0, 0.5), n_points=100
        )
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_integrate_methods(self):
        """Test different integration methods give similar results."""
        domain = IntervalDomain(0, 1)

        def f(x):
            return x**2

        simpson = domain.integrate(f, method="simpson", n_points=1000)
        trapz = domain.integrate(f, method="trapz", n_points=1000)

        # Both should be close to 1/3
        assert simpson == pytest.approx(1/3, rel=1e-8)
        assert trapz == pytest.approx(1/3, rel=1e-6)


class TestIntervalDomainOperations:
    """Test domain operations."""

    def test_interior(self):
        """Test interior returns open interval."""
        domain = IntervalDomain(0, 1, boundary_type="closed")
        interior = domain.interior()
        assert interior.boundary_type == "open"
        assert interior.a == domain.a
        assert interior.b == domain.b

    def test_closure(self):
        """Test closure returns closed interval."""
        domain = IntervalDomain(0, 1, boundary_type="open")
        closure = domain.closure()
        assert closure.boundary_type == "closed"

    def test_restriction_valid(self):
        """Test valid subdomain restriction."""
        domain = IntervalDomain(0, 1)
        sub = domain.restriction_to_subinterval(0.2, 0.8)
        assert sub.a == 0.2
        assert sub.b == 0.8

    def test_restriction_invalid(self):
        """Test invalid subdomain raises error."""
        domain = IntervalDomain(0, 1)
        with pytest.raises(ValueError):
            domain.restriction_to_subinterval(0.8, 0.2)  # a >= b
        with pytest.raises(ValueError):
            domain.restriction_to_subinterval(-0.1, 0.5)  # outside

    def test_split_at_discontinuities(self):
        """Test domain splitting."""
        domain = IntervalDomain(0, 1)
        subdomains = domain.split_at_discontinuities([0.3, 0.7])
        assert len(subdomains) == 3
        assert subdomains[0].a == 0
        assert subdomains[0].b == 0.3
        assert subdomains[1].a == 0.3
        assert subdomains[1].b == 0.7
        assert subdomains[2].a == 0.7
        assert subdomains[2].b == 1

    def test_equality(self):
        """Test domain equality."""
        d1 = IntervalDomain(0, 1)
        d2 = IntervalDomain(0, 1)
        d3 = IntervalDomain(0, 2)
        d4 = IntervalDomain(0, 1, boundary_type="open")

        assert d1 == d2
        assert d1 != d3
        assert d1 != d4

    def test_repr(self):
        """Test string representation."""
        assert repr(IntervalDomain(0, 1)) == "[0.0, 1.0]"
        assert repr(IntervalDomain(0, 1, boundary_type="open")) == "(0.0, 1.0)"
        assert (
            repr(
                IntervalDomain(0, 1, boundary_type="left_open")
            ) == "(0.0, 1.0]"
        )
        assert (
            repr(
                IntervalDomain(0, 1, boundary_type="right_open")
            ) == "[0.0, 1.0)"
        )
