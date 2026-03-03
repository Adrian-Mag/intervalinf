"""Tests for Function class."""

import pytest
import numpy as np
from intervalinf.core import IntervalDomain, Function
import math
import itertools


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

    def test_mul_disjoint_support_gives_zero_with_empty_support(self, space):
        f = Function(
            space,
            evaluate_callable=lambda x: np.ones_like(x),
            support=(0.1, 0.2),
        )
        g = Function(
            space,
            evaluate_callable=lambda x: 3.0 * np.ones_like(x),
            support=(0.8, 0.9),
        )

        h = f * g
        assert h.support == []
        assert h.has_compact_support
        x = np.array([0.0, 0.15, 0.5, 0.85, 1.0])
        np.testing.assert_allclose(h(x), 0.0)


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
        f = Function(
            space, evaluate_callable=lambda x: x + 1, support=(0.25, 0.75)
        )
        # Inside support
        assert f(0.5) == 1.5
        # Outside support should be zero
        np.testing.assert_allclose(f(0.1), 0.0)
        np.testing.assert_allclose(f(0.9), 0.0)

    def test_support_array_evaluation(self, space):
        """Test array evaluation respects support."""
        f = Function(
            space,
            evaluate_callable=lambda x: np.ones_like(x),
            support=(0.25, 0.75),
        )
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = f(x)
        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected)


class TestFunctionRestrict:
    """Test Function.restrict() behavior, especially with compact support."""

    def test_restrict_intersects_support_with_domain(self):
        dom = IntervalDomain(0.0, 1.0)
        space = MockSpace(dom)
        restricted_space = MockSpace(IntervalDomain(0.0, 0.5))

        f = Function(
            space,
            evaluate_callable=lambda x: np.ones_like(x),
            support=(0.25, 0.75),
        )
        fr = f.restrict(restricted_space)

        assert fr.support == [(0.25, 0.5)]
        assert fr.has_compact_support
        assert fr(0.1) == 0.0
        assert fr(0.3) == 1.0

        with pytest.raises(ValueError, match="not in domain"):
            fr(0.6)

    def test_restrict_can_yield_empty_support(self):
        dom = IntervalDomain(0.0, 1.0)
        space = MockSpace(dom)
        restricted_space = MockSpace(IntervalDomain(0.0, 0.5))

        f = Function(
            space,
            evaluate_callable=lambda x: 2.0 * np.ones_like(x),
            support=(0.75, 0.9),
        )
        fr = f.restrict(restricted_space)

        assert fr.support == []
        np.testing.assert_allclose(fr(np.array([0.1, 0.4])), 0.0)

    def test_restrict_preserves_no_compact_support(self):
        dom = IntervalDomain(0.0, 1.0)
        space = MockSpace(dom)
        restricted_space = MockSpace(IntervalDomain(0.0, 0.5))

        f = Function(space, evaluate_callable=lambda x: x + 1.0)
        fr = f.restrict(restricted_space)

        assert fr.support is None
        assert not fr.has_compact_support
        x = np.array([0.1, 0.4])
        np.testing.assert_allclose(fr(x), x + 1.0)


class TestFunctionValidation:
    """Test Function validation."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_requires_callable_or_coefficients(self, space):
        """Test must have callable or coefficients."""
        with pytest.raises(
            ValueError, match="coefficients.*evaluate_callable"
        ):
            Function(space)

    def test_cannot_have_both(self, space):
        """Test cannot have both callable and coefficients."""
        with pytest.raises(
            ValueError, match="coefficients.*evaluate_callable"
        ):
            Function(
                space,
                coefficients=np.array([1, 2]),
                evaluate_callable=lambda x: x,
            )


class TestFunctionIntegrate:
    """Test Function integration."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_integrate_constant(self, space):
        """Test integration of constant function."""
        f = Function(space, evaluate_callable=lambda x: np.ones_like(x))
        result = f.integrate()
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_integrate_linear(self, space):
        """Test integration of linear function."""
        f = Function(space, evaluate_callable=lambda x: x)
        result = f.integrate()
        np.testing.assert_allclose(result, 0.5, rtol=1e-10)

    def test_integrate_with_weight(self, space):
        """Test integration with weight function."""
        f = Function(space, evaluate_callable=lambda x: np.ones_like(x))
        result = f.integrate(weight=lambda x: x)
        # ∫[0,1] x dx = 0.5
        np.testing.assert_allclose(result, 0.5, rtol=1e-10)


class TestHighFrequencyIntegration:
    """High-frequency oscillatory integrals for Fourier-like bases."""

    @pytest.fixture
    def domain_space(self):
        dom = IntervalDomain(0.0, 2.0 * math.pi)
        return MockSpace(dom)

    # Note: composite Simpson can underperform on very high-frequency
    # oscillatory integrands depending on grid alignment and floating
    # point cancellation — tests below relax Simpson tolerances slightly.
    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    @pytest.mark.parametrize(
        "k,tol",
        [
            (10, 1e-10),
            (50, 1e-8),
            (200, 1e-6),
            (500, 1e-4),
        ],
    )
    def test_high_freq_vectorized(self, domain_space, method, k, tol):
        """Vectorized callable (numpy) for sin(k*x) should integrate to ~0.

        Tolerances are method-aware: Simpson tolerances are relaxed to
        account for observed higher cancellation error on oscillatory
        integrands.
        """
        f = Function(domain_space, evaluate_callable=lambda x: np.sin(k * x))
        # coarse and fine grids to check convergence
        res_coarse = f.integrate(method=method, n_points=1024)
        res_fine = f.integrate(method=method, n_points=4096)
        # fine result should be closer to zero and within tolerance
        assert abs(res_fine) <= abs(res_coarse) + 1e-12
        tol_eff = tol * (3.0 if method == "simpson" else 1.0)
        assert abs(res_fine) < tol_eff

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    @pytest.mark.parametrize("k,tol", [(10, 1e-8), (50, 1e-6), (200, 1e-3)])
    def test_high_freq_scalar_callable(self, domain_space, method, k, tol):
        """Scalar (non-vectorized) callable should behave similarly.

        Use a Python scalar `math.sin` based callable and set vectorized=False
        so the integrator evaluates point-by-point.
        """

        def scalar_fn(x):
            return math.sin(k * float(x))
        f = Function(domain_space, evaluate_callable=scalar_fn)
        res_coarse = f.integrate(
            method=method, n_points=1024, vectorized=False
        )
        res_fine = f.integrate(
            method=method, n_points=4096, vectorized=False
        )
        assert abs(res_fine) <= abs(res_coarse) + 1e-12
        tol_eff = tol * (3.0 if method == "simpson" else 1.0)
        assert abs(res_fine) < tol_eff


class TestCosineOffset:
    """Test B: cos(kx) with nonzero average offset to check aliasing bias."""

    @pytest.fixture
    def domain_space(self):
        dom = IntervalDomain(0.0, 2.0 * math.pi)
        return MockSpace(dom)

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    @pytest.mark.parametrize("k", [10, 50, 200])
    def test_cosine_plus_offset(self, domain_space, method, k):
        """Integrate cos(kx) + 1; should equal domain length (2π)."""
        f = Function(
            domain_space, evaluate_callable=lambda x: np.cos(k * x) + 1.0
        )
        result = f.integrate(method=method, n_points=4096)
        expected = 2.0 * math.pi
        # cos(kx) integrates to ~0 on [0,2π], so result ≈ ∫1 dx = 2π
        tol = 1e-6 if method == "trapz" else 3e-6
        np.testing.assert_allclose(result, expected, rtol=tol)


class TestNarrowGaussians:
    """Test C: Narrow Gaussians with small sigma to check peak capture."""

    @pytest.fixture
    def unit_space(self):
        return MockSpace(IntervalDomain(0.0, 1.0))

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    @pytest.mark.parametrize("sigma", [1e-2, 5e-3, 1e-3])
    def test_narrow_gaussian_center(self, unit_space, method, sigma):
        """Gaussian centered at x0=0.5 with small sigma."""
        x0 = 0.5
        norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        def gauss(x):
            return norm * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

        f = Function(unit_space, evaluate_callable=gauss)
        # Use fine grid for narrow peak
        result = f.integrate(method=method, n_points=8192)
        # On infinite domain integral = 1; on [0,1] ≈ 1 for small sigma
        np.testing.assert_allclose(result, 1.0, rtol=5e-2)

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_narrow_gaussian_near_boundary(self, unit_space, method):
        """Gaussian near boundary x0=0.1 with small sigma."""
        x0 = 0.1
        sigma = 2e-3
        norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        def gauss(x):
            return norm * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

        f = Function(unit_space, evaluate_callable=gauss)
        result = f.integrate(method=method, n_points=8192)
        # With sigma=2e-3 and x0=0.1, domain [0,1] captures nearly all mass
        np.testing.assert_allclose(result, 1.0, rtol=5e-2)


class TestDiscontinuousFunctions:
    """Test D: Step/Heaviside functions with jumps inside domain."""

    @pytest.fixture
    def unit_space(self):
        return MockSpace(IntervalDomain(0.0, 1.0))

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_heaviside_midpoint(self, unit_space, method):
        """Step at x=0.5: f=1 for x<0.5, f=2 for x>=0.5."""

        def step(x):
            return np.where(x < 0.5, 1.0, 2.0)

        f = Function(unit_space, evaluate_callable=step)
        result = f.integrate(method=method, n_points=4096)
        # Exact: 0.5*1 + 0.5*2 = 1.5
        np.testing.assert_allclose(result, 1.5, rtol=1e-4)

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_multiple_jumps(self, unit_space, method):
        """Piecewise constant with multiple jumps."""

        def piecewise(x):
            return np.where(
                x < 0.25, 1.0, np.where(x < 0.75, 3.0, 2.0)
            )

        f = Function(unit_space, evaluate_callable=piecewise)
        result = f.integrate(method=method, n_points=8192)
        # Exact: 0.25*1 + 0.5*3 + 0.25*2 = 0.25+1.5+0.5 = 2.25
        np.testing.assert_allclose(result, 2.25, rtol=1e-4)


class TestMultipleDisjointSupports:
    """Test E: Functions with multiple disjoint compact supports."""

    @pytest.fixture
    def unit_space(self):
        return MockSpace(IntervalDomain(0.0, 1.0))

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_two_disjoint_bumps(self, unit_space, method):
        """Two narrow bumps on disjoint intervals."""
        f = Function(
            unit_space,
            evaluate_callable=lambda x: np.ones_like(x),
            support=[(0.1, 0.2), (0.7, 0.9)],
        )
        result = f.integrate(method=method, n_points=2048)
        # Integral = 0.1 + 0.2 = 0.3
        np.testing.assert_allclose(result, 0.3, rtol=1e-6)

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_three_disjoint_intervals(self, unit_space, method):
        """Three small intervals with constant value."""
        f = Function(
            unit_space,
            evaluate_callable=lambda x: 2.0 * np.ones_like(x),
            support=[(0.1, 0.15), (0.4, 0.5), (0.8, 0.85)],
        )
        result = f.integrate(method=method, n_points=4096)
        # Integral = 2*(0.05 + 0.1 + 0.05) = 2*0.2 = 0.4
        np.testing.assert_allclose(result, 0.4, rtol=1e-6)


class TestBoundarySemantics:
    """Test H: Support endpoint semantics (open/closed/clopen)."""

    def test_closed_support_includes_endpoints(self):
        """Support (a,b) is treated as closed [a,b] (current behavior)."""
        dom = IntervalDomain(0.0, 1.0)
        space = MockSpace(dom)
        f = Function(
            space, evaluate_callable=lambda x: np.ones_like(x), support=(0.2, 0.8)
        )
        # Evaluate exactly at endpoints
        assert f(0.2) == 1.0
        assert f(0.8) == 1.0
        # Outside
        assert f(0.1) == 0.0
        assert f(0.9) == 0.0

    def test_support_boundary_consistency(self):
        """Integration over support matches callable values at boundaries."""
        dom = IntervalDomain(0.0, 1.0)
        space = MockSpace(dom)
        # Linear function on closed support
        f = Function(space, evaluate_callable=lambda x: x, support=(0.25, 0.75))
        result = f.integrate(method="simpson", n_points=2048)
        # Exact integral: ∫[0.25,0.75] x dx = [x²/2] = (0.75²-0.25²)/2
        expected = (0.75**2 - 0.25**2) / 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestNonVectorizedCallables:
    """Test I: Non-vectorized scalar-only callables."""

    @pytest.fixture
    def unit_space(self):
        return MockSpace(IntervalDomain(0.0, 1.0))

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_scalar_polynomial(self, unit_space, method):
        """Polynomial using scalar math operations."""

        def poly_scalar(x):
            return float(x) ** 2 + 2 * float(x) + 1

        f = Function(unit_space, evaluate_callable=poly_scalar)
        result = f.integrate(method=method, n_points=512, vectorized=False)
        # ∫[0,1] (x²+2x+1) dx = [x³/3 + x² + x] = 1/3 + 1 + 1 = 7/3
        expected = 7.0 / 3.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_scalar_vs_vectorized_equivalence(self, unit_space, method):
        """Scalar and vectorized callables should produce same integral."""

        def vec_fn(x):
            return np.exp(x)

        def scalar_fn(x):
            return math.exp(float(x))

        f_vec = Function(unit_space, evaluate_callable=vec_fn)
        f_scalar = Function(unit_space, evaluate_callable=scalar_fn)

        res_vec = f_vec.integrate(method=method, n_points=1024)
        res_scalar = f_scalar.integrate(
            method=method, n_points=1024, vectorized=False
        )
        np.testing.assert_allclose(res_vec, res_scalar, rtol=1e-10)


class TestExtremeFrequencies:
    """Test K: Extremely large k (aliasing) to document failure modes."""

    @pytest.fixture
    def domain_space(self):
        dom = IntervalDomain(0.0, 2.0 * math.pi)
        return MockSpace(dom)

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_very_high_frequency_documented_error(self, domain_space, method):
        """k=1000: aliasing becomes severe; document required n_points."""
        k = 1000
        f = Function(domain_space, evaluate_callable=lambda x: np.sin(k * x))
        # With moderate n_points, error is large
        res_coarse = f.integrate(method=method, n_points=1024)
        # Should be ~0 but aliasing produces large error
        assert abs(res_coarse) < 1.0  # just document it doesn't blow up

        # With very fine grid should improve
        res_fine = f.integrate(method=method, n_points=32768)
        # Expect improvement but may still have error
        assert abs(res_fine) < 0.1


class TestNaNInfHandling:
    """Test L: Functions producing NaN/Inf at sample points."""

    @pytest.fixture
    def unit_space(self):
        return MockSpace(IntervalDomain(0.0, 1.0))

    def test_nan_propagation(self, unit_space):
        """Function returning NaN should propagate to integral."""

        def nan_fn(x):
            return np.nan * np.ones_like(x)

        f = Function(unit_space, evaluate_callable=nan_fn)
        result = f.integrate(method="simpson", n_points=128)
        assert np.isnan(result)

    def test_inf_propagation(self, unit_space):
        """Function returning Inf should propagate."""

        def inf_fn(x):
            return np.inf * np.ones_like(x)

        f = Function(unit_space, evaluate_callable=inf_fn)
        result = f.integrate(method="trapz", n_points=128)
        assert np.isinf(result)

    def test_mixed_finite_nan(self, unit_space):
        """Function with some NaN values propagates NaN to result."""

        def mixed_fn(x):
            return np.where(x < 0.5, 1.0, np.nan)

        f = Function(unit_space, evaluate_callable=mixed_fn)
        result = f.integrate(method="simpson", n_points=256)
        assert np.isnan(result)


class TestLargeNPoints:
    """Test M: Very large n_points to exercise performance/memory."""

    @pytest.fixture
    def unit_space(self):
        return MockSpace(IntervalDomain(0.0, 1.0))

    @pytest.mark.parametrize("method", ["simpson", "trapz"])
    def test_large_n_points_completes(self, unit_space, method):
        """Integration with n_points=100k should complete successfully."""
        f = Function(unit_space, evaluate_callable=lambda x: x**2)
        result = f.integrate(method=method, n_points=100000)
        # ∫[0,1] x² dx = 1/3
        np.testing.assert_allclose(result, 1.0 / 3.0, rtol=1e-8)

    def test_very_large_n_points_memory(self, unit_space):
        """n_points=1M should complete (checks memory allocation)."""
        f = Function(unit_space, evaluate_callable=np.sin)
        result = f.integrate(method="simpson", n_points=1000000)
        # ∫[0,1] sin(x) dx = -cos(1) + cos(0) = 1 - cos(1) ≈ 0.4597
        expected = 1.0 - math.cos(1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


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
        f2.coefficients[0] = 99  # type: ignore
        assert f.coefficients[0] == 1  # type: ignore


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


# =============================================================================
# New tests for standalone (domain-only) functions
# =============================================================================


class TestStandaloneFunction:
    """Test Function creation directly on a domain (without space)."""

    @pytest.fixture
    def domain(self):
        return IntervalDomain(0, 1)

    def test_create_standalone_function(self, domain):
        """Test creating function directly on domain."""
        f = Function(domain, evaluate_callable=lambda x: x**2)
        assert f.is_attached is False
        assert f.space is None
        assert f.function_domain == domain

    def test_standalone_evaluate(self, domain):
        """Test standalone function can be evaluated."""
        f = Function(domain, evaluate_callable=np.sin)
        x = np.linspace(0, 1, 10)
        np.testing.assert_allclose(f(x), np.sin(x))

    def test_standalone_single_point(self, domain):
        """Test standalone evaluation at single point."""
        f = Function(domain, evaluate_callable=lambda x: 2*x + 1)
        assert f(0.5) == 2.0

    def test_standalone_coefficients_rejected(self, domain):
        """Test that coefficient-based functions require a space."""
        with pytest.raises(ValueError, match="Coefficient-based"):
            Function(domain, coefficients=np.array([1, 2, 3]))

    def test_standalone_integrate(self, domain):
        """Test standalone function integration."""
        f = Function(domain, evaluate_callable=lambda x: x)
        result = f.integrate(method="simpson", n_points=1000)
        np.testing.assert_allclose(result, 0.5, rtol=1e-6)

    def test_standalone_copy(self, domain):
        """Test copying standalone function."""
        f = Function(domain, evaluate_callable=np.cos, name="cosine")
        f2 = f.copy()
        assert f2.is_attached is False
        assert f2.function_domain == domain
        assert f2.name == "cosine"
        np.testing.assert_allclose(f(0.5), f2(0.5))


class TestFunctionAttachment:
    """Test attaching standalone functions to spaces."""

    @pytest.fixture
    def domain(self):
        return IntervalDomain(0, 1)

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_attach_to_space(self, domain, space):
        """Test attaching standalone function to a space."""
        f = Function(domain, evaluate_callable=lambda x: x**2)
        assert f.is_attached is False

        f_attached = f.attach_to_space(space)
        assert f_attached.is_attached is True
        assert f_attached.space is space
        # Original unchanged
        assert f.is_attached is False

    def test_attach_in_place(self, domain, space):
        """Test in-place attachment."""
        f = Function(domain, evaluate_callable=np.sin)
        f.attach_to_space(space, copy=False)
        assert f.is_attached is True
        assert f.space is space

    def test_attach_preserves_evaluation(self, domain, space):
        """Test that attachment preserves function values."""
        f = Function(domain, evaluate_callable=lambda x: x**3)
        f_attached = f.attach_to_space(space)
        x = np.linspace(0, 1, 20)
        np.testing.assert_allclose(f(x), f_attached(x))

    def test_attach_domain_mismatch_rejected(self):
        """Test attachment fails for mismatched domains."""
        domain = IntervalDomain(0, 1)
        space = MockSpace(IntervalDomain(0, 2))  # Different domain

        f = Function(domain, evaluate_callable=np.sin)
        with pytest.raises(ValueError, match="Domain mismatch"):
            f.attach_to_space(space)


class TestFunctionDetachment:
    """Test detaching functions from spaces."""

    @pytest.fixture
    def space(self):
        return MockSpace(IntervalDomain(0, 1))

    def test_detach_from_space(self, space):
        """Test detaching function from space."""
        f = Function(space, evaluate_callable=np.cos)
        assert f.is_attached is True

        f_detached = f.detach()
        assert f_detached.is_attached is False
        assert f_detached.space is None
        # Original unchanged
        assert f.is_attached is True

    def test_detach_in_place(self, space):
        """Test in-place detachment."""
        f = Function(space, evaluate_callable=np.sin)
        f.detach(copy=False)
        assert f.is_attached is False

    def test_detach_preserves_evaluation(self, space):
        """Test detachment preserves function values."""
        f = Function(space, evaluate_callable=lambda x: np.exp(-x))
        f_detached = f.detach()
        x = np.linspace(0, 1, 20)
        np.testing.assert_allclose(f(x), f_detached(x))

    def test_detach_coefficient_function_rejected(self, space):
        """Test that coefficient-based functions cannot be detached."""
        f = Function(space, coefficients=np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="Cannot detach"):
            f.detach()


class TestStandaloneArithmetic:
    """Test arithmetic operations with standalone functions."""

    @pytest.fixture
    def domain(self):
        return IntervalDomain(0, 1)

    def test_standalone_add(self, domain):
        """Test adding standalone functions."""
        f = Function(domain, evaluate_callable=lambda x: x)
        g = Function(domain, evaluate_callable=lambda x: x**2)
        h = f + g
        assert h.is_attached is False
        x = 0.5
        np.testing.assert_allclose(h(x), x + x**2)

    def test_standalone_scalar_mul(self, domain):
        """Test scalar multiplication of standalone function."""
        f = Function(domain, evaluate_callable=lambda x: x)
        h = 3 * f
        assert h.is_attached is False
        assert h(0.5) == 1.5

    def test_standalone_neg(self, domain):
        """Test negation of standalone function."""
        f = Function(domain, evaluate_callable=lambda x: x)
        g = -f
        assert g.is_attached is False
        assert g(0.5) == -0.5

    def test_mixed_attached_unattached_add(self, domain):
        """Test adding attached + unattached functions."""
        space = MockSpace(domain)
        f = Function(space, evaluate_callable=lambda x: x)
        g = Function(domain, evaluate_callable=lambda x: x**2)

        h = f + g
        # Result should be attached (one had a space)
        assert h.is_attached is True
        x = 0.5
        np.testing.assert_allclose(h(x), x + x**2)

    def test_mixed_arithmetic_preserves_space(self, domain):
        """Test that arithmetic with space preserves it when appropriate."""
        space = MockSpace(domain)
        f = Function(space, evaluate_callable=lambda x: x)
        g = Function(space, evaluate_callable=lambda x: 2*x)

        h = f + g
        assert h.space is space  # Same space preserved

