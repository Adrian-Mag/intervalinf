"""
Baseline correctness and robustness tests for SOLAOperator.

The tests here establish the behavioral baseline described in the Phase 1
audit before any optimization work occurs.  Nothing in this module should
change production SOLAOperator behavior.

Coverage includes:
- Analytic forward-integral checks (constant, polynomial, trigonometric kernels)
- Linearity of the forward map
- Adjoint-consistency checks (⟨G(f), y⟩_D = ⟨f, G*(y)⟩_M)
- Provider-backed kernels (SineFunctionProvider, BumpFunctionProvider)
- Direct Function-list kernels
- Direct callable-list kernels
- cache_kernels behavior
- Integration-method coverage (simpson, trapz)
- Compact-support behavior (BumpFunctionProvider kernels)
- for_direct_sum construction
- Gram-matrix symmetry
- get_cache_info / clear_cache accessors
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.operators import SOLAOperator
from pygeoinf.hilbert_space import EuclideanSpace


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_domain():
    return IntervalDomain(0.0, 1.0)


@pytest.fixture
def lebesgue_space(unit_domain):
    """Minimal Lebesgue space on [0, 1] with cosine basis."""
    return Lebesgue(20, unit_domain, basis="cosine")


@pytest.fixture
def simple_sola(lebesgue_space):
    """
    SOLAOperator with two analytic kernels on [0,1]:
        k_0(x) = 1          →  G(f)[0] = ∫₀¹ f(x) dx
        k_1(x) = 2x         →  G(f)[1] = 2 ∫₀¹ x f(x) dx
    """
    domain = lebesgue_space.function_domain
    D = EuclideanSpace(2)
    kernels = [
        Function(domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0),
        Function(domain, evaluate_callable=lambda x: 2.0 * x),
    ]
    return SOLAOperator(
        lebesgue_space, D, kernels=kernels,
        integration_config=IntegrationConfig(method="simpson", n_points=2000),
    )


# ---------------------------------------------------------------------------
# 1. Analytic forward-integral checks
# ---------------------------------------------------------------------------

class TestForwardAnalytic:
    """G(f) = (∫ f k_i dx)_i for analytic kernels."""

    def test_constant_kernel_constant_input(self, lebesgue_space, unit_domain):
        """k=1, f=1 → G(f) = [∫₀¹ 1 dx] = [1.0]."""
        D = EuclideanSpace(1)
        k = [Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)]
        G = SOLAOperator(lebesgue_space, D, kernels=k,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        result = G(f)
        assert_allclose(result, [1.0], rtol=1e-4)

    def test_constant_kernel_polynomial_input(self, lebesgue_space, unit_domain):
        """k=1, f=x → G(f) = [∫₀¹ x dx] = [0.5]."""
        D = EuclideanSpace(1)
        k = [Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)]
        G = SOLAOperator(lebesgue_space, D, kernels=k,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        assert_allclose(result, [0.5], rtol=1e-4)

    def test_linear_kernel_polynomial_input(self, lebesgue_space, unit_domain):
        """k=2x, f=x² → G(f) = 2∫₀¹ x³ dx = 2·(1/4) = 0.5."""
        D = EuclideanSpace(1)
        k = [Function(unit_domain, evaluate_callable=lambda x: 2.0 * x)]
        G = SOLAOperator(lebesgue_space, D, kernels=k,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x ** 2)
        result = G(f)
        assert_allclose(result, [0.5], rtol=1e-4)

    def test_two_kernels_analytic(self, simple_sola, lebesgue_space, unit_domain):
        """Two kernels: k₀=1, k₁=2x and f=x.
        G(f)[0] = ∫₀¹ x dx = 0.5
        G(f)[1] = 2∫₀¹ x² dx = 2/3
        """
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = simple_sola(f)
        assert_allclose(result[0], 0.5, rtol=1e-4)
        assert_allclose(result[1], 2.0 / 3.0, rtol=1e-4)

    def test_trig_kernel_trig_input(self, lebesgue_space, unit_domain):
        """k=sin(πx), f=sin(πx) → G(f) = ∫₀¹ sin²(πx) dx = 0.5."""
        D = EuclideanSpace(1)
        k = [Function(unit_domain, evaluate_callable=lambda x: np.sin(np.pi * x))]
        G = SOLAOperator(lebesgue_space, D, kernels=k,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        result = G(f)
        assert_allclose(result, [0.5], rtol=1e-4)

    def test_zero_input_gives_zero(self, simple_sola, lebesgue_space):
        """G(0) = 0."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0)
        result = simple_sola(f)
        assert_allclose(result, np.zeros(2), atol=1e-12)

    def test_output_shape(self, lebesgue_space, unit_domain):
        """Output has shape matching EuclideanSpace dimension."""
        N_d = 5
        D = EuclideanSpace(N_d)
        kernels = [Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
                   for _ in range(N_d)]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="simpson", n_points=500))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        assert result.shape == (N_d,)


# ---------------------------------------------------------------------------
# 2. Linearity
# ---------------------------------------------------------------------------

class TestLinearity:
    """G(αf + βg) = αG(f) + βG(g)."""

    def test_linearity_scalar_multiple(self, simple_sola, lebesgue_space):
        """G(3f) = 3·G(f)."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(2 * np.pi * x))
        alpha = 3.0
        result_scaled = simple_sola(Function(lebesgue_space, evaluate_callable=lambda x: alpha * np.sin(2 * np.pi * x)))
        result_direct = alpha * simple_sola(f)
        # Use atol to handle elements that are numerically near zero
        assert_allclose(result_scaled, result_direct, rtol=1e-5, atol=1e-10)

    def test_linearity_superposition(self, simple_sola, lebesgue_space):
        """G(f + g) = G(f) + G(g)."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        g = Function(lebesgue_space, evaluate_callable=lambda x: x ** 2)
        fg = Function(lebesgue_space, evaluate_callable=lambda x: x + x ** 2)
        assert_allclose(simple_sola(fg), simple_sola(f) + simple_sola(g), rtol=1e-6)

    def test_linearity_negative(self, simple_sola, lebesgue_space):
        """G(-f) = -G(f)."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.cos(np.pi * x))
        neg_f = Function(lebesgue_space, evaluate_callable=lambda x: -np.cos(np.pi * x))
        assert_allclose(simple_sola(neg_f), -simple_sola(f), rtol=1e-8)


# ---------------------------------------------------------------------------
# 3. Adjoint consistency
# ---------------------------------------------------------------------------

class TestAdjointConsistency:
    """
    Verify ⟨G(f), y⟩_D = ⟨f, G*(y)⟩_M for various f, y.

    In pygeoinf: D = EuclideanSpace, so ⟨·,·⟩_D is the dot product.
    G*(y) is a Function and ⟨·,·⟩_M is the L² inner product (integration).
    """

    @pytest.fixture
    def sola_2kernels(self, lebesgue_space, unit_domain):
        """Two orthonormal-ish kernels for adjoint testing."""
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: np.sin(np.pi * x)),
            Function(unit_domain, evaluate_callable=lambda x: np.cos(np.pi * x)),
        ]
        return SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )

    def _check_adjoint(self, G, f, y, rtol=5e-4):
        """Check ⟨G(f), y⟩_D = ⟨f, G*(y)⟩_M."""
        Gf = G(f)
        # ⟨G(f), y⟩_D = dot product for EuclideanSpace
        inner_D = float(np.dot(Gf, y))
        # G*(y) is a Function
        Gstar_y = G.adjoint(y)
        # ⟨f, G*(y)⟩_M = ∫ f(x) * G*(y)(x) dx
        product_fn = Function(
            G.domain.function_domain,
            evaluate_callable=lambda x, _f=f, _g=Gstar_y: _f.evaluate(x) * _g.evaluate(x)
        )
        inner_M = product_fn.integrate(method="simpson", n_points=2000)
        assert_allclose(inner_D, inner_M, rtol=rtol,
                        err_msg=f"Adjoint inconsistency: <Gf,y>={inner_D}, <f,G*y>={inner_M}")

    def test_adjoint_constant_f_unit_y(self, sola_2kernels, lebesgue_space):
        """f=1, y=[1,0]: both sides should equal ∫ sin(πx) dx = 2/π."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        y = np.array([1.0, 0.0])
        self._check_adjoint(sola_2kernels, f, y)

    def test_adjoint_poly_f_both_y(self, sola_2kernels, lebesgue_space):
        """f=x², y=[1, -1]."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: x ** 2)
        y = np.array([1.0, -1.0])
        self._check_adjoint(sola_2kernels, f, y)

    def test_adjoint_trig_f_random_y(self, sola_2kernels, lebesgue_space):
        """f=sin(2πx), y=[0.3, -0.7]."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(2 * np.pi * x))
        y = np.array([0.3, -0.7])
        self._check_adjoint(sola_2kernels, f, y)

    def test_adjoint_zero_y(self, sola_2kernels, lebesgue_space):
        """y=0 → G*(0) == 0 function, inner product == 0."""
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        y = np.zeros(2)
        Gf = sola_2kernels(f)
        inner_D = float(np.dot(Gf, y))
        Gstar_y = sola_2kernels.adjoint(y)
        product_fn = Function(
            sola_2kernels.domain.function_domain,
            evaluate_callable=lambda x, _f=f, _g=Gstar_y: _f.evaluate(x) * _g.evaluate(x)
        )
        inner_M = product_fn.integrate(method="simpson", n_points=1000)
        assert_allclose(inner_D, 0.0, atol=1e-12)
        assert_allclose(inner_M, 0.0, atol=1e-4)

    def test_adjoint_linearity_in_y(self, sola_2kernels, lebesgue_space):
        """G*(α·y₁ + β·y₂) = α·G*(y₁) + β·G*(y₂) at a sample point."""
        alpha, beta = 2.0, -0.5
        y1 = np.array([1.0, 0.0])
        y2 = np.array([0.0, 1.0])
        y_combo = alpha * y1 + beta * y2

        x_test = 0.4
        Gstar_combo = sola_2kernels.adjoint(y_combo)
        Gstar_y1 = sola_2kernels.adjoint(y1)
        Gstar_y2 = sola_2kernels.adjoint(y2)
        lhs = Gstar_combo.evaluate(x_test)
        rhs = alpha * Gstar_y1.evaluate(x_test) + beta * Gstar_y2.evaluate(x_test)
        assert_allclose(lhs, rhs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Provider-backed kernels
# ---------------------------------------------------------------------------

class TestProviderBackedKernels:
    """SOLAOperator constructed from IndexedFunctionProvider."""

    def test_sine_provider_forward(self, lebesgue_space, unit_domain):
        """Using SineFunctionProvider: k_i = √2 sin(iπx).
        For f(x) = sin(πx) = φ₁/√2, G(f)[0] = ∫ sin(πx)·√2·sin(πx) dx = 1/√2.
        More precisely: G(f)[0] = ∫₀¹ sin(πx)·(√2·sin(πx)) dx = √2·(1/2) = 1/√2.
        """
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        result = G(f)
        # G(f)[0] = ∫₀¹ sin(πx) · √2 sin(πx) dx = √2/2
        expected_0 = np.sqrt(2.0) / 2.0
        assert_allclose(result[0], expected_0, rtol=1e-4)

    def test_bump_provider_forward(self, lebesgue_space, unit_domain):
        """Using BumpFunctionProvider: bump kernels are normalized bump functions.
        For f=1: G(f)[i] ≈ ∫ bump_i(x) dx ≈ 1 (normalized bumps have ∫=1).
        """
        from intervalinf.providers import BumpFunctionProvider
        N_d = 3
        D = EuclideanSpace(N_d)
        centers = np.linspace(0.2, 0.8, N_d)
        provider = BumpFunctionProvider(unit_domain, centers=centers, default_width=0.15)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        result = G(f)
        # Each normalized bump integrates to 1 over its support; f=1 so G(f)[i] ≈ 1
        assert_allclose(result, np.ones(N_d), rtol=2e-3)

    def test_provider_output_shape(self, lebesgue_space, unit_domain):
        """Provider-backed operator output shape equals codomain dim."""
        from intervalinf.providers import CosineFunctionProvider
        N_d = 5
        D = EuclideanSpace(N_d)
        provider = CosineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=500))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        assert result.shape == (N_d,)

    def test_provider_adjoint_consistency(self, lebesgue_space, unit_domain):
        """Adjoint-consistency for a provider-backed operator."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(2)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x * (1 - x))
        y = np.array([1.0, -1.0])
        Gf = G(f)
        inner_D = float(np.dot(Gf, y))
        Gstar_y = G.adjoint(y)
        product_fn = Function(
            unit_domain,
            evaluate_callable=lambda x, _f=f, _g=Gstar_y: _f.evaluate(x) * _g.evaluate(x)
        )
        inner_M = product_fn.integrate(method="simpson", n_points=2000)
        assert_allclose(inner_D, inner_M, rtol=5e-4)


# ---------------------------------------------------------------------------
# 5. Direct callable-list kernels
# ---------------------------------------------------------------------------

class TestCallableKernels:
    """SOLAOperator constructed from a list of plain callables."""

    def test_callable_list_construction(self, lebesgue_space):
        """Callable list is accepted and converted to Functions."""
        D = EuclideanSpace(2)
        kernels = [lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                   lambda x: x]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels)
        assert G.N_d == 2

    def test_callable_forward_correctness(self, lebesgue_space, unit_domain):
        """Forward values match analytic integrals for callable kernels.
        k₀ = lambda x: 1,  f = x  →  G(f)[0] = 0.5
        """
        D = EuclideanSpace(1)
        G = SOLAOperator(lebesgue_space, D,
                         kernels=[lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0],
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        assert_allclose(result, [0.5], rtol=1e-4)

    def test_callable_mismatch_raises(self, lebesgue_space):
        """Mismatched number of callables and codomain dim raises ValueError."""
        D = EuclideanSpace(3)
        with pytest.raises(ValueError, match="Number of kernels"):
            SOLAOperator(lebesgue_space, D, kernels=[lambda x: x])


# ---------------------------------------------------------------------------
# 6. Direct Function-list kernels
# ---------------------------------------------------------------------------

class TestFunctionListKernels:
    """SOLAOperator constructed from a list of Function objects."""

    def test_function_list_construction(self, lebesgue_space, unit_domain):
        """Function list is accepted directly."""
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: x),
            Function(unit_domain, evaluate_callable=lambda x: x ** 2),
        ]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels)
        assert G.N_d == 2

    def test_function_list_forward(self, lebesgue_space, unit_domain):
        """k₀=x, k₁=x², f=1 → G(f) = [0.5, 1/3]."""
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: x),
            Function(unit_domain, evaluate_callable=lambda x: x ** 2),
        ]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        result = G(f)
        assert_allclose(result[0], 0.5, rtol=1e-4)
        assert_allclose(result[1], 1.0 / 3.0, rtol=1e-4)

    def test_function_list_mismatch_raises(self, lebesgue_space, unit_domain):
        """Wrong number of Functions raises ValueError."""
        D = EuclideanSpace(3)
        kernels = [Function(unit_domain, evaluate_callable=lambda x: x)]
        with pytest.raises(ValueError, match="Number of kernels"):
            SOLAOperator(lebesgue_space, D, kernels=kernels)


# ---------------------------------------------------------------------------
# 7. Caching behavior
# ---------------------------------------------------------------------------

class TestCaching:
    """cache_kernels=True / False behavior."""

    def test_cache_disabled_by_default(self, simple_sola):
        """Default: cache_kernels=False, get_cache_info reflects this."""
        info = simple_sola.get_cache_info()
        assert info["caching_enabled"] is False

    def test_cache_enabled_info(self, lebesgue_space, unit_domain):
        """cache_kernels=True: get_cache_info shows enabled state."""
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: x),
            Function(unit_domain, evaluate_callable=lambda x: 1 - x),
        ]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels, cache_kernels=True)
        info = G.get_cache_info()
        assert info["caching_enabled"] is True

    def test_cache_populated_after_get_kernel(self, lebesgue_space, unit_domain):
        """After get_kernel(i) calls, cached_functions > 0 when provider-backed."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider, cache_kernels=True)
        # Access two kernels
        _ = G.get_kernel(0)
        _ = G.get_kernel(2)
        info = G.get_cache_info()
        assert info["cached_functions"] == 2

    def test_clear_cache(self, lebesgue_space, unit_domain):
        """clear_cache() empties the kernel cache."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(2)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider, cache_kernels=True)
        _ = G.get_kernel(0)
        _ = G.get_kernel(1)
        G.clear_cache()
        info = G.get_cache_info()
        assert info["cached_functions"] == 0

    def test_cache_does_not_change_result(self, lebesgue_space, unit_domain):
        """Cache-on and cache-off produce identical forward values."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G_no_cache = SOLAOperator(lebesgue_space, D, kernels=provider, cache_kernels=False, integration_config=cfg)
        G_cache = SOLAOperator(lebesgue_space, D, kernels=provider, cache_kernels=True, integration_config=cfg)
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        # First call populates the cache
        r1 = G_cache(f)
        # Second call uses cache
        r2 = G_cache(f)
        r_no_cache = G_no_cache(f)
        assert_allclose(r1, r_no_cache, rtol=1e-10)
        assert_allclose(r1, r2, rtol=1e-10)


# ---------------------------------------------------------------------------
# 8. Integration method coverage
# ---------------------------------------------------------------------------

class TestIntegrationMethods:
    """
    Forward results are consistent across fixed-grid and adaptive methods.

    Phase 3 makes ``quad`` a supported legacy alias for the canonical
    adaptive path.
    """

    @pytest.fixture
    def test_f(self, lebesgue_space):
        return Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))

    @pytest.fixture
    def unit_kernels(self, lebesgue_space, unit_domain):
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0),
            Function(unit_domain, evaluate_callable=lambda x: x),
        ]
        return lebesgue_space, D, kernels

    def test_simpson_gives_analytic_result(self, unit_kernels, test_f):
        M, D, kernels = unit_kernels
        G = SOLAOperator(M, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        result = G(test_f)
        # ∫₀¹ sin(πx) dx = 2/π
        assert_allclose(result[0], 2.0 / np.pi, rtol=1e-4)

    def test_trapz_gives_analytic_result(self, unit_kernels, test_f):
        M, D, kernels = unit_kernels
        G = SOLAOperator(M, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="trapz", n_points=2000))
        result = G(test_f)
        assert_allclose(result[0], 2.0 / np.pi, rtol=5e-4)

    def test_simpson_trapz_close(self, unit_kernels, test_f):
        """Simpson and trapz at 2000 points agree to high relative accuracy."""
        M, D, kernels = unit_kernels
        G_simp = SOLAOperator(M, D, kernels=kernels,
                               integration_config=IntegrationConfig(method="simpson", n_points=2000))
        G_trapz = SOLAOperator(M, D, kernels=kernels,
                                integration_config=IntegrationConfig(method="trapz", n_points=2000))
        r_simp = G_simp(test_f)
        r_trapz = G_trapz(test_f)
        assert_allclose(r_simp, r_trapz, rtol=1e-3)

    def test_quad_works_as_adaptive_alias(self, unit_kernels, test_f):
        """
        Phase 3: 'quad' in IntegrationConfig now aliases the adaptive
        (scipy.integrate.quad) path.  G(f) must succeed and match the
        analytic result.
        """
        M, D, kernels = unit_kernels
        G = SOLAOperator(M, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="quad", n_points=500))
        # Must not raise; ∫₀¹ sin(πx) dx = 2/π
        result = G(test_f)
        assert_allclose(result[0], 2.0 / np.pi, rtol=1e-4)

    def test_adaptive_method_works(self, unit_kernels, test_f):
        """
        Phase 3: 'adaptive' is now the canonical name for the
        scipy.integrate.quad path.  It must produce correct results.
        """
        M, D, kernels = unit_kernels
        G = SOLAOperator(M, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="adaptive", n_points=500))
        result = G(test_f)
        assert_allclose(result[0], 2.0 / np.pi, rtol=1e-4)

    def test_quad_and_adaptive_give_same_result(self, unit_kernels, test_f):
        """Both alias names must produce identical outputs."""
        M, D, kernels = unit_kernels
        G_quad = SOLAOperator(M, D, kernels=kernels,
                              integration_config=IntegrationConfig(method="quad", n_points=500))
        G_adapt = SOLAOperator(M, D, kernels=kernels,
                               integration_config=IntegrationConfig(method="adaptive", n_points=500))
        assert_allclose(G_quad(test_f), G_adapt(test_f), rtol=1e-10)

    def test_adaptive_agrees_with_simpson(self, unit_kernels, test_f):
        """adaptive/quad results match simpson at sufficient accuracy."""
        M, D, kernels = unit_kernels
        G_adapt = SOLAOperator(M, D, kernels=kernels,
                               integration_config=IntegrationConfig(method="adaptive"))
        G_simp = SOLAOperator(M, D, kernels=kernels,
                              integration_config=IntegrationConfig(method="simpson", n_points=5000))
        assert_allclose(G_adapt(test_f), G_simp(test_f), rtol=1e-5)


# ---------------------------------------------------------------------------
# 8b. IntegrationConfig property tests (Phase 3)
# ---------------------------------------------------------------------------

class TestIntegrationConfigProperties:
    """Unit tests for the is_fixed_grid / is_adaptive properties added in Phase 3."""

    def test_simpson_is_fixed_grid(self):
        assert IntegrationConfig(method="simpson").is_fixed_grid is True

    def test_trapz_is_fixed_grid(self):
        assert IntegrationConfig(method="trapz").is_fixed_grid is True

    def test_adaptive_is_not_fixed_grid(self):
        assert IntegrationConfig(method="adaptive").is_fixed_grid is False

    def test_quad_is_not_fixed_grid(self):
        assert IntegrationConfig(method="quad").is_fixed_grid is False

    def test_is_adaptive_complement_of_is_fixed_grid(self):
        for method in ("simpson", "trapz", "adaptive", "quad"):
            cfg = IntegrationConfig(method=method)
            assert cfg.is_adaptive == (not cfg.is_fixed_grid)

    def test_fixed_grid_methods_constant(self):
        from intervalinf.core.config import FIXED_GRID_METHODS, ADAPTIVE_METHODS
        assert "simpson" in FIXED_GRID_METHODS
        assert "trapz" in FIXED_GRID_METHODS
        assert "adaptive" in ADAPTIVE_METHODS
        assert "quad" in ADAPTIVE_METHODS
        assert FIXED_GRID_METHODS.isdisjoint(ADAPTIVE_METHODS)

    def test_adaptive_quad_preset(self):
        """IntegrationConfig.adaptive_quad() returns an adaptive-method config."""
        cfg = IntegrationConfig.adaptive_quad()
        assert cfg.is_adaptive
        assert cfg.method in ("adaptive", "quad")


# ---------------------------------------------------------------------------
# 9. Compact-support behavior
# ---------------------------------------------------------------------------

class TestCompactSupportBehavior:
    """
    Compact-support correctness tests.

    Phase 3 adds support-propagation inside _apply_kernels: when both the
    input function and the kernel carry compact-support metadata, the
    integration range is narrowed to the support intersection.  These tests
    verify:
    1. Existing correctness is preserved (original results unchanged).
    2. Disjoint-support case returns exactly 0.
    3. Results with explicit support metadata match results without it.
    """

    def test_bump_kernel_localized_at_left(self, lebesgue_space, unit_domain):
        """
        Bump kernel centered at 0.1 (width 0.15): for f=1 the integral is ≈1
        (bump is normalized).  For f=x the integral should be ≈ 0.1 (center).
        """
        from intervalinf.providers import BumpFunctionProvider
        D = EuclideanSpace(1)
        centers = np.array([0.1])
        provider = BumpFunctionProvider(unit_domain, centers=centers, default_width=0.15)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f_one = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        f_x = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result_one = G(f_one)
        result_x = G(f_x)
        # Bump is normalized → ∫ k·1 dx  = 1
        assert_allclose(result_one, [1.0], rtol=2e-3)
        # ∫ k(x)·x dx ≈ center of bump ≈ 0.1
        assert_allclose(result_x[0], 0.1, atol=0.02)

    def test_bump_kernel_localized_at_right(self, lebesgue_space, unit_domain):
        """
        Bump centered at 0.9: for f=x the integral should be ≈ 0.9.
        """
        from intervalinf.providers import BumpFunctionProvider
        D = EuclideanSpace(1)
        centers = np.array([0.9])
        provider = BumpFunctionProvider(unit_domain, centers=centers, default_width=0.15)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f_x = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f_x)
        assert_allclose(result[0], 0.9, atol=0.02)

    def test_bump_kernels_disjoint_locality(self, lebesgue_space, unit_domain):
        """
        Two bump kernels at 0.2 and 0.8.  For f(x) = x:
        G(f)[0] ≈ 0.2,  G(f)[1] ≈ 0.8
        """
        from intervalinf.providers import BumpFunctionProvider
        D = EuclideanSpace(2)
        centers = np.array([0.2, 0.8])
        provider = BumpFunctionProvider(unit_domain, centers=centers, default_width=0.15)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        assert_allclose(result[0], 0.2, atol=0.02)
        assert_allclose(result[1], 0.8, atol=0.02)

    def test_disjoint_support_explicit_kernel_returns_zero(self, lebesgue_space, unit_domain):
        """
        Phase 3: kernel with compact support entirely in [0.7, 1.0], function
        with compact support entirely in [0.0, 0.3].  Support intersection is
        empty → G(f)[0] must be exactly 0.
        """
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.where(
                (np.asarray(x) >= 0.7) & (np.asarray(x) <= 1.0), 1.0, 0.0
            ),
            support=[(0.7, 1.0)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(lebesgue_space, D, kernels=[k],
                         integration_config=IntegrationConfig(method="simpson", n_points=1000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x,
                     support=[(0.0, 0.3)])
        result = G(f)
        assert_allclose(result[0], 0.0, atol=1e-12)

    def test_support_propagation_matches_no_support_metadata(self, lebesgue_space, unit_domain):
        """
        Phase 3: a kernel with support=[0.2, 0.8] should give the same
        forward value as an identical kernel without support metadata, because
        both the integrand value and the integration range are equivalent when
        the kernel is zero outside its support.
        """
        def k_callable(x):
            x = np.asarray(x)
            return np.where((x >= 0.2) & (x <= 0.8), np.sin(np.pi * x), 0.0)

        k_with_support = Function(
            unit_domain,
            evaluate_callable=k_callable,
            support=[(0.2, 0.8)],
        )
        k_no_support = Function(
            unit_domain,
            evaluate_callable=k_callable,
        )
        D = EuclideanSpace(2)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G_with = SOLAOperator(lebesgue_space, D, kernels=[k_with_support, k_no_support], integration_config=cfg)
        G_none = SOLAOperator(lebesgue_space, D, kernels=[k_no_support, k_no_support], integration_config=cfg)
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        # Both kernels are identical callables, so results should be identical
        r_with = G_with(f)
        r_none = G_none(f)
        assert_allclose(r_with, r_none, rtol=1e-6)


# ---------------------------------------------------------------------------
# 10. Gram matrix
# ---------------------------------------------------------------------------

class TestGramMatrix:
    """The Gram matrix G_ij = ∫ k_i(x) k_j(x) dx should be symmetric."""

    def test_gram_symmetric_trig(self, lebesgue_space, unit_domain):
        """Gram matrix is symmetric for trig kernels."""
        from intervalinf.providers import SineFunctionProvider
        N_d = 3
        D = EuclideanSpace(N_d)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        gram = G.compute_gram_matrix()
        assert gram.shape == (N_d, N_d)
        assert_allclose(gram, gram.T, atol=1e-6, err_msg="Gram matrix not symmetric")

    def test_gram_orthonormal_sine_basis(self, lebesgue_space, unit_domain):
        """
        Normalized sine functions are orthonormal on [0,1]:
        G_ij = δ_ij.
        """
        from intervalinf.providers import SineFunctionProvider
        N_d = 3
        D = EuclideanSpace(N_d)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=2000))
        gram = G.compute_gram_matrix()
        assert_allclose(gram, np.eye(N_d), atol=1e-4)

    def test_gram_diagonal_positive(self, lebesgue_space, unit_domain):
        """Diagonal of Gram matrix (kernel self-integrals) is positive."""
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: x + 0.1),
            Function(unit_domain, evaluate_callable=lambda x: np.sin(np.pi * x) + 0.5),
        ]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels,
                         integration_config=IntegrationConfig(method="simpson", n_points=1000))
        gram = G.compute_gram_matrix()
        assert np.all(np.diag(gram) > 0)


# ---------------------------------------------------------------------------
# 11. for_direct_sum construction
# ---------------------------------------------------------------------------

class TestForDirectSum:
    """Smoke tests for SOLAOperator.for_direct_sum."""

    def test_for_direct_sum_constructs(self, unit_domain):
        """for_direct_sum should return a RowLinearOperator."""
        from pygeoinf.direct_sum import RowLinearOperator
        from intervalinf.providers import SineFunctionProvider
        split_domain = unit_domain.split_at_discontinuities([0.5])
        sub_spaces = [Lebesgue(10, d, basis="cosine") for d in split_domain]
        from intervalinf.spaces.lebesgue import LebesgueSpaceDirectSum
        M = LebesgueSpaceDirectSum(sub_spaces)
        D = EuclideanSpace(2)
        # Use a provider on the full domain - restrict is called per subspace
        # Use SineFunctionProvider on the first sub_space for simplicity
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator.for_direct_sum(
            M, D, kernels=provider,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        assert isinstance(G, RowLinearOperator)

    def test_for_direct_sum_forward_shape(self, unit_domain):
        """Output shape matches codomain dim for for_direct_sum."""
        from intervalinf.providers import SineFunctionProvider
        split_domain = unit_domain.split_at_discontinuities([0.5])
        sub_spaces = [Lebesgue(10, d, basis="cosine") for d in split_domain]
        from intervalinf.spaces.lebesgue import LebesgueSpaceDirectSum
        M = LebesgueSpaceDirectSum(sub_spaces)
        D = EuclideanSpace(2)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator.for_direct_sum(M, D, kernels=provider,
                                        integration_config=IntegrationConfig(method="simpson", n_points=500))
        # In pygeoinf direct-sum spaces, vectors are plain Python lists of per-subspace elements
        zeros = [Function(s, evaluate_callable=lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0)
                 for s in sub_spaces]
        result = G(zeros)
        assert result.shape == (D.dim,)

    def test_for_direct_sum_non_direct_sum_raises(self, lebesgue_space, unit_domain):
        """Passing a plain Lebesgue space as domain raises TypeError."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(2)
        provider = SineFunctionProvider(unit_domain)
        with pytest.raises(TypeError):
            SOLAOperator.for_direct_sum(lebesgue_space, D, kernels=provider)


# ---------------------------------------------------------------------------
# 12. Miscellaneous / robustness
# ---------------------------------------------------------------------------

class TestMiscellaneous:
    """Additional robustness checks."""

    def test_str_representation_no_errors(self, simple_sola):
        """__str__ should not raise."""
        s = str(simple_sola)
        assert "SOLAOperator" in s

    def test_get_kernels_materializes_list(self, simple_sola):
        """get_kernels() returns a list of length N_d."""
        kernels = simple_sola.get_kernels()
        assert len(kernels) == simple_sola.N_d

    def test_domain_codomain_dims(self, lebesgue_space, unit_domain):
        """domain and codomain dims match construction arguments."""
        N_d = 4
        D = EuclideanSpace(N_d)
        kernels = [Function(unit_domain, evaluate_callable=lambda x: x)
                   for _ in range(N_d)]
        G = SOLAOperator(lebesgue_space, D, kernels=kernels)
        assert G.N_d == N_d
        assert G.codomain.dim == N_d

    def test_single_kernel_single_output(self, lebesgue_space, unit_domain):
        """N_d=1 case: output is a length-1 array."""
        D = EuclideanSpace(1)
        G = SOLAOperator(lebesgue_space, D,
                         kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
                         integration_config=IntegrationConfig(method="simpson", n_points=1000))
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        assert result.shape == (1,)
        # ∫₀¹ x·x dx = 1/3
        assert_allclose(result[0], 1.0 / 3.0, rtol=1e-4)

    def test_large_n_d_forward(self, lebesgue_space, unit_domain):
        """Smoke test: N_d=50 without errors."""
        from intervalinf.providers import SineFunctionProvider
        N_d = 50
        D = EuclideanSpace(N_d)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(lebesgue_space, D, kernels=provider,
                         integration_config=IntegrationConfig(method="simpson", n_points=500))
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        result = G(f)
        assert result.shape == (N_d,)
