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
- Phase 4: automatic batched fixed-grid forward path
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


# ---------------------------------------------------------------------------
# 13. Phase 4 – Automatic batched fixed-grid forward path
# ---------------------------------------------------------------------------

class TestFastPathFixedGrid:
    """
    Tests for the Phase 4 automatic batched fixed-grid forward path.

    The fast path is triggered automatically when
    ``self.integration.is_fixed_grid`` is True (methods 'simpson' and
    'trapz').  It builds the quadrature mesh once, evaluates f once on the
    shared mesh, assembles a kernel matrix, and integrates all products with
    a single batched scipy call.

    Correctness contract: results must match the established Phase 3
    semantics. Full-domain fixed-grid cases use the batched shared-mesh
    path, while support-restricted cases fall back per kernel so narrowed
    support quadrature behavior is preserved.

    All tests inherit the shared fixtures defined at module level and
    deliberately reuse analytic cases from earlier classes to confirm the
    fast path does not change answers.
    """

    # ------------------------------------------------------------------
    # Analytic correctness – fast path gives expected integrals
    # ------------------------------------------------------------------

    def test_fast_path_simpson_single_kernel_analytic(
        self, lebesgue_space, unit_domain
    ):
        """k=1, f=x  →  G(f)[0] = 0.5  (fast simpson path)."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        assert_allclose(G(f), [0.5], rtol=1e-5)

    def test_fast_path_trapz_single_kernel_analytic(
        self, lebesgue_space, unit_domain
    ):
        """k=1, f=x  →  G(f)[0] ≈ 0.5  (fast trapz path)."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)],
            integration_config=IntegrationConfig(method="trapz", n_points=2000),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        assert_allclose(G(f), [0.5], rtol=1e-4)

    def test_fast_path_multi_kernel_analytic(self, lebesgue_space, unit_domain):
        """N_d=5 kernels; each G(f)[i] matches analytic integral.

        k_i(x) = x^i,  f(x) = 1
        G(f)[i] = ∫₀¹ x^i dx = 1 / (i+1)
        """
        N_d = 5
        D = EuclideanSpace(N_d)
        kernels = [
            Function(unit_domain, evaluate_callable=(lambda i: lambda x: x ** i)(i))
            for i in range(N_d)
        ]
        G = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=3000),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        result = G(f)
        expected = np.array([1.0 / (i + 1) for i in range(N_d)])
        assert_allclose(result, expected, rtol=1e-4)

    def test_fast_path_trig_kernel(self, lebesgue_space, unit_domain):
        """k = sin(πx), f = sin(πx) → G(f) = ∫₀¹ sin²(πx) dx = 0.5."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: np.sin(np.pi * x))],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        assert_allclose(G(f), [0.5], rtol=1e-5)

    # ------------------------------------------------------------------
    # Equivalence: fast path matches adaptive (reference) at high accuracy
    # ------------------------------------------------------------------

    def test_fast_path_matches_adaptive_reference(
        self, lebesgue_space, unit_domain
    ):
        """Fast path (simpson, 5000 pts) agrees with adaptive reference to 1e-5.

        Kernels chosen to avoid orthogonality to f so that all integrals are
        substantially non-zero, making relative-tolerance comparisons reliable.
        """
        D = EuclideanSpace(3)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: x),
            Function(unit_domain, evaluate_callable=lambda x: 1.0 - x),
            Function(unit_domain, evaluate_callable=lambda x: x * (1 - x)),
        ]
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(2 * np.pi * x))

        G_fast = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=5000),
        )
        G_ref = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="adaptive"),
        )
        assert_allclose(G_fast(f), G_ref(f), rtol=1e-5, atol=1e-12)

    def test_fast_path_trapz_matches_adaptive_reference(
        self, lebesgue_space, unit_domain
    ):
        """Fast path (trapz, 5000 pts) agrees with adaptive reference to 5e-4."""
        D = EuclideanSpace(2)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: np.sin(np.pi * x)),
            Function(unit_domain, evaluate_callable=lambda x: x),
        ]
        f = Function(lebesgue_space, evaluate_callable=lambda x: x ** 2)

        G_fast = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="trapz", n_points=5000),
        )
        G_ref = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="adaptive"),
        )
        assert_allclose(G_fast(f), G_ref(f), rtol=5e-4)

    # ------------------------------------------------------------------
    # Adaptive path is unchanged (generic path)
    # ------------------------------------------------------------------

    def test_adaptive_still_works_after_phase4(self, lebesgue_space, unit_domain):
        """Adaptive method still produces correct results (uses generic path)."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)],
            integration_config=IntegrationConfig(method="adaptive"),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        assert_allclose(G(f), [0.5], rtol=1e-6)

    # ------------------------------------------------------------------
    # Non-vectorized callable fallback
    # ------------------------------------------------------------------

    def test_fast_path_nonvectorized_kernel_fallback(
        self, lebesgue_space, unit_domain
    ):
        """A kernel that is not vectorized falls back gracefully.

        The callable raises TypeError for array input but handles scalars.
        The _eval_on_mesh helper must catch the failure and use per-point
        evaluation instead, giving the correct integral.
        """
        def scalar_only_kernel(x):
            if isinstance(x, np.ndarray) and x.ndim > 0:
                raise TypeError("not vectorized")
            return 1.0  # constant kernel k=1

        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=scalar_only_kernel)],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        result = G(f)
        # G(f)[0] = ∫₀¹ x · 1 dx = 0.5
        assert_allclose(result, [0.5], rtol=1e-4)

    def test_fast_path_nonvectorized_input_fallback(
        self, lebesgue_space, unit_domain
    ):
        """Input function f that is not vectorized also falls back gracefully."""
        def scalar_only_f(x):
            if isinstance(x, np.ndarray) and x.ndim > 0:
                raise TypeError("not vectorized")
            return float(x) ** 2  # f(x) = x²

        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain,
                              evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=scalar_only_f)
        result = G(f)
        # G(f)[0] = ∫₀¹ x² · 1 dx = 1/3
        assert_allclose(result, [1.0 / 3.0], rtol=1e-4)

    # ------------------------------------------------------------------
    # Support handling in the fast path
    # ------------------------------------------------------------------

    def test_fast_path_disjoint_support_returns_zero(
        self, lebesgue_space, unit_domain
    ):
        """Disjoint-support kernel still returns exactly 0 in the fast path."""
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.where(
                (np.asarray(x) >= 0.7) & (np.asarray(x) <= 1.0), 1.0, 0.0
            ),
            support=[(0.7, 1.0)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x,
                     support=[(0.0, 0.3)])
        result = G(f)
        assert_allclose(result[0], 0.0, atol=1e-12)

    def test_fast_path_compact_support_correct_value(
        self, lebesgue_space, unit_domain
    ):
        """Compact-support kernel gives correct integral in fast path.

        k supported on [0.4, 0.6], callable is 1.0 inside support.
        f = 1 → G(f)[0] = ∫_{0.4}^{0.6} 1 dx = 0.2.
        """
        def k_callable(x):
            x_arr = np.asarray(x)
            return np.where((x_arr >= 0.4) & (x_arr <= 0.6), 1.0, 0.0)

        k = Function(unit_domain, evaluate_callable=k_callable,
                     support=[(0.4, 0.6)])
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        result = G(f)
        assert_allclose(result[0], 0.2, rtol=1e-3)

    def test_fast_path_narrow_support_matches_generic(
        self, lebesgue_space, unit_domain
    ):
        """Narrow compact supports preserve the generic Phase 3 result.

        This guards against losing effective point density by integrating on
        the full-domain mesh when the support intersection is tiny.
        """
        support = [(0.499, 0.501)]

        def narrow_kernel(x):
            x_arr = np.asarray(x)
            return np.where(
                (x_arr >= 0.499) & (x_arr <= 0.501),
                1.0,
                0.0,
            )

        kernel = Function(
            unit_domain,
            evaluate_callable=narrow_kernel,
            support=support,
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space,
            D,
            kernels=[kernel],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x)
            if isinstance(x, np.ndarray)
            else 1.0,
            support=support,
        )
        fast_result = G(f)
        generic_result = G._apply_kernels_generic(f)
        assert_allclose(fast_result, generic_result, rtol=1e-12, atol=1e-12)

    # ------------------------------------------------------------------
    # Provider-backed kernels in the fast path
    # ------------------------------------------------------------------

    def test_fast_path_provider_backed_sine(self, lebesgue_space, unit_domain):
        """Provider-backed kernels give correct results via fast path.

        SineFunctionProvider k_0 = √2 sin(πx).
        f(x) = sin(πx) → G(f)[0] = ∫₀¹ sin(πx)·√2·sin(πx) dx = √2/2.
        """
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        result = G(f)
        assert_allclose(result[0], np.sqrt(2.0) / 2.0, rtol=1e-4)

    # ------------------------------------------------------------------
    # Large N_d – smoke test for batched path
    # ------------------------------------------------------------------

    def test_fast_path_large_nd_shape_and_correctness(
        self, lebesgue_space, unit_domain
    ):
        """N_d=30 batched run: shape is correct and all-zero input gives zeros."""
        from intervalinf.providers import SineFunctionProvider
        N_d = 30
        D = EuclideanSpace(N_d)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        # f = 0 → G(f) = 0
        f_zero = Function(lebesgue_space,
                          evaluate_callable=lambda x: np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0)
        result = G(f_zero)
        assert result.shape == (N_d,)
        assert_allclose(result, np.zeros(N_d), atol=1e-14)

    # ------------------------------------------------------------------
    # eval_on_mesh helper
    # ------------------------------------------------------------------

    def test_eval_on_mesh_vectorized_function(self, unit_domain):
        """_eval_on_mesh returns correct values for a vectorized Function."""
        f = Function(unit_domain, evaluate_callable=lambda x: x ** 2)
        xs = np.linspace(0.0, 1.0, 50)
        result = SOLAOperator._eval_on_mesh(f, xs)
        assert result.shape == xs.shape
        assert_allclose(result, xs ** 2, rtol=1e-12)

    def test_eval_on_mesh_nonvectorized_function(self, unit_domain):
        """_eval_on_mesh fallback works for non-vectorized Function."""
        def scalar_fn(x):
            if isinstance(x, np.ndarray) and x.ndim > 0:
                raise TypeError("not vectorized")
            return float(x) ** 2

        f = Function(unit_domain, evaluate_callable=scalar_fn)
        xs = np.linspace(0.0, 1.0, 50)
        result = SOLAOperator._eval_on_mesh(f, xs)
        assert result.shape == xs.shape
        assert_allclose(result, xs ** 2, rtol=1e-12)

    def test_eval_on_mesh_preserves_complex_dtype(self, unit_domain):
        """Complex-valued functions are not coerced to real in the fast path."""
        f = Function(
            unit_domain,
            evaluate_callable=lambda x: np.exp(1j * np.pi * np.asarray(x)),
        )
        xs = np.linspace(0.0, 1.0, 25)
        result = SOLAOperator._eval_on_mesh(f, xs)
        assert result.shape == xs.shape
        assert np.iscomplexobj(result)
        assert_allclose(result, np.exp(1j * np.pi * xs), rtol=1e-12, atol=1e-12)

    def test_fast_path_preserves_complex_forward_values(
        self, lebesgue_space, unit_domain
    ):
        """Fixed-grid batched forward path preserves complex-valued outputs."""
        D = EuclideanSpace(1)
        kernel = Function(
            unit_domain,
            evaluate_callable=lambda x: np.exp(1j * np.pi * np.asarray(x)),
        )
        G = SOLAOperator(
            lebesgue_space,
            D,
            kernels=[kernel],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x)
            if isinstance(x, np.ndarray)
            else 1.0,
        )
        result = G(f)
        expected = np.array([2j / np.pi])
        assert np.iscomplexobj(result)
        assert_allclose(result, expected, rtol=1e-4, atol=1e-10)

    def test_support_restricted_fixed_grid_preserves_complex_values(
        self, lebesgue_space, unit_domain
    ):
        """Support-restricted fixed-grid cases preserve complex outputs too."""
        support = [(0.25, 0.75)]
        D = EuclideanSpace(1)
        kernel = Function(
            unit_domain,
            evaluate_callable=lambda x: np.exp(1j * np.pi * np.asarray(x)),
            support=support,
        )
        G = SOLAOperator(
            lebesgue_space,
            D,
            kernels=[kernel],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x)
            if isinstance(x, np.ndarray)
            else 1.0,
            support=support,
        )
        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert np.iscomplexobj(result)
        assert_allclose(result, generic, rtol=1e-12, atol=1e-12)

    def test_support_restricted_nonvectorized_complex_kernel(
        self, lebesgue_space, unit_domain
    ):
        """Support-restricted non-vectorized complex kernels keep their phase."""
        support = [(0.25, 0.75)]

        def scalar_complex_kernel(x):
            if isinstance(x, np.ndarray) and x.ndim > 0:
                raise TypeError("not vectorized")
            return np.exp(1j * np.pi * float(x))

        D = EuclideanSpace(1)
        kernel = Function(
            unit_domain,
            evaluate_callable=scalar_complex_kernel,
            support=support,
        )
        G = SOLAOperator(
            lebesgue_space,
            D,
            kernels=[kernel],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x)
            if isinstance(x, np.ndarray)
            else 1.0,
            support=support,
        )
        result = G(f)
        expected = np.array(
            [
                (
                    np.exp(1j * np.pi * 0.75)
                    - np.exp(1j * np.pi * 0.25)
                )
                / (1j * np.pi)
            ]
        )
        assert np.iscomplexobj(result)
        assert_allclose(result, expected, rtol=1e-4, atol=1e-10)


# ---------------------------------------------------------------------------
# 14. Phase 5 – Reuse and caching for repeated workloads
# ---------------------------------------------------------------------------

class TestPhase5ReuseAndCaching:
    """
    Tests for the Phase 5 shared mesh reuse and kernel mesh evaluation cache.

    Design rules under test
    -----------------------
    1. The shared fixed-grid mesh (xs) is built once on the first
       ``_apply_kernels_fixed_grid`` call and reused on all subsequent calls.
       It is never cleared; it depends only on immutable construction params.
    2. When ``cache_kernels=True``, per-kernel mesh evaluations ``k_i(xs)``
       are stored in ``_kernel_eval_cache`` on first use and reused on all
       subsequent calls, eliminating repeated kernel evaluations.
    3. Kernel mesh evaluations are NOT cached when ``cache_kernels=False``.
    4. Compact-support kernels (generic-path fallback) are never stored in
       ``_kernel_eval_cache``.
    5. ``clear_cache()`` clears both ``_kernels_cache`` and
       ``_kernel_eval_cache``; next call re-evaluates from scratch.
    6. ``clear_mesh_cache()`` clears ``_kernel_eval_cache`` only; the shared
       mesh (``_shared_mesh``) is preserved.
    7. ``get_cache_info()`` reports both the shared-mesh state and the count
       of kernel mesh eval entries currently stored.
    8. Caching must not change numerical results; all G(f) calls give
       bitwise-identical outputs regardless of cache state.
    """

    # ------------------------------------------------------------------
    # 1. Shared mesh reuse
    # ------------------------------------------------------------------

    def test_mesh_unbuilt_before_first_call(self, lebesgue_space, unit_domain):
        """_shared_mesh is None before any forward call."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        assert G._shared_mesh is None

    def test_mesh_built_after_first_call(self, lebesgue_space, unit_domain):
        """_shared_mesh is populated after the first fixed-grid forward call."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        assert G._shared_mesh is not None
        assert G._shared_mesh.shape == (500,)

    def test_mesh_same_object_across_calls(self, lebesgue_space, unit_domain):
        """The shared mesh is the identical array object on repeated calls."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        mesh_id_first = id(G._shared_mesh)
        _ = G(f)
        assert id(G._shared_mesh) == mesh_id_first, \
            "Shared mesh should be the same object on every call"

    def test_mesh_not_built_for_adaptive_method(self, lebesgue_space, unit_domain):
        """Adaptive method never builds the shared mesh."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
            integration_config=IntegrationConfig(method="adaptive"),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        assert G._shared_mesh is None

    def test_mesh_correct_bounds(self, lebesgue_space, unit_domain):
        """Shared mesh spans [0, 1] with exactly n_points points."""
        n_pts = 301
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
            integration_config=IntegrationConfig(method="simpson", n_points=n_pts),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        xs = G._shared_mesh
        assert xs[0] == pytest.approx(0.0)
        assert xs[-1] == pytest.approx(1.0)
        assert len(xs) == n_pts

    def test_get_cache_info_reports_mesh_not_built(self, lebesgue_space, unit_domain):
        """get_cache_info reports shared_mesh_built=False before any call."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
        )
        info = G.get_cache_info()
        assert info["shared_mesh_built"] is False

    def test_get_cache_info_reports_mesh_built_after_call(
        self, lebesgue_space, unit_domain
    ):
        """get_cache_info reports shared_mesh_built=True after a forward call."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        info = G.get_cache_info()
        assert info["shared_mesh_built"] is True

    # ------------------------------------------------------------------
    # 2. Kernel mesh evaluation cache – disabled path
    # ------------------------------------------------------------------

    def test_kernel_eval_cache_none_when_caching_disabled(
        self, lebesgue_space, unit_domain
    ):
        """_kernel_eval_cache is None when cache_kernels=False."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=False,
        )
        assert G._kernel_eval_cache is None

    def test_kernel_eval_cache_stays_none_after_call_when_disabled(
        self, lebesgue_space, unit_domain
    ):
        """_kernel_eval_cache remains None after a forward call when cache_kernels=False."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=False,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        assert G._kernel_eval_cache is None

    # ------------------------------------------------------------------
    # 3. Kernel mesh evaluation cache – enabled path
    # ------------------------------------------------------------------

    def test_kernel_eval_cache_empty_dict_before_first_call(
        self, lebesgue_space, unit_domain
    ):
        """_kernel_eval_cache is an empty dict before any forward call."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=True,
        )
        assert G._kernel_eval_cache == {}

    def test_kernel_eval_cache_populated_after_forward_call(
        self, lebesgue_space, unit_domain
    ):
        """After a fixed-grid forward call, full-domain kernels are cached."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        info = G.get_cache_info()
        assert info["kernel_eval_cache_entries"] == 2

    def test_kernel_eval_cache_entries_shape(self, lebesgue_space, unit_domain):
        """Cached kernel eval entries are ndarrays with shape (n_points,)."""
        n_pts = 300
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=n_pts),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        assert G._kernel_eval_cache is not None
        for idx, arr in G._kernel_eval_cache.items():
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (n_pts,), \
                f"kernel {idx}: expected shape ({n_pts},), got {arr.shape}"

    def test_cached_and_uncached_forward_results_identical(
        self, lebesgue_space, unit_domain
    ):
        """Pre- and post-cache warm calls must give bitwise-identical outputs."""
        D = EuclideanSpace(3)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: np.sin(np.pi * x)),
            Function(unit_domain, evaluate_callable=lambda x: x),
            Function(unit_domain, evaluate_callable=lambda x: x * (1 - x)),
        ]
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.cos(np.pi * x))

        G_no_cache = SOLAOperator(
            lebesgue_space, D, kernels=kernels, cache_kernels=False, integration_config=cfg
        )
        G_cached = SOLAOperator(
            lebesgue_space, D, kernels=kernels, cache_kernels=True, integration_config=cfg
        )
        r_first = G_cached(f)    # warms cache
        r_second = G_cached(f)   # uses cached evals
        r_ref = G_no_cache(f)

        assert_allclose(r_first, r_ref, rtol=0, atol=0,
                        err_msg="First (warm) cached call differs from no-cache result")
        assert_allclose(r_second, r_ref, rtol=0, atol=0,
                        err_msg="Second (hot) cached call differs from no-cache result")

    def test_provider_backed_caching_correct(self, lebesgue_space, unit_domain):
        """Provider-backed kernels with cache_kernels=True give correct values."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            cache_kernels=True, integration_config=cfg,
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        r1 = G(f)   # warms eval cache
        r2 = G(f)   # uses cached evals
        assert_allclose(r1[0], np.sqrt(2.0) / 2.0, rtol=1e-4)
        assert_allclose(r1, r2, rtol=0, atol=0,
                        err_msg="Provider-backed cached second call gave different result")

    # ------------------------------------------------------------------
    # 4. Compact-support kernels excluded from eval cache
    # ------------------------------------------------------------------

    def test_compact_support_kernel_not_in_eval_cache(
        self, lebesgue_space, unit_domain
    ):
        """Support-restricted kernels must not appear in _kernel_eval_cache."""
        k_compact = Function(
            unit_domain,
            evaluate_callable=lambda x: np.where(
                (np.asarray(x) >= 0.2) & (np.asarray(x) <= 0.8), 1.0, 0.0
            ),
            support=[(0.2, 0.8)],
        )
        k_full = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
        )
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[k_compact, k_full],
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        # f must also have compact support overlapping k_compact so that
        # _intersect_supports(f.support, k_compact.support) returns a non-None
        # list and the kernel is routed through the generic (non-batched) path.
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 0.5)],
        )
        _ = G(f)
        assert G._kernel_eval_cache is not None
        assert 0 not in G._kernel_eval_cache, \
            "Compact-support kernel (index 0) must NOT appear in eval cache "\
            "when both func and kernel have overlapping compact support (generic path)"
        assert 1 in G._kernel_eval_cache, \
            "Full-domain kernel (index 1) must appear in eval cache (batched path)"

    def test_compact_support_result_unchanged_by_caching(
        self, lebesgue_space, unit_domain
    ):
        """Compact-support kernel forward result is identical with/without caching."""
        def k_callable(x):
            x_arr = np.asarray(x)
            return np.where((x_arr >= 0.3) & (x_arr <= 0.7), 1.0, 0.0)

        k_compact = Function(
            unit_domain, evaluate_callable=k_callable, support=[(0.3, 0.7)]
        )
        D = EuclideanSpace(1)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G_cached = SOLAOperator(
            lebesgue_space, D, kernels=[k_compact], cache_kernels=True, integration_config=cfg
        )
        G_no_cache = SOLAOperator(
            lebesgue_space, D, kernels=[k_compact], cache_kernels=False, integration_config=cfg
        )
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
        r_cached = G_cached(f)
        r_ref = G_no_cache(f)
        assert_allclose(r_cached, r_ref, rtol=0, atol=0)
        assert_allclose(r_cached[0], 0.4, rtol=1e-3)

    # ------------------------------------------------------------------
    # 5. Cache invalidation
    # ------------------------------------------------------------------

    def test_clear_cache_empties_kernel_eval_entries(
        self, lebesgue_space, unit_domain
    ):
        """clear_cache() empties _kernel_eval_cache as well as _kernels_cache."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        assert G.get_cache_info()["kernel_eval_cache_entries"] == 2
        G.clear_cache()
        assert G.get_cache_info()["kernel_eval_cache_entries"] == 0

    def test_clear_mesh_cache_clears_eval_entries_not_shared_mesh(
        self, lebesgue_space, unit_domain
    ):
        """clear_mesh_cache() clears eval entries but preserves the shared mesh."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        G = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        mesh_before = G._shared_mesh
        assert G.get_cache_info()["kernel_eval_cache_entries"] == 3
        G.clear_mesh_cache()
        assert G.get_cache_info()["kernel_eval_cache_entries"] == 0
        assert G._shared_mesh is mesh_before

    def test_result_after_clear_cache_is_identical(self, lebesgue_space, unit_domain):
        """After clear_cache(), the next call gives the same result as before."""
        from intervalinf.providers import SineFunctionProvider
        D = EuclideanSpace(3)
        provider = SineFunctionProvider(unit_domain)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            cache_kernels=True, integration_config=cfg,
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        r_before = G(f)
        G.clear_cache()
        r_after = G(f)
        assert_allclose(r_before, r_after, rtol=0, atol=0,
                        err_msg="Post clear_cache result differs from pre-clear result")

    def test_clear_mesh_cache_noop_when_disabled(self, lebesgue_space, unit_domain):
        """clear_mesh_cache() is a no-op when cache_kernels=False."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
            cache_kernels=False,
        )
        G.clear_mesh_cache()  # must not raise
        assert G._kernel_eval_cache is None

    # ------------------------------------------------------------------
    # 6. get_cache_info Phase 5 fields
    # ------------------------------------------------------------------

    def test_cache_info_no_cache_has_mesh_key(self, lebesgue_space, unit_domain):
        """get_cache_info when disabled includes shared_mesh_built key."""
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[Function(unit_domain, evaluate_callable=lambda x: x)],
        )
        info = G.get_cache_info()
        assert "caching_enabled" in info
        assert "shared_mesh_built" in info
        assert info["caching_enabled"] is False
        assert info["shared_mesh_built"] is False

    def test_cache_info_full_stats_after_call(self, lebesgue_space, unit_domain):
        """With cache_kernels=True and after a call, all Phase 5 stats present."""
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[
                Function(unit_domain, evaluate_callable=lambda x: x),
                Function(unit_domain, evaluate_callable=lambda x: 1 - x),
            ],
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        _ = G(f)
        info = G.get_cache_info()
        assert info["caching_enabled"] is True
        assert info["shared_mesh_built"] is True
        assert info["kernel_eval_cache_entries"] == 2
        assert info["total_functions"] == 2

    # ------------------------------------------------------------------
    # 7. Repeated workload smoke test (50 calls)
    # ------------------------------------------------------------------

    def test_repeated_workload_correctness(self, lebesgue_space, unit_domain):
        """50 G(f_i) calls with varying f and cached kernels — all match reference.

        Simulates an iterative inverse problem: the operator is fixed, only
        the input function changes at each iteration.
        """
        np.random.seed(42)
        N_d = 10
        D = EuclideanSpace(N_d)
        from intervalinf.providers import SineFunctionProvider
        provider = SineFunctionProvider(unit_domain)
        cfg = IntegrationConfig(method="simpson", n_points=1000)

        G_cached = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            cache_kernels=True, integration_config=cfg,
        )
        G_ref = SOLAOperator(
            lebesgue_space, D, kernels=provider,
            cache_kernels=False, integration_config=cfg,
        )

        for _ in range(50):
            coeffs = np.random.randn(5)

            def make_f(c=coeffs):
                return Function(
                    lebesgue_space,
                    evaluate_callable=lambda x, _c=c: sum(
                        _c[k] * np.sin((k + 1) * np.pi * x)
                        for k in range(len(_c))
                    ),
                )

            f = make_f()
            r_cached = G_cached(f)
            r_ref = G_ref(f)
            assert_allclose(
                r_cached, r_ref, rtol=0, atol=0,
                err_msg="Cached result differed from reference on repeated call",
            )


# ---------------------------------------------------------------------------
# 14. Phase 2: SOLAOperator instrumentation (stats / reset_stats)
# ---------------------------------------------------------------------------

class TestPhase2Instrumentation:
    """Unit tests for the lightweight passive instrumentation added in Phase 2.

    Verifies:
    - ``stats`` property returns an accurate copy as a plain dict.
    - ``reset_stats()`` zeros all counters.
    - correct counter increments for each of the three fixed-grid sub-paths:
        * batched fixed-grid path (full-domain / no support metadata)
        * compact-support fallback path (overlapping supports)
        * disjoint-support skip path (disjoint supports → exact zero)
    - counters accumulate correctly across multiple forward calls.
    - generic (adaptive) path also increments disjoint_skips.
    - timing fields are non-negative and forward_time_total_s > 0 after calls.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_kernel(domain, support=None):
        """Constant-1 kernel, optionally with compact-support metadata."""
        return Function(
            domain,
            evaluate_callable=lambda x: (
                np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
            ),
            support=support,
        )

    @staticmethod
    def _make_operator(domain, space, kernels, method="simpson"):
        D = EuclideanSpace(len(kernels))
        cfg = IntegrationConfig(method=method, n_points=500)
        return SOLAOperator(space, D, kernels=kernels, integration_config=cfg)

    # ------------------------------------------------------------------
    # 14.1  stats dict structure
    # ------------------------------------------------------------------

    def test_stats_returns_dict_with_expected_keys(self, lebesgue_space, unit_domain):
        """stats property returns a dict with all Phase 2 counter keys."""
        D = EuclideanSpace(1)
        k = self._flat_kernel(unit_domain)
        G = self._make_operator(unit_domain, lebesgue_space, [k])
        s = G.stats
        expected_keys = {
            "forward_calls", "disjoint_skips", "compact_support_fallbacks",
            "batched_fixed_grid_kernels", "forward_time_total_s",
            "compact_support_fallback_time_total_s", "fast_path_hits",
        }
        assert expected_keys == set(s.keys()), f"Unexpected keys: {set(s.keys())}"

    def test_stats_initial_values_all_zero(self, lebesgue_space, unit_domain):
        """All counters start at zero before any forward call."""
        D = EuclideanSpace(1)
        k = self._flat_kernel(unit_domain)
        G = self._make_operator(unit_domain, lebesgue_space, [k])
        s = G.stats
        for key, val in s.items():
            assert val == 0 or val == 0.0, f"{key}={val} should be 0 initially"

    def test_stats_returns_copy(self, lebesgue_space, unit_domain):
        """Mutating the returned dict does not affect the operator's internal state."""
        D = EuclideanSpace(1)
        k = self._flat_kernel(unit_domain)
        G = self._make_operator(unit_domain, lebesgue_space, [k])
        s = G.stats
        s["forward_calls"] = 9999
        assert G.stats["forward_calls"] == 0, "stats should return an independent copy"

    # ------------------------------------------------------------------
    # 14.2  reset_stats
    # ------------------------------------------------------------------

    def test_reset_clears_all_counters(self, lebesgue_space, unit_domain):
        """reset_stats() brings all counters back to zero after forward calls."""
        D = EuclideanSpace(2)
        kernels = [self._flat_kernel(unit_domain) for _ in range(2)]
        G = self._make_operator(unit_domain, lebesgue_space, kernels)
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        G(f)
        G(f)
        G.reset_stats()
        s = G.stats
        for key, val in s.items():
            assert val == 0 or val == 0.0, f"After reset, {key}={val} should be 0"

    # ------------------------------------------------------------------
    # 14.3  full-domain / batched path
    # ------------------------------------------------------------------

    def test_batched_path_increments_forward_calls(self, lebesgue_space, unit_domain):
        """Each forward call increments forward_calls by 1."""
        D = EuclideanSpace(3)
        kernels = [self._flat_kernel(unit_domain) for _ in range(3)]
        G = self._make_operator(unit_domain, lebesgue_space, kernels)
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        for n in range(1, 4):
            G(f)
            assert G.stats["forward_calls"] == n

    def test_batched_path_no_fallback_no_skip(self, lebesgue_space, unit_domain):
        """No-support-metadata kernels go entirely through the batched path."""
        N_d = 4
        D = EuclideanSpace(N_d)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x, _i=i: np.sin((_i + 1) * np.pi * np.asarray(x)))
            for i in range(N_d)
        ]
        G = self._make_operator(unit_domain, lebesgue_space, kernels)
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        G(f)
        s = G.stats
        assert s["batched_fixed_grid_kernels"] == N_d
        assert s["compact_support_fallbacks"] == 0
        assert s["disjoint_skips"] == 0

    def test_batched_path_accumulates_across_calls(self, lebesgue_space, unit_domain):
        """batched_fixed_grid_kernels accumulates over multiple forward calls."""
        N_d = 3
        D = EuclideanSpace(N_d)
        kernels = [self._flat_kernel(unit_domain) for _ in range(N_d)]
        G = self._make_operator(unit_domain, lebesgue_space, kernels)
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        n_calls = 5
        for _ in range(n_calls):
            G(f)
        assert G.stats["batched_fixed_grid_kernels"] == N_d * n_calls

    # ------------------------------------------------------------------
    # 14.4  disjoint-support skip path
    # ------------------------------------------------------------------

    def test_disjoint_skips_counter(self, lebesgue_space, unit_domain):
        """Disjoint-support kernels increment disjoint_skips and give zero result."""
        k_right = self._flat_kernel(unit_domain, support=[(0.7, 1.0)])
        G = self._make_operator(unit_domain, lebesgue_space, [k_right])
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: x,
                     support=[(0.0, 0.3)])
        result = G(f)
        assert_allclose(result, [0.0], atol=1e-12)
        s = G.stats
        assert s["disjoint_skips"] == 1
        assert s["compact_support_fallbacks"] == 0
        assert s["batched_fixed_grid_kernels"] == 0

    def test_disjoint_skips_multiple_kernels(self, lebesgue_space, unit_domain):
        """Multiple disjoint kernels each increment disjoint_skips by 1."""
        N_d = 5
        kernels = [self._flat_kernel(unit_domain, support=[(0.7, 1.0)])
                   for _ in range(N_d)]
        G = self._make_operator(unit_domain, lebesgue_space, kernels)
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: x,
                     support=[(0.0, 0.3)])
        G(f)
        assert G.stats["disjoint_skips"] == N_d

    # ------------------------------------------------------------------
    # 14.5  compact-support fallback path
    # ------------------------------------------------------------------

    def test_fallback_counter_for_overlapping_support(self, lebesgue_space, unit_domain):
        """Overlapping compact-support kernel increments compact_support_fallbacks."""
        k_overlap = self._flat_kernel(unit_domain, support=[(0.1, 0.5)])
        G = self._make_operator(unit_domain, lebesgue_space, [k_overlap])
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: x,
                     support=[(0.0, 0.6)])
        G(f)
        s = G.stats
        assert s["compact_support_fallbacks"] == 1
        assert s["batched_fixed_grid_kernels"] == 0
        assert s["disjoint_skips"] == 0

    def test_fallback_time_positive_after_fallback(self, lebesgue_space, unit_domain):
        """compact_support_fallback_time_total_s > 0 after at least one fallback."""
        k_overlap = self._flat_kernel(unit_domain, support=[(0.1, 0.5)])
        G = self._make_operator(unit_domain, lebesgue_space, [k_overlap])
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: x,
                     support=[(0.0, 0.6)])
        G(f)
        assert G.stats["compact_support_fallback_time_total_s"] > 0.0

    def test_forward_time_positive_after_call(self, lebesgue_space, unit_domain):
        """forward_time_total_s > 0 after at least one forward call."""
        G = self._make_operator(unit_domain, lebesgue_space,
                                [self._flat_kernel(unit_domain)])
        f = Function(lebesgue_space, evaluate_callable=lambda x: x)
        G(f)
        assert G.stats["forward_time_total_s"] > 0.0

    # ------------------------------------------------------------------
    # 14.6  mixed-path scenario
    # ------------------------------------------------------------------

    def test_mixed_path_counter_split(self, lebesgue_space, unit_domain):
        """Mix of no-metadata and overlapping-support kernels splits correctly."""
        # 2 full-domain (batched) + 2 overlapping-support (fallback) + 1 disjoint (skip)
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: x),          # batched
            Function(unit_domain, evaluate_callable=lambda x: x * 2),      # batched
            self._flat_kernel(unit_domain, support=[(0.1, 0.45)]),          # fallback
            self._flat_kernel(unit_domain, support=[(0.1, 0.45)]),          # fallback
            self._flat_kernel(unit_domain, support=[(0.7, 1.0)]),           # disjoint
        ]
        G = self._make_operator(unit_domain, lebesgue_space, kernels)
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: x,
                     support=[(0.0, 0.5)])
        G(f)
        s = G.stats
        assert s["batched_fixed_grid_kernels"] == 2, f"Expected batched=2, got {s['batched_fixed_grid_kernels']}"
        assert s["compact_support_fallbacks"] == 2, f"Expected fallbacks=2, got {s['compact_support_fallbacks']}"
        assert s["disjoint_skips"] == 1, f"Expected skips=1, got {s['disjoint_skips']}"

    # ------------------------------------------------------------------
    # 14.7  generic (adaptive) path — disjoint_skips still counted
    # ------------------------------------------------------------------

    def test_generic_path_disjoint_skips(self, lebesgue_space, unit_domain):
        """disjoint_skips is also counted on the adaptive (generic) path."""
        k_right = self._flat_kernel(unit_domain, support=[(0.7, 1.0)])
        G = self._make_operator(unit_domain, lebesgue_space, [k_right],
                                method="adaptive")
        f = Function(lebesgue_space,
                     evaluate_callable=lambda x: x,
                     support=[(0.0, 0.3)])
        result = G(f)
        assert_allclose(result, [0.0], atol=1e-12)
        s = G.stats
        assert s["forward_calls"] == 1
        assert s["disjoint_skips"] == 1

    # ------------------------------------------------------------------
    # 14.8  numerical invariance — instrumentation must not change results
    # ------------------------------------------------------------------

    def test_instrumentation_does_not_change_forward_result(
        self, lebesgue_space, unit_domain
    ):
        """Results from instrumented operator match a freshly built reference."""
        N_d = 4
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x, _i=i: np.sin((_i + 1) * np.pi * np.asarray(x)))
            for i in range(N_d)
        ]
        D = EuclideanSpace(N_d)
        cfg = IntegrationConfig(method="simpson", n_points=1000)
        G1 = SOLAOperator(lebesgue_space, D, kernels=kernels, integration_config=cfg)
        G2 = SOLAOperator(lebesgue_space, D, kernels=kernels, integration_config=cfg)
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * x))
        y1 = G1(f)
        y2 = G2(f)
        assert_allclose(y1, y2, rtol=0, atol=0,
                        err_msg="Instrumentation changed forward result")

    # ------------------------------------------------------------------
    # 14.9  consolidated smoke test: batched + fallback + reset
    # ------------------------------------------------------------------

    def test_instrumentation_and_reset_smoke(self, lebesgue_space, unit_domain):
        """Smoke: one call exercises batched and fallback paths; reset_stats zeros all.

        One forward call on a mixed operator (one full-domain kernel, one
        overlapping-support kernel) must increment:
        - forward_calls by 1
        - batched_fixed_grid_kernels by 1
        - compact_support_fallbacks by 1
        After reset_stats() every counter must be 0.
        """
        k_batched = Function(
            unit_domain,
            evaluate_callable=lambda x: np.sin(np.pi * np.asarray(x)),
        )
        k_overlap = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.1, 0.4)],
        )
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[k_batched, k_overlap],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        # f has compact support overlapping k_overlap but k_batched has no support
        f = Function(lebesgue_space, evaluate_callable=lambda x: x,
                     support=[(0.0, 0.5)])
        G(f)
        s = G.stats
        assert s["forward_calls"] == 1
        assert s["batched_fixed_grid_kernels"] == 1, (
            f"Expected batched=1, got {s['batched_fixed_grid_kernels']}"
        )
        assert s["compact_support_fallbacks"] == 1, (
            f"Expected fallbacks=1, got {s['compact_support_fallbacks']}"
        )
        G.reset_stats()
        for key, val in G.stats.items():
            assert val == 0 or val == 0.0, f"After reset_stats, {key}={val} should be 0"


# ---------------------------------------------------------------------------
# 15. Phase 4 – Private support-mesh helper (_build_support_mesh)
# ---------------------------------------------------------------------------

class TestBuildSupportMesh:
    """
    Unit tests for SOLAOperator._build_support_mesh.

    The helper reproduces IntervalDomain.integrate proportional-allocation
    meshing semantics for support lists without running the integrand, so the
    same mesh can be reused across multiple function evaluations in future
    batched support-restricted kernel passes (Phase 6 preparation).

    Five canonical scenarios verified:
    1. Single-interval support → mesh equals np.linspace exactly.
    2. Multi-interval proportional allocation + deterministic remainder
       distribution (longer interval wins the extra point).
    3. Effective-total enforcement when n_points < 3*n_sub.
    4. Stable tie-break on equal fractional remainders (first index wins).
    5. Adjacent intervals keep the duplicated shared boundary point.
    """

    # ------------------------------------------------------------------
    # Ancillary: empty support
    # ------------------------------------------------------------------

    def test_empty_support_returns_empty_array(self):
        """Empty support list returns an empty array."""
        xs = SOLAOperator._build_support_mesh([], 100)
        assert isinstance(xs, np.ndarray)
        assert len(xs) == 0

    # ------------------------------------------------------------------
    # 1. Single interval → plain linspace
    # ------------------------------------------------------------------

    def test_single_interval_matches_linspace(self):
        """Single-interval list: mesh equals np.linspace(a, b, n_points) exactly.

        n_sub=1 → effective_total = max(100, 3) = 100;
        raw=[100], alloc=[100], remainder=0.
        """
        support = [(0.0, 1.0)]
        xs = SOLAOperator._build_support_mesh(support, 100)
        expected = np.linspace(0.0, 1.0, 100)
        assert_allclose(xs, expected, rtol=0, atol=0)

    def test_single_interval_small_n_points_clipped_to_three(self):
        """n_points=1 → effective_total=max(1,3)=3, alloc=[max(3,3)]=3."""
        support = [(0.5, 0.8)]
        xs = SOLAOperator._build_support_mesh(support, 1)
        expected = np.linspace(0.5, 0.8, 3)
        assert_allclose(xs, expected, rtol=0, atol=0)

    # ------------------------------------------------------------------
    # 2. Multi-interval: proportional allocation + remainder distribution
    # ------------------------------------------------------------------

    def test_two_equal_intervals_split_evenly(self):
        """Two equal-length intervals with even n_points split into halves.

        lengths=[0.5, 0.5], n_points=100 →
        effective_total=100, raw=[50.0, 50.0], alloc=[50,50], remainder=0.
        """
        support = [(0.0, 0.5), (0.5, 1.0)]
        xs = SOLAOperator._build_support_mesh(support, 100)
        expected = np.concatenate([
            np.linspace(0.0, 0.5, 50),
            np.linspace(0.5, 1.0, 50),
        ])
        assert_allclose(xs, expected, rtol=0, atol=0)

    def test_unequal_intervals_proportional_with_correct_remainder(self):
        """Unequal intervals: longer interval gets larger allocation; remainder
        goes to interval with highest fractional part.

        support=[(0,1),(1,3)], lengths=[1,2], n_points=10 →
        effective_total=10, raw=[10/3≈3.33, 20/3≈6.67], alloc=[3,6],
        remainder=1; fracs desc=[(0.67,1),(0.33,0)] → alloc[1]+=1 → [3,7].
        """
        support = [(0.0, 1.0), (1.0, 3.0)]
        xs = SOLAOperator._build_support_mesh(support, 10)
        expected = np.concatenate([
            np.linspace(0.0, 1.0, 3),
            np.linspace(1.0, 3.0, 7),
        ])
        assert_allclose(xs, expected, rtol=0, atol=0)
        assert len(xs) == 10

    # ------------------------------------------------------------------
    # 3. Effective-total enforcement when n_points < 3*n_sub
    # ------------------------------------------------------------------

    def test_effective_total_enforced_when_n_points_too_small(self):
        """n_points < 3*n_sub → effective_total raised to 3*n_sub.

        3 equal sub-intervals, n_points=5 < 9=3*3 → effective_total=9;
        alloc=[3,3,3], total mesh length=9.
        """
        support = [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)]
        xs = SOLAOperator._build_support_mesh(support, 5)
        assert len(xs) == 9  # 3 * n_sub
        expected = np.concatenate([
            np.linspace(0.0, 1/3, 3),
            np.linspace(1/3, 2/3, 3),
            np.linspace(2/3, 1.0, 3),
        ])
        assert_allclose(xs, expected, rtol=1e-15)

    # ------------------------------------------------------------------
    # 4. Stable tie-breaker on equal fractional remainders
    # ------------------------------------------------------------------

    def test_stable_tiebreak_first_index_wins(self):
        """Equal fractional parts: Python stable sort gives remainder to index 0.

        Two equal-length intervals, n_points=7:
        effective_total=max(7,6)=7, raw=[3.5, 3.5], alloc=[3,3], remainder=1.
        Both fracs are exactly 0.5 in IEEE-754, so stable sort preserves the
        original order and alloc[0] wins the extra point → alloc=[4,3].
        """
        support = [(0.0, 0.5), (0.5, 1.0)]
        xs = SOLAOperator._build_support_mesh(support, 7)
        expected = np.concatenate([
            np.linspace(0.0, 0.5, 4),   # index 0 gets the extra point
            np.linspace(0.5, 1.0, 3),
        ])
        assert_allclose(xs, expected, rtol=0, atol=0)
        assert len(xs) == 7

    # ------------------------------------------------------------------
    # 5. Adjacent intervals preserve duplicated shared boundary point
    # ------------------------------------------------------------------

    def test_adjacent_intervals_boundary_point_duplicated(self):
        """Shared boundary between adjacent sub-intervals appears exactly twice
        in the concatenated mesh (no deduplication)."""
        support = [(0.0, 0.5), (0.5, 1.0)]
        xs = SOLAOperator._build_support_mesh(support, 100)
        n_at_boundary = int(np.sum(np.abs(xs - 0.5) < 1e-15))
        assert n_at_boundary == 2, (
            f"Expected 0.5 to appear exactly twice; found {n_at_boundary} times"
        )

    def test_three_intervals_two_shared_boundaries_each_duplicated(self):
        """Three adjacent sub-intervals: each internal boundary appears twice."""
        support = [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)]
        xs = SOLAOperator._build_support_mesh(support, 30)
        b1, b2 = 1/3, 2/3
        n_b1 = int(np.sum(np.abs(xs - b1) < 1e-15))
        n_b2 = int(np.sum(np.abs(xs - b2) < 1e-15))
        assert n_b1 == 2, f"Boundary 1/3 found {n_b1} times, expected 2"
        assert n_b2 == 2, f"Boundary 2/3 found {n_b2} times, expected 2"


# ---------------------------------------------------------------------------
# 16. Phase 5 – Grouped support-restricted batched integration
# ---------------------------------------------------------------------------

class TestPhase5GroupedSupportBatching:
    """
    Regression tests for the Phase 5 grouped support-restricted batched path.

    Compact-support kernels that share the same intersected support are now
    integrated in a single grouped batched pass rather than one at a time.

    Behavioral contracts:
    1. Correct values: grouped kernels give analytic integrals.
    2. Multi-interval support: grouped fixed-grid result matches generic path
       at tight tolerance.
    3. No cache pollution: grouped kernels do NOT appear in _kernel_eval_cache.
    4. Stats coherence: compact_support_fallbacks counts N kernels in a group
       (per-kernel, not per-group), so N kernels → N increments.
    5. Batching proof: _build_support_mesh is called once per unique support
       group, not once per kernel.
    6. Complex dtype preservation in the grouped support-restricted path.
    """

    # ------------------------------------------------------------------
    # 1. Correct values for shared-support groups
    # ------------------------------------------------------------------

    def test_two_shared_support_kernels_analytic(self, lebesgue_space, unit_domain):
        """Two kernels with the same compact support give correct analytic integrals.

        k0(x) = 1 on [0.2, 0.8],  k1(x) = x on [0.2, 0.8].
        f = 1 with support [(0.0, 1.0)].

        G(f)[0] = ∫_{0.2}^{0.8} 1 dx = 0.6
        G(f)[1] = ∫_{0.2}^{0.8} x dx = (0.8² − 0.2²) / 2 = 0.3
        """
        support = [(0.2, 0.8)]
        k0 = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        k1 = Function(
            unit_domain,
            evaluate_callable=lambda x: np.asarray(x),
            support=support,
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(2)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G = SOLAOperator(lebesgue_space, D, kernels=[k0, k1], integration_config=cfg)

        result = G(f)
        assert_allclose(result[0], 0.6, rtol=1e-4)
        assert_allclose(result[1], 0.3, rtol=1e-4)

    def test_three_shared_support_kernels_match_generic(
        self, lebesgue_space, unit_domain
    ):
        """Three kernels with identical support match the generic path."""
        support = [(0.3, 0.7)]
        kernels = [
            Function(
                unit_domain,
                evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                support=support,
            ),
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x), support=support),
            Function(
                unit_domain,
                evaluate_callable=lambda x: np.sin(np.pi * np.asarray(x)),
                support=support,
            ),
        ]
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.asarray(x) + 0.5,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(3)
        cfg = IntegrationConfig(method="simpson", n_points=3000)
        G = SOLAOperator(lebesgue_space, D, kernels=kernels, integration_config=cfg)

        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert_allclose(result, generic, rtol=1e-6)

    def test_shared_support_trapz_matches_analytic(self, lebesgue_space, unit_domain):
        """Grouped path works with 'trapz' method as well as 'simpson'."""
        support = [(0.1, 0.9)]
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="trapz", n_points=3000),
        )
        result = G(f)
        assert_allclose(result[0], 0.8, rtol=1e-3)

    # ------------------------------------------------------------------
    # 2. Multi-interval support
    # ------------------------------------------------------------------

    def test_multi_interval_support_single_kernel_matches_generic(
        self, lebesgue_space, unit_domain
    ):
        """Multi-interval compact support: fixed-grid result matches generic.

        Kernel supported on [(0.1, 0.4), (0.6, 0.9)], value 1.0 inside.
        f = 1 with support [(0.0, 1.0)].

        Generic result = ∫_{0.1}^{0.4} 1 dx + ∫_{0.6}^{0.9} 1 dx = 0.3 + 0.3 = 0.6
        """
        support = [(0.1, 0.4), (0.6, 0.9)]

        def k_callable(x):
            x_arr = np.asarray(x)
            in_s1 = (x_arr >= 0.1) & (x_arr <= 0.4)
            in_s2 = (x_arr >= 0.6) & (x_arr <= 0.9)
            return np.where(in_s1 | in_s2, 1.0, 0.0)

        k = Function(unit_domain, evaluate_callable=k_callable, support=support)
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(1)
        cfg = IntegrationConfig(method="simpson", n_points=3000)
        G = SOLAOperator(lebesgue_space, D, kernels=[k], integration_config=cfg)

        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert_allclose(result, generic, rtol=1e-6, atol=1e-12)
        assert_allclose(result[0], 0.6, rtol=1e-4)

    def test_multi_interval_support_two_kernels_match_generic(
        self, lebesgue_space, unit_domain
    ):
        """Two kernels with DIFFERENT multi-interval supports both match generic."""
        support_a = [(0.1, 0.4), (0.6, 0.9)]
        support_b = [(0.2, 0.5), (0.7, 0.95)]

        def make_indicator(supp):
            def k_callable(x):
                x_arr = np.asarray(x)
                mask = np.zeros_like(x_arr, dtype=float)
                for a_s, b_s in supp:
                    mask[(x_arr >= a_s) & (x_arr <= b_s)] = 1.0
                return mask
            return k_callable

        ka = Function(
            unit_domain, evaluate_callable=make_indicator(support_a), support=support_a
        )
        kb = Function(
            unit_domain, evaluate_callable=make_indicator(support_b), support=support_b
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(2)
        cfg = IntegrationConfig(method="simpson", n_points=3000)
        G = SOLAOperator(lebesgue_space, D, kernels=[ka, kb], integration_config=cfg)

        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert_allclose(result, generic, rtol=1e-6, atol=1e-12)

    def test_multi_interval_two_kernels_same_support_match_generic(
        self, lebesgue_space, unit_domain
    ):
        """Two kernels with identical multi-interval support are grouped and match generic."""
        support = [(0.1, 0.3), (0.7, 0.9)]

        def k0_callable(x):
            x_arr = np.asarray(x)
            return np.where(((x_arr >= 0.1) & (x_arr <= 0.3)) | ((x_arr >= 0.7) & (x_arr <= 0.9)), 1.0, 0.0)

        def k1_callable(x):
            x_arr = np.asarray(x)
            return np.where(((x_arr >= 0.1) & (x_arr <= 0.3)) | ((x_arr >= 0.7) & (x_arr <= 0.9)), float(x_arr) if x_arr.ndim == 0 else x_arr, 0.0)

        k0 = Function(unit_domain, evaluate_callable=k0_callable, support=support)
        k1 = Function(unit_domain, evaluate_callable=lambda x: np.asarray(x), support=support)
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(2)
        cfg = IntegrationConfig(method="simpson", n_points=3000)
        G = SOLAOperator(lebesgue_space, D, kernels=[k0, k1], integration_config=cfg)

        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert_allclose(result, generic, rtol=1e-6, atol=1e-12)

    # ------------------------------------------------------------------
    # 3. No cache pollution after grouped batching
    # ------------------------------------------------------------------

    def test_grouped_support_kernels_not_in_eval_cache(
        self, lebesgue_space, unit_domain
    ):
        """Grouped compact-support kernels must not populate _kernel_eval_cache."""
        support = [(0.2, 0.8)]
        k0 = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        k1 = Function(unit_domain, evaluate_callable=lambda x: np.asarray(x), support=support)
        k_full = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(3)
        G = SOLAOperator(
            lebesgue_space, D,
            kernels=[k0, k1, k_full],
            cache_kernels=True,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        _ = G(f)

        assert G._kernel_eval_cache is not None
        assert 0 not in G._kernel_eval_cache, \
            "Grouped compact-support kernel 0 must NOT appear in eval cache"
        assert 1 not in G._kernel_eval_cache, \
            "Grouped compact-support kernel 1 must NOT appear in eval cache"
        assert 2 in G._kernel_eval_cache, \
            "Full-domain kernel 2 must appear in eval cache"

    def test_grouped_support_result_identical_with_and_without_caching(
        self, lebesgue_space, unit_domain
    ):
        """Grouped batching result is identical regardless of cache_kernels setting."""
        support = [(0.3, 0.7)]
        kernels = [
            Function(unit_domain,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=support),
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x), support=support),
        ]
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(2)
        cfg = IntegrationConfig(method="simpson", n_points=2000)
        G_cached = SOLAOperator(lebesgue_space, D, kernels=kernels, cache_kernels=True, integration_config=cfg)
        G_nocache = SOLAOperator(lebesgue_space, D, kernels=kernels, cache_kernels=False, integration_config=cfg)

        r_cached = G_cached(f)
        r_nocache = G_nocache(f)
        assert_allclose(r_cached, r_nocache, rtol=0, atol=0,
                        err_msg="Grouped batching result differs with/without caching")

    # ------------------------------------------------------------------
    # 4. Stats coherence for grouped batching
    # ------------------------------------------------------------------

    def test_grouped_batching_fallbacks_count_per_kernel(
        self, lebesgue_space, unit_domain
    ):
        """compact_support_fallbacks counts N kernels in a group, not 1 group.

        Three kernels with the same support → compact_support_fallbacks == 3.
        This verifies per-kernel counting is maintained in the grouped path.
        """
        support = [(0.2, 0.8)]
        kernels = [
            Function(unit_domain,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=support),
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x), support=support),
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x) ** 2, support=support),
        ]
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(3)
        G = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=1000),
        )
        G.reset_stats()
        _ = G(f)

        s = G.stats
        assert s["compact_support_fallbacks"] == 3, (
            f"Expected compact_support_fallbacks=3 for 3 shared-support kernels; "
            f"got {s['compact_support_fallbacks']}"
        )
        assert s["batched_fixed_grid_kernels"] == 0
        assert s["disjoint_skips"] == 0

    def test_grouped_batching_fallback_time_positive(
        self, lebesgue_space, unit_domain
    ):
        """compact_support_fallback_time_total_s > 0 after grouped batched integration."""
        support = [(0.2, 0.8)]
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.0, 1.0)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=1000),
        )
        G.reset_stats()
        _ = G(f)
        assert G.stats["compact_support_fallback_time_total_s"] > 0.0

    def test_grouped_and_full_domain_counter_split(self, lebesgue_space, unit_domain):
        """Mix of grouped-support and full-domain kernels splits counters correctly.

        Layout: 2 full-domain (batched) + 3 same-support (grouped fallback) + 1 disjoint.
        Expected: batched=2, fallbacks=3, skips=1.
        """
        support = [(0.1, 0.4)]
        kernels = [
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x)),           # batched
            Function(unit_domain, evaluate_callable=lambda x: 1 - np.asarray(x)),       # batched
            Function(unit_domain,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=support),                                                    # grouped
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x), support=support),  # grouped
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x)**2, support=support),  # grouped
            Function(unit_domain,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=[(0.7, 1.0)]),                                              # disjoint
        ]
        D = EuclideanSpace(len(kernels))
        G = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.0, 0.5)])
        G.reset_stats()
        _ = G(f)

        s = G.stats
        assert s["batched_fixed_grid_kernels"] == 2, f"Expected batched=2; got {s['batched_fixed_grid_kernels']}"
        assert s["compact_support_fallbacks"] == 3, f"Expected fallbacks=3; got {s['compact_support_fallbacks']}"
        assert s["disjoint_skips"] == 1, f"Expected skips=1; got {s['disjoint_skips']}"

    # ------------------------------------------------------------------
    # 5. Batching proof: f is evaluated once per support group, not per kernel
    # ------------------------------------------------------------------

    def test_shared_support_f_evaluated_once_per_group(
        self, lebesgue_space, unit_domain
    ):
        """N kernels with the same single-interval support → f is evaluated once
        on that group's restricted mesh, not once per kernel.

        Before Phase 5: N kernels would require N separate domain.integrate calls
        (each evaluating f internally).
        After Phase 5: one _eval_on_mesh(func, xs_i) call per subinterval per group.
        For a single-interval group: exactly 1 f evaluation regardless of N.
        """
        support = [(0.2, 0.8)]
        N = 4
        kernels = [
            Function(
                unit_domain,
                evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                support=support,
            )
            for _ in range(N)
        ]
        # Count vectorized callable-level invocations of f.
        eval_calls: list = []

        def f_callable(x):
            eval_calls.append(1)
            return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

        f = Function(lebesgue_space, evaluate_callable=f_callable, support=[(0.0, 1.0)])
        D = EuclideanSpace(N)
        G = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        _ = G(f)

        assert len(eval_calls) == 1, (
            f"Expected f evaluated once for {N} same-support kernels in 1 group; "
            f"got {len(eval_calls)}"
        )

    def test_two_different_supports_f_evaluated_twice(
        self, lebesgue_space, unit_domain
    ):
        """Two kernels with different single-interval supports → f is evaluated
        once per support group = twice total.
        """
        support_a = [(0.1, 0.4)]
        support_b = [(0.6, 0.9)]
        ka = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support_a,
        )
        kb = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support_b,
        )
        eval_calls: list = []

        def f_callable(x):
            eval_calls.append(1)
            return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

        f = Function(lebesgue_space, evaluate_callable=f_callable, support=[(0.0, 1.0)])
        D = EuclideanSpace(2)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[ka, kb],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        _ = G(f)

        assert len(eval_calls) == 2, (
            f"Expected f evaluated twice (once per support group); got {len(eval_calls)}"
        )

    # ------------------------------------------------------------------
    # 6. Complex dtype preservation
    # ------------------------------------------------------------------

    def test_grouped_support_restricted_complex_dtype_preserved(
        self, lebesgue_space, unit_domain
    ):
        """Grouped compact-support path preserves complex dtype."""
        support = [(0.25, 0.75)]
        kernel = Function(
            unit_domain,
            evaluate_callable=lambda x: np.exp(1j * np.pi * np.asarray(x)),
            support=support,
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[kernel],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert np.iscomplexobj(result), "Result should be complex-valued"
        assert_allclose(result, generic, rtol=1e-10, atol=1e-12)

    def test_two_grouped_complex_kernels_correct(
        self, lebesgue_space, unit_domain
    ):
        """Two complex kernels in the same support group give correct complex results."""
        support = [(0.0, 1.0)]
        k0 = Function(
            unit_domain,
            evaluate_callable=lambda x: np.exp(1j * np.pi * np.asarray(x)),
            support=support,
        )
        k1 = Function(
            unit_domain,
            evaluate_callable=lambda x: np.exp(-1j * np.pi * np.asarray(x)),
            support=support,
        )
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        D = EuclideanSpace(2)
        cfg = IntegrationConfig(method="simpson", n_points=3000)
        G = SOLAOperator(lebesgue_space, D, kernels=[k0, k1], integration_config=cfg)

        result = G(f)
        generic = G._apply_kernels_generic(f)
        assert np.iscomplexobj(result)
        assert_allclose(result, generic, rtol=1e-8, atol=1e-12)

    # ------------------------------------------------------------------
    # 7. Equivalence with generic path for existing compact-support tests
    # ------------------------------------------------------------------

    def test_narrow_support_grouped_matches_generic(
        self, lebesgue_space, unit_domain
    ):
        """Narrow compact support maintained via grouped path still matches generic."""
        support = [(0.499, 0.501)]

        def narrow_kernel(x):
            x_arr = np.asarray(x)
            return np.where((x_arr >= 0.499) & (x_arr <= 0.501), 1.0, 0.0)

        kernel = Function(unit_domain, evaluate_callable=narrow_kernel, support=support)
        f = Function(
            lebesgue_space,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[kernel],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        fast_result = G(f)
        generic_result = G._apply_kernels_generic(f)
        assert_allclose(fast_result, generic_result, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 17. Phase 5 fix – Lazy full-domain f-evaluation
# ---------------------------------------------------------------------------


class TestLazyFullDomainEvaluation:
    """Guard tests for the review-blocker fix: full-domain mesh construction
    and ``f(xs)`` evaluation are deferred until after kernel classification
    and only computed when at least one full-domain kernel exists.

    Behavioral contracts:
    1. Disjoint-only workload: ``_shared_mesh`` stays ``None`` after forward call
       (shared mesh never built → f never evaluated on full domain).
    2. Support-only workload: ``_shared_mesh`` stays ``None`` after forward call.
    3. Mixed workload (one full-domain kernel present): mesh IS built and f IS
       evaluated (regression guard ensuring the fix does not break the batched path).
    4. Correctness: disjoint-only and support-only results are still exact.
    5. Support-only: f callable is called only on restricted meshes, never
       on the full-domain mesh (verified by callable-level instrumentation).
    """

    # ------------------------------------------------------------------
    # 1. Disjoint-only workload – mesh never built
    # ------------------------------------------------------------------

    def test_disjoint_only_shared_mesh_stays_none(
        self, lebesgue_space, unit_domain
    ):
        """All kernels disjoint: _shared_mesh must still be None after the call."""
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.7, 1.0)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.0, 0.3)])

        assert G._shared_mesh is None, "Pre-condition: mesh not built yet"
        _ = G(f)
        assert G._shared_mesh is None, (
            "Shared mesh must not be built for a disjoint-only workload"
        )

    def test_disjoint_only_multiple_kernels_mesh_stays_none(
        self, lebesgue_space, unit_domain
    ):
        """Multiple disjoint kernels: _shared_mesh still None after the call."""
        D = EuclideanSpace(3)
        kernels = [
            Function(unit_domain,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=[(0.7, 1.0)]),
            Function(unit_domain,
                     evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.8, 1.0)]),
            Function(unit_domain,
                     evaluate_callable=lambda x: np.asarray(x) ** 2,
                     support=[(0.6, 1.0)]),
        ]
        G = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.0, 0.2)])
        _ = G(f)
        assert G._shared_mesh is None, (
            "Shared mesh must not be built when all kernels are disjoint"
        )

    # ------------------------------------------------------------------
    # 2. Support-only workload – mesh never built
    # ------------------------------------------------------------------

    def test_support_only_shared_mesh_stays_none(
        self, lebesgue_space, unit_domain
    ):
        """All kernels have compact (non-disjoint) support: _shared_mesh stays None."""
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.2, 0.8)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.0, 1.0)])

        assert G._shared_mesh is None
        _ = G(f)
        assert G._shared_mesh is None, (
            "Shared mesh must not be built for a support-only workload "
            "(all kernels have compact support, none is full-domain)"
        )

    def test_support_only_multiple_kernels_mesh_stays_none(
        self, lebesgue_space, unit_domain
    ):
        """Multiple compact-support kernels: _shared_mesh still None."""
        D = EuclideanSpace(3)
        kernels = [
            Function(unit_domain,
                     evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=[(0.1, 0.4)]),
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.2, 0.5)]),
            Function(unit_domain, evaluate_callable=lambda x: np.asarray(x) ** 2,
                     support=[(0.3, 0.6)]),
        ]
        G = SOLAOperator(
            lebesgue_space, D, kernels=kernels,
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.asarray(x),
                     support=[(0.0, 1.0)])
        _ = G(f)
        assert G._shared_mesh is None, (
            "Shared mesh must not be built when all kernels have compact support"
        )

    # ------------------------------------------------------------------
    # 3. Mixed workload – mesh IS built (regression guard)
    # ------------------------------------------------------------------

    def test_mixed_workload_shared_mesh_is_built(
        self, lebesgue_space, unit_domain
    ):
        """When at least one full-domain kernel is present, the shared mesh must
        be constructed (regression guard: fix must not suppress the batched path)."""
        k_full = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k_full],
            integration_config=IntegrationConfig(method="simpson", n_points=200),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.asarray(x))
        assert G._shared_mesh is None
        _ = G(f)
        assert G._shared_mesh is not None, (
            "Shared mesh must be built when a full-domain kernel is present"
        )

    # ------------------------------------------------------------------
    # 4. Correctness – lazy path still gives exact/correct results
    # ------------------------------------------------------------------

    def test_disjoint_only_result_exactly_zero(
        self, lebesgue_space, unit_domain
    ):
        """Disjoint workload gives exactly 0 (no numerical noise) even with lazy fix."""
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.sin(np.pi * np.asarray(x)),
            support=[(0.6, 1.0)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.sin(np.pi * np.asarray(x)),
                     support=[(0.0, 0.4)])
        result = G(f)
        assert result[0] == 0.0, f"Disjoint result must be exactly 0; got {result[0]}"
        s = G.stats
        assert s["disjoint_skips"] == 1
        assert s["batched_fixed_grid_kernels"] == 0

    def test_support_only_result_matches_analytic(
        self, lebesgue_space, unit_domain
    ):
        """Support-only workload gives correct analytic result without full-domain eval."""
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=[(0.3, 0.7)],
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=2000),
        )
        f = Function(lebesgue_space, evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
                     support=[(0.0, 1.0)])
        result = G(f)
        # ∫_{0.3}^{0.7} 1 dx = 0.4
        assert_allclose(result[0], 0.4, rtol=1e-4)
        assert G._shared_mesh is None, "Shared mesh must not be built for support-only"

    # ------------------------------------------------------------------
    # 5. Support-only: f never evaluated on full-domain mesh
    # ------------------------------------------------------------------

    def test_support_only_f_not_evaluated_on_full_mesh(
        self, lebesgue_space, unit_domain
    ):
        """Support-only workload: f's callable is only invoked with x values inside
        the kernel's support range — never called on the full-domain mesh [0, 1].

        With the lazy fix, ``_get_or_build_mesh()`` is never called, so ``f``
        cannot be evaluated on the full-domain linspace.  The restricted mesh
        for a support of [(0.2, 0.8)] only contains values in [0.2, 0.8].
        """
        support = [(0.2, 0.8)]
        k = Function(
            unit_domain,
            evaluate_callable=lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
            support=support,
        )
        D = EuclideanSpace(1)
        G = SOLAOperator(
            lebesgue_space, D, kernels=[k],
            integration_config=IntegrationConfig(method="simpson", n_points=500),
        )

        call_xs: list = []

        def f_callable(x):
            x_arr = np.asarray(x).ravel()
            call_xs.extend(x_arr.tolist())
            return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

        f = Function(lebesgue_space, evaluate_callable=f_callable,
                     support=[(0.0, 1.0)])
        _ = G(f)

        # f must have been called at least once (on the restricted mesh).
        assert len(call_xs) >= 1, "f_callable was never invoked"
        # Every x value passed to f must lie within the kernel support [0.2, 0.8].
        # A regression (eager full-domain eval) would include 0.0 and 1.0.
        assert all(0.2 <= x <= 0.8 for x in call_xs), (
            f"f was evaluated outside kernel support [0.2, 0.8]: "
            f"min={min(call_xs):.6f}, max={max(call_xs):.6f}"
        )


# ---------------------------------------------------------------------------
# Phase N: _reconstruct_function fast path
# ---------------------------------------------------------------------------

class TestReconstructFunctionFastPath:
    """Fast-path in _reconstruct_function: when kernel eval cache is fully
    populated, G*(z).evaluate(shared_mesh) uses K_mat.T @ z instead of
    looping over M kernel.evaluate calls.

    Fast path is tracked via stats['fast_path_hits'].
    """

    @pytest.fixture
    def cached_sola(self):
        """SOLAOperator with cache_kernels=True, sine kernels, pre-warmed."""
        domain = IntervalDomain(0.0, 1.0)
        M = Lebesgue(0, domain, basis=None,
                     integration_config=None)
        N_d = 8
        D = EuclideanSpace(N_d)
        # Simple analytic kernels: k_i(x) = sin((i+1) * pi * x)
        kernels = [
            Function(domain,
                     evaluate_callable=(lambda n: (lambda x: np.sin((n + 1) * np.pi * x)))(i))
            for i in range(N_d)
        ]
        G = SOLAOperator(M, D, kernels=kernels, cache_kernels=True,
                         integration_config=IntegrationConfig("simpson", 1000))
        # Warm the kernel eval cache with one forward pass
        m = Function(domain, evaluate_callable=lambda x: np.sin(np.pi * x))
        _ = G(m)
        return G, M, D, N_d

    # ------------------------------------------------------------------
    # 1. stats['fast_path_hits'] exists and is incremented by G(G*(z))
    # ------------------------------------------------------------------

    def test_fast_path_stats_key_exists(self, cached_sola):
        """stats dict must contain 'fast_path_hits' key (added by implementation)."""
        G, M, D, N_d = cached_sola
        assert 'fast_path_hits' in G.stats, (
            "'fast_path_hits' key missing from SOLAOperator.stats — "
            "fast path not yet implemented"
        )

    def test_fast_path_increments_stats(self, cached_sola):
        """G(G*(z)) must increment stats['fast_path_hits'] exactly once."""
        G, M, D, N_d = cached_sola
        G.reset_stats()
        z = np.ones(N_d) / N_d
        u_adj = G.adjoint(z)
        _ = G(u_adj)
        hits = G.stats.get('fast_path_hits', 0)
        assert hits >= 1, (
            f"Expected fast_path_hits >= 1 after G(G*(z)), got {hits}. "
            "Fast path not taken — kernel loop is used instead."
        )

    def test_fast_path_not_hit_without_cache(self):
        """Without cache_kernels=True, fast path must NOT be taken."""
        domain = IntervalDomain(0.0, 1.0)
        M = Lebesgue(0, domain, basis=None, integration_config=None)
        N_d = 4
        D = EuclideanSpace(N_d)
        kernels = [
            Function(domain,
                     evaluate_callable=(lambda n: (lambda x: np.sin((n + 1) * np.pi * x)))(i))
            for i in range(N_d)
        ]
        G = SOLAOperator(M, D, kernels=kernels, cache_kernels=False,
                         integration_config=IntegrationConfig("simpson", 500))
        m = Function(domain, evaluate_callable=lambda x: np.sin(np.pi * x))
        _ = G(m)
        G.reset_stats()
        z = np.ones(N_d) / N_d
        _ = G(G.adjoint(z))
        assert G.stats.get('fast_path_hits', 0) == 0, (
            "Fast path must not be taken when cache_kernels=False"
        )

    # ------------------------------------------------------------------
    # 2. Correctness: G(G*(z)) == P_mat @ z
    # ------------------------------------------------------------------

    def test_fast_path_correctness_direct(self, cached_sola):
        """G(G*(z)) equals the Gram matrix times z."""
        G, M, D, N_d = cached_sola
        P_mat = (G @ G.adjoint).matrix(dense=True)
        np.random.seed(7)
        z = np.random.randn(N_d)
        expected = P_mat @ z
        result = G(G.adjoint(z))
        assert_allclose(result, expected, rtol=1e-5,
                        err_msg="G(G*(z)) != P_mat @ z (fast path breaks correctness)")

    def test_fast_path_correctness_scaled(self, cached_sola):
        """G(a * G*(z)) == a * P_mat @ z for scalar a."""
        G, M, D, N_d = cached_sola
        P_mat = (G @ G.adjoint).matrix(dense=True)
        np.random.seed(8)
        z = np.random.randn(N_d)
        a = 3.14
        u = M.multiply(a, G.adjoint(z))
        result = G(u)
        assert_allclose(result, a * (P_mat @ z), rtol=1e-5,
                        err_msg="G(a * G*(z)) != a * P_mat @ z")

    def test_fast_path_correctness_sum(self, cached_sola):
        """G(G*(z1) + G*(z2)) == P_mat @ (z1 + z2)."""
        G, M, D, N_d = cached_sola
        P_mat = (G @ G.adjoint).matrix(dense=True)
        np.random.seed(9)
        z1 = np.random.randn(N_d)
        z2 = np.random.randn(N_d)
        u = M.add(G.adjoint(z1), G.adjoint(z2))
        result = G(u)
        assert_allclose(result, P_mat @ (z1 + z2), rtol=1e-5,
                        err_msg="G(G*(z1) + G*(z2)) != P_mat @ (z1 + z2)")

    # ------------------------------------------------------------------
    # 3. Correctness: arbitrary mesh falls back without corrupting result
    # ------------------------------------------------------------------

    def test_fast_path_fallback_arbitrary_mesh(self, cached_sola):
        """Evaluating G*(z) on a mesh that is NOT the shared mesh still works."""
        G, M, D, N_d = cached_sola
        np.random.seed(10)
        z = np.random.randn(N_d)
        u_adj = G.adjoint(z)
        # Use a different mesh (different object, different size)
        xs_other = np.linspace(0.0, 1.0, 333)
        vals = u_adj.evaluate(xs_other, check_domain=False)
        # Verify manually: should equal sum_i z_i * K_i(xs)
        kernels = G.get_kernels()
        expected = sum(z[i] * kernels[i].evaluate(xs_other, check_domain=False)
                       for i in range(N_d))
        assert_allclose(vals, expected, rtol=1e-10,
                        err_msg="Fallback path (non-shared mesh) returns wrong values")

    # ------------------------------------------------------------------
    # 4. Fast path also taken inside G(G*(z)) composition chain
    # ------------------------------------------------------------------

    def test_fast_path_stats_via_G_call(self, cached_sola):
        """stats['fast_path_hits'] increments when G is called on G*(z),
        because _apply_kernels_fixed_grid evaluates G*(z) on the shared mesh."""
        G, M, D, N_d = cached_sola
        G.reset_stats()
        z = np.random.default_rng(11).standard_normal(N_d)
        _ = G(G.adjoint(z))
        assert G.stats.get('fast_path_hits', 0) >= 1, (
            "Fast path not taken inside G(G*(z)) — "
            "shared mesh id check may be broken"
        )
