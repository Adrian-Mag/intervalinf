"""
Tests for Lebesgue space.

Tests the Lebesgue class and related functionality including:
- Basic space creation
- Inner products
- Dual mappings
- Configuration management
- Restriction operations
- Known regions and partitioned spaces
"""

import pytest
import numpy as np

from intervalinf.core import IntervalDomain, Function
from intervalinf.core.config import IntegrationConfig, ParallelConfig
from intervalinf.spaces.lebesgue import (
    Lebesgue,
    LebesgueSpaceDirectSum,
    LebesgueIntegrationConfig,
    LebesgueParallelConfig,
    KnownRegion,
    PartitionedLebesgueSpace,
)


class TestLebesgueInit:
    """Test Lebesgue space initialization."""

    def test_basic_init(self):
        """Test basic initialization with required arguments."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        assert space.dim == 50
        assert space.function_domain == domain

    def test_init_baseless(self):
        """Test initialization with basis='none'."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain, basis='none')

        assert space._basis_type == 'none'

    def test_init_with_callable_basis(self):
        """Test initialization with list of callables."""
        domain = IntervalDomain(0, 1)
        basis = [
            lambda x: np.ones_like(x),
            lambda x: x,
            lambda x: x**2,
        ]
        space = Lebesgue(3, domain, basis=basis)

        assert space._basis_type == 'direct_functions'
        assert len(space._basis_functions) == 3

    def test_init_callable_basis_dimension_mismatch(self):
        """Test that mismatched basis count raises error."""
        domain = IntervalDomain(0, 1)
        basis = [lambda x: 1, lambda x: x]

        with pytest.raises(ValueError, match="must match dimension"):
            Lebesgue(5, domain, basis=basis)

    def test_init_invalid_basis_type(self):
        """Test that invalid basis type raises error."""
        domain = IntervalDomain(0, 1)

        with pytest.raises(TypeError, match="must be a string or list"):
            Lebesgue(5, domain, basis=42)  # type: ignore


class TestLebesgueProperties:
    """Test Lebesgue space properties."""

    def test_dim_property(self):
        """Test dim property."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(100, domain)
        assert space.dim == 100

    def test_function_domain_property(self):
        """Test function_domain property."""
        domain = IntervalDomain(-1, 2)
        space = Lebesgue(50, domain)
        assert space.function_domain.a == -1
        assert space.function_domain.b == 2

    def test_zero_property(self):
        """Test zero function property."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)
        zero = space.zero

        assert isinstance(zero, Function)
        x = np.linspace(0, 1, 10)
        np.testing.assert_array_almost_equal(zero.evaluate(x), 0)


class TestLebesgueInnerProduct:
    """Test Lebesgue space inner product."""

    def test_inner_product_orthogonal(self):
        """Test inner product of orthogonal functions (sin/cos)."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        f = Function(space, evaluate_callable=lambda x: np.sin(2 * np.pi * x))
        g = Function(space, evaluate_callable=lambda x: np.cos(2 * np.pi * x))

        inner = space.inner_product(f, g)
        # sin and cos are orthogonal on [0, 1]
        assert abs(inner) < 1e-6

    def test_inner_product_norm_squared(self):
        """Test <f, f> equals norm squared."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        # sin²(πx) integrates to 0.5 on [0, 1]
        f = Function(space, evaluate_callable=lambda x: np.sin(np.pi * x))
        inner = space.inner_product(f, f)

        expected = 0.5  # ∫₀¹ sin²(πx) dx = 0.5
        assert abs(inner - expected) < 1e-3

    def test_inner_product_constant(self):
        """Test inner product of constant functions."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        f = Function(space, evaluate_callable=lambda x: np.full_like(x, 2.0))
        g = Function(space, evaluate_callable=lambda x: np.full_like(x, 3.0))

        inner = space.inner_product(f, g)
        expected = 6.0  # 2 * 3 * length
        assert abs(inner - expected) < 1e-6


class TestLebesgueEquality:
    """Test Lebesgue space equality."""

    def test_equal_spaces(self):
        """Test that equivalent spaces are equal."""
        domain1 = IntervalDomain(0, 1)
        domain2 = IntervalDomain(0, 1)
        space1 = Lebesgue(50, domain1)
        space2 = Lebesgue(50, domain2)

        assert space1 == space2

    def test_unequal_dimension(self):
        """Test that different dimensions are unequal."""
        domain = IntervalDomain(0, 1)
        space1 = Lebesgue(50, domain)
        space2 = Lebesgue(100, domain)

        assert space1 != space2

    def test_unequal_domain(self):
        """Test that different domains are unequal."""
        space1 = Lebesgue(50, IntervalDomain(0, 1))
        space2 = Lebesgue(50, IntervalDomain(0, 2))

        assert space1 != space2

    def test_unequal_basis_representation(self):
        """Different coordinate bases are not operationally compatible."""
        domain = IntervalDomain(0, 1)
        sine_space = Lebesgue(3, domain, basis="sine")
        cosine_space = Lebesgue(3, domain, basis="cosine")

        assert sine_space != cosine_space

    def test_equal_matching_basis_representation(self):
        """Equivalent independently built standard bases remain equal."""
        domain = IntervalDomain(0, 1)
        space1 = Lebesgue(3, domain, basis="sine")
        space2 = Lebesgue(3, domain, basis="sine")

        assert space1 == space2

    def test_unequal_direct_callable_bases(self):
        """Arbitrary callable bases require the same callable identities."""
        domain = IntervalDomain(0, 1)
        constant = lambda x: np.ones_like(x)
        linear = lambda x: np.asarray(x)
        quadratic = lambda x: np.asarray(x) ** 2
        space1 = Lebesgue(2, domain, basis=[constant, linear])
        space2 = Lebesgue(2, domain, basis=[constant, quadratic])

        assert space1 != space2

        space3 = Lebesgue(2, domain, basis=[constant, linear])
        assert space1 == space3

    def test_not_equal_to_non_lebesgue(self):
        """Test that Lebesgue is not equal to other types."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        assert space != "not a space"
        assert space != 42


class TestLebesgueVectorOperations:
    """Test vector space operations."""

    def test_add(self):
        """Test function addition."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        f = Function(space, evaluate_callable=lambda x: x)
        g = Function(space, evaluate_callable=lambda x: x**2)

        h = space.add(f, g)
        x = np.array([0.0, 0.5, 1.0])
        expected = x + x**2
        np.testing.assert_array_almost_equal(h.evaluate(x), expected)

    def test_multiply(self):
        """Test scalar multiplication."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        f = Function(space, evaluate_callable=lambda x: x**2)
        g = space.multiply(3.0, f)

        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(g.evaluate(x), 3 * x**2)


class TestLebesgueRestriction:
    """Test restriction to subintervals."""

    def test_restrict_to_subinterval(self):
        """Test restriction creates proper space."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        subdomain = IntervalDomain(0.25, 0.75)
        restricted = space.restrict_to_subinterval(subdomain)

        assert restricted.dim == space.dim
        assert restricted.function_domain.a == 0.25
        assert restricted.function_domain.b == 0.75

    def test_restrict_invalid_subdomain(self):
        """Test that invalid subdomain raises error."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        # Subdomain extends beyond original
        with pytest.raises(ValueError, match="must be contained"):
            space.restrict_to_subinterval(IntervalDomain(-1, 0.5))


class TestLebesgueIntegrationConfig:
    """Test integration configuration."""

    def test_default_config(self):
        """Test default integration config."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        assert hasattr(space, 'integration')
        assert hasattr(space.integration, 'inner_product')
        assert hasattr(space.integration, 'dual')
        assert hasattr(space.integration, 'general')

    def test_from_single_config(self):
        """Test from_single creates uniform config."""
        config = IntegrationConfig(method='simpson', n_points=10000)
        leb_config = LebesgueIntegrationConfig.from_single(config)

        assert leb_config.inner_product.method == 'simpson'
        assert leb_config.dual.method == 'simpson'
        assert leb_config.general.method == 'simpson'

    def test_high_accuracy_galerkin(self):
        """Test high accuracy preset."""
        config = LebesgueIntegrationConfig.high_accuracy_galerkin()

        assert config.inner_product.n_points == 20000
        assert config.dual.n_points == 10000
        assert config.general.n_points == 5000

    def test_backward_compatible_properties(self):
        """Test backward-compatible integration properties."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain)

        # Get default values
        method = space.integration_method
        npoints = space.integration_npoints

        assert method in ['trapz', 'simpson']
        assert npoints > 0

        # Set new values
        space.integration_method = 'simpson'
        space.integration_npoints = 5000

        assert space.integration.general.method == 'simpson'
        assert space.integration.general.n_points == 5000


class TestLebesgueParallelConfig:
    """Test parallel configuration."""

    def test_default_parallel_disabled(self):
        """Test default parallel is disabled."""
        config = LebesgueParallelConfig()
        assert config.inner_product.enabled is False
        assert config.dual.enabled is False
        assert config.general.enabled is False

    def test_parallel_dual_preset(self):
        """Test parallel_dual preset."""
        config = LebesgueParallelConfig.parallel_dual(n_jobs=4)

        assert config.inner_product.enabled is False
        assert config.dual.enabled is True
        assert config.dual.n_jobs == 4

    def test_full_parallel_preset(self):
        """Test full_parallel preset."""
        config = LebesgueParallelConfig.full_parallel(n_jobs=8)

        assert config.inner_product.enabled is True
        assert config.dual.enabled is True
        assert config.general.enabled is True
        assert config.inner_product.n_jobs == 8


class TestKnownRegion:
    """Test KnownRegion class."""

    def test_zero_factory(self):
        """Test KnownRegion.zero factory."""
        interval = IntervalDomain(0, 1)
        region = KnownRegion.zero(interval)

        assert region.interval == interval
        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(region.value.evaluate(x), 0)

    def test_constant_factory(self):
        """Test KnownRegion.constant factory."""
        interval = IntervalDomain(0, 1)
        region = KnownRegion.constant(interval, 5.0)

        x = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(region.value.evaluate(x), 5.0)

    def test_invalid_function_type(self):
        """Test that non-Function value raises TypeError."""
        interval = IntervalDomain(0, 1)

        with pytest.raises(TypeError, match="must be a Function"):
            KnownRegion(interval, "not a function")  # type: ignore

    def test_domain_mismatch(self):
        """Test that domain mismatch raises ValueError."""
        interval1 = IntervalDomain(0, 1)
        interval2 = IntervalDomain(0, 2)
        space = Lebesgue(10, interval2)
        func = Function(space, evaluate_callable=lambda x: x)

        with pytest.raises(ValueError, match="doesn't match"):
            KnownRegion(interval1, func)


class TestPartitionedLebesgueSpace:
    """Test PartitionedLebesgueSpace."""

    def test_single_known_region(self):
        """Test space with one known region in middle."""
        full = IntervalDomain(0, 3)
        known = KnownRegion.zero(IntervalDomain(1, 2))

        partitioned = PartitionedLebesgueSpace(
            full, [known], dims=[10, 10], basis='none'
        )

        assert partitioned.n_unknown_regions == 2
        assert partitioned.n_known_regions == 1
        assert len(partitioned.unknown_intervals) == 2
        assert partitioned.unknown_intervals[0].a == 0
        assert partitioned.unknown_intervals[0].b == 1
        assert partitioned.unknown_intervals[1].a == 2
        assert partitioned.unknown_intervals[1].b == 3

    def test_no_known_regions(self):
        """Test space with no known regions."""
        full = IntervalDomain(0, 1)

        partitioned = PartitionedLebesgueSpace(
            full, [], dims=[20], basis='none'
        )

        assert partitioned.n_unknown_regions == 1
        assert partitioned.n_known_regions == 0
        assert partitioned.unknown_intervals[0] == full

    def test_dims_mismatch_raises(self):
        """Test that wrong number of dims raises error."""
        full = IntervalDomain(0, 3)
        known = KnownRegion.zero(IntervalDomain(1, 2))

        with pytest.raises(ValueError, match="must have 2 elements"):
            PartitionedLebesgueSpace(
                full, [known], dims=[10, 10, 10], basis='none'
            )

    def test_overlapping_regions_raises(self):
        """Test that overlapping known regions raise error."""
        full = IntervalDomain(0, 3)
        known1 = KnownRegion.zero(IntervalDomain(0.5, 1.5))
        known2 = KnownRegion.zero(IntervalDomain(1.0, 2.0))

        with pytest.raises(ValueError, match="overlap"):
            PartitionedLebesgueSpace(
                full, [known1, known2], dims=[10, 10], basis='none'
            )

    def test_get_unknown_space(self):
        """Test get_unknown_space method."""
        full = IntervalDomain(0, 2)
        known = KnownRegion.zero(IntervalDomain(0.5, 1.5))

        partitioned = PartitionedLebesgueSpace(
            full, [known], dims=[10, 15], basis='none'
        )

        space0 = partitioned.get_unknown_space(0)
        space1 = partitioned.get_unknown_space(1)

        assert space0.dim == 10
        assert space1.dim == 15

    def test_extend_to_full_domain(self):
        """Test extending model to full domain."""
        full = IntervalDomain(0, 2)
        known = KnownRegion.constant(IntervalDomain(0.5, 1.5), 10.0)

        partitioned = PartitionedLebesgueSpace(
            full, [known], dims=[5, 5], basis='none'
        )

        # Create functions on unknown regions
        f1 = Function(
            partitioned.unknown_spaces[0],
            evaluate_callable=lambda x: np.full_like(x, 1.0)
        )
        f2 = Function(
            partitioned.unknown_spaces[1],
            evaluate_callable=lambda x: np.full_like(x, 2.0)
        )

        extended = partitioned.extend_to_full_domain([f1, f2])

        # Test values in different regions
        assert extended.evaluate(0.25) == pytest.approx(1.0)  # Unknown 1
        assert extended.evaluate(1.0) == pytest.approx(10.0)  # Known
        assert extended.evaluate(1.75) == pytest.approx(2.0)  # Unknown 2


class TestLebesgueSpaceDirectSum:
    """Test LebesgueSpaceDirectSum."""

    def test_create_direct_sum(self):
        """Test creating direct sum of Lebesgue spaces."""
        domain1 = IntervalDomain(0, 1)
        domain2 = IntervalDomain(1, 2)

        space1 = Lebesgue(10, domain1)
        space2 = Lebesgue(15, domain2)

        direct_sum = LebesgueSpaceDirectSum([space1, space2])

        assert direct_sum.number_of_subspaces == 2
        assert direct_sum.dim == 25

    def test_to_dual_direct_sum(self):
        """Test to_dual mapping for direct sum."""
        domain1 = IntervalDomain(0, 1)
        domain2 = IntervalDomain(1, 2)

        space1 = Lebesgue(10, domain1)
        space2 = Lebesgue(15, domain2)

        direct_sum = LebesgueSpaceDirectSum([space1, space2])

        f1 = Function(space1, evaluate_callable=lambda x: x)
        f2 = Function(space2, evaluate_callable=lambda x: x**2)

        dual = direct_sum.to_dual([f1, f2])

        assert isinstance(dual._kernel, list)
        assert len(dual._kernel) == 2

    def test_from_dual_direct_sum(self):
        """Test from_dual for direct sum."""
        domain1 = IntervalDomain(0, 1)
        domain2 = IntervalDomain(1, 2)

        space1 = Lebesgue(10, domain1)
        space2 = Lebesgue(15, domain2)

        direct_sum = LebesgueSpaceDirectSum([space1, space2])

        f1 = Function(space1, evaluate_callable=lambda x: x)
        f2 = Function(space2, evaluate_callable=lambda x: x**2)

        dual = direct_sum.to_dual([f1, f2])
        recovered = direct_sum.from_dual(dual)

        assert len(recovered) == 2


class TestLebesgueWithDiscontinuities:
    """Test Lebesgue.with_discontinuities factory."""

    def test_single_discontinuity(self):
        """Test creating space with single discontinuity."""
        domain = IntervalDomain(0, 2)

        space = Lebesgue.with_discontinuities(
            20, domain, [1.0], basis='none'
        )

        assert isinstance(space, LebesgueSpaceDirectSum)
        assert space.number_of_subspaces == 2
        assert space.dim == 20

    def test_multiple_discontinuities(self):
        """Test creating space with multiple discontinuities."""
        domain = IntervalDomain(0, 3)

        space = Lebesgue.with_discontinuities(
            30, domain, [1.0, 2.0], basis='none'
        )

        assert space.number_of_subspaces == 3
        assert space.dim == 30

    def test_custom_dims_per_subspace(self):
        """Test custom dimensions per subspace."""
        domain = IntervalDomain(0, 2)

        space = Lebesgue.with_discontinuities(
            30, domain, [1.0],
            dim_per_subspace=[10, 20],
            basis='none'
        )

        assert space.subspace(0).dim == 10
        assert space.subspace(1).dim == 20

    def test_list_basis_raises(self):
        """Test that list basis raises error."""
        domain = IntervalDomain(0, 2)

        with pytest.raises(ValueError, match="list of basis functions"):
            Lebesgue.with_discontinuities(
                20, domain, [1.0],
                basis=[lambda x: x, lambda x: x**2]  # type: ignore
            )


class TestLebesgueBasisless:
    """Test baseless Lebesgue space operations."""

    def test_require_basis_raises(self):
        """Test that basis-requiring operations raise on baseless space."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain, basis='none')

        with pytest.raises(RuntimeError, match="requires a basis"):
            space.to_components(space.zero)

    def test_from_components_requires_basis(self):
        """Test from_components requires basis."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain, basis='none')

        with pytest.raises(RuntimeError, match="requires a basis"):
            space.from_components(np.zeros(50))

    def test_basis_functions_property_raises(self):
        """Test basis_functions property raises on baseless space."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain, basis='none')

        with pytest.raises(RuntimeError, match="No basis functions"):
            _ = space.basis_functions

    def test_get_basis_function_raises(self):
        """Test get_basis_function raises on baseless space."""
        domain = IntervalDomain(0, 1)
        space = Lebesgue(50, domain, basis='none')

        with pytest.raises(RuntimeError, match="No basis functions"):
            space.get_basis_function(0)


class TestLebesgueFromComponentsSupport:
    """Test Phase 2: support-metadata inference in from_components and algebra."""

    @pytest.fixture
    def boxcar_space(self):
        """Lebesgue space backed by BoxCarFunctionProvider (compact support)."""
        from intervalinf.providers import BoxCarFunctionProvider, CustomBasisProvider

        domain = IntervalDomain(0, 1)
        n = 5
        space = Lebesgue(n, domain)
        centers = np.linspace(0.1, 0.9, n)
        boxcar_fp = BoxCarFunctionProvider(
            space, centers=centers, default_width=0.15
        )
        basis_provider = CustomBasisProvider(space, boxcar_fp, basis_type='step')
        space.set_basis_provider(basis_provider)
        return space

    @pytest.fixture
    def fourier_space(self):
        """Lebesgue space with Fourier basis (globally supported)."""
        domain = IntervalDomain(0, 1)
        return Lebesgue(5, domain, basis='fourier')

    # ------------------------------------------------------------------
    # zero property
    # ------------------------------------------------------------------

    def test_zero_property_has_empty_support(self, boxcar_space):
        """space.zero should have support=[] (no nonzero contribution)."""
        zero = boxcar_space.zero
        assert zero.support == []

    # ------------------------------------------------------------------
    # from_components support inference
    # ------------------------------------------------------------------

    def test_from_components_all_zero_gives_empty_support(self, boxcar_space):
        """All-zero coefficient array → Function with support=[]."""
        result = boxcar_space.from_components(np.zeros(5))
        assert result.support == []

    def test_from_components_boxcar_single_nonzero_gives_basis_support(
        self, boxcar_space
    ):
        """Unit vector at index i → support matches the i-th basis function."""
        i = 2
        coeffs = np.zeros(5)
        coeffs[i] = 1.0
        result = boxcar_space.from_components(coeffs)
        bf_support = boxcar_space.get_basis_function(i).support
        assert result.support == bf_support

    def test_from_components_boxcar_two_nonzero_gives_union_support(
        self, boxcar_space
    ):
        """Two nonzero entries → support is union of their basis-function supports."""
        coeffs = np.zeros(5)
        coeffs[0] = 1.0
        coeffs[4] = 1.0
        result = boxcar_space.from_components(coeffs)
        bf0 = boxcar_space.get_basis_function(0).support
        bf4 = boxcar_space.get_basis_function(4).support
        expected = Function._union_supports(bf0, bf4)
        assert result.support == expected

    def test_from_components_fourier_any_nonzero_gives_none_support(
        self, fourier_space
    ):
        """Active Fourier (global) basis function → result.support is None."""
        coeffs = np.zeros(5)
        coeffs[1] = 1.0
        result = fourier_space.from_components(coeffs)
        assert result.support is None

    # ------------------------------------------------------------------
    # multiply support propagation
    # ------------------------------------------------------------------

    def test_multiply_zero_scalar_gives_empty_support(self, boxcar_space):
        """Multiplying a coefficient function by 0 → support=[]."""
        coeffs = np.zeros(5)
        coeffs[2] = 1.0
        f = boxcar_space.from_components(coeffs)
        result = boxcar_space.multiply(0.0, f)
        assert result.support == []

    def test_multiply_nonzero_scalar_preserves_support(self, boxcar_space):
        """Multiplying by a nonzero scalar keeps original support."""
        coeffs = np.zeros(5)
        coeffs[2] = 1.0
        f = boxcar_space.from_components(coeffs)
        result = boxcar_space.multiply(2.0, f)
        assert result.support == f.support

    # ------------------------------------------------------------------
    # add support propagation
    # ------------------------------------------------------------------

    def test_add_coefficient_union_support(self, boxcar_space):
        """add(f, g) with coefficient functions → union of supports."""
        ca = np.zeros(5)
        ca[0] = 1.0
        cb = np.zeros(5)
        cb[4] = 1.0
        fa = boxcar_space.from_components(ca)
        fb = boxcar_space.from_components(cb)
        result = boxcar_space.add(fa, fb)
        expected = Function._union_supports(fa.support, fb.support)
        assert result.support == expected

    # ------------------------------------------------------------------
    # axpy support propagation
    # ------------------------------------------------------------------

    def test_axpy_updates_support_in_place(self, boxcar_space):
        """axpy(a, x, y) updates y.support to union(y.support, x.support)."""
        cx = np.zeros(5)
        cx[0] = 1.0
        cy = np.zeros(5)
        cy[4] = 1.0
        x = boxcar_space.from_components(cx)
        y = boxcar_space.from_components(cy)
        expected_support = Function._union_supports(y.support, x.support)
        result = boxcar_space.axpy(1.0, x, y)
        assert result.support == expected_support

    def test_multiply_zero_globally_supported_function_gives_empty_support(
        self, fourier_space
    ):
        """multiply(0, f) → support=[] even when f has no compact support."""
        coeffs = np.zeros(5)
        coeffs[1] = 1.0
        f = fourier_space.from_components(coeffs)
        assert f.support is None  # globally supported basis
        result = fourier_space.multiply(0.0, f)
        assert result.support == []


# =============================================================================
# GaussianMeasure.from_covariance_matrix non-orthonormal basis regression test
# =============================================================================


class TestGaussianMeasureFromCovarianceMatrixNonOrthonormal:
    """
    Regression tests for GaussianMeasure.from_covariance_matrix with a
    Lebesgue space backed by a non-orthonormal boxcar basis (Gram matrix G = h·I).

    Bug: from_covariance_matrix was treating its input as the component-space
    covariance E[ccᵀ] instead of the Galerkin covariance K_gal[i,j] = ⟨φᵢ, Cφⱼ⟩.
    For spaces with G ≠ I this produced samples h× too large in std-dev,
    because K_comp was set equal to K_gal rather than G⁻¹ K_gal G⁻¹.
    """

    @pytest.fixture
    def boxcar_space(self):
        """Lebesgue space with non-overlapping unit-height boxcar basis.

        N=5 cells on [0, 1], cell width h = 0.2, Gram matrix G = 0.2 · I.
        """
        from intervalinf.providers import BoxCarFunctionProvider, CustomBasisProvider

        N = 5
        domain = IntervalDomain(0, 1)
        space = Lebesgue(N, domain)
        centers = np.linspace(0.1, 0.9, N)  # 0.1, 0.3, 0.5, 0.7, 0.9
        boxcar_fp = BoxCarFunctionProvider(
            space, centers=centers, default_width=0.2
        )
        basis_provider = CustomBasisProvider(space, boxcar_fp, basis_type="step")
        space.set_basis_provider(basis_provider)
        return space

    def test_gram_matrix_is_diagonal_and_nonidentity(self, boxcar_space):
        """Gram matrix G[i,j] = ⟨φᵢ, φⱼ⟩ should be (1/cell_width)·I for
        non-overlapping normalized boxcars (height = 1/width, width = 0.2).

        Normalized boxcars have height = 1/width so that ∫ φᵢ dx = 1, giving
        G_ii = height² × width = (1/width)² × width = 1/width = 5.  The key
        property under test is G ≠ I, which is what triggers the bug.
        """
        G = boxcar_space.metric
        cell_width = 0.2
        g_expected = (1.0 / cell_width) * np.eye(boxcar_space.dim)
        np.testing.assert_allclose(
            G,
            g_expected,
            atol=1e-2,  # numerical integration tolerance
            err_msg="Gram matrix is not (1/cell_width)·I for normalized boxcars",
        )
        # Confirm G ≠ I so the bug is triggered
        assert not np.allclose(G, np.eye(boxcar_space.dim))

    def test_from_covariance_matrix_galerkin_round_trip(self, boxcar_space):
        """from_covariance_matrix(M, K_gal) must produce samples whose empirical
        Galerkin covariance E[⟨φᵢ, X⟩⟨φⱼ, X⟩] matches K_gal.

        With K_gal = h²·I (unit component variance), the Galerkin covariance of
        the generated measure should equal h²·I.  Before the fix the empirical
        Galerkin covariance equalled h⁴·I (h× over-scaled in variance).
        """
        from pygeoinf.gaussian_measure import GaussianMeasure

        np.random.seed(42)
        M = boxcar_space
        G = M.metric
        g_diag = np.diag(G)  # [0.2, 0.2, 0.2, 0.2, 0.2]
        h = g_diag[0]

        # K_gal = h²·I: Galerkin covariance corresponding to unit component variance
        K_gal = (h**2) * np.eye(M.dim)

        measure = GaussianMeasure.from_covariance_matrix(M, K_gal)

        # Draw samples and compute the empirical Galerkin covariance.
        # For diagonal G: ⟨φᵢ, f⟩ = G_ii · comp_i  (since φᵢ is a basis vector
        # with to_components returning the coefficient directly for diagonal G).
        N_samples = 5000
        samples = measure.samples(N_samples)
        comps = np.array([M.to_components(s) for s in samples])  # (N_samples, N)
        galerkin_ips = comps * g_diag[np.newaxis, :]  # ⟨φᵢ, sample⟩ for each draw
        empirical_K_gal = np.cov(galerkin_ips.T)

        np.testing.assert_allclose(
            empirical_K_gal,
            K_gal,
            atol=3.0,  # ~6× std of estimator (σ ≈ 0.5 for K_gal=25·I, N=5000)
            err_msg=(
                "Empirical Galerkin covariance does not match K_gal. "
                "from_covariance_matrix is not correctly interpreting its "
                "input as the Galerkin representation (K_comp ≠ G⁻¹ K_gal G⁻¹)."
            ),
        )
