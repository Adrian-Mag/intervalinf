"""
Lebesgue spaces on interval domains.

This module provides L² and more general Lebesgue spaces on intervals as
proper Hilbert spaces that inherit from the main pygeoinf HilbertSpace
abstract base class.

The Lebesgue space L²([a,b]) consists of square-integrable functions on
the interval [a,b] with the standard inner product:

    ⟨u, v⟩ = ∫_a^b u(x) v(x) dx

or with a weight function w(x):

    ⟨u, v⟩_w = ∫_a^b u(x) v(x) w(x) dx

This module provides:
- `Lebesgue`: The main L² space class
- `LebesgueSpaceDirectSum`: Direct sums of Lebesgue spaces
- `KnownRegion`: Specification of fixed-value regions
- `PartitionedLebesgueSpace`: Spaces with known/unknown regions
"""

from __future__ import annotations

import copy as _copy
import logging
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np

from pygeoinf import HilbertSpace, HilbertSpaceDirectSum, LinearForm

from intervalinf.core.config import (
    IntegrationConfig,
    ParallelConfig,
)
from intervalinf.core.functions import Function
from intervalinf.core.materialization import RepresentationSpec
from intervalinf.spaces.forms import LinearFormKernel
from intervalinf.providers.base import BasisProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from intervalinf.core.domain import IntervalDomain


# =============================================================================
# Hierarchical Configuration Classes (Lebesgue-specific)
# =============================================================================


class LebesgueIntegrationConfig:
    """
    Hierarchical integration configuration for Lebesgue spaces.

    Different operations in Lebesgue spaces may require different
    integration precision:
    - inner_product: For computing Gram matrices (high precision)
    - dual: For dual space mappings (to_dual, from_dual)
    - general: Default for other operations

    Attributes:
        inner_product: Config for Gram matrix computation.
        dual: Config for dual space operations.
        general: Default config for other operations.

    Example:
        >>> config = LebesgueIntegrationConfig()
        >>> config.inner_product.n_points = 10000  # High for Gram
        >>> config.dual.method = 'simpson'
    """

    def __init__(
        self,
        inner_product: Optional[IntegrationConfig] = None,
        dual: Optional[IntegrationConfig] = None,
        general: Optional[IntegrationConfig] = None,
    ):
        """Initialize with optional per-subsystem configs."""
        self.inner_product = inner_product or IntegrationConfig()
        self.dual = dual or IntegrationConfig()
        self.general = general or IntegrationConfig()

    @classmethod
    def from_single(
        cls,
        config: IntegrationConfig
    ) -> 'LebesgueIntegrationConfig':
        """Create from a single config used for all subsystems."""
        return cls(
            inner_product=_copy.copy(config),
            dual=_copy.copy(config),
            general=_copy.copy(config)
        )

    @classmethod
    def high_accuracy_galerkin(cls) -> 'LebesgueIntegrationConfig':
        """Preset for high-accuracy Galerkin methods."""
        return cls(
            inner_product=IntegrationConfig(method='simpson', n_points=20000),
            dual=IntegrationConfig(method='simpson', n_points=10000),
            general=IntegrationConfig(method='simpson', n_points=5000)
        )

    @classmethod
    def adaptive_spectral(cls, dim: int) -> 'LebesgueIntegrationConfig':
        """Create config adapted to space dimension."""
        # Higher dimension needs more integration points
        base_points = max(1000, 10 * dim)
        return cls(
            inner_product=IntegrationConfig(
                method='simpson', n_points=2 * base_points
            ),
            dual=IntegrationConfig(method='simpson', n_points=base_points),
            general=IntegrationConfig(method='trapz', n_points=base_points)
        )


class LebesgueParallelConfig:
    """
    Hierarchical parallelization configuration for Lebesgue spaces.

    Different operations may benefit from different parallelization:
    - inner_product: Gram matrix rows can be parallelized
    - dual: Dual mappings for multiple functions
    - general: Other operations

    Attributes:
        inner_product: Parallel config for Gram computation.
        dual: Parallel config for dual operations.
        general: Default parallel config.

    Example:
        >>> config = LebesgueParallelConfig()
        >>> config.dual.enabled = True
        >>> config.dual.n_jobs = 4
    """

    def __init__(
        self,
        inner_product: Optional[ParallelConfig] = None,
        dual: Optional[ParallelConfig] = None,
        general: Optional[ParallelConfig] = None,
    ):
        """Initialize with optional per-subsystem configs."""
        self.inner_product = inner_product or ParallelConfig()
        self.dual = dual or ParallelConfig()
        self.general = general or ParallelConfig()

    @classmethod
    def from_single(cls, config: ParallelConfig) -> 'LebesgueParallelConfig':
        """Create from a single config used for all subsystems."""
        return cls(
            inner_product=_copy.copy(config),
            dual=_copy.copy(config),
            general=_copy.copy(config)
        )

    @classmethod
    def parallel_dual(cls, n_jobs: int = -1) -> 'LebesgueParallelConfig':
        """Preset with parallel dual operations only."""
        return cls(
            inner_product=ParallelConfig(enabled=False, n_jobs=1),
            dual=ParallelConfig(enabled=True, n_jobs=n_jobs),
            general=ParallelConfig(enabled=False, n_jobs=1)
        )

    @classmethod
    def full_parallel(cls, n_jobs: int = -1) -> 'LebesgueParallelConfig':
        """Preset with all operations parallelized."""
        return cls(
            inner_product=ParallelConfig(enabled=True, n_jobs=n_jobs),
            dual=ParallelConfig(enabled=True, n_jobs=n_jobs),
            general=ParallelConfig(enabled=True, n_jobs=n_jobs)
        )


# =============================================================================
# Main Lebesgue Space Class
# =============================================================================


class Lebesgue(HilbertSpace):
    """
    Lebesgue space L² on an interval [a,b] with inner product
    ⟨u,v⟩ = ∫_a^b u(x)v(x) dx.

    This class properly inherits from the pygeoinf HilbertSpace abstract base
    class, using Function objects as the Vector type. It provides:

    - L² inner product and norm via integration
    - Basis-free functional operations on callable Function objects
    - Optional basis function management (Fourier, hat functions, etc.)
    - Function evaluation and coefficient transformations
    - Proper dual space relationships via Riesz representation
    - Full integration with pygeoinf operators and linear forms

    The mathematical foundation is the Lebesgue space L²([a,b]) with the
    standard inner product defined by integration. Computationally, the class
    supports two modes:

    1. Basis-free mode (`basis='none'` or `basis=None`), where functions are
       manipulated directly through evaluation and quadrature without an
       explicit finite-dimensional basis representation.
    2. Basis-backed mode, where `dim` basis functions are used for projection,
       coefficient transforms, Gram matrices, and related spectral/FEM-style
       workflows.

    Attributes:
        dim: Number of basis functions available when a basis-backed workflow
            is used. In basis-free mode this can be zero and does not control
            direct function-evaluation/integration operations.
        function_domain: The IntervalDomain [a, b].
        integration: Hierarchical integration configuration.
        parallel: Hierarchical parallelization configuration.

    Example:
        >>> from intervalinf.core import IntervalDomain
        >>> from intervalinf.spaces import Lebesgue
        >>> domain = IntervalDomain(0, 1)
        >>> space = Lebesgue(0, domain, basis=None)
        >>> # Use callable Functions directly in a basis-free workflow
        >>> f = Function(space, evaluate_callable=lambda x: x**2)
        >>> norm = space.norm(f)

        >>> # Or opt into a basis-backed representation
        >>> space = Lebesgue(50, domain, basis='fourier')
        >>> # Create a function from basis coefficients
        >>> f = space.from_components(np.random.randn(50))
        >>> norm = space.norm(f)
    """

    def __init__(
        self,
        dim: int,
        function_domain: 'IntervalDomain',
        /,
        *,
        basis: Optional[Union[str, list]] = None,
        weight: Optional[Callable] = None,
        integration_config: Optional[Union[
            IntegrationConfig,
            LebesgueIntegrationConfig
        ]] = None,
        parallel_config: Optional[Union[
            ParallelConfig,
            LebesgueParallelConfig
        ]] = None,
    ):
        """
        Initialize a Lebesgue space L²([a,b]).

        Args:
            dim: Number of basis functions for basis-backed workflows.
                This may be zero when using the space in basis-free mode with
                `basis=None`/`'none'`.
            function_domain: IntervalDomain object specifying [a,b] and
                boundary conditions.
            basis: Basis specification, can be:
                - str: 'fourier', 'hat', 'sine', 'cosine', 'DN', 'ND', etc.
                - str: 'none' (basis-free functional mode)
                - list: [func1, func2, ...] (custom callable functions)
                - None: defaults to 'none' (basis-free functional mode)
            weight: Optional weight function w(x) for weighted L² space.
            integration_config: Hierarchical integration configuration.
                Can be IntegrationConfig (same for all) or
                LebesgueIntegrationConfig (per-subsystem).
            parallel_config: Hierarchical parallelization configuration.

        Example:
            >>> # Basis-free functional usage
            >>> space = Lebesgue(0, domain, basis=None)
            >>> f = Function(space, evaluate_callable=lambda x: x)
            >>> value = space.norm(f)

            >>> # Basis-backed usage
            >>> space = Lebesgue(50, domain, basis='sine')

            >>> # With hierarchical configs
            >>> int_cfg = LebesgueIntegrationConfig.high_accuracy_galerkin()
            >>> par_cfg = LebesgueParallelConfig.parallel_dual()
            >>> space = Lebesgue(100, domain, basis='sine',
            ...                  integration_config=int_cfg,
            ...                  parallel_config=par_cfg)
        """
        self._dim = dim
        self._function_domain = function_domain
        self._weight = weight

        # Integration configuration (hierarchical)
        if integration_config is None:
            self.integration = LebesgueIntegrationConfig()
        elif isinstance(integration_config, IntegrationConfig):
            self.integration = LebesgueIntegrationConfig.from_single(
                integration_config
            )
        else:
            self.integration = integration_config

        # Parallelization configuration
        if parallel_config is None:
            self.parallel = LebesgueParallelConfig()
        elif isinstance(parallel_config, ParallelConfig):
            self.parallel = LebesgueParallelConfig.from_single(
                parallel_config
            )
        else:
            self.parallel = parallel_config

        # Initialize basis
        self._initialize_basis(basis or 'none')

        # Cached computations
        self._metric = None
        self._inverse_metric_chol = None

    # ================================================================
    # Abstract methods from HilbertSpace
    # ================================================================

    @property
    def dim(self) -> int:
        """Number of basis functions configured for basis-backed operations."""
        return self._dim

    # ================================================================
    # Backward-compatible properties
    # ================================================================

    @property
    def integration_method(self) -> str:
        """Get integration method (backward compatible)."""
        return self.integration.general.method

    @integration_method.setter
    def integration_method(self, value: str):
        """Set integration method for all subsystems."""
        self.integration.inner_product.method = value
        self.integration.dual.method = value
        self.integration.general.method = value

    @property
    def integration_npoints(self) -> int:
        """Get integration points (backward compatible)."""
        return self.integration.general.n_points

    @integration_npoints.setter
    def integration_npoints(self, value: int):
        """Set integration points for all subsystems."""
        self.integration.inner_product.n_points = value
        self.integration.dual.n_points = value
        self.integration.general.n_points = value

    # ================================================================
    # Inner product and metric
    # ================================================================

    def inner_product(self, u: 'Function', v: 'Function') -> float:
        """
        Compute the L² inner product ⟨u, v⟩.

        Args:
            u, v: Functions in this space.

        Returns:
            The inner product ⟨u, v⟩ = ∫ u(x) v(x) w(x) dx.
        """
        return self._continuous_l2_inner_product(u, v)

    def distance(self, u: 'Function', v: 'Function') -> float:
        """
        Compute the L² distance between two functions.

        The distance is defined as ||u - v|| = sqrt(⟨u-v, u-v⟩).

        Args:
            u, v: Functions in this space.

        Returns:
            The L² distance ||u - v||.

        Example:
            >>> space = Lebesgue(domain, basis_functions)
            >>> f = Function(domain, lambda x: x)
            >>> g = Function(domain, lambda x: x + 1)
            >>> dist = space.distance(f, g)  # ||f - g||
        """
        diff = self.subtract(u, v)
        return self.norm(diff)

    @property
    def metric(self) -> np.ndarray:
        """
        The metric tensor (Gram matrix) G[i,j] = ⟨φᵢ, φⱼ⟩.

        For orthonormal bases, this is approximately the identity.
        Computed once and cached.
        """
        if self._metric is None:
            self._compute_metric()
        return self._metric  # type: ignore

    # ================================================================
    # Dual space mappings (Riesz representation)
    # ================================================================

    def to_dual(self, x: 'Function') -> LinearFormKernel:
        """
        Map a function to its dual (Riesz representation).

        In L², the Riesz map is the identity on functions: the linear
        form φ_x(y) = ⟨x, y⟩ is represented by x itself.

        Args:
            x: A Function in this space.

        Returns:
            LinearFormKernel representing the dual element.
        """
        return LinearFormKernel(
            self,
            kernel=x,
            integration_config=self.integration.dual,
            parallel_config=self.parallel.dual
        )

    def from_dual(self, xp: LinearForm) -> 'Function':
        """
        Map a dual element back to a function via the Riesz isomorphism.

        For a ``LinearFormKernel`` the kernel function is returned directly
        (it already IS the L² Riesz representative).

        For a generic ``LinearForm`` with component vector ``c``, the Riesz
        representative satisfies ``⟨f, g⟩ = φ(g)`` for all ``g``, i.e.
        ``f_comp^T G g_comp = c^T g_comp``, giving ``f_comp = G⁻¹ c``.

        Args:
            xp: A LinearForm (typically LinearFormKernel).

        Returns:
            The Function representing this dual element.
        """
        if isinstance(xp, LinearFormKernel):
            kernel = xp.kernel
            if kernel is not None:
                return kernel  # type: ignore
            else:
                raise ValueError("LinearFormKernel has no kernel")
        else:
            # Generic LinearForm: components c represent φ(φᵢ) = cᵢ.
            # Riesz representative has components G⁻¹ c.
            return self.from_components(
                np.linalg.solve(self.metric, xp.components)
            )

    # ================================================================
    # Coefficient transformations
    # ================================================================

    def to_components(self, f: 'Function') -> np.ndarray:
        """
        Project a function onto the basis and return coefficients.

        Computes c_i = ⟨φᵢ, f⟩ / ⟨φᵢ, φᵢ⟩ for each basis function.

        For a basis-free space (dim=0, basis=None) this always returns an
        empty array of length 0.

        Args:
            f: A Function to project.

        Returns:
            Coefficient array of length dim.
        """
        if self._dim == 0:
            return np.zeros(0)
        self._require_basis()

        # If function already has coefficients and is in this space, use them
        has_coeffs = hasattr(f, 'coefficients') and f.coefficients is not None
        if has_coeffs and f.space is self:
            return f.coefficients.copy()

        # Otherwise, project via inner products
        n = self.dim
        coeffs = np.zeros(n)

        for i in range(n):
            phi_i = self.get_basis_function(i)
            # c_i = ⟨φᵢ, f⟩ / ⟨φᵢ, φᵢ⟩
            # For orthonormal basis, denominator is 1
            inner_phi_f = self._continuous_l2_inner_product(phi_i, f)
            inner_phi_phi = self._continuous_l2_inner_product(phi_i, phi_i)
            coeffs[i] = inner_phi_f / inner_phi_phi

        return coeffs

    def from_components(self, coefficients: np.ndarray) -> 'Function':
        """
        Construct a function from its basis coefficients.

        Infers compact support from the active (non-negligible) basis
        functions.  A basis function is considered active when
        ``|cᵢ| > tol`` where ``tol = 1e-14``.

        - If all coefficients are within tolerance → ``support=[]``.
        - If any active basis function has ``support=None`` (globally
          supported) → ``support=None``.
        - Otherwise → union of the active basis-function supports.

        For a basis-free space (dim=0, basis=None) the only valid input is an
        empty array; this returns the zero function.

        Args:
            coefficients: Array of length dim.

        Returns:
            Function f = Σ cᵢ φᵢ with inferred support metadata.
        """
        if self._dim == 0:
            if len(coefficients) != 0:
                raise ValueError(
                    f"Expected 0 coefficients for basis-free space, "
                    f"got {len(coefficients)}"
                )
            return self.zero
        self._require_basis()

        if len(coefficients) != self.dim:
            raise ValueError(
                f"Expected {self.dim} coefficients, got {len(coefficients)}"
            )

        tol = 1e-14
        active_intervals = []
        inferred_support = []  # default: empty (all-zero)

        for i, c in enumerate(coefficients):
            if abs(c) > tol:
                bf_support = self.get_basis_function(i).support
                if bf_support is None:
                    # Globally-supported basis function → no compact support
                    inferred_support = None
                    break
                active_intervals.extend(bf_support)

        if inferred_support is not None and active_intervals:
            inferred_support = Function._union_supports([], active_intervals)

        return Function(
            self,
            coefficients=coefficients.copy(),
            support=inferred_support,
        )

    # ================================================================
    # Equality and properties
    # ================================================================

    def __eq__(self, other: object) -> bool:
        """Check if two Lebesgue spaces are equal."""
        if not isinstance(other, Lebesgue):
            return False
        if self._dim != other._dim:
            return False
        if not (self._function_domain == other._function_domain):
            return False
        return True

    @property
    def zero(self) -> 'Function':
        """The zero function in this space (support=[])."""
        return Function(
            self,
            evaluate_callable=lambda x: np.zeros_like(x, dtype=float),
            support=[],
        )

    # ================================================================
    # Vector space operations (override for coefficient consistency)
    # ================================================================

    def multiply(self, a: float, x: 'Function') -> 'Function':
        """Compute scalar multiplication a*x, propagating support."""
        if hasattr(x, 'coefficients') and x.coefficients is not None:
            new_coefficients = a * x.coefficients
            new_support = [] if a == 0 else x.support
            return Function(
                self, coefficients=new_coefficients.copy(), support=new_support
            )
        else:
            return a * x

    def add(self, x: 'Function', y: 'Function') -> 'Function':
        """Compute vector addition x + y, propagating support as union."""
        x_has = hasattr(x, 'coefficients') and x.coefficients is not None
        y_has = hasattr(y, 'coefficients') and y.coefficients is not None
        if x_has and y_has:
            new_coefficients = x.coefficients + y.coefficients
            new_support = Function._union_supports(x.support, y.support)
            return Function(
                self, coefficients=new_coefficients.copy(), support=new_support
            )
        else:
            return x + y

    def ax(self, a: float, x: 'Function') -> None:
        """Perform in-place scaling x := a*x."""
        if hasattr(x, 'coefficients') and x.coefficients is not None:
            x.coefficients *= a
            if a == 0:
                x.support = []
            x.clear_materializations()
        else:
            raise ValueError(
                "Cannot perform in-place operation on function "
                "without coefficients"
            )

    def axpy(self, a: float, x: 'Function', y: 'Function') -> 'Function':
        """Performs y := y + a*x in-place, updating support to union(y, x)."""
        y_has = hasattr(y, 'coefficients') and y.coefficients is not None
        x_has = hasattr(x, 'coefficients') and x.coefficients is not None
        if y_has and x_has:
            y.coefficients += a * x.coefficients
            if a != 0:
                y.support = Function._union_supports(y.support, x.support)
            y.clear_materializations()
            return y
        else:
            return self.add(y, self.multiply(a, x))

    # ================================================================
    # Domain and restriction
    # ================================================================

    @property
    def function_domain(self) -> 'IntervalDomain':
        """The interval domain [a,b] for this space."""
        return self._function_domain

    def restrict_to_subinterval(
        self,
        subdomain: 'IntervalDomain',
    ) -> 'Lebesgue':
        """
        Create a new Lebesgue space restricted to a subinterval.

        Args:
            subdomain: An IntervalDomain contained in function_domain.

        Returns:
            A new baseless Lebesgue space on the subinterval.

        Raises:
            ValueError: If subdomain is not contained in function_domain.
        """
        a_ok = self.function_domain.a <= subdomain.a
        b_ok = subdomain.b <= self.function_domain.b
        if not (a_ok and b_ok):
            raise ValueError(
                f"Subdomain {subdomain} must be contained in "
                f"function domain {self.function_domain}"
            )

        restricted_space = Lebesgue(
            self.dim,
            subdomain,
            basis='none',
            weight=self._weight
        )

        # Copy configs
        restricted_space.integration = _copy.deepcopy(self.integration)
        restricted_space.parallel = _copy.deepcopy(self.parallel)

        return restricted_space

    # ================================================================
    # Factory method for discontinuous spaces
    # ================================================================

    @classmethod
    def with_discontinuities(
        cls,
        dim: int,
        function_domain: 'IntervalDomain',
        discontinuity_points: list,
        /,
        *,
        basis: Optional[Union[str, list]] = None,
        weight: Optional[Callable] = None,
        dim_per_subspace: Optional[list] = None,
        basis_per_subspace: Optional[list] = None,
        integration_config: Optional[IntegrationConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> 'LebesgueSpaceDirectSum':
        """
        Create a LebesgueSpaceDirectSum with discontinuities.

        This factory creates a direct sum of Lebesgue spaces, where
        each component is defined on a subinterval separated by
        discontinuity points.

        Args:
            dim: Total dimension across all subspaces.
            function_domain: The full interval domain.
            discontinuity_points: Points where discontinuities occur.
            basis: Basis type for all subspaces.
            weight: Weight function for inner product.
            dim_per_subspace: Optional dimensions per subspace.
            basis_per_subspace: Optional basis per subspace.
            integration_config: Integration config for all subspaces.
            parallel_config: Parallel config for all subspaces.

        Returns:
            LebesgueSpaceDirectSum with component spaces.
        """
        if isinstance(basis, list):
            raise ValueError(
                "Providing a list of basis functions is not supported. "
                "Use basis_per_subspace instead."
            )

        # Split domain at discontinuities
        subdomains = function_domain.split_at_discontinuities(
            discontinuity_points
        )
        n_subspaces = len(subdomains)

        # Determine dimensions
        if dim_per_subspace is None:
            lengths = [sd.length for sd in subdomains]
            total_length = sum(lengths)
            dims = [int(dim * length / total_length) for length in lengths]
            remainder = dim - sum(dims)
            length_indices = sorted(
                range(n_subspaces),
                key=lambda i: lengths[i],
                reverse=True
            )
            for i in range(remainder):
                dims[length_indices[i % n_subspaces]] += 1
        else:
            if len(dim_per_subspace) != n_subspaces:
                raise ValueError(
                    f"dim_per_subspace must have length {n_subspaces}"
                )
            if sum(dim_per_subspace) != dim:
                raise ValueError(f"dim_per_subspace must sum to {dim}")
            dims = list(dim_per_subspace)

        # Determine bases
        if basis_per_subspace is not None:
            if len(basis_per_subspace) != n_subspaces:
                raise ValueError(
                    f"basis_per_subspace must have length {n_subspaces}"
                )
            bases = list(basis_per_subspace)
        else:
            bases = [basis] * n_subspaces

        # Create subspaces
        subspaces = [
            cls(
                d, subdomain, basis=b, weight=weight,
                integration_config=integration_config,
                parallel_config=parallel_config
            )
            for d, subdomain, b in zip(dims, subdomains, bases)
        ]

        return LebesgueSpaceDirectSum(subspaces)

    # ================================================================
    # Projection
    # ================================================================

    def project(self, f: 'Function') -> 'Function':
        """
        Project a function onto this finite-dimensional space.

        The L² projection minimizes ||f - g||_{L²} over g in the space.

        Args:
            f: Function to project.

        Returns:
            The L² projection onto this space.
        """
        coefficients = self.to_components(f)
        return self.from_components(coefficients)

    # ================================================================
    # Basis management
    # ================================================================

    def _require_basis(self):
        """Raise if no basis is available."""
        if self._basis_type == 'none':
            raise RuntimeError(
                "This operation requires a basis; set a BasisProvider "
                "or direct basis first."
            )

    def _initialize_basis(self, basis):
        """Initialize basis from parameter."""
        if isinstance(basis, str):
            if basis == 'none':
                self._basis_type = 'none'
                self._basis_functions = None
                self.basis_provider = None
                self._use_basis_provider = False
            else:
                # String-based basis types using provider factory
                try:
                    basis_provider = create_basis_provider(self, basis)
                    self._basis_type = basis
                    self._basis_functions = None
                    self.basis_provider = basis_provider
                    self._use_basis_provider = True
                except ValueError as e:
                    raise ValueError(f"Unsupported basis type '{basis}': {e}")

        elif isinstance(basis, list):
            if len(basis) != self._dim:
                raise ValueError(
                    f"Number of basis functions ({len(basis)}) "
                    f"must match dimension ({self._dim})"
                )
            self._basis_type = 'direct_functions'
            self._basis_functions = [
                self._create_function_from_callable(f) for f in basis
            ]
            self.basis_provider = None
            self._use_basis_provider = False

        elif isinstance(basis, BasisProvider):
            self.basis_provider = basis
            self._use_basis_provider = True
            self._basis_type = getattr(basis, 'type', 'custom_provider')
            self._basis_functions = None

        else:
            raise TypeError(
                f"basis must be a string or list of callables, "
                f"got {type(basis)}"
            )

    def _create_function_from_callable(self, callable_func):
        """Create Function from callable."""
        return Function(self, evaluate_callable=callable_func)

    def set_basis_provider(self, basis_provider: 'BasisProvider'):
        """
        Set a BasisProvider after space creation.

        Args:
            basis_provider: The BasisProvider to use.
        """
        self.basis_provider = basis_provider
        self._use_basis_provider = True
        self._basis_type = getattr(basis_provider, 'type', 'custom_provider')
        self._basis_functions = None
        self._metric = None
        self._inverse_metric_chol = None

    @property
    def basis_functions(self) -> List['Function']:
        """List of basis functions for this space."""
        if self._basis_type == 'none':
            raise RuntimeError(
                "No basis functions available - space is baseless"
            )

        if self._use_basis_provider:
            if self.basis_provider is None:
                raise RuntimeError("No basis provider available")
            if hasattr(self.basis_provider, 'get_basis_function'):
                return [
                    self.basis_provider.get_basis_function(i)
                    for i in range(self.dim)
                ]
            else:
                raise RuntimeError("BasisProvider missing get_basis_function")
        else:
            if self._basis_functions is None:
                raise RuntimeError("No basis functions available")
            return self._basis_functions

    def get_basis_function(self, index: int) -> 'Function':
        """Get the i-th basis function."""
        if self._basis_type == 'none':
            raise RuntimeError(
                "No basis functions - space is baseless"
            )

        if self._use_basis_provider:
            if self.basis_provider is None:
                raise RuntimeError("No basis provider available")
            return self.basis_provider.get_basis_function(index)
        else:
            if not (0 <= index < self.dim):
                raise IndexError(f"Index {index} out of range [0, {self.dim})")
            if self._basis_functions is None:
                raise RuntimeError("No basis functions available")
            return self._basis_functions[index]

    # ================================================================
    # Private helpers
    # ================================================================

    def _continuous_l2_inner_product(
        self,
        u: 'Function',
        v: 'Function'
    ) -> float:
        """Compute continuous L² inner product via integration."""
        product = u * v
        method = self.integration_method
        if method not in {"simpson", "trapz"} or product.has_compact_support:
            return product.integrate(
                method=method,
                n_points=self.integration_npoints,
                weight=self._weight
            )

        spec = RepresentationSpec(
            kind="fixed_grid",
            n_points=self.integration_npoints,
            interval=(self.function_domain.a, self.function_domain.b),
            method="uniform",
        )
        u_materialized = u.materialize(spec)
        v_materialized = v.materialize(spec)
        integrand = u_materialized.values * v_materialized.values

        if self._weight is not None:
            integrand = integrand * self._evaluate_array_callable(
                self._weight,
                u_materialized.grid,
            )

        return float(
            self._integrate_fixed_grid(
                integrand,
                u_materialized.grid,
                method,
            )
        )

    @staticmethod
    def _evaluate_array_callable(
        callable_obj: Callable,
        grid: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a scalar or vectorized callable on a fixed grid."""
        try:
            values = np.asarray(callable_obj(grid), dtype=float)
            if values.shape == grid.shape:
                return values
            if values.shape == ():
                return np.full_like(grid, float(values), dtype=float)
            if values.ndim == 1 and values.size == grid.size:
                return values.reshape(grid.shape)
            raise ValueError("callable returned incompatible shape")
        except Exception:
            return np.asarray([callable_obj(x) for x in grid], dtype=float)

    @staticmethod
    def _integrate_fixed_grid(
        values: np.ndarray,
        grid: np.ndarray,
        method: str,
    ) -> float:
        """Integrate array values on a fixed grid with the configured rule."""
        if method == "simpson":
            from scipy.integrate import simpson

            return float(simpson(values, x=grid))

        try:
            from scipy.integrate import trapezoid as trapz
        except ImportError:
            from scipy.integrate import trapz  # type: ignore

        return float(trapz(values, x=grid))

    def _compute_metric(self):
        """Compute and cache the metric tensor (Gram matrix)."""
        if self._metric is not None:
            return

        if self._basis_type == 'none':
            raise RuntimeError("Cannot compute metric: no basis available.")

        n = self.dim
        self._metric = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                basis_i = self.get_basis_function(i)
                basis_j = self.get_basis_function(j)
                inner = self._continuous_l2_inner_product(basis_i, basis_j)
                self._metric[i, j] = inner
                self._metric[j, i] = inner

    def _clear_metric_caches(self):
        """Clear cached metric computations."""
        self._metric = None
        self._chol = None


# =============================================================================
# Basis Provider Factory
# =============================================================================


def create_basis_provider(
    space: Lebesgue,
    basis_type: str
) -> 'BasisProvider':
    """
    Factory function to create a BasisProvider for a Lebesgue space.

    Args:
        space: The Lebesgue space to create the basis for.
        basis_type: Type of basis ('fourier', 'hat', 'sine', etc.).

    Returns:
        A BasisProvider instance.

    Raises:
        ValueError: If basis_type is not supported.
    """
    from intervalinf.providers import (
        SineFunctionProvider,
        CosineFunctionProvider,
        FourierFunctionProvider,
        HatFunctionProvider,
        MixedDNFunctionProvider,
        MixedNDFunctionProvider,
        SplineFunctionProvider,
        WaveletFunctionProvider,
        CustomBasisProvider,
    )

    # Map basis type to (provider_factory, orthonormal)
    provider_map = {
        'sine': (SineFunctionProvider, True),
        'cosine': (CosineFunctionProvider, True),
        'cosine_non_constant': (
            lambda s: CosineFunctionProvider(s, non_constant_only=True),
            True
        ),
        'fourier': (FourierFunctionProvider, True),
        'fourier_non_constant': (
            lambda s: FourierFunctionProvider(s, non_constant_only=True),
            True
        ),
        'hat': (HatFunctionProvider, False),
        'spline': (SplineFunctionProvider, False),
        'wavelet': (WaveletFunctionProvider, True),
        'DN': (MixedDNFunctionProvider, True),
        'ND': (MixedNDFunctionProvider, True),
    }

    if basis_type not in provider_map:
        supported = ', '.join(sorted(provider_map.keys()))
        raise ValueError(
            f"Unknown basis type: '{basis_type}'. "
            f"Supported types: {supported}"
        )

    provider_factory, orthonormal = provider_map[basis_type]

    # Check if factory is a lambda or class
    if callable(provider_factory) and not isinstance(provider_factory, type):
        function_provider = provider_factory(space)
    else:
        function_provider = provider_factory(space)

    # Wrap in CustomBasisProvider
    return CustomBasisProvider(
        space,
        function_provider,
        orthonormal=orthonormal,
        basis_type=basis_type
    )


# =============================================================================
# Direct Sum of Lebesgue Spaces
# =============================================================================


class LebesgueSpaceDirectSum(HilbertSpaceDirectSum):
    """
    Direct sum of Lebesgue spaces using LinearFormKernel.

    This extends HilbertSpaceDirectSum to work with Lebesgue spaces
    without requiring explicit basis functions, using integration-based
    inner products via LinearFormKernel.
    """

    def to_dual(self, xs: List[Function]) -> LinearFormKernel:
        """
        Map a list of functions to a dual element.

        Args:
            xs: List of Functions, one per subspace.

        Returns:
            LinearFormKernel for integration-based inner products.
        """
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")

        # Get config from first subspace
        subspace = self.subspace(0)
        if hasattr(subspace, 'integration'):
            int_cfg = subspace.integration.dual  # type: ignore
            par_cfg = subspace.parallel.dual  # type: ignore
        else:
            int_cfg = IntegrationConfig(method='trapz', n_points=1000)
            par_cfg = ParallelConfig(enabled=False, n_jobs=-1)

        return LinearFormKernel(
            self,
            kernel=xs,
            integration_config=int_cfg,
            parallel_config=par_cfg,
        )

    def from_dual(self, xp) -> List[Function]:
        """Map a dual element back to functions."""
        if isinstance(xp, LinearFormKernel):
            return xp.kernel  # type: ignore
        else:
            return super().from_dual(xp)


# =============================================================================
# Known Region and Partitioned Space
# =============================================================================


class KnownRegion:
    """
    Specification of a region where the model is known (fixed).

    A known region is a sub-interval where the model value is fixed
    and not part of the inference. For example, shear wave velocity
    in the Earth's outer core is known to be zero.

    Attributes:
        interval: The IntervalDomain where the model is known.
        value: A Function representing the known value.

    Example:
        >>> from intervalinf.core import IntervalDomain
        >>> outer_core = IntervalDomain(1217.5, 3480.0)
        >>> known = KnownRegion.zero(outer_core)
    """

    def __init__(
        self,
        interval: 'IntervalDomain',
        value: 'Function',
    ):
        """
        Initialize a KnownRegion.

        Args:
            interval: The IntervalDomain where the model is known.
            value: A Function representing the known value.

        Raises:
            TypeError: If value is not a Function.
            ValueError: If function domain doesn't match interval.
        """
        if not isinstance(value, Function):
            raise TypeError(
                f"value must be a Function, got {type(value).__name__}"
            )

        func_domain = value.space.function_domain
        if not (func_domain.a == interval.a and func_domain.b == interval.b):
            raise ValueError(
                f"Function domain {func_domain} doesn't match "
                f"interval {interval}"
            )

        self.interval = interval
        self.value = value

    @classmethod
    def zero(
        cls,
        interval: 'IntervalDomain',
        dim: int = 10,
        **lebesgue_kwargs
    ) -> 'KnownRegion':
        """Factory for a known region with zero value."""
        space = Lebesgue(dim, interval, basis='none', **lebesgue_kwargs)

        def zero_callable(x):
            return np.zeros_like(np.asarray(x), dtype=float)

        zero_func = Function(
            space, evaluate_callable=zero_callable, name="zero"
        )
        return cls(interval, zero_func)

    @classmethod
    def constant(
        cls,
        interval: 'IntervalDomain',
        constant_value: float,
        dim: int = 10,
        **lebesgue_kwargs
    ) -> 'KnownRegion':
        """Factory for a known region with constant value."""
        space = Lebesgue(dim, interval, basis='none', **lebesgue_kwargs)
        const_func = Function(
            space,
            evaluate_callable=lambda x: np.full_like(
                np.asarray(x), constant_value, dtype=float
            ),
            name=f"constant_{constant_value}"
        )
        return cls(interval, const_func)

    def __repr__(self) -> str:
        return f"KnownRegion(interval={self.interval}, value={self.value})"


class PartitionedLebesgueSpace:
    """
    A Lebesgue space partitioned into known and unknown regions.

    This handles the common case where the model is known (fixed) on
    some sub-intervals and unknown on others.

    Attributes:
        full_domain: The complete IntervalDomain.
        known_regions: List of KnownRegion specifications.
        unknown_intervals: List of IntervalDomain for unknown regions.
        unknown_spaces: List of Lebesgue spaces for unknown regions.
        model_space: DirectSum of unknown spaces (use for inference).

    Example:
        >>> from intervalinf.core import IntervalDomain
        >>> full = IntervalDomain(0, 6371)
        >>> outer_core = KnownRegion.zero(IntervalDomain(1217.5, 3480.0))
        >>> partitioned = PartitionedLebesgueSpace(
        ...     full, [outer_core], dims=[50, 50], basis='cosine'
        ... )
        >>> M = partitioned.model_space  # Use for inference
    """

    def __init__(
        self,
        full_domain: 'IntervalDomain',
        known_regions: List[KnownRegion],
        dims: List[int],
        *,
        basis: Optional[Union[str, list]] = None,
        bases: Optional[List[Optional[Union[str, list]]]] = None,
        integration_config: Optional[Union[
            IntegrationConfig,
            LebesgueIntegrationConfig
        ]] = None,
        parallel_config: Optional[Union[
            ParallelConfig,
            LebesgueParallelConfig
        ]] = None,
        weight: Optional[Callable] = None,
    ):
        """Initialize a PartitionedLebesgueSpace."""
        self.full_domain = full_domain
        self.known_regions = sorted(
            known_regions, key=lambda kr: kr.interval.a
        )

        self._validate_known_regions()
        self.unknown_intervals = self._compute_unknown_intervals()

        n_unknown = len(self.unknown_intervals)
        if len(dims) != n_unknown:
            raise ValueError(
                f"dims must have {n_unknown} elements, got {len(dims)}"
            )

        if bases is not None:
            if len(bases) != n_unknown:
                raise ValueError(
                    f"bases must have {n_unknown} elements"
                )
            region_bases = bases
        else:
            region_bases = [basis] * n_unknown

        self.unknown_spaces: List[Lebesgue] = []
        for interval, dim, region_basis in zip(
            self.unknown_intervals, dims, region_bases
        ):
            space = Lebesgue(
                dim, interval, basis=region_basis, weight=weight,
                integration_config=integration_config,
                parallel_config=parallel_config,
            )
            self.unknown_spaces.append(space)

        self.model_space = LebesgueSpaceDirectSum(self.unknown_spaces)

    def _validate_known_regions(self) -> None:
        """Validate known regions are non-overlapping and within domain."""
        for i, kr in enumerate(self.known_regions):
            if kr.interval.a < self.full_domain.a:
                raise ValueError(
                    f"Known region {i} starts before full_domain"
                )
            if kr.interval.b > self.full_domain.b:
                raise ValueError(
                    f"Known region {i} ends after full_domain"
                )
            if i > 0:
                prev_kr = self.known_regions[i - 1]
                if kr.interval.a < prev_kr.interval.b:
                    raise ValueError(
                        f"Known regions {i-1} and {i} overlap"
                    )

    def _compute_unknown_intervals(self) -> List['IntervalDomain']:
        """Compute unknown intervals by subtracting known regions."""
        from intervalinf.core.domain import IntervalDomain

        if not self.known_regions:
            return [self.full_domain]

        unknown = []
        current_start = self.full_domain.a

        for kr in self.known_regions:
            if kr.interval.a > current_start:
                unknown.append(IntervalDomain(
                    current_start, kr.interval.a,
                    boundary_type=self.full_domain.boundary_type
                ))
            current_start = kr.interval.b

        if current_start < self.full_domain.b:
            unknown.append(IntervalDomain(
                current_start, self.full_domain.b,
                boundary_type=self.full_domain.boundary_type
            ))

        return unknown

    @property
    def n_unknown_regions(self) -> int:
        """Number of unknown regions."""
        return len(self.unknown_intervals)

    @property
    def n_known_regions(self) -> int:
        """Number of known regions."""
        return len(self.known_regions)

    def get_unknown_space(self, index: int) -> Lebesgue:
        """Get the Lebesgue space for the i-th unknown region."""
        return self.unknown_spaces[index]

    def get_known_region(self, index: int) -> KnownRegion:
        """Get the i-th known region."""
        return self.known_regions[index]

    def extend_to_full_domain(
        self,
        unknown_model: List['Function'],
        name: Optional[str] = None
    ) -> 'Function':
        """
        Extend a model from unknown regions to the full domain.

        Args:
            unknown_model: List of Functions, one per unknown region.
            name: Optional name for the result.

        Returns:
            Function on the full domain.
        """
        if len(unknown_model) != self.n_unknown_regions:
            raise ValueError(
                f"unknown_model must have {self.n_unknown_regions} elements"
            )

        for i, func in enumerate(unknown_model):
            if not isinstance(func, Function):
                raise ValueError(
                    f"unknown_model[{i}] must be a Function"
                )

        min_dim = min(space.dim for space in self.unknown_spaces)
        full_space = Lebesgue(min_dim, self.full_domain, basis='none')

        def extended_callable(x):
            x_arr = np.atleast_1d(np.asarray(x))
            result = np.zeros_like(x_arr, dtype=float)

            for interval, func in zip(self.unknown_intervals, unknown_model):
                mask = (x_arr >= interval.a) & (x_arr <= interval.b)
                if np.any(mask):
                    result[mask] = func.evaluate(
                        x_arr[mask], check_domain=False
                    )

            for kr in self.known_regions:
                mask = (x_arr >= kr.interval.a) & (x_arr <= kr.interval.b)
                if np.any(mask):
                    result[mask] = kr.value.evaluate(
                        x_arr[mask], check_domain=False
                    )

            if np.ndim(x) == 0:
                return float(result[0])
            return result

        return Function(
            full_space,
            evaluate_callable=extended_callable,
            name=name or "extended_model"
        )

    def restrict_function(
        self,
        full_function: 'Function',
        region_index: int
    ) -> 'Function':
        """
        Restrict a function from full domain to an unknown region.

        Args:
            full_function: Function on the full domain.
            region_index: Index of the unknown region.

        Returns:
            Function on the specified unknown region's space.
        """
        if region_index < 0 or region_index >= self.n_unknown_regions:
            raise IndexError(
                f"region_index {region_index} out of range"
            )

        target_space = self.unknown_spaces[region_index]
        return full_function.restrict(target_space)

    def __repr__(self) -> str:
        return (
            f"PartitionedLebesgueSpace(\n"
            f"  full_domain={self.full_domain},\n"
            f"  n_unknown_regions={self.n_unknown_regions},\n"
            f"  unknown_intervals={self.unknown_intervals},\n"
            f"  n_known_regions={self.n_known_regions}\n"
            f")"
        )


__all__ = [
    'Lebesgue',
    'LebesgueSpaceDirectSum',
    'LebesgueIntegrationConfig',
    'LebesgueParallelConfig',
    'KnownRegion',
    'PartitionedLebesgueSpace',
    'create_basis_provider',
]
