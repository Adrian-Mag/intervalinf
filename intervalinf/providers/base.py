"""
Base classes for providers.

This module contains the abstract base classes that define the provider
interfaces for generating functions, eigenvalues, bases, and spectra.

Hierarchy:
- Level 0 (this file): Abstract bases and simple concrete wrappers
- Level 1 (eigenvalues.py): Simple eigenvalue providers
- Level 2 (functions/): Concrete function providers
- Level 3+ (laplacian.py, radial.py): Composite providers

Providers can be initialized with either:
- A function space (traditional mode): Functions are attached to the space
- An IntervalDomain (standalone mode): Functions are created standalone

This flexibility solves the bootstrapping problem where basis functions
needed a space, but the space needed basis functions to exist.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from intervalinf.core.functions import Function
    from intervalinf.core.domain import IntervalDomain


def _is_interval_domain(obj) -> bool:
    """Check if object is an IntervalDomain (avoids circular import)."""
    return type(obj).__name__ == 'IntervalDomain'


def _get_domain(space_or_domain) -> 'IntervalDomain':
    """Extract domain from space or return domain directly."""
    if _is_interval_domain(space_or_domain):
        return space_or_domain
    if hasattr(space_or_domain, "_function_domain"):
        return space_or_domain._function_domain
    if hasattr(space_or_domain, "function_domain"):
        return space_or_domain.function_domain
    raise AttributeError(
        f"Cannot extract domain from {type(space_or_domain).__name__}"
    )


# =============================================================================
# Function Provider Base Classes
# =============================================================================


class FunctionProvider(ABC):
    """
    Abstract base class for function providers.

    Function providers create Function objects from various families
    with a composable, lazy approach.

    Providers can be initialized with either:
    - A function space: Created functions are attached to the space
    - An IntervalDomain: Created functions are standalone (unattached)

    This dual-mode design solves the bootstrapping problem where basis
    functions previously needed a space to exist, but the space needed
    basis functions.

    Attributes:
        space_or_domain: The space or domain passed to constructor
        domain: The IntervalDomain (always available)
        space: The function space (None if initialized with domain only)
        is_standalone: True if initialized with domain only
    """

    def __init__(self, space_or_domain):
        """
        Initialize provider.

        Args:
            space_or_domain: Either a function space (Lebesgue, Sobolev, etc.)
                or an IntervalDomain. If domain, created functions will be
                standalone. If space, functions will be attached to it.

        Raises:
            ValueError: If space_or_domain is None
        """
        if space_or_domain is None:
            raise ValueError(
                (
                    "Space or domain must be "
                    f"provided to {self.__class__.__name__}. "
                    "Cannot be None."
                )
            )

        self._space_or_domain = space_or_domain
        self._is_standalone = _is_interval_domain(space_or_domain)

    @property
    def domain(self) -> 'IntervalDomain':
        """Get the IntervalDomain (always available)."""
        return _get_domain(self._space_or_domain)

    @property
    def space(self):
        """
        Get the function space (None if standalone mode).

        Use this when you need space-specific operations.
        """
        if self._is_standalone:
            return None
        return self._space_or_domain

    @property
    def is_standalone(self) -> bool:
        """True if provider creates standalone (unattached) functions."""
        return self._is_standalone

    @property
    def function_context(self):
        """
        Get the context to pass to Function constructor.

        Returns the space if available, otherwise the domain.
        This is what should be passed as the first argument to Function().
        """
        return self._space_or_domain


class IndexedFunctionProvider(FunctionProvider):
    """
    Provider for functions that can be accessed by index.

    Useful for basis functions, orthogonal families, etc.
    """

    @abstractmethod
    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get function by index."""
        pass

    def get_functions(self, indices: List[int], **kwargs) -> List['Function']:
        """Get multiple functions by indices."""
        return [self.get_function_by_index(i, **kwargs) for i in indices]

    def restrict(self, restricted_space):
        """
        Create a restricted version of this provider.

        Returns a new provider that generates functions on the restricted
        space by taking the restriction of functions from the original
        provider.

        Args:
            restricted_space: The target space to restrict to.

        Returns:
            RestrictedFunctionProvider: A provider that returns restricted
                versions of the original provider's functions.
        """
        return RestrictedFunctionProvider(self, restricted_space)


class RestrictedFunctionProvider(IndexedFunctionProvider):
    """
    A provider that returns restricted versions of another provider's
    functions.
    """

    def __init__(
        self,
        original_provider: IndexedFunctionProvider,
        restricted_space_or_domain
    ):
        """
        Initialize restricted provider.

        Args:
            original_provider: The provider to restrict
            restricted_space_or_domain: The target space or domain for
                restrictions
        """
        super().__init__(restricted_space_or_domain)
        self.original_provider = original_provider

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get restricted version of function at given index."""
        original_func = self.original_provider.get_function_by_index(
            index, **kwargs
        )
        return original_func.restrict(self._space_or_domain)


class ParametricFunctionProvider(FunctionProvider):
    """
    Provider for parametric function families.

    Functions are defined by a parameter dictionary.
    """

    @abstractmethod
    def get_function_by_parameters(
        self,
        parameters: Dict[str, Any],
        **kwargs
    ) -> 'Function':
        """Get a function with specific parameters."""
        pass

    def get_function(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Function':
        """Get function with given or default parameters."""
        if parameters is None:
            parameters = self.get_default_parameters()
        return self.get_function_by_parameters(parameters, **kwargs)

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this family."""
        pass


class RandomFunctionProvider(FunctionProvider):
    """
    Provider for random function generation.
    """

    def __init__(self, space, random_state=None):
        """
        Initialize random provider.

        Args:
            space: Function space
            random_state: Random seed or RandomState for reproducibility
        """
        super().__init__(space)
        if random_state is None:
            self.rng = np.random.default_rng()
        elif isinstance(random_state, int):
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = random_state

    @abstractmethod
    def get_random_function(self, **kwargs) -> 'Function':
        """Generate a random function."""
        pass


class NullFunctionProvider(IndexedFunctionProvider):
    """
    A provider that returns zero functions.

    Useful as a placeholder or for testing.
    """

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Return zero function regardless of index."""
        from intervalinf.core.functions import Function

        if self.space is not None:
            return self.space.zero
        else:
            # Standalone mode: create zero function on domain
            return Function(
                self.domain,
                evaluate_callable=lambda x: np.zeros_like(
                    np.asarray(x), dtype=float
                )
            )


# =============================================================================
# Eigenvalue Provider Base Classes
# =============================================================================


class EigenvalueProvider(ABC):
    """
    Abstract base class for eigenvalue providers.

    Eigenvalue providers compute eigenvalues for corresponding function
    providers, typically for specific differential operators.
    """

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """
        Get eigenvalue for given index.

        Args:
            index: Index of the eigenfunction

        Returns:
            float: Eigenvalue at the given index
        """
        pass

    def get_eigenvalues(self, n: int) -> np.ndarray:
        """
        Get array of eigenvalues up to index n.

        Args:
            n: Number of eigenvalues to return

        Returns:
            np.ndarray: Array of eigenvalues
        """
        return np.array([self.get_eigenvalue(i) for i in range(n)])


class CustomEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider with user-specified eigenvalues.

    Allows users to provide their own eigenvalue array or callable.
    """

    def __init__(
        self, eigenvalues: Union[np.ndarray, list, Callable[[int], float]]
    ):
        """
        Initialize with eigenvalue array or callable.

        Args:
            eigenvalues: Array of eigenvalues, or a callable f(index) -> float
        """
        if callable(eigenvalues):
            self._eigenvalue_func = eigenvalues
            self._eigenvalues = None
        else:
            self._eigenvalues = np.asarray(eigenvalues)
            self._eigenvalue_func = None

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue from the stored array or callable."""
        if self._eigenvalue_func is not None:
            return float(self._eigenvalue_func(index))

        if (
            self._eigenvalues is None
            or not (0 <= index < len(self._eigenvalues))
        ):
            raise IndexError(
                f"Eigenvalue index {index} out of range "
                f"[0, {len(self._eigenvalues) if self._eigenvalues is not None else 0})"  # noqa
            )
        return float(self._eigenvalues[index])


# =============================================================================
# Basis Provider Base Classes
# =============================================================================


class BasisProvider(ABC):
    """
    Abstract base class for basis providers.

    Basis providers wrap function providers and add space-specific
    functionality:
    - Ensures the number of functions matches space dimension
    - Provides space-aware caching and validation
    - Implements convenient access patterns (indexing, iteration)

    All computation is lazy with caching for efficiency.
    """

    def __init__(
        self,
        space,
        orthonormal: bool = False,
        basis_type: Optional[str] = None
    ):
        """
        Initialize basis provider.

        Args:
            space: The function space that owns this provider
            orthonormal: True if the basis functions are orthonormal with
                        respect to the L² inner product
            basis_type: String identifier for the type of basis
        """
        self.space = space
        self.orthonormal = orthonormal
        self.type = basis_type

    @abstractmethod
    def get_basis_function(self, index: int) -> 'Function':
        """
        Get basis function for given index.

        Args:
            index: Index of the basis function

        Returns:
            Function: The basis function at the given index
        """
        pass


class CustomBasisProvider(BasisProvider):
    """
    Basis provider with user-specified basis functions.

    Wraps any IndexedFunctionProvider to provide basis functionality.
    """

    def __init__(
        self,
        space,
        function_provider: IndexedFunctionProvider,
        orthonormal: bool = False,
        basis_type: Optional[str] = None
    ):
        """
        Initialize with function provider.

        Args:
            space: The function space that owns this provider
            function_provider: IndexedFunctionProvider for the basis functions
            orthonormal: True if the basis functions are orthonormal
            basis_type: String identifier for the type of basis
        """
        super().__init__(space, orthonormal, basis_type)
        self.function_provider = function_provider

    def get_basis_function(self, index: int) -> 'Function':
        """Get basis function from the wrapped provider."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Function index {index} out of range "
                f"[0, {self.space.dim})"
            )
        return self.function_provider.get_function_by_index(index)


# =============================================================================
# Spectrum Provider Base Classes
# =============================================================================


class SpectrumProvider(ABC):
    """
    Abstract base class for spectrum providers.

    Spectrum providers combine eigenvalues with eigenfunctions for
    spectral decomposition. This is used for spectral inner products
    in Sobolev spaces and for operators with known eigenbases.

    All computation is lazy with caching for efficiency.
    """

    def __init__(
        self,
        space,
        orthonormal: bool = False,
        basis_type: Optional[str] = None
    ):
        """
        Initialize spectrum provider.

        Args:
            space: The function space
            orthonormal: True if the basis functions are orthonormal
            basis_type: String identifier for the type of basis
        """
        self.space = space
        self.orthonormal = orthonormal
        self.type = basis_type

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """
        Get eigenvalue for given basis function index.

        Args:
            index: Index of the eigenfunction

        Returns:
            float: The eigenvalue at the given index
        """
        pass

    @abstractmethod
    def get_eigenfunction(self, index: int) -> 'Function':
        """
        Get eigenfunction for given index.

        This is the same as get_basis_function but emphasizes that
        these are eigenfunctions of some operator.

        Args:
            index: Index of the eigenfunction

        Returns:
            Function: The eigenfunction at the given index
        """
        pass


class CustomSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider with user-specified functions and eigenvalues.

    Wraps an IndexedFunctionProvider and an EigenvalueProvider to
    provide spectrum functionality.
    """

    def __init__(
        self,
        space,
        function_provider: IndexedFunctionProvider,
        eigenvalue_provider: EigenvalueProvider,
        orthonormal: bool = False,
        basis_type: Optional[str] = None
    ):
        """
        Initialize with function and eigenvalue providers.

        Args:
            space: The function space
            function_provider: IndexedFunctionProvider for eigenfunctions
            eigenvalue_provider: EigenvalueProvider for eigenvalues
            orthonormal: True if the basis functions are orthonormal
            basis_type: String identifier for the type of basis
        """
        super().__init__(space, orthonormal, basis_type)
        self.function_provider = function_provider
        self.eigenvalue_provider = eigenvalue_provider

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue from the wrapped provider."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Eigenvalue index {index} out of range "
                f"[0, {self.space.dim})"
            )
        return self.eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> 'Function':
        """Get eigenfunction from the wrapped provider."""
        if not (0 <= index < self.space.dim):
            raise IndexError(
                f"Function index {index} out of range "
                f"[0, {self.space.dim})"
            )
        return self.function_provider.get_function_by_index(index)
