"""
Base classes for function providers.

This module contains the abstract base classes that define the provider
interface for generating functions from various families.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class FunctionProvider(ABC):
    """
    Abstract base class for function providers.

    Function providers create Function objects from various families
    with a composable, lazy approach. All providers require explicit
    space specification from the user.
    """

    def __init__(self, space):
        """
        Initialize provider.

        Args:
            space: Lebesgue instance (contains domain information)
        """
        if space is None:
            raise ValueError(
                f"Space must be provided to {self.__class__.__name__}. "
                "Space cannot be None."
            )
        self.space = space

    @property
    def domain(self):
        """Get the domain from the space."""
        return self.space.function_domain


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
        restricted_space
    ):
        """
        Initialize restricted provider.

        Args:
            original_provider: The provider to restrict
            restricted_space: The target space for restrictions
        """
        super().__init__(restricted_space)
        self.original_provider = original_provider

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get restricted version of function at given index."""
        original_func = self.original_provider.get_function_by_index(
            index, **kwargs
        )
        return original_func.restrict(self.space)


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
        return self.space.zero
