"""Base classes for operators on interval domains."""

from abc import ABC, abstractmethod

from pygeoinf.linear_operators import LinearOperator

from intervalinf.core.functions import Function


class SpectralOperator(LinearOperator, ABC):
    """
    Abstract base class for spectral operators on interval domains.

    Spectral operators are linear operators that can be diagonalized in terms
    of eigenfunctions. They provide methods for accessing eigenvalues and
    eigenfunctions, which are used in spectral methods for applying the
    operator and its inverse.

    This is the base class for operators like:
    - Laplacian: -d²/dx² with boundary conditions
    - InverseLaplacian: Green's function operator
    - RadialLaplacian: Laplacian in spherical coordinates

    Subclasses must implement:
    - get_eigenvalue(index): Return eigenvalue at given index
    - get_eigenfunction(index): Return eigenfunction at given index
    - _apply(f): Apply the operator to a function
    """

    def __init__(
        self,
        domain,
        codomain,
        mapping
    ):
        """
        Initialize spectral operator.

        Args:
            domain: Input function space
            codomain: Output function space
            mapping: The operator mapping function
        """
        super().__init__(domain, codomain, mapping)

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """
        Get the eigenvalue at a specific index.

        Args:
            index: Zero-based index of the eigenvalue

        Returns:
            The eigenvalue λ_index
        """
        pass

    @abstractmethod
    def get_eigenfunction(self, index: int) -> Function:
        """
        Get the eigenfunction at a specific index.

        Args:
            index: Zero-based index of the eigenfunction

        Returns:
            The eigenfunction φ_index as a Function object
        """
        pass

    @abstractmethod
    def _apply(self, f: Function) -> Function:
        """
        Apply the operator to a function.

        Args:
            f: Input function

        Returns:
            Result of applying the operator: L(f)
        """
        pass
