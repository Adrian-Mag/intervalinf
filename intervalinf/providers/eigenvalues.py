"""
Simple eigenvalue providers for common differential operators.

This module contains eigenvalue providers for the negative Laplacian
with various boundary conditions. These are Level 1 providers that
depend only on the base EigenvalueProvider class.

For composite providers like LaplacianEigenvalueProvider that build
on these, see laplacian.py.
"""

import math
import numpy as np

from intervalinf.providers.base import EigenvalueProvider


class SineEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for sine eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (kπ/L)² where L is the domain length.
    These correspond to Dirichlet boundary conditions.

    Index mapping: index 0 → k=1, index 1 → k=2, etc.
    """

    def __init__(self, domain_length: float):
        """
        Initialize sine eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for the negative Laplacian with Dirichlet BC."""
        k = index + 1  # Sine functions start from k=1
        return (k * math.pi / self.domain_length) ** 2


class CosineEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for cosine eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (kπ/L)² where L is the domain length.
    The first eigenvalue (k=0) is zero corresponding to the constant mode.
    These correspond to Neumann boundary conditions.

    Index mapping: index 0 → k=0 (constant), index 1 → k=1, etc.
    """

    def __init__(self, domain_length: float):
        """
        Initialize cosine eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for the negative Laplacian with Neumann BC."""
        if index == 0:
            return 0.0  # Constant mode has eigenvalue 0
        else:
            k = index
            return (k * math.pi / self.domain_length) ** 2


class FourierEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for Fourier eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (2πk/L)² where L is the domain length.
    These correspond to periodic boundary conditions.

    Index mapping:
        index 0 → constant (λ=0)
        index 1 → cos(2πx/L) (k=1)
        index 2 → sin(2πx/L) (k=1)
        index 3 → cos(4πx/L) (k=2)
        index 4 → sin(4πx/L) (k=2)
        etc.

    Both cos and sin modes with the same k have identical eigenvalues.
    """

    def __init__(self, domain_length: float):
        """
        Initialize Fourier eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for the negative Laplacian with periodic BC."""
        if index == 0:
            return 0.0  # Constant term has eigenvalue 0
        else:
            # For index > 0, we alternate between cosine and sine
            k = (index + 1) // 2  # Frequency index
            # Both cos and sin modes have the same eigenvalue
            return (2 * k * math.pi / self.domain_length) ** 2


class ZeroEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider that returns zero for all indices.

    Useful for basis functions that aren't eigenfunctions of a specific
    operator, or when eigenvalue information isn't needed.
    """

    def get_eigenvalue(self, index: int) -> float:
        """Return zero eigenvalue."""
        return 0.0


class MixedDNEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for mixed Dirichlet-Neumann boundary conditions.

    For -d²/dx² on (a,b) with u(a)=0, u'(b)=0:
    Eigenvalues: λₖ = ((k+1/2)π/L)² for k=0,1,2,...
    """

    def __init__(self, domain_length: float):
        """
        Initialize mixed DN eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for mixed Dirichlet-Neumann BC."""
        return (((index + 0.5) * math.pi) / self.domain_length) ** 2


class MixedNDEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for mixed Neumann-Dirichlet boundary conditions.

    For -d²/dx² on (a,b) with u'(a)=0, u(b)=0:
    Eigenvalues: λₖ = ((k+1/2)π/L)² for k=0,1,2,...

    Note: Same eigenvalues as DN case, but different eigenfunctions.
    """

    def __init__(self, domain_length: float):
        """
        Initialize mixed ND eigenvalue provider.

        Args:
            domain_length: Length of the domain (b - a)
        """
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue for mixed Neumann-Dirichlet BC."""
        return (((index + 0.5) * math.pi) / self.domain_length) ** 2
