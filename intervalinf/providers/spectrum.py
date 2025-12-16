"""
Eigenvalue and spectrum providers for interval domains.

This module contains providers for eigenvalues and spectra of common
differential operators, particularly the Laplacian with various
boundary conditions.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

from intervalinf.utils.robin_utils import RobinRootFinder

if TYPE_CHECKING:
    from intervalinf.spaces.lebesgue import Lebesgue
    from intervalinf.spaces.sobolev import Sobolev


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


class SineEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for sine eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (kπ/L)² where L is the domain length.
    """

    def __init__(self, domain_length: float):
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        k = index + 1  # Sine functions start from k=1
        return (k * math.pi / self.domain_length) ** 2


class CosineEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for cosine eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (kπ/L)² where L is the domain length.
    The first eigenvalue (k=0) is zero corresponding to the constant mode.
    """

    def __init__(self, domain_length: float):
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        if index == 0:
            return 0.0
        else:
            k = index
            return (k * math.pi / self.domain_length) ** 2


class FourierEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for Fourier eigenfunctions of the negative Laplacian.

    Computes eigenvalues λₖ = (2πk/L)² where L is the domain length.
    """

    def __init__(self, domain_length: float):
        self.domain_length = domain_length

    def get_eigenvalue(self, index: int) -> float:
        if index == 0:
            return 0.0
        else:
            k = (index + 1) // 2
            return (2 * k * math.pi / self.domain_length) ** 2


class ZeroEigenvalueProvider(EigenvalueProvider):
    """Eigenvalue provider that returns zero for all indices."""

    def get_eigenvalue(self, index: int) -> float:
        return 0.0


class CustomEigenvalueProvider(EigenvalueProvider):
    """Eigenvalue provider with user-specified eigenvalues."""

    def __init__(self, eigenvalues: Union[np.ndarray, list]):
        self.eigenvalues = np.asarray(eigenvalues)

    def get_eigenvalue(self, index: int) -> float:
        if not (0 <= index < len(self.eigenvalues)):
            raise IndexError(
                f"Eigenvalue index {index} out of range "
                f"[0, {len(self.eigenvalues)})"
            )
        return self.eigenvalues[index]


class LaplacianEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for the negative Laplacian operator.

    Computes eigenvalues λₖ based on boundary conditions and domain.
    For the inverse Laplacian, eigenvalues are 1/λₖ.
    """

    def __init__(
        self,
        function_domain,
        boundary_conditions,
        inverse=False,
        alpha=1.0
    ):
        """
        Initialize the eigenvalue provider.

        Args:
            function_domain: Interval domain
            boundary_conditions: Boundary conditions object
            inverse: If True, compute eigenvalues of (-Δ)⁻¹
            alpha: Scaling factor for the eigenvalues
        """
        self._function_domain = function_domain
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        self._alpha = alpha
        self._eigenvalue_cache = {}
        self._robin_mu: list = []

    def get_eigenvalue(self, index: int) -> float:
        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int) -> float:
        length = self._function_domain.b - self._function_domain.a

        if self._boundary_conditions.type == 'dirichlet':
            sine_provider = SineEigenvalueProvider(length)
            eigenval = sine_provider.get_eigenvalue(index)

        elif self._boundary_conditions.type == 'neumann':
            if self._inverse:
                index += 1  # Skip zero eigenvalue for inverse
            cosine_provider = CosineEigenvalueProvider(length)
            eigenval = cosine_provider.get_eigenvalue(index)

        elif self._boundary_conditions.type == 'periodic':
            if self._inverse:
                index += 1  # Skip zero eigenvalue for inverse
            fourier_provider = FourierEigenvalueProvider(length)
            eigenval = fourier_provider.get_eigenvalue(index)

        elif self._boundary_conditions.type == 'mixed_dirichlet_neumann':
            eigenval = (((index + 0.5) * np.pi) / length)**2

        elif self._boundary_conditions.type == 'mixed_neumann_dirichlet':
            eigenval = (((index + 0.5) * np.pi) / length)**2

        elif self._boundary_conditions.type == 'robin':
            mu = self._robin_mu_at(index)
            eigenval = mu * mu

        else:
            raise ValueError(
                f"Unknown boundary condition type: "
                f"{self._boundary_conditions.type}"
            )

        # Apply alpha scaling and inverse if needed
        if self._inverse:
            return 1.0 / (eigenval * self._alpha)
        else:
            return eigenval * self._alpha

    def _robin_mu_at(self, k: int) -> float:
        while len(self._robin_mu) <= k:
            self._append_next_robin_root(k)
        return self._robin_mu[k]

    def _append_next_robin_root(self, target_index: int):
        L = self._function_domain.b - self._function_domain.a
        alpha_0 = float(
            self._boundary_conditions.get_parameter('left_alpha')
        )
        beta_0 = float(
            self._boundary_conditions.get_parameter('left_beta')
        )
        alpha_L = float(
            self._boundary_conditions.get_parameter('right_alpha')
        )
        beta_L = float(
            self._boundary_conditions.get_parameter('right_beta')
        )

        mu = RobinRootFinder.compute_robin_eigenvalue(
            target_index, alpha_0, beta_0, alpha_L, beta_L, L,
            tol=1e-12, maxit=100
        )
        self._robin_mu.append(mu)


class SpectrumProvider(ABC):
    """
    Abstract base class for spectrum providers.

    Combines eigenvalues with eigenfunctions for spectral decomposition.
    """

    def __init__(
        self,
        space,
        orthonormal: bool = False,
        basis_type: str = ''
    ):
        self.space = space
        self.orthonormal = orthonormal
        self.type = basis_type

    @abstractmethod
    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue for given index."""
        pass

    @abstractmethod
    def get_eigenfunction(self, index: int):
        """Get eigenfunction for given index."""
        pass


class LaplacianSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider for Laplacian eigenfunctions.

    Combines a function provider (for eigenfunctions) with an eigenvalue
    provider (for corresponding eigenvalues of -Δ or (-Δ)⁻¹).
    """

    def __init__(
        self,
        space: "Union[Lebesgue, Sobolev]",
        boundary_conditions,
        alpha: float = 1,
        inverse: bool = False,
    ):
        """
        Initialize Laplacian spectrum provider.

        Args:
            space: The function space
            boundary_conditions: Boundary conditions object
            alpha: Scaling factor
            inverse: If True, use inverse eigenvalues
        """
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        super().__init__(space, orthonormal=True, basis_type='')

        self._eigenvalue_provider = LaplacianEigenvalueProvider(
            space.function_domain,
            boundary_conditions,
            inverse,
            alpha,
        )
        self._function_provider = self._choose_basis()

    def get_eigenvalue(self, index: int) -> float:
        return self._eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int):
        if self._boundary_conditions.type == 'dirichlet':
            return self._function_provider.get_function_by_index(index)
        elif self._boundary_conditions.type == 'neumann':
            if self._inverse:
                index += 1
            return self._function_provider.get_function_by_index(index)
        elif self._boundary_conditions.type == 'periodic':
            if self._inverse:
                index += 1
            return self._function_provider.get_function_by_index(index)
        elif self._boundary_conditions.type in [
            'mixed_dirichlet_neumann',
            'mixed_neumann_dirichlet',
            'robin'
        ]:
            return self._function_provider.get_function_by_index(index)

    def _choose_basis(self):
        from intervalinf.providers.functions import (
            SineFunctionProvider,
            CosineFunctionProvider,
            FourierFunctionProvider,
            MixedDNFunctionProvider,
            MixedNDFunctionProvider,
            RobinFunctionProvider,
        )

        if self._boundary_conditions.type == 'dirichlet':
            return SineFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'neumann':
            return CosineFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'periodic':
            return FourierFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'mixed_dirichlet_neumann':
            return MixedDNFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'mixed_neumann_dirichlet':
            return MixedNDFunctionProvider(self.space)
        elif self._boundary_conditions.type == 'robin':
            return RobinFunctionProvider(self.space, self._boundary_conditions)
        else:
            raise ValueError(
                f"Unsupported boundary condition: "
                f"{self._boundary_conditions.type}"
            )
