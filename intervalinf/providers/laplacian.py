"""
Laplacian eigenvalue and spectrum providers.

This module contains composite providers for the Laplacian operator
that build on the simple eigenvalue providers and function providers:

- LaplacianEigenvalueProvider: Eigenvalues for -Δ or (-Δ)⁻¹
- LaplacianSpectrumProvider: Complete eigenbasis for the Laplacian
"""

import numpy as np
from typing import Union, TYPE_CHECKING

from intervalinf.providers.base import EigenvalueProvider, SpectrumProvider
from intervalinf.providers.eigenvalues import (
    SineEigenvalueProvider,
    CosineEigenvalueProvider,
    FourierEigenvalueProvider,
)
from intervalinf.utils.robin_utils import RobinRootFinder

if TYPE_CHECKING:
    from intervalinf.spaces.lebesgue import Lebesgue
    from intervalinf.spaces.sobolev import Sobolev


class LaplacianEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for the negative Laplacian operator.

    Computes eigenvalues λₖ based on boundary conditions and domain.
    For the inverse Laplacian, eigenvalues are 1/λₖ.

    Supported boundary conditions:
    - 'dirichlet': Uses SineEigenvalueProvider
    - 'neumann': Uses CosineEigenvalueProvider
    - 'periodic': Uses FourierEigenvalueProvider
    - 'mixed_dirichlet_neumann': λₖ = ((k+1/2)π/L)²
    - 'mixed_neumann_dirichlet': λₖ = ((k+1/2)π/L)²
    - 'robin': Numerically computed from transcendental equation
    """

    def __init__(
        self,
        function_domain,
        boundary_conditions,
        inverse: bool = False,
        alpha: float = 1.0
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
        """Get eigenvalue for given index (with caching)."""
        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int) -> float:
        """Compute eigenvalue based on boundary conditions."""
        length = self._function_domain.b - self._function_domain.a
        bc_type = self._boundary_conditions.type

        if bc_type == 'dirichlet':
            sine_provider = SineEigenvalueProvider(length)
            eigenval = sine_provider.get_eigenvalue(index)

        elif bc_type == 'neumann':
            if self._inverse:
                index += 1  # Skip zero eigenvalue for inverse
            cosine_provider = CosineEigenvalueProvider(length)
            eigenval = cosine_provider.get_eigenvalue(index)

        elif bc_type == 'periodic':
            if self._inverse:
                index += 1  # Skip zero eigenvalue for inverse
            fourier_provider = FourierEigenvalueProvider(length)
            eigenval = fourier_provider.get_eigenvalue(index)

        elif bc_type == 'mixed_dirichlet_neumann':
            eigenval = (((index + 0.5) * np.pi) / length)**2

        elif bc_type == 'mixed_neumann_dirichlet':
            eigenval = (((index + 0.5) * np.pi) / length)**2

        elif bc_type == 'robin':
            mu = self._robin_mu_at(index)
            eigenval = mu * mu

        else:
            raise ValueError(
                f"Unknown boundary condition type: {bc_type}"
            )

        # Apply alpha scaling and inverse if needed
        if self._inverse:
            return 1.0 / (eigenval * self._alpha)
        else:
            return eigenval * self._alpha

    def _robin_mu_at(self, k: int) -> float:
        """Get μₖ for Robin boundary conditions (with caching)."""
        while len(self._robin_mu) <= k:
            self._append_next_robin_root(k)
        return self._robin_mu[k]

    def _append_next_robin_root(self, target_index: int):
        """Compute and cache the next Robin eigenvalue root."""
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


class LaplacianSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider for Laplacian eigenfunctions.

    Combines a function provider (for eigenfunctions) with an eigenvalue
    provider (for corresponding eigenvalues of -Δ or (-Δ)⁻¹).

    This is the main entry point for spectral methods on intervals.
    """

    def __init__(
        self,
        space: "Union[Lebesgue, Sobolev]",
        boundary_conditions,
        alpha: float = 1.0,
        inverse: bool = False,
    ):
        """
        Initialize Laplacian spectrum provider.

        Args:
            space: The function space (Lebesgue or Sobolev)
            boundary_conditions: Boundary conditions object
            alpha: Scaling factor for eigenvalues
            inverse: If True, use eigenvalues of (-Δ)⁻¹
        """
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        super().__init__(space, orthonormal=True, basis_type='laplacian')

        self._eigenvalue_provider = LaplacianEigenvalueProvider(
            space.function_domain,
            boundary_conditions,
            inverse,
            alpha,
        )
        self._function_provider = self._choose_basis()

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        return self._eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int):
        """
        Get eigenfunction at given index.

        Handles index adjustment for inverse operators where zero
        eigenvalue modes are excluded.
        """
        bc_type = self._boundary_conditions.type

        if bc_type == 'dirichlet':
            return self._function_provider.get_function_by_index(index)

        elif bc_type == 'neumann':
            if self._inverse:
                index += 1  # Skip constant mode for inverse
            return self._function_provider.get_function_by_index(index)

        elif bc_type == 'periodic':
            if self._inverse:
                index += 1  # Skip constant mode for inverse
            return self._function_provider.get_function_by_index(index)

        elif bc_type in ['mixed_dirichlet_neumann',
                         'mixed_neumann_dirichlet',
                         'robin']:
            return self._function_provider.get_function_by_index(index)

        else:
            raise ValueError(
                f"Unsupported boundary condition: {bc_type}"
            )

    def _choose_basis(self):
        """Select appropriate function provider based on boundary conditions."""
        from intervalinf.providers.functions import (
            SineFunctionProvider,
            CosineFunctionProvider,
            FourierFunctionProvider,
            MixedDNFunctionProvider,
            MixedNDFunctionProvider,
            RobinFunctionProvider,
        )

        bc_type = self._boundary_conditions.type

        if bc_type == 'dirichlet':
            return SineFunctionProvider(self.space)

        elif bc_type == 'neumann':
            return CosineFunctionProvider(self.space)

        elif bc_type == 'periodic':
            return FourierFunctionProvider(self.space)

        elif bc_type == 'mixed_dirichlet_neumann':
            return MixedDNFunctionProvider(self.space)

        elif bc_type == 'mixed_neumann_dirichlet':
            return MixedNDFunctionProvider(self.space)

        elif bc_type == 'robin':
            return RobinFunctionProvider(
                self.space,
                self._boundary_conditions
            )

        else:
            raise ValueError(
                f"Unsupported boundary condition: {bc_type}"
            )
