"""
Sobolev spaces on interval domains.

This module provides Sobolev spaces H^s([a,b]) on intervals, which are
Hilbert spaces of functions with s derivatives in L². The inner product
involves both the function and its derivatives, providing additional
smoothness structure beyond L².

The Sobolev space uses a Bessel potential representation:
    H^s = (k² I + Δ)^{-s/2} L²

where Δ is the Laplacian operator with appropriate boundary conditions.

This module provides:
- `Sobolev`: The main Sobolev space class (inherits MassWeightedHilbertSpace)
- `SobolevSpaceDirectSum`: Direct sums of Sobolev spaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from pygeoinf import MassWeightedHilbertSpace, HilbertSpaceDirectSum

from intervalinf.core.config import IntegrationConfig, ParallelConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.forms import LinearFormKernel

if TYPE_CHECKING:
    from intervalinf.core.domain import IntervalDomain
    from intervalinf.core.boundary import BoundaryConditions
    from intervalinf.spaces.lebesgue import Lebesgue


class Sobolev(MassWeightedHilbertSpace):
    """
    Sobolev space H^s([a,b]) on an interval.

    Sobolev spaces are Hilbert spaces of functions with s derivatives
    in L². The inner product is:

        ⟨u, v⟩_{H^s} = ⟨(k² I + Δ)^s u, v⟩_{L²}

    where Δ is a Laplacian operator with specified boundary conditions.

    This implementation uses a mass-weighted structure where:
    - The underlying space is L² (Lebesgue)
    - The mass operator is M = (k² I + Δ)^{2s}
    - Inner products are computed as ⟨Mu, v⟩_{L²}

    Attributes:
        function_domain: The interval [a, b].
        dim: Dimension of the finite-dimensional approximation.

    Example:
        >>> from intervalinf.core import IntervalDomain, BoundaryConditions
        >>> domain = IntervalDomain(0, 1)
        >>> bcs = BoundaryConditions(bc_type='dirichlet', left=0, right=0)
        >>> # Create with Laplacian operator (Phase 3)
        >>> # space = Sobolev(50, domain, s=1, k=1, L=laplacian)

    Note:
        This class depends on operators from Phase 3. Full functionality
        requires the operators module to be migrated.
    """

    def __init__(
        self,
        dim: int,
        function_domain: 'IntervalDomain',
        s: float,
        k: float,
        L,  # SpectralOperator - will be typed in Phase 3
        /,
        *,
        basis: Optional[Union[str, list]] = None,
        integration_config: Optional[IntegrationConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """
        Initialize a Sobolev space.

        Args:
            dim: Dimension of the finite-dimensional approximation.
            function_domain: IntervalDomain [a, b].
            s: Regularity parameter (Sobolev order). Higher s means
                smoother functions.
            k: Scaling parameter in Bessel potential (k² I + Δ)^s.
            L: Spectral operator (typically Laplacian) defining the
                differential structure.
            basis: Basis type for underlying Lebesgue space.
            integration_config: Integration configuration.
            parallel_config: Parallelization configuration.
        """
        # Extract configs from Laplacian if not provided
        if integration_config is None and hasattr(L, '_integration_config'):
            integration_config = L._integration_config
        if parallel_config is None and hasattr(L, '_parallel_config'):
            parallel_config = L._parallel_config

        # Create underlying L² space
        self._underlying_space = self._create_underlying_space(
            dim, function_domain, basis, integration_config, parallel_config
        )

        # Sobolev parameters
        self._s = s
        self._k = k
        self._L = L
        self._dofs = getattr(L, '_dofs', dim)

        # Domain from underlying space
        self._function_domain = self._underlying_space.function_domain

        # Create mass operators (requires Phase 3 operators).
        # If no Laplacian/operator `L` was provided, defer operator
        # construction and use placeholders that raise when invoked.
        if self._L is None:
            def _not_implemented(*args, **kwargs):
                raise NotImplementedError(
                    "Sobolev mass operators require the operators module; "
                    "provide a Laplacian or install pygeoinf with operators."
                )

            # simple callables are sufficient at runtime for mass ops
            M_op = lambda x: _not_implemented()
            M_op_inv = lambda x: _not_implemented()
        else:
            M_op, M_op_inv = self._create_mass_operators()

        super().__init__(
            self._underlying_space,
            M_op,
            M_op_inv,
        )

    def _create_underlying_space(
        self,
        dim: int,
        function_domain: 'IntervalDomain',
        basis: Optional[Union[str, list]],
        integration_config: Optional[IntegrationConfig],
        parallel_config: Optional[ParallelConfig],
    ) -> 'Lebesgue':
        """Create the underlying Lebesgue space."""
        from intervalinf.spaces.lebesgue import Lebesgue
        return Lebesgue(
            dim,
            function_domain,
            basis=basis,
            integration_config=integration_config,
            parallel_config=parallel_config,
        )

    def _create_mass_operators(self):
        """
        Create Bessel-Sobolev mass operators.

        Returns:
            Tuple of (mass_operator, inverse_mass_operator).

        Note:
            This uses pygeoinf.interval.operators for now.
            Will be migrated in Phase 3.
        """
        try:
            from intervalinf.operators import (
                BesselSobolev, BesselSobolevInverse
            )

            M_op = BesselSobolev(
                self._underlying_space,
                self._underlying_space,
                k=self._k,
                s=2 * self._s,
                L=self._L,
                dofs=self._dofs
            )
            M_op_inv = BesselSobolevInverse(
                self._underlying_space,
                self._underlying_space,
                k=self._k,
                s=2 * self._s,
                L=self._L,
                dofs=self._dofs
            )
            return M_op, M_op_inv

        except ImportError:
            raise NotImplementedError(
                "Sobolev spaces require the operators module (Phase 3). "
                "Install pygeoinf for interim support."
            )

    @property
    def function_domain(self) -> 'IntervalDomain':
        """The interval domain [a, b]."""
        return self._function_domain

    @property
    def zero(self) -> Function:
        """The zero function in this space."""
        return Function(
            self._underlying_space,
            evaluate_callable=lambda x: np.zeros_like(x)
        )

    @property
    def s(self) -> float:
        """Sobolev regularity parameter."""
        return self._s

    @property
    def k(self) -> float:
        """Bessel potential scaling parameter."""
        return self._k

    def to_dual(self, x: Function) -> LinearFormKernel:
        """
        Map a function to its dual via the mass operator.

        In Sobolev spaces, the dual mapping applies the mass operator:
            φ_x(y) = ⟨x, y⟩_{H^s} = ⟨Mx, y⟩_{L²}

        Args:
            x: A Function in this space.

        Returns:
            LinearFormKernel with kernel = Mx.
        """
        if not isinstance(x, Function):
            raise TypeError("Expected Function for primal element")

        kernel = self._mass_operator(x)

        int_cfg = self._underlying_space.integration.dual
        par_cfg = self._underlying_space.parallel.dual

        return LinearFormKernel(
            self,
            kernel=kernel,
            integration_config=int_cfg,
            parallel_config=par_cfg
        )

    def from_dual(self, xp: LinearFormKernel) -> Function:
        """
        Map a dual element back to a function via the inverse mass.

        Args:
            xp: A LinearFormKernel (or generic LinearForm).

        Returns:
            Function x = M⁻¹ kernel.
        """
        if isinstance(xp, LinearFormKernel):
            # Direct kernel - apply inverse mass
            return self._inverse_mass_operator(xp.kernel)
        else:
            # Generic LinearForm - extract kernel from underlying space
            kernel = self._underlying_space.from_dual(xp)
            return self._inverse_mass_operator(kernel)

    @property
    def mass_operator_factor(self):
        """Get M^{1/2} = (k² I + Δ)^s."""
        from intervalinf.operators.bessel import BesselSobolev
        return BesselSobolev(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=self._L,
            dofs=self._dofs
        )

    @property
    def inverse_mass_operator_factor(self):
        """Get M^{-1/2} = (k² I + Δ)^{-s}."""
        from intervalinf.operators.bessel import BesselSobolevInverse
        return BesselSobolevInverse(
            self._underlying_space,
            self._underlying_space,
            k=self._k,
            s=self._s,
            L=self._L,
            dofs=self._dofs
        )

    def restrict(
        self,
        restricted_space: 'Sobolev',
        new_bcs: Optional['BoundaryConditions'] = None
    ) -> 'Sobolev':
        """
        Restrict Sobolev space to a subspace with new boundary conditions.

        Args:
            restricted_space: Target Sobolev space with same s, k parameters.
            new_bcs: New boundary conditions for the Laplacian. If None,
                uses the original boundary conditions.

        Returns:
            The restricted_space with updated Laplacian.

        Raises:
            ValueError: If s or k parameters don't match.
        """
        # Validate regularity parameters
        if restricted_space._s != self._s:
            raise ValueError(
                f"Regularity s must match: "
                f"original={self._s}, restricted={restricted_space._s}"
            )
        if restricted_space._k != self._k:
            raise ValueError(
                f"Scaling k must match: "
                f"original={self._k}, restricted={restricted_space._k}"
            )

        # Restrict Laplacian
        L_restricted = self._L.restrict(
            restricted_space._underlying_space,
            new_bcs=new_bcs
        )
        restricted_space._L = L_restricted

        # Update mass operators
        from intervalinf.operators.bessel import BesselSobolev, BesselSobolevInverse

        M_op = BesselSobolev(
            restricted_space._underlying_space,
            restricted_space._underlying_space,
            k=restricted_space._k,
            s=2 * restricted_space._s,
            L=L_restricted,
            dofs=L_restricted._dofs
        )
        M_op_inv = BesselSobolevInverse(
            restricted_space._underlying_space,
            restricted_space._underlying_space,
            k=restricted_space._k,
            s=2 * restricted_space._s,
            L=L_restricted,
            dofs=L_restricted._dofs
        )

        restricted_space._mass_operator = M_op
        restricted_space._inverse_mass_operator = M_op_inv

        return restricted_space

    @classmethod
    def with_discontinuities(
        cls,
        dim: int,
        function_domain: 'IntervalDomain',
        discontinuity_points: list,
        s: float,
        k: float,
        bcs: 'BoundaryConditions',
        alpha: float,
        *,
        basis: Optional[Union[str, list]] = None,
        dim_per_subspace: Optional[list] = None,
        basis_per_subspace: Optional[list] = None,
        bcs_per_subspace: Optional[list] = None,
        laplacian_method: str = 'spectral',
        dofs: int = 100,
        n_samples: int = 2048,
        integration_config: Optional[IntegrationConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> 'SobolevSpaceDirectSum':
        """
        Create a SobolevSpaceDirectSum with discontinuities.

        This factory creates a direct sum of Sobolev spaces on subintervals
        separated by discontinuity points.

        Args:
            dim: Total dimension across all subspaces.
            function_domain: The full interval domain.
            discontinuity_points: Points where discontinuities occur.
            s: Sobolev regularity parameter.
            k: Bessel potential scaling.
            bcs: Default boundary conditions.
            alpha: Laplacian scaling parameter.
            basis: Basis type for all subspaces.
            dim_per_subspace: Optional dimensions per subspace.
            basis_per_subspace: Optional basis per subspace.
            bcs_per_subspace: Optional boundary conditions per subspace.
            laplacian_method: Method for Laplacian ('spectral', etc.).
            dofs: Degrees of freedom for Laplacian.
            n_samples: Samples for spectral operators.
            integration_config: Integration configuration.
            parallel_config: Parallelization configuration.

        Returns:
            SobolevSpaceDirectSum with component spaces.
        """
        from intervalinf.spaces.lebesgue import Lebesgue

        if isinstance(basis, list):
            raise ValueError(
                "List of basis functions not supported for discontinuous "
                "spaces. Use basis_per_subspace instead."
            )

        # Split domain
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

        # Determine boundary conditions
        if bcs_per_subspace is not None:
            if len(bcs_per_subspace) != n_subspaces:
                raise ValueError(
                    f"bcs_per_subspace must have length {n_subspaces}"
                )
            bcs_list = list(bcs_per_subspace)
        else:
            bcs_list = [bcs] * n_subspaces

        # Default integration config
        if integration_config is None:
            integration_config = IntegrationConfig(
                method='simpson', n_points=1000
            )

        # Create subspaces
        from intervalinf.operators.laplacian import Laplacian

        subspaces = []
        for d, subdomain, b, bc in zip(dims, subdomains, bases, bcs_list):
            # Create Lebesgue space for Laplacian
            M_lebesgue = Lebesgue(
                0, subdomain, basis=None,
                integration_config=integration_config,
                parallel_config=parallel_config
            )

            # Create Laplacian
            laplacian = Laplacian(
                M_lebesgue, bc, alpha,
                method=laplacian_method,
                dofs=dofs,
                n_samples=n_samples,
                integration_config=integration_config
            )

            # Create Sobolev subspace
            sobolev_subspace = cls(
                d, subdomain, s, k, laplacian,
                basis=b,
                integration_config=integration_config
            )
            subspaces.append(sobolev_subspace)

        return SobolevSpaceDirectSum(subspaces)


class SobolevSpaceDirectSum(HilbertSpaceDirectSum):
    """
    Direct sum of Sobolev spaces using LinearFormKernel.

    This extends HilbertSpaceDirectSum for Sobolev spaces without
    requiring explicit basis functions. Uses integration-based
    inner products via LinearFormKernel.
    """

    def to_dual(self, xs: List[Function]) -> LinearFormKernel:
        """
        Map a list of functions to a dual element.

        Applies mass operator to each component.

        Args:
            xs: List of Functions, one per subspace.

        Returns:
            LinearFormKernel with mass-weighted kernels.
        """
        if len(xs) != self.number_of_subspaces:
            raise ValueError("Input list has incorrect number of vectors.")

        # Apply to_dual on each subspace (applies mass operator)
        kernels = [
            space.to_dual(x).kernel
            for space, x in zip(self._spaces, xs)
        ]

        # Get config from first subspace
        first_subspace = self._spaces[0]
        underlying = getattr(first_subspace, '_underlying_space', None)
        if underlying:
            int_cfg = underlying.integration.dual
            par_cfg = underlying.parallel.dual
        else:
            int_cfg = IntegrationConfig()
            par_cfg = ParallelConfig()

        return LinearFormKernel(
            self,
            kernel=kernels,
            integration_config=int_cfg,
            parallel_config=par_cfg
        )

    def from_dual(self, xp: LinearFormKernel) -> List[Function]:
        """
        Map a dual element back to functions.

        Args:
            xp: LinearFormKernel with list of kernels.

        Returns:
            List of Functions, one per subspace.
        """
        if isinstance(xp, LinearFormKernel):
            if isinstance(xp.kernel, list):
                int_cfg = getattr(xp, 'integration', IntegrationConfig())
                par_cfg = getattr(xp, 'parallel', ParallelConfig())
                return [
                    space.from_dual(
                        LinearFormKernel(
                            space, kernel=k,
                            integration_config=int_cfg,
                            parallel_config=par_cfg
                        )
                    )
                    for space, k in zip(self._spaces, xp.kernel)
                ]
            else:
                raise ValueError("Expected kernel to be a list for direct sum")
        else:
            return super().from_dual(xp)


__all__ = [
    'Sobolev',
    'SobolevSpaceDirectSum',
]
