"""Laplacian and inverse Laplacian operators for interval domains."""

import logging
from typing import Union, Optional, Literal, List

import numpy as np

from intervalinf.operators.base import SpectralOperator
from intervalinf.operators.spectral_helpers import (
    build_eigenfunction_expansion,
    compute_spectral_coefficients_fast,
    compute_spectral_coefficients_slow,
)
from intervalinf.operators._impl.fast_spectral import (
    fast_spectral_coefficients,
    create_uniform_samples,
)
from intervalinf.operators._impl.fem_solvers import GeneralFEMSolver
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.spaces.sobolev import Sobolev
from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.functions import Function
from intervalinf.core.config import IntegrationConfig
from intervalinf.providers.spectrum import LaplacianSpectrumProvider


class Laplacian(SpectralOperator):
    """
    The Laplacian operator (-Δ) on interval domains.

    This operator computes -d²u/dx² with specified boundary conditions.
    Multiple discretization methods are available:

    - 'spectral': Uses analytical eigendecomposition with
                  Fourier/sine/cosine basis
    - 'fd': Centered finite differences

    The operator maps between function spaces based on the method:
    - Spectral: L² → L² (preserves regularity via eigenfunction expansion)
    - FD: H^s → H^(s-2) (requires domain with sufficient regularity)
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['spectral', 'fd'] = 'spectral',
        dofs: Optional[int] = None,
        fd_order: int = 2,
        n_samples: int = 512,
        integration_config: IntegrationConfig = None,
    ):
        """
        Initialize the negative Laplacian operator.

        Args:
            domain: Function space (Lebesgue or Sobolev)
            boundary_conditions: Boundary conditions for the operator
            alpha: Scaling factor (default: 1.0)
            method: Discretization method ('spectral', 'fd')
            dofs: Number of degrees of freedom (default: domain.dim)
            fd_order: Order of finite difference stencil (2, 4)
            n_samples: Number of samples for fast spectral transforms
            integration_config: Integration configuration
        """
        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._dofs = dofs if dofs is not None else domain.dim
        self._fd_order = fd_order
        self._method = method
        self._n_samples = max(n_samples, self._dofs)

        # Store integration config
        if integration_config is None:
            integration_config = IntegrationConfig(
                method='simpson', n_points=1000
            )
        self.integration = integration_config

        super().__init__(domain, domain, self._apply)

        # Initialize spectrum provider
        self._spectrum_provider = LaplacianSpectrumProvider(
            domain,
            boundary_conditions,
            alpha
        )

        # Check if fast transforms are available for this BC type
        self._can_use_fast_transforms = boundary_conditions.type in [
            'dirichlet', 'neumann', 'periodic',
            'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
        ]

        if self._method == 'fd':
            self._setup_finite_difference()

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        return self._alpha * self._spectrum_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        if isinstance(self._domain, Sobolev):
            func = self._spectrum_provider.get_eigenfunction(index)
            return self._domain.inverse_mass_operator_factor(func)
        else:
            return self._spectrum_provider.get_eigenfunction(index)

    def _setup_finite_difference(self):
        """Setup finite difference discretization."""
        a = self._domain.function_domain.a
        b = self._domain.function_domain.b
        self._x_grid = np.linspace(a, b, self._dofs)
        self._dx = (b - a) / (self._dofs - 1)
        self._fd_matrix = self._create_fd_matrix()

        logger = logging.getLogger(__name__)
        logger.info(
            "Laplacian (finite difference, order %s) "
            "initialized with %s grid points",
            self._fd_order,
            self._dofs,
        )

    def _create_fd_matrix(self):
        """Create finite difference matrix for the negative Laplacian."""
        n = self._dofs
        dx2 = self._dx**2

        def _second_derivative_stencil(order: int, location: str = "center"):
            if order == 2:
                if location == "center":
                    return np.array([-1, 0, 1]), np.array([1.0, -2.0, 1.0])
                if location == "near_left":
                    return np.array([0, 1, 2]), np.array([1.0, -2.0, 1.0])
                if location == "near_right":
                    return np.array([-2, -1, 0]), np.array([1.0, -2.0, 1.0])
            if order == 4:
                if location == "center":
                    offsets = np.array([-2, -1, 0, 1, 2])
                    coeffs = np.array([1.0, -16.0, 30.0, -16.0, 1.0]) / 12.0
                    return offsets, coeffs
                if location in ("near_left", "near_right"):
                    return _second_derivative_stencil(2, location)
            raise ValueError(
                "Unsupported fd_order or location for second derivative"
            )

        if self._fd_order == 2:
            matrix = np.zeros((n, n))

            offsets, coeffs = _second_derivative_stencil(2, "center")
            for i in range(1, n-1):
                for off, c in zip(offsets, coeffs):
                    matrix[i, i + off] = -c / dx2

            offsets, coeffs = _second_derivative_stencil(2, "near_left")
            for off, c in zip(offsets, coeffs):
                matrix[0, 0 + off] = -c / dx2

            offsets, coeffs = _second_derivative_stencil(2, "near_right")
            for off, c in zip(offsets, coeffs):
                matrix[n-1, n-1 + off] = -c / dx2

        elif self._fd_order == 4:
            matrix = np.zeros((n, n))

            offsets, coeffs = _second_derivative_stencil(4, "center")
            for i in range(2, n-2):
                for off, c in zip(offsets, coeffs):
                    matrix[i, i + off] = -c / dx2

            offsets_left, coeffs_left = _second_derivative_stencil(
                2, "near_left"
            )
            offsets_right, coeffs_right = _second_derivative_stencil(
                2, "near_right"
            )

            for off, c in zip(offsets_left, coeffs_left):
                matrix[1, 1 + off] = -c / dx2

            for off, c in zip(offsets_right, coeffs_right):
                matrix[n-2, n-2 + off] = -c / dx2

        else:
            raise ValueError(
                f"Finite difference order {self._fd_order} not implemented"
            )

        return matrix

    def _apply(self, f: Function) -> Function:
        """Apply the negative Laplacian operator to function f."""
        if self._method == 'spectral':
            return self._apply_spectral(f)
        elif self._method == 'fd':
            return self._apply_finite_difference(f)
        else:
            raise ValueError(f"Unknown method '{self._method}'")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply Laplacian using spectral method."""
        if self._can_use_fast_transforms:
            return self._apply_spectral_fast(f)
        else:
            return self._apply_spectral_slow(f)

    def _apply_spectral_fast(self, f: Function) -> Function:
        """Apply Laplacian using fast transforms."""
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        f_samples = create_uniform_samples(
            f, domain_tuple, self._n_samples,
            self._boundary_conditions.type
        )

        coefficients = fast_spectral_coefficients(
            f_samples, self._boundary_conditions.type, domain_length,
            self._dofs
        )

        terms = compute_spectral_coefficients_fast(
            self, f, coefficients,
            scale_func=lambda i, ev: ev
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._domain
        )

    def _apply_spectral_slow(self, f: Function) -> Function:
        """Apply Laplacian using numerical integration."""
        terms = compute_spectral_coefficients_slow(
            self, f, self._dofs,
            self.integration.method,
            self.integration.n_points,
            scale_func=lambda i, ev: ev
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._domain
        )

    def _apply_finite_difference(self, f: Function) -> Function:
        """Apply Laplacian using finite difference method."""
        f_values = f(self._x_grid)
        laplacian_values = self._fd_matrix @ f_values

        def laplacian_func(x):
            return np.interp(x, self._x_grid, laplacian_values)

        return Function(self.codomain, evaluate_callable=laplacian_func)

    def restrict(self, restricted_space, new_bcs=None):
        """Restrict Laplacian operator to a subspace."""
        if not isinstance(restricted_space, (Lebesgue, Sobolev)):
            raise TypeError("restricted_space must be Lebesgue or Sobolev")

        orig_domain = self.domain.function_domain
        rest_domain = restricted_space.function_domain

        if not (orig_domain.a <= rest_domain.a and
                rest_domain.b <= orig_domain.b):
            raise ValueError(
                f"Restricted domain {rest_domain} is not contained "
                f"in original domain {orig_domain}"
            )

        bcs_to_use = new_bcs if new_bcs is not None else self._boundary_conditions

        return Laplacian(
            restricted_space,
            bcs_to_use,
            self._alpha,
            method=self._method,
            dofs=self._dofs,
            fd_order=self._fd_order,
            n_samples=self._n_samples,
            integration_config=self.integration
        )


class InverseLaplacian(SpectralOperator):
    """
    Inverse Laplacian operator that acts as a covariance operator.

    This operator solves -Δu = f with homogeneous boundary conditions,
    providing a self-adjoint operator suitable for Gaussian measures.

    Uses native Python FEM implementation (no external dependencies).
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['fem', 'spectral'],
        dofs: int = 100,
        fem_type: str = "hat",
        n_samples: int = 512,
        integration_config: IntegrationConfig = None,
    ):
        """
        Initialize the Laplacian inverse operator.

        Args:
            domain: The function space (Lebesgue or Sobolev)
            boundary_conditions: Boundary conditions for the operator
            alpha: Scaling factor (default: 1.0)
            method: 'fem' or 'spectral'
            dofs: Number of degrees of freedom (default: 100)
            fem_type: FEM implementation ('hat' or 'general')
            n_samples: Number of samples for fast spectral transforms
            integration_config: Integration configuration
        """
        if not isinstance(domain, (Lebesgue, Sobolev)):
            raise TypeError(
                f"domain must be Lebesgue or Sobolev, got {type(domain)}"
            )

        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._method = method
        self._dofs = dofs if dofs is not None else domain.dim
        self._fem_type = fem_type
        self._n_samples = max(n_samples, self._dofs)

        if integration_config is None:
            integration_config = IntegrationConfig(
                method='simpson', n_points=1000
            )
        self.integration = integration_config

        if fem_type not in ["hat", "general"]:
            raise ValueError(
                f"fem_type must be 'hat' or 'general', got '{fem_type}'"
            )

        self._codomain = domain

        super().__init__(domain, domain, self._apply)

        self._spectrum_provider = LaplacianSpectrumProvider(
            domain,
            boundary_conditions,
            alpha,
            inverse=True
        )

        self._can_use_fast_transforms = boundary_conditions.type in [
            'dirichlet', 'neumann', 'periodic',
            'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
        ]

        if method == 'fem':
            self._initialize_fem_solver()

        self._log = logging.getLogger(__name__)
        self._log.info(
            "InverseLaplacian initialized: dofs=%s, fem_type=%s, alpha=%s",
            self._dofs,
            self._fem_type,
            self._alpha,
        )

    def _initialize_fem_solver(self):
        self._fem_solver = GeneralFEMSolver(
            function_domain=self._domain._function_domain,
            dofs=self._dofs,
            operator_domain=self._domain,
            boundary_conditions=self._boundary_conditions
        )

    def _apply(self, f: Function) -> Function:
        """Apply the inverse Laplacian operator."""
        if self._method == 'fem':
            return self._apply_fem(f)
        elif self._method == 'spectral':
            return self._apply_spectral(f)

    def _apply_spectral(self, f: Function) -> Function:
        """Apply inverse Laplacian using spectral method."""
        if self._can_use_fast_transforms:
            return self._apply_spectral_fast(f)
        else:
            return self._apply_spectral_slow(f)

    def _apply_spectral_fast(self, f: Function) -> Function:
        """Apply inverse Laplacian using fast transforms."""
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        if (self._boundary_conditions.type == 'neumann' or
                self._boundary_conditions.type == 'periodic'):
            f_samples = create_uniform_samples(
                f, domain_tuple, self._n_samples + 1,
                self._boundary_conditions.type
            )
            coefficients = fast_spectral_coefficients(
                f_samples, self._boundary_conditions.type,
                domain_length, self._dofs + 1
            )
        else:
            f_samples = create_uniform_samples(
                f, domain_tuple, self._n_samples,
                self._boundary_conditions.type
            )
            coefficients = fast_spectral_coefficients(
                f_samples, self._boundary_conditions.type,
                domain_length, self._dofs
            )

        if (self._boundary_conditions.type == 'neumann' or
                self._boundary_conditions.type == 'periodic'):
            coefficients = coefficients[1:]

        terms = compute_spectral_coefficients_fast(
            self, f, coefficients,
            scale_func=lambda i, ev: ev
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def _apply_spectral_slow(self, f: Function) -> Function:
        """Apply inverse Laplacian using numerical integration."""
        terms = compute_spectral_coefficients_slow(
            self, f, self._dofs,
            self.integration.method,
            self.integration.n_points,
            scale_func=lambda i, ev: ev,
            skip_zero_eigenvalues=True
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def _apply_fem(self, f: Function) -> Function:
        f = (1/self._alpha) * f
        solution_coeffs = self._fem_solver.solve_poisson(f)
        return self._fem_solver.solution_to_function(solution_coeffs)

    @property
    def boundary_conditions(self) -> BoundaryConditions:
        """Get the boundary conditions object."""
        return self._boundary_conditions

    @property
    def fem_type(self) -> str:
        """Get the FEM implementation type."""
        return self._fem_type

    @property
    def dofs(self) -> int:
        """Get the mesh resolution."""
        return self._dofs

    @property
    def spectrum_provider(self):
        """Get the spectrum provider for eigenvalue computations."""
        return self._spectrum_provider

    def restrict(self, restricted_space, new_bcs=None):
        """Restrict InverseLaplacian operator to a subspace."""
        restricted_domain = restricted_space.function_domain
        if not self._domain.function_domain.contains(restricted_domain.a):
            raise ValueError(
                f"Restricted domain {restricted_domain} is not a subdomain "
                f"of original domain {self._domain.function_domain}"
            )
        if not self._domain.function_domain.contains(restricted_domain.b):
            raise ValueError(
                f"Restricted domain {restricted_domain} is not a subdomain "
                f"of original domain {self._domain.function_domain}"
            )

        bcs_to_use = (new_bcs if new_bcs is not None
                      else self._boundary_conditions)

        return InverseLaplacian(
            restricted_space,
            bcs_to_use,
            self._alpha,
            method=self._method,
            dofs=self._dofs,
            fem_type=self._fem_type,
            n_samples=self._n_samples,
            integration_config=self.integration
        )

    def get_eigenvalue(self, index: int) -> float:
        return self._spectrum_provider.get_eigenvalue(index)

    def get_eigenvalues(self, indices: List[int]) -> List[float]:
        return [
            self._spectrum_provider.get_eigenvalue(i)
            for i in indices
        ]

    def get_eigenfunction(self, index: int) -> Function:
        if isinstance(self._domain, Sobolev):
            func = self._spectrum_provider.get_eigenfunction(index)
            return self._domain.inverse_mass_operator_factor(func)
        else:
            return self._spectrum_provider.get_eigenfunction(index)
