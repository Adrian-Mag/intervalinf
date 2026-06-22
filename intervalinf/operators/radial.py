"""
Radial Laplacian operators for 3D spherical coordinates.

These operators are self-adjoint with respect to the weighted inner product
⟨f,g⟩ = ∫ f(r)g(r) r² dr, which arises naturally in spherical coordinates.
"""

import logging
from typing import Union, Optional, Literal

import numpy as np
from scipy.sparse import diags

from .base import SpectralOperator
from ..spaces.lebesgue import Lebesgue
from ..spaces.sobolev import Sobolev
from ..core.boundary import BoundaryConditions
from ..core.functions import Function
from ..core.config import IntegrationConfig
from ..providers.base import EigenvalueProvider, SpectrumProvider
from ..providers.radial import (
    GeneralRadialLaplacianModeSolver,
    GeneralRadialLaplacianProvider,
    RadialLaplacianDirichletProvider,
    RadialLaplacianNeumannProvider,
    RadialLaplacianDDProvider,
    RadialLaplacianDNProvider,
    RadialLaplacianNDProvider,
    RadialLaplacianNNProvider,
)
from ..utils.robin_utils import RobinRootFinder


class RadialLaplacianEigenvalueProvider(EigenvalueProvider):
    """
    Eigenvalue provider for the radial Laplacian operator.

    The radial Laplacian in 3D spherical coordinates is:
        L = -d²/dr² - (2/r)d/dr = -(1/r²)d/dr(r² d/dr)

    For the weighted inner product ⟨f,g⟩ = ∫ f(r)g(r) r² dr,
    this operator is self-adjoint.

    Eigenvalues and eigenfunctions depend on boundary conditions:
    - Dirichlet at r=0 and r=R: Related to spherical Bessel functions
    - Neumann at r=R (with regularity at r=0): Different eigenstructure
    """

    def __init__(
        self,
        function_domain,
        boundary_conditions: BoundaryConditions,
        inverse: bool = False,
        alpha: float = 1.0,
        ell: int = 0,
        mode_solver: Optional[GeneralRadialLaplacianModeSolver] = None,
    ):
        """
        Initialize the radial Laplacian eigenvalue provider.

        Parameters
        ----------
        function_domain : IntervalDomain
            Interval domain [0, R] or [a, b]
        boundary_conditions : BoundaryConditions
            Boundary conditions at r=R (or both endpoints)
        inverse : bool, default=False
            If True, compute eigenvalues of L⁻¹, else of L
        alpha : float, default=1.0
            Scaling factor for the eigenvalues
        ell : int, default=0
            Angular momentum quantum number.
        """
        self._function_domain = function_domain
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        self._alpha = alpha
        self._ell = ell
        self._mode_solver = mode_solver
        self._eigenvalue_cache = {}

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue for given index."""
        if index not in self._eigenvalue_cache:
            self._eigenvalue_cache[index] = self._compute_eigenvalue(index)
        return self._eigenvalue_cache[index]

    def _compute_eigenvalue(self, index: int) -> float:
        """
        Compute eigenvalue based on boundary conditions.

        For ℓ=0 (s-wave):

        Case A: Domain (0, R) with regularity at r=0:
            1. regularity-Dirichlet: λₖ = (kπ/R)² for k=1,2,3,...
            2. regularity-Neumann: solve tan(kR) = kR numerically

        Case B: Domain (a, b) with 0 < a < b:
            The wavenumbers k=√λ satisfy various equations depending on BC.
        """
        a = self._function_domain.a
        b = self._function_domain.b
        L = b - a

        if self._ell == 0:
            # s-wave (ℓ=0): analytical or semi-analytical eigenvalues

            # Case A: Domain includes r=0 (regularity condition at left)
            if np.isclose(a, 0.0, atol=1e-10):
                if self._boundary_conditions.type == 'dirichlet':
                    # regularity-Dirichlet: φ(r) ∝ sin(kr)/r, φ(R)=0
                    k = index + 1
                    eigenval = (k * np.pi / L) ** 2

                elif self._boundary_conditions.type == 'neumann':
                    # regularity-Neumann: tan(kR) = kR
                    eigenval = self._compute_regularity_neumann_eigenvalue(
                        index, L
                    )

                else:
                    raise ValueError(
                        f"For domain containing r=0, only 'dirichlet' "
                        f"(regularity-Dirichlet) or 'neumann' "
                        f"(regularity-Neumann) are supported. "
                        f"Got: '{self._boundary_conditions.type}'"
                    )

            # Case B: Domain does not include r=0 (two-endpoint BCs)
            else:
                if self._boundary_conditions.type == 'dirichlet':
                    # DD: k_n = nπ/L for n=1,2,3,...
                    k = (index + 1) * np.pi / L
                    eigenval = k ** 2

                elif self._boundary_conditions.type == \
                        'mixed_dirichlet_neumann':
                    # DN: tan(kL) = kb
                    eigenval = self._compute_dn_eigenvalue(index, a, b, L)

                elif self._boundary_conditions.type == \
                        'mixed_neumann_dirichlet':
                    # ND: tan(kL) = -ak
                    eigenval = self._compute_nd_eigenvalue(index, a, b, L)

                elif self._boundary_conditions.type == 'neumann':
                    # NN: λ_0=0, then numerical roots
                    if index == 0:
                        eigenval = 0.0
                    else:
                        eigenval = self._compute_nn_eigenvalue(index, a, b, L)

                elif self._boundary_conditions.type == 'robin':
                    raise NotImplementedError(
                        "Robin boundary conditions not yet implemented "
                        "for radial Laplacian"
                    )

                else:
                    raise ValueError(
                        f"Unsupported boundary condition type "
                        f"'{self._boundary_conditions.type}'"
                    )
        else:
            if self._mode_solver is None:
                self._mode_solver = GeneralRadialLaplacianModeSolver(
                    self._function_domain, self._boundary_conditions, self._ell
                )
            eigenval = self._mode_solver.eigenvalue(index)

        # Apply alpha scaling and inverse if needed
        if self._inverse:
            if eigenval == 0:
                raise ValueError("Cannot invert zero eigenvalue")
            return 1.0 / (eigenval * self._alpha)
        else:
            return eigenval * self._alpha

    def _compute_regularity_neumann_eigenvalue(
        self, index: int, R: float
    ) -> float:
        """Compute regularity-Neumann eigenvalue for ℓ=0."""
        if index == 0:
            return 0.0  # First root is k=0 → λ=0
        else:
            def F(k):
                return k * R
            k_root = RobinRootFinder.solve_tan_equation(F, R, index - 1)
            return k_root ** 2

    def _compute_dn_eigenvalue(
        self, index: int, a: float, b: float, L: float
    ) -> float:
        """Compute Dirichlet-Neumann eigenvalue for ℓ=0 on (a,b)."""
        def F(k):
            return k * b
        k_root = RobinRootFinder.solve_tan_equation(F, L, index)
        return k_root ** 2

    def _compute_nd_eigenvalue(
        self, index: int, a: float, b: float, L: float
    ) -> float:
        """Compute Neumann-Dirichlet eigenvalue for ℓ=0 on (a,b)."""
        def F(k):
            return -a * k
        k_root = RobinRootFinder.solve_tan_equation(F, L, index)
        return k_root ** 2

    def _compute_nn_eigenvalue(
        self, index: int, a: float, b: float, L: float
    ) -> float:
        """Compute Neumann-Neumann eigenvalue for ℓ=0 on (a,b)."""
        numerator = 1.0/b - 1.0/a

        def F(k):
            return numerator / (k + 1.0/(a * b * k))

        k_root = RobinRootFinder.solve_tan_equation(F, L, index - 1)
        return k_root ** 2


class RadialLaplacianSpectrumProvider(SpectrumProvider):
    """
    Spectrum provider for radial Laplacian eigenfunctions and eigenvalues.

    This provider delegates to the appropriate function provider based on
    the domain and boundary conditions.
    """

    def __init__(
        self,
        space: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        inverse: bool = False,
        ell: int = 0
    ):
        """
        Initialize radial Laplacian spectrum provider.

        Parameters
        ----------
        space : Lebesgue or Sobolev
            Function space
        boundary_conditions : BoundaryConditions
            Boundary conditions
        alpha : float, default=1.0
            Scaling factor
        inverse : bool, default=False
            If True, spectrum of inverse operator
        ell : int, default=0
            Angular momentum quantum number
        """
        self._boundary_conditions = boundary_conditions
        self._inverse = inverse
        self._ell = ell
        self._general_mode_solver = (
            GeneralRadialLaplacianModeSolver(
                space.function_domain, boundary_conditions, ell
            )
            if ell != 0 else None
        )
        super().__init__(
            space, orthonormal=True, basis_type='radial_laplacian'
        )

        self._eigenvalue_provider = RadialLaplacianEigenvalueProvider(
            space.function_domain,
            boundary_conditions,
            inverse,
            alpha,
            ell,
            self._general_mode_solver,
        )

        # Initialize the appropriate function provider
        self._function_provider = self._create_function_provider()

    def _create_function_provider(self):
        """Create the appropriate function provider based on domain and BC."""
        a = self.space.function_domain.a
        bc_type = self._boundary_conditions.type

        if self._ell != 0:
            return GeneralRadialLaplacianProvider(
                self.space,
                self._boundary_conditions,
                self._ell,
                mode_solver=self._general_mode_solver,
            )

        # Case A: Domain (0, R) with regularity at r=0
        if np.isclose(a, 0.0, atol=1e-10):
            if bc_type == 'dirichlet':
                return RadialLaplacianDirichletProvider(self.space)
            elif bc_type == 'neumann':
                return RadialLaplacianNeumannProvider(self.space)
            else:
                raise NotImplementedError(
                    f"Radial Laplacian on (0,R) with BC type '{bc_type}' "
                    "not implemented"
                )

        # Case B: Domain (a, b) with 0 < a < b
        else:
            if bc_type == 'dirichlet':
                return RadialLaplacianDDProvider(self.space)
            elif bc_type == 'mixed_dirichlet_neumann':
                return RadialLaplacianDNProvider(self.space)
            elif bc_type == 'mixed_neumann_dirichlet':
                return RadialLaplacianNDProvider(self.space)
            elif bc_type == 'neumann':
                return RadialLaplacianNNProvider(self.space)
            else:
                raise NotImplementedError(
                    f"Radial Laplacian on (a,b) with BC type '{bc_type}' "
                    "not implemented"
                )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue at given index."""
        return self._eigenvalue_provider.get_eigenvalue(index)

    def get_eigenfunction(self, index: int) -> Function:
        """
        Get eigenfunction at given index.

        These are orthonormal with respect to ⟨f,g⟩ = ∫ f(r)g(r) r² dr.
        """
        return self._function_provider.get_function_by_index(index)


class RadialLaplacian(SpectralOperator):
    """
    Radial Laplacian operator for 3D spherical coordinates.

    The radial Laplacian is:
        L = -d²/dr² - (2/r)d/dr = -(1/r²)d/dr(r² d/dr)

    This operator is self-adjoint with respect to the weighted inner product:
        ⟨f,g⟩ = ∫ f(r)g(r) r² dr

    The weight r² comes from the Jacobian in spherical coordinates.

    Multiple discretization methods are available:
    - 'spectral': Uses analytical eigendecomposition (ℓ=0 only for now)
    - 'fd': Finite difference on a radial grid
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
        ell: int = 0,
        fd_order: int = 2,
        n_samples: int = 512,
        integration_config: IntegrationConfig = IntegrationConfig(
            method='simpson', n_points=1000
        ),
    ):
        """
        Initialize the radial Laplacian operator.

        Parameters
        ----------
        domain : Lebesgue or Sobolev
            Function space (must have weight=r²)
        boundary_conditions : BoundaryConditions
            Boundary conditions at outer radius
        alpha : float, default=1.0
            Scaling factor
        method : {'spectral', 'fd'}, default='spectral'
            Discretization method
        dofs : int, optional
            Number of degrees of freedom
        ell : int, default=0
            Angular momentum quantum number
        fd_order : int, default=2
            Order of finite difference stencil
        n_samples : int, default=512
            Number of samples for spectral transforms
        integration_config : IntegrationConfig
            Integration configuration
        """
        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._dofs = dofs if dofs is not None else domain.dim
        self._ell = ell
        self._fd_order = fd_order
        self._method = method
        self._n_samples = max(n_samples, self._dofs)

        # Store integration config
        self.integration = integration_config

        super().__init__(domain, domain, self._apply)

        # Initialize spectrum provider for spectral method
        if method == 'spectral':
            self._spectrum_provider = RadialLaplacianSpectrumProvider(
                domain,
                boundary_conditions,
                alpha,
                inverse=False,
                ell=ell
            )
        elif method == 'fd':
            self._setup_finite_difference()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "RadialLaplacian initialized: method=%s, dofs=%s, ℓ=%s, alpha=%s",
            method, self._dofs, ell, alpha
        )

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenvalue(index)
        else:
            raise NotImplementedError(
                "Eigenvalues not available for finite difference method"
            )

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenfunction(index)
        else:
            raise NotImplementedError(
                "Eigenfunctions not available for finite difference method"
            )

    def _setup_finite_difference(self):
        """Setup finite difference discretization for radial Laplacian."""
        a, b = (self._domain.function_domain.a,  # type: ignore
                self._domain.function_domain.b)  # type: ignore

        # Create radial grid - handle r=0 carefully
        if a == 0:
            eps = (b - a) / (10 * self._dofs)
            self._r_grid = np.linspace(eps, b, self._dofs)
        else:
            self._r_grid = np.linspace(a, b, self._dofs)

        self._dr = self._r_grid[1] - self._r_grid[0]

        # Create FD matrix for radial Laplacian
        self._fd_matrix = self._create_radial_fd_matrix()

        self._log.info(
            "RadialLaplacian FD setup: grid from r=%.3e to r=%.3e, dr=%.3e",
            self._r_grid[0], self._r_grid[-1], self._dr
        )

    def _create_radial_fd_matrix(self) -> np.ndarray:
        """
        Create finite difference matrix for the radial Laplacian.

        L = -d²/dr² - (2/r)d/dr + ell(ell+1)/r²

        Using centered differences:
            L_i f ≈ -[(1 + dr/r_i)f_{i+1} - 2f_i + (1 - dr/r_i)f_{i-1}]/dr²
                    + ell(ell+1)f_i/r_i²
        """
        n = self._dofs
        dr = self._dr
        r = self._r_grid

        # Main diagonal
        main_diag = np.full(n, 2.0 / dr**2)
        if self._ell:
            main_diag += self._ell * (self._ell + 1.0) / (r * r)

        # Upper diagonal: -(1 + dr/r_i)/dr²
        upper_diag = -(1.0 + dr / r[:-1]) / dr**2

        # Lower diagonal: -(1 - dr/r_i)/dr²
        lower_diag = -(1.0 - dr / r[1:]) / dr**2

        # Create sparse matrix
        matrix = diags(
            [lower_diag, main_diag, upper_diag],
            offsets=[-1, 0, 1],  # type: ignore
            shape=(n, n),
            format='csr'
        )

        return matrix.toarray()

    def _apply(self, f: Function) -> Function:
        """Apply the radial Laplacian operator to a function."""
        if self._method == 'spectral':
            return self._apply_spectral(f)
        elif self._method == 'fd':
            return self._apply_fd(f)
        else:
            raise ValueError(f"Unknown method: {self._method}")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply radial Laplacian using spectral decomposition."""
        # Expand f in eigenbasis
        coeffs = []
        for k in range(self._dofs):
            phi_k = self.get_eigenfunction(k)
            # Project: c_k = ⟨φ_k, f⟩ with weighted inner product
            c_k = self._domain.inner_product(phi_k, f)
            coeffs.append(c_k)

        # Apply operator: L f = Σ λ_k c_k φ_k
        result = self._domain.zero
        for k, c_k in enumerate(coeffs):
            if abs(c_k) > 1e-14:
                lambda_k = self.get_eigenvalue(k)
                phi_k = self.get_eigenfunction(k)
                result = result + (lambda_k * c_k) * phi_k

        return result

    def _apply_fd(self, f: Function) -> Function:
        """Apply radial Laplacian using finite differences."""
        # Evaluate function on grid
        f_values = f.evaluate(self._r_grid)

        # Apply FD matrix
        laplacian_values = self._fd_matrix @ f_values

        # Create result function by interpolation
        def laplacian_func(r, _r_grid=self._r_grid, _vals=laplacian_values):
            return np.interp(r, _r_grid, _vals)

        return Function(
            self.codomain.function_domain,  # type: ignore
            evaluate_callable=laplacian_func
        )


class InverseRadialLaplacian(SpectralOperator):
    """
    Inverse radial Laplacian operator for use as prior covariance.

    This operator solves:
        L u = f
    where L is the radial Laplacian.

    It provides a self-adjoint, positive-definite operator suitable
    for defining Gaussian measures on spaces with r² weight.
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        boundary_conditions: BoundaryConditions,
        alpha: float = 1.0,
        /,
        *,
        method: Literal['fem', 'spectral'] = 'spectral',
        dofs: int = 100,
        ell: int = 0,
        fem_type: str = "hat",
        n_samples: int = 512,
        integration_config: IntegrationConfig = IntegrationConfig(
            method='simpson', n_points=1000
        ),
    ):
        """
        Initialize the inverse radial Laplacian operator.

        Parameters
        ----------
        domain : Lebesgue or Sobolev
            Function space (must have weight=r²)
        boundary_conditions : BoundaryConditions
            Boundary conditions
        alpha : float, default=1.0
            Scaling factor
        method : {'fem', 'spectral'}, default='spectral'
            Solution method
        dofs : int, default=100
            Number of degrees of freedom
        ell : int, default=0
            Angular momentum quantum number
        fem_type : str, default='hat'
            FEM type ('hat' or 'general')
        n_samples : int, default=512
            Number of samples for spectral transforms
        integration_config : IntegrationConfig
            Integration configuration
        """
        if not isinstance(domain, (Lebesgue, Sobolev)):
            raise TypeError(
                f"domain must be a Lebesgue or Sobolev space, "
                f"got {type(domain)}"
            )

        self._domain = domain
        self._boundary_conditions = boundary_conditions
        self._alpha = alpha
        self._method = method
        self._dofs = dofs if dofs is not None else domain.dim
        self._ell = ell
        self._fem_type = fem_type
        self._n_samples = max(n_samples, self._dofs)

        # Store integration config
        self.integration = integration_config

        super().__init__(domain, domain, self._apply)

        # Initialize spectrum provider for spectral method
        if method == 'spectral':
            self._spectrum_provider = RadialLaplacianSpectrumProvider(
                domain,
                boundary_conditions,
                alpha,
                inverse=True,
                ell=ell
            )
        elif method == 'fem':
            self._initialize_fem_solver()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Logger
        self._log = logging.getLogger(__name__)
        self._log.info(
            "InverseRadialLaplacian initialized: method=%s, dofs=%s, "
            "ℓ=%s, alpha=%s",
            method, dofs, ell, alpha
        )

    def _initialize_fem_solver(self):
        """Initialize FEM solver for the inverse radial Laplacian."""
        raise NotImplementedError(
            "FEM solver for inverse radial Laplacian not yet implemented. "
            "Use method='spectral' instead."
        )

    def get_eigenvalue(self, index: int) -> float:
        """Get eigenvalue of the inverse operator."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenvalue(index)
        else:
            raise NotImplementedError(
                "Eigenvalues not available for FEM method"
            )

    def get_eigenfunction(self, index: int) -> Function:
        """Get eigenfunction (same as forward operator)."""
        if self._method == 'spectral':
            return self._spectrum_provider.get_eigenfunction(index)
        else:
            raise NotImplementedError(
                "Eigenfunctions not available for FEM method"
            )

    def get_eigenvalues(self, indices) -> np.ndarray:
        """Get multiple eigenvalues."""
        return np.array([self.get_eigenvalue(i) for i in indices])

    def _apply(self, f: Function) -> Function:
        """Apply the inverse radial Laplacian to a function."""
        if self._method == 'spectral':
            return self._apply_spectral(f)
        elif self._method == 'fem':
            return self._apply_fem(f)
        else:
            raise ValueError(f"Unknown method: {self._method}")

    def _apply_spectral(self, f: Function) -> Function:
        """Apply inverse using spectral decomposition."""
        # Expand f in eigenbasis
        coeffs = []
        for k in range(self._dofs):
            phi_k = self.get_eigenfunction(k)
            c_k = self._domain.inner_product(phi_k, f)
            coeffs.append(c_k)

        # Apply inverse operator: L⁻¹ f = Σ (1/λ_k) c_k φ_k
        result = self._domain.zero
        for k, c_k in enumerate(coeffs):
            if abs(c_k) > 1e-14:
                inv_lambda_k = self.get_eigenvalue(k)  # Already inverted
                phi_k = self.get_eigenfunction(k)
                result = result + (inv_lambda_k * c_k) * phi_k

        return result

    def _apply_fem(self, f: Function) -> Function:
        """Apply inverse using FEM solver."""
        raise NotImplementedError(
            "FEM solver for inverse radial Laplacian not yet implemented"
        )
