"""
Trigonometric function providers.

This module contains providers for orthogonal basis functions based on
trigonometric functions:

- SineFunctionProvider: Dirichlet eigenfunctions
- CosineFunctionProvider: Neumann eigenfunctions
- FourierFunctionProvider: Periodic eigenfunctions
- MixedDNFunctionProvider: Dirichlet-Neumann eigenfunctions
- MixedNDFunctionProvider: Neumann-Dirichlet eigenfunctions
- RobinFunctionProvider: General Robin eigenfunctions
"""

import math
import numpy as np
from typing import TYPE_CHECKING

from intervalinf.providers.base import IndexedFunctionProvider
from intervalinf.utils.robin_utils import RobinRootFinder

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class SineFunctionProvider(IndexedFunctionProvider):
    """
    Provider for sine functions: sin(kπ(x-a)/L).

    These are the eigenfunctions for Dirichlet boundary conditions
    on the negative Laplacian operator.

    Normalized: ||φₖ||₂ = 1 with φₖ(x) = √(2/L) sin(kπ(x-a)/L)
    """

    def __init__(self, space_or_domain):
        """Initialize the sine function provider."""
        super().__init__(space_or_domain)
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get sine function with index k = index + 1."""
        if index not in self._cache:
            from intervalinf.core.functions import Function

            a, b = self.domain.a, self.domain.b
            length = b - a
            k = index + 1  # Sine functions start from k=1
            normalization = np.sqrt(2 / length)

            def sine_func(x):
                if isinstance(x, np.ndarray):
                    return normalization * np.sin(
                        k * np.pi * (x - a) / length
                    )
                else:
                    return normalization * math.sin(
                        k * np.pi * (x - a) / length
                    )

            func = Function(
                self.function_context,
                evaluate_callable=sine_func,
                name=f"sin({k}π(x-{a})/{length})"
            )
            self._cache[index] = func

        return self._cache[index]


class CosineFunctionProvider(IndexedFunctionProvider):
    """
    Provider for cosine functions: cos(kπ(x-a)/L).

    These are the eigenfunctions for Neumann boundary conditions
    on the negative Laplacian operator (with constant mode for k=0).

    Normalized: ||φₖ||₂ = 1
        φ₀(x) = 1/√L (constant)
        φₖ(x) = √(2/L) cos(kπ(x-a)/L) for k≥1
    """

    def __init__(self, space_or_domain, non_constant_only: bool = False):
        """
        Initialize the cosine function provider.

        Args:
            space_or_domain: Space or IntervalDomain
            non_constant_only: If True, skip constant mode (start at k=1)
        """
        super().__init__(space_or_domain)
        self._cache = {}
        self.non_constant_only = non_constant_only

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get cosine function or constant for index 0."""
        from intervalinf.core.functions import Function

        if self.non_constant_only:
            index += 1

        if index not in self._cache:
            a, b = self.domain.a, self.domain.b
            length = b - a

            if index == 0:
                # Constant mode for Neumann BC
                def constant_func(x):
                    return ((np.ones_like(x) if isinstance(x, np.ndarray)
                             else 1.0) / np.sqrt(length))

                func = Function(
                    self.function_context,
                    evaluate_callable=constant_func,
                    name="1 (constant)"
                )
            else:
                k = index
                normalization = np.sqrt(2 / length)

                def cosine_func(x):
                    if isinstance(x, np.ndarray):
                        return normalization * np.cos(
                            k * np.pi * (x - a) / length
                        )
                    else:
                        return normalization * math.cos(
                            k * np.pi * (x - a) / length
                        )

                func = Function(
                    self.function_context,
                    evaluate_callable=cosine_func,
                    name=f"cos({k}π(x-{a})/{length})"
                )

            self._cache[index] = func

        return self._cache[index]


class FourierFunctionProvider(IndexedFunctionProvider):
    """
    Provider for Fourier basis functions (periodic BCs).

    Ordering:
        index 0: 1/√L (constant)
        index 1: √(2/L) cos(2πx/L)
        index 2: √(2/L) sin(2πx/L)
        index 3: √(2/L) cos(4πx/L)
        index 4: √(2/L) sin(4πx/L)
        etc.
    """

    def __init__(self, space_or_domain, non_constant_only: bool = False):
        """
        Initialize Fourier provider.

        Args:
            space_or_domain: Space or IntervalDomain
            non_constant_only: If True, skip constant function
        """
        super().__init__(space_or_domain)
        self.non_constant_only = non_constant_only
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get Fourier basis function by index."""
        from intervalinf.core.functions import Function

        if self.non_constant_only:
            index += 1

        if index in self._cache:
            return self._cache[index]

        a, b = self.domain.a, self.domain.b
        L = b - a

        if index == 0:
            def const_func(x):
                return np.ones_like(x) / np.sqrt(L)

            func = Function(
                self.function_context,
                evaluate_callable=const_func,
                name='fourier_const'
            )
        else:
            k = (index + 1) // 2

            if index % 2 == 1:  # Odd index: cosine
                def cosine_func(x):
                    return (np.sqrt(2/L)
                            * np.cos(2 * k * np.pi * (x - a) / L))

                func = Function(
                    self.function_context,
                    evaluate_callable=cosine_func,
                )
            else:  # Even index: sine
                def sine_func(x):
                    return (np.sqrt(2/L)
                            * np.sin(2 * k * np.pi * (x - a) / L))

                func = Function(
                    self.function_context,
                    evaluate_callable=sine_func,
                    name=f'fourier_sin_{k}'
                )

        self._cache[index] = func
        return func


class MixedDNFunctionProvider(IndexedFunctionProvider):
    """
    Mixed Dirichlet-Neumann eigenfunctions for -d²/dx² on (a,b).

    Boundary conditions: u(a)=0, u'(b)=0
    Eigenfunctions: φₖ(x) = √(2/L) · sin((k+1/2)π(x-a)/L), k≥0
    """

    def __init__(self, space_or_domain):
        super().__init__(space_or_domain)
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        if index < 0:
            raise ValueError("index must be ≥ 0")

        if index not in self._cache:
            from intervalinf.core.functions import Function

            a, b = self.domain.a, self.domain.b
            L = b - a
            mu = (index + 0.5) * math.pi / L
            c = math.sqrt(2.0 / L)

            def phi(x):
                y = np.asarray(x) - a
                return c * np.sin(mu * y)

            func = Function(
                self.function_context,
                evaluate_callable=phi,
                name=f"mixed_DN_k{index}"
            )
            self._cache[index] = func

        return self._cache[index]


class MixedNDFunctionProvider(IndexedFunctionProvider):
    """
    Mixed Neumann-Dirichlet eigenfunctions for -d²/dx² on (a,b).

    Boundary conditions: u'(a)=0, u(b)=0
    Eigenfunctions: φₖ(x) = √(2/L) · cos((k+1/2)π(x-a)/L), k≥0
    """

    def __init__(self, space_or_domain):
        super().__init__(space_or_domain)
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        if index < 0:
            raise ValueError("index must be ≥ 0")

        if index not in self._cache:
            from intervalinf.core.functions import Function

            a, b = self.domain.a, self.domain.b
            L = b - a
            mu = (index + 0.5) * math.pi / L
            c = math.sqrt(2.0 / L)

            def phi(x):
                y = np.asarray(x) - a
                return c * np.cos(mu * y)

            func = Function(
                self.function_context,
                evaluate_callable=phi,
                name=f"mixed_ND_k{index}"
            )
            self._cache[index] = func

        return self._cache[index]


class RobinFunctionProvider(IndexedFunctionProvider):
    """
    Robin eigenfunctions for -d²/dx² on (a,b) with separated BCs.

    Boundary conditions:
        α₀ u(a) + β₀ u'(a) = 0
        α_L u(b) + β_L u'(b) = 0

    The eigenvalues μₖ are determined by solving the transcendental
    equation arising from the boundary conditions.
    """

    def __init__(
        self,
        space_or_domain,
        bcs,
        integration_method: str = 'simpson',
        n_points: int = 2000,
        root_tol: float = 1e-12,
        max_bisect_iter: int = 100
    ):
        """
        Initialize Robin function provider.

        Args:
            space_or_domain: Space or IntervalDomain
            bcs: BoundaryConditions object with Robin parameters
            integration_method: Method for L² normalization
            n_points: Number of points for integration
            root_tol: Tolerance for root finding
            max_bisect_iter: Maximum bisection iterations
        """
        super().__init__(space_or_domain)
        self.alpha0 = float(bcs.get_parameter('left_alpha'))
        self.beta0 = float(bcs.get_parameter('left_beta'))
        self.alphaL = float(bcs.get_parameter('right_alpha'))
        self.betaL = float(bcs.get_parameter('right_beta'))
        self.value0 = float(bcs.get_parameter('left_value'))
        self.valueL = float(bcs.get_parameter('right_value'))

        self.integration_method = integration_method
        self.n_points = n_points
        self.root_tol = root_tol
        self.max_bisect_iter = max_bisect_iter

        self._mu_cache: list = []
        self._func_cache: dict = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        if index < 0:
            raise ValueError("index must be ≥ 0")

        if index in self._func_cache:
            return self._func_cache[index]

        from intervalinf.core.functions import Function

        a, b = self.domain.a, self.domain.b
        L = b - a

        # Get μₖ
        mu = self._mu_at(index)

        # Special pure Neumann case -> constant mode
        if mu == 0.0:
            def const(x):
                return ((np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
                        / math.sqrt(L))

            f0 = Function(
                self.function_context,
                evaluate_callable=const,
                name="robin_constant"
            )
            self._func_cache[index] = f0
            return f0

        # Build (A,B) from left BC
        A, B = RobinRootFinder.compute_coefficients_from_left_bc(
            mu, self.alpha0, self.beta0, self.alphaL, self.betaL, L
        )

        # Raw eigenfunction (unnormalized)
        def raw(x):
            y = np.asarray(x) - a
            return A * np.cos(mu * y) + B * np.sin(mu * y)

        # Normalize in L²(a,b)
        raw_func = Function(self.function_context, evaluate_callable=raw)
        norm2 = (raw_func * raw_func).integrate(
            method=self.integration_method,
            n_points=self.n_points
        )
        c = 1.0 / math.sqrt(max(norm2, 1e-300))

        def phi(x):
            return c * raw(x)

        f = Function(
            self.function_context,
            evaluate_callable=phi,
            name=f"robin_mu={mu:.8g}"
        )
        self._func_cache[index] = f
        return f

    def _mu_at(self, k: int) -> float:
        while len(self._mu_cache) <= k:
            self._append_next_mu()
        return self._mu_cache[k]

    def _append_next_mu(self):
        index = len(self._mu_cache)
        mu = RobinRootFinder.compute_robin_eigenvalue(
            index,
            self.alpha0,
            self.beta0,
            self.alphaL,
            self.betaL,
            self.domain.length,
            tol=self.root_tol,
            maxit=self.max_bisect_iter
        )
        self._mu_cache.append(mu)
