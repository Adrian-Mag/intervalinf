"""
Concrete function providers for interval domains.

This module contains function providers for common orthogonal basis functions:
- SineFunctionProvider: Dirichlet eigenfunctions
- CosineFunctionProvider: Neumann eigenfunctions
- FourierFunctionProvider: Periodic eigenfunctions
- MixedDNFunctionProvider: Dirichlet-Neumann eigenfunctions
- MixedNDFunctionProvider: Neumann-Dirichlet eigenfunctions
- RobinFunctionProvider: General Robin eigenfunctions
- HatFunctionProvider: Piecewise linear basis for FEM
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
    """

    def __init__(self, space):
        """Initialize the sine function provider."""
        super().__init__(space)
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get sine function with index k = index + 1."""
        if index not in self._cache:
            from intervalinf.core.functions import Function

            a, b = self.space.function_domain.a, self.space.function_domain.b
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
                self.space,
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
    """

    def __init__(self, space, non_constant_only: bool = False):
        """
        Initialize the cosine function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            non_constant_only: If True, skip constant mode (start at k=1)
        """
        super().__init__(space)
        self._cache = {}
        self.non_constant_only = non_constant_only

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get cosine function or constant for index 0."""
        from intervalinf.core.functions import Function

        if self.non_constant_only:
            index += 1

        if index not in self._cache:
            a, b = self.space.function_domain.a, self.space.function_domain.b
            length = b - a

            if index == 0:
                # Constant mode for Neumann BC
                def constant_func(x):
                    return ((np.ones_like(x) if isinstance(x, np.ndarray)
                             else 1.0) / np.sqrt(length))

                func = Function(
                    self.space,
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
                    self.space,
                    evaluate_callable=cosine_func,
                    name=f"cos({k}π(x-{a})/{length})"
                )

            self._cache[index] = func

        return self._cache[index]


class FourierFunctionProvider(IndexedFunctionProvider):
    """Provider for Fourier basis functions (periodic BCs)."""

    def __init__(self, space, non_constant_only: bool = False):
        """
        Initialize Fourier provider.

        Args:
            space: Lebesgue instance
            non_constant_only: If True, skip constant function
        """
        super().__init__(space)
        self.non_constant_only = non_constant_only
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get Fourier basis function by index.

        Index 0: constant function
        Odd index (2k-1):  √(2/L) · cos(2πk(x−a)/L)
        Even index (2k):   √(2/L) · sin(2πk(x−a)/L)
        """
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
                self.space,
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
                    self.space,
                    evaluate_callable=cosine_func,
                    name=f'fourier_cos_{k}'
                )
            else:  # Even index: sine
                def sine_func(x):
                    return (np.sqrt(2/L)
                            * np.sin(2 * k * np.pi * (x - a) / L))

                func = Function(
                    self.space,
                    evaluate_callable=sine_func,
                    name=f'fourier_sin_{k}'
                )

        self._cache[index] = func
        return func


class MixedDNFunctionProvider(IndexedFunctionProvider):
    """
    Mixed DN eigenfunctions for -d2/dx2 on (a,b):
      u(a)=0, u'(b)=0  ⇒  φ_k(x) = √(2/L) · sin((k+1/2)π(x-a)/L), k≥0
    """

    def __init__(self, space):
        super().__init__(space)
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
                self.space,
                evaluate_callable=phi,
                name=f"mixed_DN_k{index}"
            )
            self._cache[index] = func

        return self._cache[index]


class MixedNDFunctionProvider(IndexedFunctionProvider):
    """
    Mixed ND eigenfunctions for -d2/dx2 on (a,b):
      u'(a)=0, u(b)=0  ⇒  φ_k(x) = √(2/L) · cos((k+1/2)π(x-a)/L), k≥0
    """

    def __init__(self, space):
        super().__init__(space)
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
                self.space,
                evaluate_callable=phi,
                name=f"mixed_ND_k{index}"
            )
            self._cache[index] = func

        return self._cache[index]


class RobinFunctionProvider(IndexedFunctionProvider):
    """
    Robin eigenfunctions for -d2/dx2 on (a,b) with separated BCs:
      alpha_0 u(a) + beta_0 u'(a) = 0,
      alpha_L u(b) + beta_L u'(b) = 0.
    """

    def __init__(
        self,
        space,
        bcs,
        integration_method: str = 'simpson',
        n_points: int = 2000,
        root_tol: float = 1e-12,
        max_bisect_iter: int = 100
    ):
        super().__init__(space)
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

        # get μ_k
        mu = self._mu_at(index)

        # Special pure Neumann case -> constant mode
        if mu == 0.0:
            def const(x):
                return ((np.ones_like(x) if isinstance(x, np.ndarray) else 1.0)
                        / math.sqrt(L))

            f0 = Function(
                self.space,
                evaluate_callable=const,
                name="robin_constant"
            )
            self._func_cache[index] = f0
            return f0

        # Build (A,B) from left BC
        A, B = RobinRootFinder.compute_coefficients_from_left_bc(
            mu, self.alpha0, self.beta0, self.alphaL, self.betaL, L
        )

        # raw eigenfunction (unnormalized)
        def raw(x):
            y = np.asarray(x) - a
            return A * np.cos(mu * y) + B * np.sin(mu * y)

        # normalize in L²(a,b)
        raw_func = Function(self.space, evaluate_callable=raw)
        norm2 = (raw_func * raw_func).integrate(
            method=self.integration_method,
            n_points=self.n_points
        )
        c = 1.0 / math.sqrt(max(norm2, 1e-300))

        def phi(x):
            return c * raw(x)

        f = Function(
            self.space,
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


class HatFunctionProvider(IndexedFunctionProvider):
    """
    Provider for hat functions (piecewise linear basis functions).

    Hat functions are continuous, piecewise linear functions that form
    a basis for finite element methods.
    """

    def __init__(self, space, homogeneous=False, n_nodes=None):
        """
        Initialize the hat function provider.

        Args:
            space: Lebesgue instance
            homogeneous: If True, omit boundary nodes (homogeneous Dirichlet)
            n_nodes: Number of nodes (default: space.dim + boundary adjustment)
        """
        super().__init__(space)
        self._cache = {}
        self.homogeneous = homogeneous

        if n_nodes is None:
            if homogeneous:
                self.n_nodes = self.space.dim + 2
            else:
                self.n_nodes = self.space.dim
        else:
            self.n_nodes = n_nodes

        a, b = self.space.function_domain.a, self.space.function_domain.b
        self.nodes = np.linspace(a, b, self.n_nodes)
        self.h = (b - a) / (self.n_nodes - 1)

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get hat function for given index."""
        if index not in self._cache:
            from intervalinf.core.functions import Function

            if self.homogeneous:
                effective_index = index + 1
            else:
                effective_index = index

            node_position = self.nodes[effective_index]

            def hat_func(x):
                x = np.asarray(x)
                result = np.zeros_like(x, dtype=float)

                left_node = effective_index - 1
                right_node = effective_index + 1

                if left_node >= 0:
                    left_x = self.nodes[left_node]
                    mask_left = (x >= left_x) & (x <= node_position)
                    if np.any(mask_left):
                        result[mask_left] = (x[mask_left] - left_x) / self.h

                if right_node < self.n_nodes:
                    right_x = self.nodes[right_node]
                    mask_right = (x >= node_position) & (x <= right_x)
                    if np.any(mask_right):
                        result[mask_right] = (
                            (right_x - x[mask_right]) / self.h
                        )

                mask_exact = np.isclose(
                    x, node_position, rtol=1e-14, atol=1e-14
                )
                result[mask_exact] = 1.0

                return result

            if self.homogeneous:
                name = f"hat_hom_{index}(x={node_position:.3f})"
            else:
                name = f"hat_{index}(x={node_position:.3f})"

            func = Function(
                self.space,
                evaluate_callable=hat_func,
                name=name
            )
            self._cache[index] = func

        return self._cache[index]

    def get_nodes(self):
        """Get the node coordinates."""
        return self.nodes.copy()

    def get_active_nodes(self):
        """Get coordinates of nodes corresponding to basis functions."""
        if self.homogeneous:
            return self.nodes[1:-1].copy()
        else:
            return self.nodes.copy()
