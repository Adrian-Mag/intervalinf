"""
Finite element method (FEM) basis function providers.

This module contains providers for piecewise polynomial basis functions
commonly used in finite element methods:

- HatFunctionProvider: Piecewise linear (P1) basis functions
- SplineFunctionProvider: B-spline basis functions
"""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

from intervalinf.providers.base import (
    IndexedFunctionProvider,
    ParametricFunctionProvider,
)

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class HatFunctionProvider(IndexedFunctionProvider):
    """
    Provider for hat functions (piecewise linear basis functions).

    Hat functions are continuous, piecewise linear functions that form
    a basis for finite element methods. Each hat function is 1 at its
    associated node and 0 at all other nodes.
    """

    def __init__(
        self,
        space_or_domain,
        homogeneous: bool = False,
        n_nodes: Optional[int] = None
    ):
        """
        Initialize the hat function provider.

        Args:
            space_or_domain: Space or IntervalDomain
            homogeneous: If True, omit boundary nodes (homogeneous Dirichlet)
            n_nodes: Number of nodes (default: space.dim + boundary adjustment)
        """
        super().__init__(space_or_domain)
        self._cache = {}
        self.homogeneous = homogeneous

        if n_nodes is None:
            if self.space is not None:
                if homogeneous:
                    self.n_nodes = self.space.dim + 2
                else:
                    self.n_nodes = self.space.dim
            else:
                # Standalone mode: require n_nodes
                raise ValueError(
                    "n_nodes must be specified when using domain-only mode"
                )
        else:
            self.n_nodes = n_nodes

        a, b = self.domain.a, self.domain.b
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
                self.function_context,
                evaluate_callable=hat_func,
                name=name
            )
            self._cache[index] = func

        return self._cache[index]

    def get_nodes(self) -> np.ndarray:
        """Get the node coordinates."""
        return self.nodes.copy()

    def get_active_nodes(self) -> np.ndarray:
        """Get coordinates of nodes corresponding to basis functions."""
        if self.homogeneous:
            return self.nodes[1:-1].copy()
        else:
            return self.nodes.copy()


class SplineFunctionProvider(
    IndexedFunctionProvider, ParametricFunctionProvider
):
    """
    Provider for B-spline basis functions.

    B-splines are piecewise polynomial functions with local support,
    useful for smooth approximations and interpolation.
    """

    def __init__(self, space_or_domain):
        """
        Initialize spline provider.

        Args:
            space_or_domain: Space or IntervalDomain
        """
        super().__init__(space_or_domain)
        self._cache = {}

    def get_function_by_index(
        self,
        index: int,
        degree: int = 3,
        n_knots: int = 10,
        **kwargs
    ) -> 'Function':
        """
        Get B-spline basis function by index.

        Args:
            index: Index of the basis function
            degree: Polynomial degree (default: 3 for cubic)
            n_knots: Number of internal knots

        Returns:
            Function: B-spline basis function
        """
        from intervalinf.core.functions import Function
        from scipy.interpolate import BSpline

        cache_key = (index, degree, n_knots)
        if cache_key in self._cache:
            return self._cache[cache_key]

        a, b = self.domain.a, self.domain.b

        # Create knot vector
        internal_knots = np.linspace(a, b, n_knots + 2)[1:-1]
        knots = np.concatenate([
            [a] * (degree + 1),
            internal_knots,
            [b] * (degree + 1)
        ])

        # Create coefficient vector (1 at index, 0 elsewhere)
        n_coeffs = len(knots) - degree - 1
        coeffs = np.zeros(n_coeffs)
        coeffs[index % n_coeffs] = 1.0

        spline = BSpline(knots, coeffs, degree)

        def spline_func(x):
            return spline(x)

        func = Function(
            self.function_context,
            evaluate_callable=spline_func,
            name=f'spline_{index}_deg{degree}'
        )
        self._cache[cache_key] = func
        return func

    def get_function_by_parameters(
        self,
        parameters: Dict[str, Any],
        **kwargs
    ) -> 'Function':
        """
        Get spline with specific parameters.

        Args:
            parameters: Dictionary containing:
                - 'degree': Polynomial degree
                - 'knots': Knot vector
                - 'coeffs': Coefficient vector
        """
        from intervalinf.core.functions import Function
        from scipy.interpolate import BSpline

        a, b = self.domain.a, self.domain.b
        degree = parameters.get('degree', 3)
        knots = parameters.get('knots', np.linspace(a, b, 10))
        coeffs = parameters.get('coeffs', np.ones(len(knots) - degree - 1))

        spline = BSpline(knots, coeffs, degree)

        def spline_func(x):
            return spline(x)

        return Function(
            self.function_context,
            evaluate_callable=spline_func,
            name=f'spline_deg{degree}'
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default spline parameters."""
        a, b = self.domain.a, self.domain.b
        return {
            'degree': 3,
            'knots': np.linspace(a, b, 10),
            'coeffs': np.ones(6)  # Compatible with degree 3 and 10 knots
        }
