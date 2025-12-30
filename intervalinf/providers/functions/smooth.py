"""
Smooth function providers.

This module contains providers for smooth, compactly supported functions:

- BumpFunctionProvider: C∞ bump functions with compact support
- BumpFunctionGradientProvider: Analytical gradients of bump functions
"""

import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from scipy import integrate

from intervalinf.providers.base import (
    ParametricFunctionProvider,
    IndexedFunctionProvider,
)

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class BumpFunctionProvider(ParametricFunctionProvider, IndexedFunctionProvider):
    """
    Provider for smooth bump functions with compact support.

    Bump functions are infinitely differentiable (C∞) functions that are
    zero outside a finite interval and positive inside. They use the
    generalized mathematical form: exp(k·t²/(t²-1)) where t is a scaled
    coordinate and k is a shape parameter.

    The shape parameter k controls the concentration of the function:
    - k = 1: Standard bump function
    - k > 1: More concentrated near the center
    - k < 1: More spread out (but k > 0 required for convergence)

    All bump functions are automatically normalized so that their integral
    over their compact support equals 1 (unimodular property).
    """

    def __init__(
        self,
        space,
        default_width: float = 0.2,
        centers: Optional[np.ndarray] = None,
        default_k: float = 1.0
    ):
        """
        Initialize bump function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            default_width: Default width for indexed access (as fraction of
                          domain)
            centers: Optional array of centers for indexed access
            default_k: Shape parameter (higher values = more concentrated)
        """
        super().__init__(space)
        self.default_width = default_width
        self.centers = np.asarray(centers) if centers is not None else None
        self.default_k = default_k
        self._cache = {}
        self._normalization_cache = {}

    def _get_normalization_constant(self, k: float, width: float) -> float:
        """Get or compute the normalization constant for given k and width."""
        if k not in self._normalization_cache:
            def unnormalized_bump(t):
                if abs(t) >= 1.0:
                    return 0.0
                return np.exp(k * t**2 / (t**2 - 1))

            integral, _ = integrate.quad(unnormalized_bump, -1, 1)
            self._normalization_cache[k] = integral

        return self._normalization_cache[k] * (width / 2)

    def get_function_by_parameters(
        self,
        parameters: Dict[str, Any],
        **kwargs
    ) -> 'Function':
        """
        Get a normalized bump function with specific parameters.

        Args:
            parameters: Dictionary containing:
                - 'center': Center of the bump function
                - 'width': Width of the compact support
                - 'k': Shape parameter (optional)
        """
        from intervalinf.core.functions import Function

        center = parameters['center']
        width = parameters['width']
        k = parameters.get('k', self.default_k)

        a_support = center - width / 2
        b_support = center + width / 2
        normalization_constant = self._get_normalization_constant(k, width)

        def normalized_bump_func(x):
            x_arr = np.asarray(x)
            t = 2.0 * (x_arr - center) / width
            result = np.zeros_like(x_arr, dtype=float)
            interior_mask = np.abs(t) < 1.0

            if np.any(interior_mask):
                t_interior = t[interior_mask]
                if np.isclose(k, 0.0):
                    result[interior_mask] = 1.0
                else:
                    denominator = t_interior**2 - 1.0
                    result[interior_mask] = np.exp(
                        k * t_interior**2 / denominator
                    )

            return result / normalization_constant

        return Function(
            self.space,
            evaluate_callable=normalized_bump_func,
            name=f'bump_center_{center:.3f}_width_{width:.3f}_k_{k:.3f}',
            support=(a_support, b_support)
        )

    def get_function_by_index(
        self,
        index: int,
        k: Optional[float] = None,
        **kwargs
    ) -> 'Function':
        """
        Get bump function by index with predetermined or distributed centers.

        Args:
            index: Index of the bump function (must be >= 0)
            k: Shape parameter (optional, defaults to default_k)
        """
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        if k is None:
            k = self.default_k

        cache_key = (index, k)

        if cache_key not in self._cache:
            a, b = self.domain.a, self.domain.b
            domain_length = b - a

            if self.centers is not None:
                if index >= len(self.centers):
                    raise IndexError(
                        f"Index {index} out of range for provided centers "
                        f"(length {len(self.centers)})"
                    )
                center = self.centers[index]
                if not (a <= center <= b):
                    raise ValueError(
                        f"Center {center} at index {index} is outside "
                        f"domain [{a}, {b}]"
                    )
            else:
                n_divisions = index + 2
                center_positions = np.linspace(
                    a + 0.1 * domain_length,
                    b - 0.1 * domain_length,
                    n_divisions
                )
                center = center_positions[index % len(center_positions)]

            width = min(self.default_width,
                        2 * min(center - a, b - center))

            parameters = {'center': center, 'width': width, 'k': k}
            func = self.get_function_by_parameters(parameters)
            func.name = f'bump_{index}_center_{center:.3f}_k_{k:.3f}'
            self._cache[cache_key] = func

        return self._cache[cache_key]

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for bump functions."""
        a, b = self.domain.a, self.domain.b
        domain_length = b - a

        return {
            'center': (a + b) / 2,
            'width': self.default_width * domain_length,
            'k': self.default_k
        }

    def get_n_functions(self) -> Optional[int]:
        """Get the number of available functions."""
        return len(self.centers) if self.centers is not None else None

    def get_centers(self) -> Optional[np.ndarray]:
        """Get the array of centers used by this provider."""
        return self.centers.copy() if self.centers is not None else None


class BumpFunctionGradientProvider(ParametricFunctionProvider,
                                   IndexedFunctionProvider):
    """
    Provider for analytical gradients of smooth bump functions.

    This class provides the exact analytical derivatives of bump functions.
    The gradient formula is:

        f'(x) = f(x) · k · (-2t)/(t²-1)² · (2/width)

    where f(x) is the bump function and t = 2(x-center)/width.
    """

    def __init__(
        self,
        space,
        default_width: float = 0.2,
        default_k: float = 1.0,
        centers: Optional[List[float]] = None
    ):
        """
        Initialize the bump function gradient provider.

        Args:
            space: Function space for the gradients
            default_width: Default width for bump functions
            default_k: Default shape parameter k
            centers: Optional list of predetermined centers
        """
        super().__init__(space)
        self.default_width = default_width
        self.default_k = default_k
        self.centers = centers
        self._domain = space.function_domain
        self._cache = {}

        centers_array = None
        if centers is not None:
            centers_array = np.array(centers)

        self._bump_provider = BumpFunctionProvider(
            space, default_width=default_width,
            centers=centers_array, default_k=default_k
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for bump function gradient."""
        a, b = self._domain.a, self._domain.b
        default_center = (a + b) / 2.0

        return {
            'center': default_center,
            'width': self.default_width,
            'k': self.default_k
        }

    def get_function_by_parameters(
        self,
        parameters: Dict[str, Any],
        **kwargs
    ) -> 'Function':
        """Get the analytical gradient of a bump function with parameters."""
        from intervalinf.core.functions import Function

        center = parameters['center']
        width = parameters['width']
        k = parameters.get('k', self.default_k)

        bump_func = self._bump_provider.get_function_by_parameters(parameters)
        a_support = center - width / 2
        b_support = center + width / 2

        def bump_gradient_func(x):
            x_arr = np.asarray(x)
            bump_values = bump_func(x_arr)
            t = 2.0 * (x_arr - center) / width
            result = np.zeros_like(x_arr, dtype=float)
            interior_mask = np.abs(t) < 1.0

            if np.any(interior_mask):
                t_interior = t[interior_mask]

                if np.isclose(k, 0.0):
                    result[interior_mask] = 0.0
                else:
                    denominator = t_interior**2 - 1.0
                    derivative_factor = (
                        k * (-2 * t_interior) /
                        (denominator**2) * (2.0 / width)
                    )

                    bump_vals_array = np.asarray(bump_values)
                    if bump_vals_array.ndim == 0:
                        bump_vals_interior = np.full(
                            len(t_interior), float(bump_vals_array)
                        )
                    else:
                        bump_vals_interior = bump_vals_array[interior_mask]

                    result[interior_mask] = (
                        bump_vals_interior * derivative_factor
                    )

            return result

        return Function(
            self.space,
            evaluate_callable=bump_gradient_func,
            name=f'bump_gradient_center_{center:.3f}_k_{k:.3f}',
            support=(a_support, b_support)
        )

    def get_function_by_index(
        self,
        index: int,
        k: Optional[float] = None,
        **kwargs
    ) -> 'Function':
        """Get bump function gradient by index."""
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        if k is None:
            k = self.default_k

        cache_key = (index, k)

        if cache_key not in self._cache:
            a, b = self._domain.a, self._domain.b
            domain_length = b - a

            if self.centers is not None:
                if index >= len(self.centers):
                    raise IndexError(
                        f"Index {index} exceeds number of centers "
                        f"{len(self.centers)}"
                    )
                center = self.centers[index]
                width = self.default_width
            else:
                n_bumps_estimate = max(
                    1, int(domain_length / self.default_width)
                )
                if index >= n_bumps_estimate:
                    extra_spacing = (index - n_bumps_estimate + 1) * 0.1
                    center = a + (index * self.default_width) + extra_spacing
                else:
                    center = (a + (index + 0.5) * domain_length /
                              n_bumps_estimate)
                width = self.default_width

            if center < a or center > b:
                raise ValueError(
                    f"Bump center {center} is outside domain [{a}, {b}]"
                )

            gradient_func = self.get_function_by_parameters({
                'center': center,
                'width': width,
                'k': k
            })

            self._cache[cache_key] = gradient_func

        return self._cache[cache_key]

    @property
    def num_functions(self) -> int:
        """Return estimated number of functions."""
        if self.centers is not None:
            return len(self.centers)
        else:
            a, b = self._domain.a, self._domain.b
            domain_length = b - a
            return max(1, int(domain_length / self.default_width))
