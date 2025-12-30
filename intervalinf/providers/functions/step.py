"""
Step function providers.

This module contains providers for piecewise constant functions:

- BoxCarFunctionProvider: Rectangular/step functions with compact support
- DiscontinuousFunctionProvider: Functions with random discontinuities
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

from intervalinf.providers.base import (
    ParametricFunctionProvider,
    IndexedFunctionProvider,
    RandomFunctionProvider,
)

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class BoxCarFunctionProvider(
    ParametricFunctionProvider, IndexedFunctionProvider
):
    """
    Provider for box-car (rectangular/step) functions.

    Box-car functions are piecewise constant functions that are zero outside
    a finite interval and have a constant value inside:

        f(x) = height for x ∈ [center - width/2, center + width/2]
        f(x) = 0 elsewhere

    By default, functions are normalized so that their integral equals 1.
    """

    def __init__(
        self,
        space_or_domain,
        default_width: float = 0.2,
        centers: Optional[np.ndarray] = None,
        default_height: float = 1.0,
        normalize: bool = True
    ):
        """
        Initialize box-car function provider.

        Args:
            space_or_domain: Space or IntervalDomain
            default_width: Default width for indexed access (as fraction of
                          domain)
            centers: Optional array of centers for indexed access
            default_height: Default height of the box-car function
            normalize: If True, normalize so integral equals 1
        """
        super().__init__(space_or_domain)
        self.default_width = default_width
        self.centers = np.asarray(centers) if centers is not None else None
        self.default_height = default_height
        self.normalize = normalize
        self._cache = {}

    def get_function_by_parameters(
        self,
        parameters: Dict[str, Any],
        **kwargs
    ) -> 'Function':
        """
        Get a box-car function with specific parameters.

        Args:
            parameters: Dictionary containing:
                - 'center': Center of the box-car function
                - 'width': Width of the box-car function
                - 'height': Height (optional)
                - 'normalize': Whether to normalize (optional)
        """
        from intervalinf.core.functions import Function

        center = parameters['center']
        width = parameters['width']
        height = parameters.get('height', self.default_height)
        normalize = parameters.get('normalize', self.normalize)

        a_support = center - width / 2
        b_support = center + width / 2

        if normalize:
            actual_height = 1.0 / width
        else:
            actual_height = height

        def boxcar_func(x):
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)
            mask = (x_arr >= a_support) & (x_arr <= b_support)
            result[mask] = actual_height
            return result

        name_suffix = "_normalized" if normalize else f"_h{height:.3f}"
        return Function(
            self.function_context,
            evaluate_callable=boxcar_func,
            name=f'boxcar_c{center:.3f}_w{width:.3f}{name_suffix}',
            support=(a_support, b_support)
        )

    def get_function_by_index(
        self,
        index: int,
        height: Optional[float] = None,
        normalize: Optional[bool] = None,
        **kwargs
    ) -> 'Function':
        """Get box-car function by index."""
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        if height is None:
            height = self.default_height
        if normalize is None:
            normalize = self.normalize

        cache_key = (index, height, normalize)

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

            width = min(self.default_width * domain_length,
                        2 * min(center - a, b - center))

            parameters = {
                'center': center,
                'width': width,
                'height': height,
                'normalize': normalize
            }
            func = self.get_function_by_parameters(parameters)
            func.name = f'boxcar_{index}_c{center:.3f}'
            self._cache[cache_key] = func

        return self._cache[cache_key]

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for box-car functions."""
        a, b = self.domain.a, self.domain.b
        domain_length = b - a

        return {
            'center': (a + b) / 2,
            'width': self.default_width * domain_length,
            'height': self.default_height,
            'normalize': self.normalize
        }

    def get_n_functions(self) -> Optional[int]:
        """Get the number of available functions."""
        return len(self.centers) if self.centers is not None else None


class DiscontinuousFunctionProvider(RandomFunctionProvider):
    """
    Provider for functions with random discontinuities.

    Generates piecewise constant functions with random jump locations
    and jump sizes.
    """

    def __init__(self, space_or_domain, random_state=None):
        """
        Initialize discontinuous function provider.

        Args:
            space_or_domain: Space or IntervalDomain
            random_state: Random seed for reproducibility
        """
        super().__init__(space_or_domain, random_state)

    def get_random_function(
        self,
        n_discontinuities: Optional[int] = None,
        jump_range: Tuple[float, float] = (-1, 1),
        **kwargs
    ) -> 'Function':
        """
        Sample a function with random discontinuities.

        Args:
            n_discontinuities: Number of discontinuities (random if None)
            jump_range: Range for jump sizes (min, max)

        Returns:
            Function: Piecewise constant function with discontinuities
        """
        from intervalinf.core.functions import Function

        a, b = self.domain.a, self.domain.b

        if n_discontinuities is None:
            n_discontinuities = self.rng.integers(1, 6)

        # Random discontinuity locations
        disc_locations = self.rng.uniform(a, b, n_discontinuities)
        disc_locations = np.sort(disc_locations)

        # Random jump sizes
        jumps = self.rng.uniform(
            jump_range[0], jump_range[1], n_discontinuities
        )

        def discontinuous_func(x):
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)

            for loc, jump in zip(disc_locations, jumps):
                result[x_arr >= loc] += jump

            return result

        return Function(
            self.function_context,
            evaluate_callable=discontinuous_func,
            name=f'discontinuous_{n_discontinuities}'
        )

    # Alias for consistency with RandomFunctionProvider interface
    sample_function = get_random_function
