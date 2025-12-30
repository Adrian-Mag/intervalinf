"""
Data-based function providers.

This module contains providers for functions loaded from external data
or generated with specific random patterns:

- KernelProvider: Kernel functions loaded from data files
- NormalModesProvider: Gaussian-modulated trigonometric functions
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from scipy.interpolate import interp1d

from intervalinf.providers.base import (
    IndexedFunctionProvider,
    RandomFunctionProvider,
    ParametricFunctionProvider,
)

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class KernelProvider(IndexedFunctionProvider):
    """
    Provider for kernel functions loaded from data files.

    This provider reads kernel sensitivity data from files and creates
    interpolated Function objects. Useful for loading pre-computed
    sensitivity kernels in geophysics applications.
    """

    def __init__(
        self,
        space_or_domain,
        kernel_type: str = "rho",
        kernel_data_dir: Optional[str] = None
    ):
        """
        Initialize kernel provider.

        Args:
            space_or_domain: Space or IntervalDomain
            kernel_type: Type of kernel (e.g., "rho", "vpv", "vph")
            kernel_data_dir: Directory containing kernel data files
        """
        super().__init__(space_or_domain)
        self._kernel_type = kernel_type
        self._data_dir = kernel_data_dir
        self._data_list = self._get_data_list()
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get kernel function by index from data files.

        Args:
            index: Index of the kernel function

        Returns:
            Function: Interpolated kernel function
        """
        if index in self._cache:
            return self._cache[index]

        from intervalinf.core.functions import Function

        if self._data_dir is None:
            raise ValueError("kernel_data_dir must be provided")
        if index < 0 or index >= len(self._data_list):
            raise IndexError(
                f"Invalid index {index}. Maximum is {len(self._data_list) - 1}"
            )

        mode = self._data_list[index]
        filename = f"{self._kernel_type}-sens_{mode}_iso.dat"
        filepath = os.path.join(self._data_dir, filename)

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Kernel file not found: {filepath}")

        data = np.loadtxt(filepath)
        radius = data[:, 0]
        values = data[:, 1]

        interp_func = interp1d(
            radius, values, bounds_error=False, fill_value=0.0
        )

        func = Function(
            self.function_context,
            evaluate_callable=interp_func,
            name=f"kernel_{self._kernel_type}_{mode}"
        )
        self._cache[index] = func
        return func

    def _get_data_list(self) -> List[str]:
        """Get a list of all kernel data files."""
        if self._data_dir is None:
            return []

        filepath = os.path.join(self._data_dir, 'data_list_SP12RTS')
        if not os.path.isfile(filepath):
            return []

        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    @property
    def n_kernels(self) -> int:
        """Get number of available kernels."""
        return len(self._data_list)


class NormalModesProvider(
    RandomFunctionProvider,
    ParametricFunctionProvider,
    IndexedFunctionProvider
):
    """
    Provider for random trigonometric functions modulated by Gaussians.

    Generates functions as linear combinations of sine functions with random
    frequencies, each modulated by a Gaussian envelope with random center
    and width.

    This provider can be used as:
    - RandomFunctionProvider: sample_function() / get_random_function()
    - ParametricFunctionProvider: get_function_by_parameters()
    - IndexedFunctionProvider: get_function_by_index() (deterministic)
    """

    def __init__(
        self,
        space_or_domain,
        random_state: Optional[int] = None,
        n_modes_range: Tuple[int, int] = (3, 8),
        coeff_range: Tuple[float, float] = (-2.0, 2.0),
        freq_range: Tuple[float, float] = (0.5, 10.0),
        gaussian_width_percent_range: Tuple[float, float] = (10.0, 50.0)
    ):
        """
        Initialize normal modes provider.

        Args:
            space_or_domain: Space or IntervalDomain
            random_state: Random seed for reproducibility
            n_modes_range: (min, max) number of sine functions to combine
            coeff_range: (min, max) range for linear combination coefficients
            freq_range: (min, max) range for sine function frequencies
            gaussian_width_percent_range: (min, max) percentage of interval
                                        length for Gaussian width
        """
        super().__init__(space_or_domain, random_state)
        self.n_modes_range = n_modes_range
        self.coeff_range = coeff_range
        self.freq_range = freq_range
        self.gaussian_width_percent_range = gaussian_width_percent_range

        self._original_seed = random_state
        self._base_seed = random_state if random_state is not None else 12345

    def get_random_function(self, **kwargs) -> 'Function':
        """Sample a random trigonometric function modulated by Gaussians."""
        from intervalinf.core.functions import Function

        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        n_modes = self.rng.integers(
            self.n_modes_range[0], self.n_modes_range[1] + 1
        )

        coefficients = self.rng.uniform(
            self.coeff_range[0], self.coeff_range[1], n_modes
        )
        frequencies = self.rng.uniform(
            self.freq_range[0], self.freq_range[1], n_modes
        )

        gaussian_center = self.rng.uniform(a, b)
        width_percentage = self.rng.uniform(
            self.gaussian_width_percent_range[0],
            self.gaussian_width_percent_range[1]
        )
        gaussian_width = (width_percentage / 100.0) * interval_length

        def combined_func(x):
            x_arr = np.asarray(x)
            trig_combination = np.zeros_like(x_arr, dtype=float)

            for i in range(n_modes):
                sine_part = np.sin(
                    2 * np.pi * frequencies[i] * (x_arr - a) / interval_length
                )
                trig_combination += coefficients[i] * sine_part

            gaussian_envelope = np.exp(
                -0.5 * ((x_arr - gaussian_center) / gaussian_width)**2
            )
            return trig_combination * gaussian_envelope

        return Function(
            self.function_context,
            evaluate_callable=combined_func,
            name=f'gaussian_modulated_trig_{n_modes}_modes'
        )

    # Alias for compatibility
    sample_function = get_random_function

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """
        Get a deterministic function by index.

        Uses a hash-like approach based on the original random_state and
        the index to generate reproducible functions.
        """
        from intervalinf.core.functions import Function

        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        # Create index-specific seed
        index_seed = (self._base_seed + index * 1000003) % (2**31 - 1)
        index_rng = np.random.RandomState(index_seed)

        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        n_modes = index_rng.randint(
            self.n_modes_range[0], self.n_modes_range[1] + 1
        )
        coefficients = index_rng.uniform(
            self.coeff_range[0], self.coeff_range[1], n_modes
        )
        frequencies = index_rng.uniform(
            self.freq_range[0], self.freq_range[1], n_modes
        )
        gaussian_center = index_rng.uniform(a, b)
        width_percentage = index_rng.uniform(
            self.gaussian_width_percent_range[0],
            self.gaussian_width_percent_range[1]
        )
        gaussian_width = (width_percentage / 100.0) * interval_length

        def combined_func(x):
            x_arr = np.asarray(x)
            trig_combination = np.zeros_like(x_arr, dtype=float)

            for i in range(n_modes):
                sine_part = np.sin(
                    2 * np.pi * frequencies[i] * (x_arr - a) / interval_length
                )
                trig_combination += coefficients[i] * sine_part

            gaussian_envelope = np.exp(
                -0.5 * ((x_arr - gaussian_center) / gaussian_width)**2
            )
            return trig_combination * gaussian_envelope

        return Function(
            self.function_context,
            evaluate_callable=combined_func,
            name=f'normal_mode_{index}_{n_modes}_modes'
        )

    def get_indexed_functions(
        self,
        n_functions: int,
        **kwargs
    ) -> List['Function']:
        """Get a deterministic sequence of n functions."""
        return [self.get_function_by_index(i, **kwargs)
                for i in range(n_functions)]

    def get_function_by_parameters(
        self,
        parameters: Dict[str, Any],
        **kwargs
    ) -> 'Function':
        """
        Get a function with specific parameters.

        Args:
            parameters: Dictionary containing 'coefficients', 'frequencies',
                       'gaussian_center', and 'gaussian_width'
        """
        from intervalinf.core.functions import Function

        a, b = self.domain.a, self.domain.b
        interval_length = b - a

        coefficients = parameters['coefficients']
        frequencies = parameters['frequencies']
        gaussian_center = parameters['gaussian_center']
        gaussian_width = parameters['gaussian_width']

        n_modes = len(coefficients)

        def combined_func(x):
            x_arr = np.asarray(x)
            trig_combination = np.zeros_like(x_arr, dtype=float)

            for i in range(n_modes):
                sine_part = np.sin(
                    2 * np.pi * frequencies[i] * (x_arr - a) / interval_length
                )
                trig_combination += coefficients[i] * sine_part

            gaussian_envelope = np.exp(
                -0.5 * ((x_arr - gaussian_center) / gaussian_width)**2
            )
            return trig_combination * gaussian_envelope

        return Function(
            self.function_context,
            evaluate_callable=combined_func,
            name=f'parametric_gaussian_modulated_{n_modes}_modes'
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this family."""
        a, b = self.domain.a, self.domain.b
        interval_length = b - a
        n_modes = (self.n_modes_range[0] + self.n_modes_range[1]) // 2

        return {
            'coefficients': np.ones(n_modes),
            'frequencies': np.linspace(
                self.freq_range[0], self.freq_range[1], n_modes
            ),
            'gaussian_center': (a + b) / 2,
            'gaussian_width': 0.3 * interval_length
        }
