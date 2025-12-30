"""
Wavelet function providers.

This module contains providers for wavelet basis functions:

- WaveletFunctionProvider: General wavelet basis functions (Haar, etc.)
"""

import numpy as np
from typing import TYPE_CHECKING

from intervalinf.providers.base import IndexedFunctionProvider

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class WaveletFunctionProvider(IndexedFunctionProvider):
    """
    Provider for wavelet basis functions.

    Currently supports Haar wavelets. The Haar wavelet basis is a complete
    orthonormal system for L²([a,b]).

    Index ordering:
        index 0: Scaling function (constant)
        index 1: Level 1, position 0
        index 2: Level 2, position 0
        index 3: Level 2, position 1
        index 4: Level 3, position 0
        etc.
    """

    def __init__(self, space, wavelet_type: str = 'haar'):
        """
        Initialize wavelet provider.

        Args:
            space: Lebesgue instance (contains domain information)
            wavelet_type: Type of wavelet ('haar', etc.)
        """
        super().__init__(space)
        self.wavelet_type = wavelet_type
        self._cache = {}

    def get_function_by_index(self, index: int, **kwargs) -> 'Function':
        """Get wavelet function by index."""
        if index in self._cache:
            return self._cache[index]

        if self.wavelet_type == 'haar':
            func = self._get_haar_wavelet(index)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        self._cache[index] = func
        return func

    def _get_haar_wavelet(self, index: int) -> 'Function':
        """Get Haar wavelet by index."""
        from intervalinf.core.functions import Function

        a, b = self.domain.a, self.domain.b

        if index == 0:
            # Scaling function (constant)
            def scaling_func(x):
                x_arr = np.asarray(x)
                return np.ones_like(x_arr, dtype=float) / np.sqrt(b - a)

            return Function(
                self.space,
                evaluate_callable=scaling_func,
                name='haar_scaling'
            )

        # Decode index to get level and position
        level = int(np.log2(index)) + 1
        index_in_level = index - (2**(level-1) - 1)

        def haar_func(x):
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)

            # Normalize x to [0, 1]
            x_norm = (x_arr - a) / (b - a)

            # Calculate shift and scale for this wavelet
            scale = 2**level
            shift = index_in_level / scale

            # Haar wavelet: +1 on first half, -1 on second half of support
            mask1 = (x_norm >= shift) & (x_norm < shift + 0.5/scale)
            mask2 = (x_norm >= shift + 0.5/scale) & (x_norm < shift + 1.0/scale)

            result[mask1] = np.sqrt(scale)
            result[mask2] = -np.sqrt(scale)

            return result

        return Function(
            self.space,
            evaluate_callable=haar_func,
            name=f'haar_L{level}_I{index_in_level}'
        )
