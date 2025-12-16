"""
Linear form kernels for interval function spaces.

This module provides LinearFormKernel, which implements linear forms
on Hilbert spaces using integration-based evaluation rather than
basis-dependent component representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable, Union, List

import numpy as np

from pygeoinf import LinearForm, HilbertSpace

from intervalinf.core.config import IntegrationConfig, ParallelConfig

if TYPE_CHECKING:
    from intervalinf.core.functions import Function


class LinearFormKernel(LinearForm):
    """
    A linear form represented by a kernel function for integration.

    This class implements linear forms on function spaces using the
    Riesz representation: the linear form φ is represented by a kernel
    function k such that φ(f) = ⟨k, f⟩ = ∫ k(x) f(x) dx.

    This is particularly useful for:
    - Dual space representations in L² spaces
    - Mass-weighted inner products in Sobolev spaces
    - Sensitivity kernels in inverse problems

    The key advantage over component-based representations is that
    this approach does not require explicit basis functions, making
    it suitable for "basis-free" function spaces.

    Attributes:
        integration: Integration configuration for inner products.
        parallel: Parallelization configuration.
        kernel: The kernel function (or list of functions for direct sums).

    Example:
        >>> from intervalinf.core import IntervalDomain, Function
        >>> from intervalinf.core.config import IntegrationConfig
        >>> domain = IntervalDomain(0, 1)
        >>> # Create kernel representing the delta functional at x=0.5
        >>> # (approximated by a narrow Gaussian)
        >>> kernel_fn = Function(space, evaluate_callable=narrow_gaussian)
        >>> form = LinearFormKernel(
        ...     space,
        ...     kernel=kernel_fn,
        ...     integration_config=IntegrationConfig(),
        ...     parallel_config=ParallelConfig()
        ... )
        >>> # Evaluate on test function
        >>> result = form(test_function)
    """

    def __init__(
        self,
        domain: HilbertSpace,
        /,
        *,
        mapping: Optional[Callable[['Function'], float]] = None,
        kernel: Optional[Union['Function', List['Function']]] = None,
        components: Optional[np.ndarray] = None,
        integration_config: IntegrationConfig = IntegrationConfig(),
        parallel_config: ParallelConfig = ParallelConfig(
            enabled=False, n_jobs=-1
        ),
    ) -> None:
        """
        Initialize a LinearFormKernel.

        Exactly one of `mapping`, `kernel`, or `components` must be provided.

        Args:
            domain: The HilbertSpace this linear form is defined on.
            mapping: Optional explicit mapping function f -> R.
            kernel: Optional kernel function(s) for integration-based
                evaluation. Can be a single Function or a list of Functions
                (for direct sum spaces).
            components: Optional coefficient array for basis representation.
            integration_config: Configuration for numerical integration.
            parallel_config: Configuration for parallel computation.

        Raises:
            ValueError: If none or more than one of mapping/kernel/components
                is provided.
        """
        # Store configs
        self.integration = integration_config
        self.parallel = parallel_config

        if kernel is not None:
            self._kernel = kernel
            mapping = self._mapping_impl
            self._components = np.zeros((domain.dim,))
        elif mapping is not None:
            self._kernel = None  # type: ignore
            self._components = np.zeros((domain.dim,))
        elif components is not None:
            self._kernel = None  # type: ignore
            self._components = components
        else:
            raise ValueError(
                "Either mapping, kernel, or components must be provided"
            )

        # Get weight from domain if it has one (e.g., weighted L² space)
        # None means standard (unweighted) inner product
        self._weight = getattr(domain, '_weight', None)

        # Cheeky trick: provide fake components so the base class doesn't
        # try to compute them (which requires basis functions)
        self._fake_components = True

        super().__init__(
            domain,
            mapping=mapping,
            components=self._components,
            parallel=self.parallel.enabled,
            n_jobs=self.parallel.n_jobs
        )

    def _mapping_impl(
        self,
        v: Union['Function', List['Function']]
    ) -> float:
        """
        Evaluate the linear form via integration with the kernel.

        For a single kernel k and function v:
            φ(v) = ∫ k(x) v(x) w(x) dx

        For a list of kernels (direct sum), sums contributions:
            φ([v₁, v₂, ...]) = Σᵢ ∫ kᵢ(x) vᵢ(x) wᵢ(x) dx
        """
        # Import here to avoid circular import at module level
        from intervalinf.core.functions import Function

        if isinstance(self._kernel, list):
            # Direct sum case: kernel and v are both lists
            if not isinstance(v, list):
                raise ValueError(
                    "Input must be a list of functions for direct sum"
                )
            return sum(
                (k * vi).integrate(
                    weight=self._weight,
                    method=self.integration.method,
                    n_points=self.integration.n_points
                )
                for k, vi in zip(self._kernel, v)
            )
        else:
            # Single space case
            if not isinstance(v, Function):
                raise ValueError("Input must be a Function")
            return (self._kernel * v).integrate(
                weight=self._weight,
                method=self.integration.method,
                n_points=self.integration.n_points
            )

    @property
    def kernel(self) -> Union['Function', List['Function'], None]:
        """
        The kernel function representing this linear form.

        If a weight function w(x) is set, returns the adjusted kernel
        k(x)/w(x) so that ⟨kernel, f⟩_w = ⟨k/w, f⟩_w recovers the
        original integral ∫ k(x) f(x) dx.

        Returns:
            The kernel Function (or list for direct sums), or None if
            this form was created with mapping or components.
        """
        # Import here to avoid circular import
        from intervalinf.core.functions import Function

        if self._kernel is None:
            return None
        elif self._weight is None:
            # No weight function - return kernel as-is
            return self._kernel
        else:
            # With weight function - adjust kernel by 1/weight
            return self._kernel * Function(
                self.domain,
                evaluate_callable=lambda x: 1 / self._weight(x)
            )

    @property
    def components(self) -> np.ndarray:
        """
        Component array in the basis representation.

        Lazily computes components on first access if not already set.
        This allows the LinearFormKernel to function without basis
        functions until components are explicitly needed.

        Returns:
            Coefficient array φ = Σᵢ cᵢ φᵢ* where φᵢ* are dual basis.
        """
        if self._fake_components:
            # Compute components using parent's implementation
            super()._compute_components(
                self._mapping_impl,
                self.parallel.enabled,
                self.parallel.n_jobs
            )
            self._fake_components = False
        return self._components


__all__ = ['LinearFormKernel']
