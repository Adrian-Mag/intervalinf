"""SOLA operator for interval domains.

The SOLA (Subtractive Optimally Localized Averages) operator
integrates input functions against a set of kernel functions,
producing a vector of data values.
"""

from typing import Union, Optional, List, Callable, TYPE_CHECKING

import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator

from ..spaces.lebesgue import Lebesgue
from ..spaces.sobolev import Sobolev
from ..core.functions import Function
from ..core.config import IntegrationConfig
from ..providers.base import IndexedFunctionProvider

if TYPE_CHECKING:
    from pygeoinf import LinearForm
    from ..spaces.forms import LinearFormKernel


class SOLAOperator(LinearOperator):
    """
    SOLA operator that applies kernel functions to input functions via
    integration.

    This operator takes a function from a Lebesgue space and computes integrals
    against a set of kernel functions, resulting in a vector in the specified
    Euclidean space.

    The operator maps: Lebesgue -> EuclideanSpace

    For each kernel function k_i, it computes: ∫ f(x) * k_i(x) dx

    The kernel functions can be provided in three ways:
    1. Via a FunctionProvider (original functionality)
    2. Via a list of Function objects
    3. Via a list of callables (automatically converted to Function objects)

    For direct sum domains (LebesgueSpaceDirectSum), use the static method
    `for_direct_sum` to create a RowLinearOperator that operates on each
    subspace independently.

    Examples
    --------
    Using a function provider:

    >>> provider = NormalModesProvider(lebesgue_space)
    >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
    ...                        kernels=provider)

    Using direct callables:

    >>> kernels = [lambda x: np.sin(x), lambda x: np.cos(x)]
    >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
    ...                        kernels=kernels)

    Using Function objects:

    >>> func1 = Function(lebesgue_space, evaluate_callable=lambda x: x**2)
    >>> func2 = Function(lebesgue_space, evaluate_callable=lambda x: x**3)
    >>> sola_op = SOLAOperator(lebesgue_space, euclidean_space,
    ...                        kernels=[func1, func2])
    """

    def __init__(
        self,
        domain: Union[Lebesgue, Sobolev],
        codomain: EuclideanSpace,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None,
        cache_kernels: bool = False,
        integration_config: IntegrationConfig = IntegrationConfig(
            method='simpson', n_points=1000
        ),
    ):
        """
        Initialize the SOLA operator.

        Parameters
        ----------
        domain : Lebesgue or Sobolev
            Lebesgue or Sobolev space (the function space)
        codomain : EuclideanSpace
            EuclideanSpace instance that defines the output dimension
        kernels : IndexedFunctionProvider or list of Function/callable, optional
            Provider or list of kernel functions. If list of callables,
            they will be converted to Function instances.
        cache_kernels : bool, default=False
            If True, cache kernels after first access
        integration_config : IntegrationConfig
            Integration configuration for computing integrals
        """
        self._domain = domain
        self._codomain = codomain
        self.N_d = codomain.dim
        self._kernels_provider = None
        self.cache_kernels = cache_kernels
        self._kernels_cache = {} if cache_kernels else None

        # Store integration config
        self.integration = integration_config

        self._initialize_kernels(kernels)

        super().__init__(
            domain,
            codomain,
            self._mapping,
            dual_mapping=self._dual_mapping
        )

    def _mapping(self, f: 'Function') -> np.ndarray:
        """Apply kernel functions to input function via integration."""
        return self._apply_kernels(f)

    def _dual_mapping(self, yp: 'LinearForm') -> 'LinearFormKernel':
        """Reconstruct function from data using kernel functions."""
        from ..spaces.forms import LinearFormKernel
        kernel = self._reconstruct_function(yp.components)
        return LinearFormKernel(
            self.domain, kernel=kernel, integration_config=self.integration
        )

    def _initialize_kernels(
        self,
        kernels: Optional[
            Union[
                IndexedFunctionProvider,
                List[Union[Function, Callable]]
            ]
        ] = None
    ):
        """Initialize kernels from provider or list."""
        # Default to None - will use provider if set
        self._kernels = None

        if isinstance(kernels, list):
            if len(kernels) != self.N_d:
                raise ValueError(
                    f"Number of kernels ({len(kernels)}) must match "
                    f"codomain dimension ({self.N_d})"
                )
            if isinstance(kernels[0], Function):
                # Directly use provided Function instances
                self._kernels = kernels
            elif callable(kernels[0]):
                # Convert callables to Function instances
                self._kernels = [
                    Function(
                        self._domain.function_domain,
                        evaluate_callable=func
                    )
                    for func in kernels
                ]
        elif isinstance(kernels, IndexedFunctionProvider):
            self._kernels_provider = kernels
            # _kernels already set to None above

    def get_kernel(self, index: int) -> Function:
        """
        Lazily get the i-th kernel with optional caching.

        Parameters
        ----------
        index : int
            Index of the kernel to retrieve

        Returns
        -------
        Function
            The i-th kernel
        """
        # If kernels are directly provided, return from list
        if self._kernels is not None:
            return self._kernels[index]

        # Check cache
        if self.cache_kernels and self._kernels_cache is not None:
            if index in self._kernels_cache:
                return self._kernels_cache[index]

        # Use the provider to get the kernel
        assert self._kernels_provider is not None
        kernel = self._kernels_provider.get_function_by_index(index)

        # Cache if enabled
        if self.cache_kernels and self._kernels_cache is not None:
            self._kernels_cache[index] = kernel

        return kernel

    def _apply_kernels(self, func: Function) -> np.ndarray:
        """
        Apply the kernel functions to a function by integrating their product.

        For each kernel k_i, computes ∫ func(x) * k_i(x) dx

        Parameters
        ----------
        func : Function
            Function from the domain space

        Returns
        -------
        numpy.ndarray
            Vector of data in R^{N_d}
        """
        data = np.zeros(self.N_d)

        for i in range(self.N_d):
            # Lazily get the i-th kernel
            kernel = self.get_kernel(i)

            # Compute integral of product: ∫ func(x) * kernel(x) dx
            def product_callable(x, _kernel=kernel):
                return func.evaluate(x) * _kernel.evaluate(x, check_domain=False)

            product_func = Function(
                self._domain.function_domain, evaluate_callable=product_callable
            )
            data[i] = product_func.integrate(
                method=self.integration.method,
                n_points=self.integration.n_points
            )

        return data

    def _reconstruct_function(self, data: np.ndarray) -> Function:
        """
        Reconstruct a function from data using lazy evaluation.

        Parameters
        ----------
        data : numpy.ndarray
            Data in R^{N_d}

        Returns
        -------
        Function
            Reconstructed function in the domain space
        """
        # Collect non-zero terms to avoid deep recursion
        terms = []
        for i, coeff in enumerate(data):
            if abs(coeff) > 1e-14:  # Avoid numerical noise
                kernel = self.get_kernel(i)
                terms.append((coeff, kernel))

        # Create a single callable that evaluates all terms
        if not terms:
            return self._domain.zero

        def evaluate_sum(x):
            result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            for coeff, kernel in terms:
                result = result + coeff * kernel.evaluate(x)
            return result

        return Function(
            self._domain.function_domain, evaluate_callable=evaluate_sum
        )

    def get_kernels(self) -> List[Function]:
        """
        Get the list of kernels used by this operator.
        
        Note: This materializes all functions and may be expensive.

        Returns
        -------
        list of Function
            List of kernels used for projection
        """
        return [self.get_kernel(i) for i in range(self.N_d)]

    def compute_gram_matrix(self) -> np.ndarray:
        """
        Compute the Gram matrix of the kernels using function integration.

        For kernels k_i, k_j, computes ∫ k_i(x) * k_j(x) dx

        Returns
        -------
        numpy.ndarray
            N_d x N_d matrix of integrals between kernels
        """
        gram = np.zeros((self.N_d, self.N_d))

        for i in range(self.N_d):
            kernel_i = self.get_kernel(i)
            for j in range(self.N_d):
                kernel_j = self.get_kernel(j)

                def product_callable(x, _ki=kernel_i, _kj=kernel_j):
                    return _ki.evaluate(x) * _kj.evaluate(x)

                product_func = Function(
                    self._domain.function_domain, evaluate_callable=product_callable
                )
                gram[i, j] = product_func.integrate(
                    method=self.integration.method,
                    n_points=self.integration.n_points
                )

        return gram

    def clear_cache(self):
        """Clear the function cache if caching is enabled."""
        if self.cache_kernels and self._kernels_cache is not None:
            self._kernels_cache.clear()

    def get_cache_info(self) -> dict:
        """
        Get information about the function cache.

        Returns
        -------
        dict
            Cache statistics including size and hit rate
        """
        if not self.cache_kernels:
            return {"caching_enabled": False}

        assert self._kernels_cache is not None
        return {
            "caching_enabled": True,
            "cached_functions": len(self._kernels_cache),
            "total_functions": self.N_d,
            "cache_coverage": len(self._kernels_cache) / self.N_d
        }

    def __str__(self) -> str:
        """String representation of the SOLA operator."""
        provider_type = (
            type(self._kernels_provider).__name__
            if self._kernels_provider else "direct functions"
        )
        return (
            f"SOLAOperator: {self._domain} -> {self._codomain}\n"
            f"  Uses {self.N_d} kernels from {provider_type}\n"
            f"  Domain dimension: {self._domain.dim}\n"
            f"  Codomain dimension: {self._codomain.dim}"
        )

    @staticmethod
    def for_direct_sum(
        domain,  # HilbertSpaceDirectSum
        codomain: EuclideanSpace,
        kernels: Union[
            IndexedFunctionProvider,
            List[Union[Function, Callable]]
        ],
        cache_kernels: bool = False,
        integration_config: IntegrationConfig = IntegrationConfig(
            method='simpson', n_points=1000
        ),
    ):
        """
        Create SOLAOperator for direct sum domain (discontinuous functions).

        This method creates a RowLinearOperator where each block operates
        on one of the subspaces of the direct sum. The kernel functions
        are used on each subdomain independently.

        Parameters
        ----------
        domain : HilbertSpaceDirectSum
            Direct sum space (e.g., LebesgueSpaceDirectSum)
        codomain : EuclideanSpace
            EuclideanSpace defining the output dimension
        kernels : IndexedFunctionProvider or list
            Provider or list of kernels defined on the full domain
        cache_kernels : bool, default=False
            If True, cache kernels after first access
        integration_config : IntegrationConfig
            Integration configuration

        Returns
        -------
        RowLinearOperator
            Operator mapping from the direct sum space to the
            codomain by integrating against kernels on each subdomain.

        Examples
        --------
        >>> # Create a space with discontinuity
        >>> M = Lebesgue.with_discontinuities(
        ...     200, domain, [0.5], basis=None
        ... )
        >>> # Create kernels that span the full domain
        >>> provider = NormalModesProvider(M, ...)
        >>> # Create the operator
        >>> G = SOLAOperator.for_direct_sum(M, D, provider)
        >>> # G can act on discontinuous functions:
        >>> # G([f_lower, f_upper])
        """
        from pygeoinf.direct_sum import (
            HilbertSpaceDirectSum,
            RowLinearOperator
        )

        if not isinstance(domain, HilbertSpaceDirectSum):
            raise TypeError(
                f"domain must be HilbertSpaceDirectSum, "
                f"got {type(domain)}"
            )

        # Create a SOLA operator for each subspace
        operators = []
        for i in range(domain.number_of_subspaces):
            subspace = domain.subspace(i)

            # Type assertion for type checker
            if not isinstance(subspace, (Lebesgue, Sobolev)):
                raise TypeError(
                    f"SOLAOperator requires Lebesgue or Sobolev subspaces,"
                    f" got {type(subspace)}"
                )

            # Restrict kernels to this subspace
            if isinstance(kernels, IndexedFunctionProvider):
                # Use provider restriction
                restricted_kernels = kernels.restrict(subspace)
            elif isinstance(kernels, list):
                # Restrict each function in the list
                restricted_kernels = []
                for kernel in kernels:
                    if isinstance(kernel, Function):
                        restricted_kernels.append(kernel.restrict(subspace))
                    elif callable(kernel):
                        raise NotImplementedError(
                            "Cannot automatically restrict callable kernels."
                            " Please provide a FunctionProvider or "
                            "pre-restricted Functions."
                        )
                    else:
                        raise TypeError(f"Unknown kernel type: {type(kernel)}")
            else:
                raise TypeError(
                    f"kernels must be IndexedFunctionProvider or list, "
                    f"got {type(kernels)}"
                )

            # Create SOLAOperator with restricted kernels
            sola_sub = SOLAOperator(
                subspace,
                codomain,
                kernels=restricted_kernels,
                cache_kernels=cache_kernels,
                integration_config=integration_config
            )
            operators.append(sola_sub)

        # Create and return row operator
        return RowLinearOperator(operators)
