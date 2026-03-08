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
        kernels : IndexedFunctionProvider or list of Function/callable,
                  optional
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

    def _apply_kernels(self, func: 'Function') -> np.ndarray:
        """
        Apply the kernel functions to a function by integrating their product.

        For each kernel k_i, computes $\\int f(x) \\, k_i(x) \\, dx$.

        Dispatches automatically to the fast batched path for fixed-grid
        methods (``'simpson'``, ``'trapz'``) and to the generic per-kernel
        path for adaptive methods.

        Parameters
        ----------
        func : Function
            Function from the domain space.

        Returns
        -------
        numpy.ndarray
            Vector of data in $\\mathbb{R}^{N_d}$.
        """
        if self.integration.is_fixed_grid:
            return self._apply_kernels_fixed_grid(func)
        return self._apply_kernels_generic(func)

    @staticmethod
    def _eval_on_mesh(func: 'Function', xs: np.ndarray) -> np.ndarray:
        """
        Evaluate *func* on a mesh, preserving scalar dtype where possible.

        Tries a vectorised call first; if that raises or returns the wrong
        shape, falls back to per-point scalar evaluation.  This preserves
        correctness for any callable, including non-vectorised ones.

        Parameters
        ----------
        func : Function
            The function to evaluate.
        xs : ndarray, shape (n,)
            Mesh points.

        Returns
        -------
        ndarray, shape (n,)
        """
        try:
            result = func.evaluate(xs, check_domain=False)
            arr = np.asarray(result)
            if arr.shape == xs.shape:
                return arr
        except Exception:
            pass
        # Per-point fallback for non-vectorised callables.
        return np.asarray(
            [func.evaluate(float(x), check_domain=False) for x in xs]
        )

    def _apply_kernels_fixed_grid(self, func: 'Function') -> np.ndarray:
        """
        Automatic accelerated forward path for fixed-grid integration methods.

        Builds the quadrature mesh **once** per call, evaluates the input
        function **once** on the shared mesh, assembles a
        ``(N_d, n_points)`` kernel matrix, and integrates all products with
        a single batched ``scipy.integrate.simpson`` or
        ``trapezoid`` call — avoiding the per-kernel mesh-builds and
        repeated evaluations of *f* present in the generic path.

        Dispatch conditions
        -------------------
        * ``self.integration.is_fixed_grid`` is True (method ``'simpson'``
          or ``'trapz'``).
        * Called automatically from :meth:`_apply_kernels`.

        Fallback for non-vectorised callables
        --------------------------------------
        :meth:`_eval_on_mesh` tries a vectorised call on the shared mesh
        first; if that fails (wrong shape, exception), it falls back to a
        per-point loop.  Correctness is preserved regardless of whether
        the callable supports array input.

        Support propagation
        -------------------
        Kernels whose support is disjoint from *func*'s support are skipped
        exactly as in the generic path (result is 0 without evaluating the
        integrand). For non-disjoint compact-support configurations, this
        method falls back to the generic Phase 3 path for that kernel so the
        quadrature mesh is still built on the narrowed support intersection.

        Parameters
        ----------
        func : Function
            Function from the domain space.

        Returns
        -------
        ndarray, shape (N_d,)
        """
        from scipy.integrate import simpson as _simpson
        try:
            from scipy.integrate import trapezoid as _trapz
        except ImportError:
            # pragma: no cover - scipy < 1.11 fallback
            from scipy.integrate import trapz as _trapz  # type: ignore

        domain = self._domain.function_domain
        method = self.integration.method
        n_points = max(3, self.integration.n_points)

        # ── Build shared mesh once ────────────────────────────────────────
        xs = np.linspace(domain.a, domain.b, n_points)

        # ── Evaluate f once on the shared mesh ───────────────────────────
        f_vals = self._eval_on_mesh(func, xs)

        # ── Evaluate all kernels on the shared mesh ───────────────────────
        # Pre-build a (N_d, n_points) kernel matrix; kernels with disjoint
        # support vs func are left as zeros and flagged in disjoint_mask.
        results = [0.0] * self.N_d
        batched_indices = []
        batched_rows = []
        disjoint_mask = np.zeros(self.N_d, dtype=bool)

        for i in range(self.N_d):
            kernel = self.get_kernel(i)
            intersected_support = Function._intersect_supports(
                func.support, kernel.support
            )
            if intersected_support == []:
                # Supports are disjoint → product is identically zero.
                disjoint_mask[i] = True
                continue

            # Preserve Phase 3 support-aware quadrature semantics whenever a
            # genuine compact-support restriction is available.
            if intersected_support is not None:
                def product_callable(x, _f=func, _k=kernel):
                    return _f.evaluate(
                        x,
                        check_domain=False,
                    ) * _k.evaluate(
                        x,
                        check_domain=False,
                    )

                results[i] = domain.integrate(
                    product_callable,
                    method=method,
                    support=intersected_support,
                    n_points=n_points,
                )
                continue

            batched_indices.append(i)
            batched_rows.append(self._eval_on_mesh(kernel, xs))

        if batched_rows:
            K_matrix = np.stack(batched_rows, axis=0)
            P_matrix = f_vals[np.newaxis, :] * K_matrix
            if method == "simpson":
                batched_data = np.asarray(_simpson(P_matrix, x=xs, axis=1))
            else:  # 'trapz'
                batched_data = np.asarray(_trapz(P_matrix, x=xs, axis=1))

            for index, value in zip(batched_indices, batched_data):
                results[index] = value

        data = np.asarray(results)
        # Force disjoint-support entries to exactly 0 (no numerical noise).
        data[disjoint_mask] = 0.0
        return data

    def _apply_kernels_generic(self, func: 'Function') -> np.ndarray:
        """
        Per-kernel integration loop for adaptive methods and as fallback.

        This is the original integration path, used when
        ``self.integration.is_adaptive`` is True.  It builds a fresh
        product callable and calls ``domain.integrate`` for each kernel
        individually.

        Support propagation (Phase 3): if both *func* and the kernel carry
        compact-support metadata the integration range is narrowed to the
        support intersection.  When the supports are disjoint the result
        is exactly 0 without evaluating the integrand.

        Parameters
        ----------
        func : Function
            Function from the domain space.

        Returns
        -------
        numpy.ndarray
            Vector of data in $\\mathbb{R}^{N_d}$.
        """
        results = [0.0] * self.N_d
        domain = self._domain.function_domain
        method = self.integration.method
        n_points = self.integration.n_points

        for i in range(self.N_d):
            # Lazily get the i-th kernel
            kernel = self.get_kernel(i)

            # Narrow the integration range to compound support intersection.
            # Function._intersect_supports returns None when either operand has
            # no compact-support hint (safe: integrates over the full domain).
            intersected_support = Function._intersect_supports(
                func.support, kernel.support
            )

            # Empty intersection → product is identically zero; no need to
            # evaluate the integrand at all.
            if intersected_support == []:
                continue  # data[i] already 0.0

            def product_callable(x, _f=func, _k=kernel):
                return _f.evaluate(
                    x,
                    check_domain=False,
                ) * _k.evaluate(
                    x,
                    check_domain=False,
                )

            results[i] = domain.integrate(
                product_callable,
                method=method,
                support=intersected_support,
                n_points=n_points,
            )

        return np.asarray(results)

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

        For kernels $k_i, k_j$, computes $G_{ij} = \\int k_i(x) k_j(x) \\, dx$.

        Support propagation is applied: if both kernels have compact-support
        metadata, integration is restricted to the support intersection.

        Returns
        -------
        numpy.ndarray
            $N_d \\times N_d$ matrix of kernel inner products.
        """
        gram = np.zeros((self.N_d, self.N_d))
        domain = self._domain.function_domain
        method = self.integration.method
        n_points = self.integration.n_points

        for i in range(self.N_d):
            kernel_i = self.get_kernel(i)
            for j in range(self.N_d):
                kernel_j = self.get_kernel(j)

                intersected_support = Function._intersect_supports(
                    kernel_i.support, kernel_j.support
                )
                if intersected_support == []:
                    continue  # gram[i, j] already 0.0

                def product_callable(x, _ki=kernel_i, _kj=kernel_j):
                    return _ki.evaluate(x) * _kj.evaluate(x)

                gram[i, j] = domain.integrate(
                    product_callable,
                    method=method,
                    support=intersected_support,
                    n_points=n_points,
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
