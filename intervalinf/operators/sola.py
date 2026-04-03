"""SOLA operator for interval domains.

The SOLA (Subtractive Optimally Localized Averages) operator
integrates input functions against a set of kernel functions,
producing a vector of data values.
"""

import math
import time
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

        # The mesh depends only on domain bounds and n_points, both immutable
        # after construction, so it is safe to build once and reuse forever.
        self._shared_mesh: Optional[np.ndarray] = None

        # Maps kernel index → ndarray of values on the shared fixed-grid mesh.
        # Populated for kernels that take the batched path. Fallback to the
        # generic path happens only when a support intersection is explicitly
        # computable from both func.support and kernel.support.
        # Only active when cache_kernels=True so that provider-backed kernels
        # whose Function objects are also cached remain the source of truth.
        self._kernel_eval_cache: Optional[dict] = (
            {} if cache_kernels else None
        )

        self._stats: dict = {
            "forward_calls": 0,
            "disjoint_skips": 0,
            "compact_support_fallbacks": 0,
            "batched_fixed_grid_kernels": 0,
            "forward_time_total_s": 0.0,
            "compact_support_fallback_time_total_s": 0.0,
        }

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

    @property
    def stats(self) -> dict:
        """Return a copy of the current instrumentation counters.

        Keys
        ----
        forward_calls : int
            Total number of times the forward map has been applied.
        disjoint_skips : int
            Kernels skipped because their support is disjoint from *f*'s
            support (result is exactly 0 without evaluating the integrand).
        compact_support_fallbacks : int
            Kernels handled by the support-restricted grouped batched path in
            ``_apply_kernels_fixed_grid`` (per-kernel count: N kernels sharing
            a support group contribute N to this counter).  These kernels are
            *not* handled by the full-domain batched matrix path and are *not*
            stored in the kernel eval cache.
        batched_fixed_grid_kernels : int
            Kernels that went through the fast full-domain batched fixed-grid
            path.
        forward_time_total_s : float
            Cumulative wall time (seconds) of all forward-map calls.
        compact_support_fallback_time_total_s : float
            Cumulative wall time (seconds) spent in support-restricted (grouped
            batched) integrations within the fixed-grid path.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all instrumentation counters to zero."""
        for key in self._stats:
            self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0

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
        self._stats["forward_calls"] += 1
        _t0 = time.perf_counter()
        if self.integration.is_fixed_grid:
            result = self._apply_kernels_fixed_grid(func)
        else:
            result = self._apply_kernels_generic(func)
        self._stats["forward_time_total_s"] += time.perf_counter() - _t0
        return result

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

    def _get_or_build_mesh(self) -> np.ndarray:
        """Return the shared fixed-grid mesh, building it lazily on first call.

        The mesh depends only on domain bounds and ``integration.n_points``,
        both of which are immutable after construction, so it is safe to
        build once and reuse indefinitely across repeated ``G(f)`` calls.

        Returns
        -------
        ndarray, shape (n_points,)
        """
        if self._shared_mesh is None:
            domain = self._domain.function_domain
            n_points = max(3, self.integration.n_points)
            self._shared_mesh = np.linspace(domain.a, domain.b, n_points)
        return self._shared_mesh

    @staticmethod
    def _build_support_mesh(support: list, n_points: int) -> np.ndarray:
        """Build a quadrature mesh over a list of support subintervals.

        Reproduces the proportional-allocation and remainder-distribution
        logic of :meth:`IntervalDomain.integrate` for support lists so that
        the same mesh can be computed once and reused across multiple function
        evaluations in future batched support-restricted kernel passes.

        Parameters
        ----------
        support : list of (float, float)
            List of ``(a_i, b_i)`` subinterval pairs.  An empty list is
            allowed and returns an empty array.
        n_points : int
            Desired total number of quadrature points.

        Returns
        -------
        ndarray
            Concatenation of per-subinterval ``np.linspace(a_i, b_i, alloc_i)``
            meshes.  Adjacent intervals share a boundary value (duplicated
            endpoints are kept, not removed).  Returns an empty array when
            *support* is empty.

        Notes
        -----
        Allocation algorithm (mirrors ``IntervalDomain.integrate``):

        * ``effective_total = max(n_points, 3 * n_sub)``
        * ``raw_i = effective_total × L_i / total_length``
        * ``alloc_i = max(3, floor(raw_i))``
        * Remaining points distributed by descending fractional part with
          stable original-order tie-break.
        """
        if not support:
            return np.empty(0)

        alloc = SOLAOperator._compute_subinterval_alloc(support, n_points)
        parts = [
            np.linspace(float(support[i][0]), float(support[i][1]), alloc[i])
            for i in range(len(support))
        ]
        return np.concatenate(parts)

    @staticmethod
    def _compute_subinterval_alloc(support_list: list, n_points: int) -> list:
        """Compute per-subinterval point allocations for a quadrature mesh.

        Shared logic used by both :meth:`_build_support_mesh` and the grouped
        support-restricted integration loop in
        :meth:`_apply_kernels_fixed_grid`.  Consolidating here prevents the
        two callers from drifting apart.

        Parameters
        ----------
        support_list : list of (float, float)
            Subinterval pairs ``(a_i, b_i)``.
        n_points : int
            Desired total number of quadrature points.

        Returns
        -------
        list of int
            Per-subinterval point counts.  Each entry is ``>= 3``.
        """
        n_sub = len(support_list)
        lengths = [float(b) - float(a) for a, b in support_list]
        total_length = sum(lengths)
        effective_total = max(n_points, 3 * n_sub)
        raw = [effective_total * (L / total_length) for L in lengths]
        alloc = [max(3, int(math.floor(r))) for r in raw]
        remainder = effective_total - sum(alloc)
        if remainder > 0:
            fracs = sorted(
                [(raw[i] - math.floor(raw[i]), i) for i in range(n_sub)],
                key=lambda x: x[0],
                reverse=True,
            )
            idx = 0
            while remainder > 0:
                alloc[fracs[idx % n_sub][1]] += 1
                remainder -= 1
                idx += 1
        return alloc

    def _apply_kernels_fixed_grid(self, func: 'Function') -> np.ndarray:
        """
        Automatic accelerated forward path for fixed-grid integration methods.

        Phase 4: builds the quadrature mesh once per call (Phase 5: reused
        across calls), evaluates the input function once on the shared mesh,
        assembles a ``(N_d, n_points)`` kernel matrix, and integrates all
        products with a single batched ``scipy.integrate.simpson`` or
        ``trapezoid`` call.

        Phase 5 additions
        -----------------
        * **Mesh reuse**: the shared mesh (xs) is built once at first call and
          reused on all subsequent calls via :meth:`_get_or_build_mesh`.  The
          mesh depends only on immutable construction parameters so no
          invalidation is needed.
        * **Lazy full-domain evaluation**: the shared mesh and ``f``-on-mesh
          evaluation are deferred until *after* the kernel-classification pass.
          They are constructed only if at least one full-domain kernel is
          present.  Disjoint-only and support-only workloads skip the full-
          domain mesh entirely, paying zero unnecessary evaluation cost.
        * **Kernel mesh evaluation cache**: when ``cache_kernels=True``, the
          per-kernel mesh evaluations ``k_i(xs)`` are stored in
          ``_kernel_eval_cache`` after the first forward call and reused on
          all subsequent calls.  Only full-domain (non-compact-support) kernels
          are cached here; support-restricted kernels are handled by the
          grouped support-restricted path and are never stored in this cache.
        * **Grouped support-restricted integration**: compact-support kernels
          (those with non-None, non-empty intersected support) are now grouped
          by their exact ``intersected_support`` key.  All kernels in the same
          group share per-subinterval meshes computed via
          :meth:`_compute_subinterval_alloc`, and their product-with-f
          integrals are computed in a single batched numpy/scipy pass per
          subinterval.  For multi-interval supports, partial batch integrals
          are summed over each subinterval (ensuring correctness for
          non-contiguous support regions).

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
        integrand).  For non-disjoint compact-support configurations, kernels
        are grouped by exact ``intersected_support`` key and processed batch-
        wise on a restricted mesh built from that support.

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

        method = self.integration.method
        n_points = max(3, self.integration.n_points)

        results = [0.0] * self.N_d
        disjoint_mask = np.zeros(self.N_d, dtype=bool)

        # Maps tuple(tuple(interval)) → list of (kernel_index, kernel).
        # Compact-support kernels sharing the same intersected support are
        # grouped here and processed in a single batched pass below.
        support_groups: dict = {}

        # Full-domain kernels accumulated during classification; xs and f_vals
        # are constructed only if this list is non-empty (see below).
        full_domain_kernels: list = []

        # ── Classification pass (no mesh construction yet) ─────────────────
        for i in range(self.N_d):
            kernel = self.get_kernel(i)
            intersected_support = Function._intersect_supports(
                func.support, kernel.support
            )
            if intersected_support == []:
                # Supports are disjoint → product is identically zero.
                disjoint_mask[i] = True
                self._stats["disjoint_skips"] += 1
                continue

            if intersected_support is not None:
                # Group by exact support key
                # (no floating-point canonicalization beyond what
                # _intersect_supports already produces).
                key = tuple(tuple(iv) for iv in intersected_support)
                if key not in support_groups:
                    support_groups[key] = []
                support_groups[key].append((i, kernel))
                continue

            full_domain_kernels.append((i, kernel))

        # ── Full-domain batched integration ───────────────────────────────
        # Build the shared mesh and evaluate f on it only when there are
        # full-domain kernels to process.  Disjoint-only and support-only
        # workloads skip this block entirely, avoiding unnecessary full-domain
        # function evaluation.
        self._stats["batched_fixed_grid_kernels"] += len(full_domain_kernels)
        if full_domain_kernels:
            xs = self._get_or_build_mesh()
            f_vals = self._eval_on_mesh(func, xs)
            batched_indices = []
            batched_rows = []
            for i, kernel in full_domain_kernels:
                if (
                    self._kernel_eval_cache is not None
                    and i in self._kernel_eval_cache
                ):
                    k_vals = self._kernel_eval_cache[i]
                else:
                    k_vals = self._eval_on_mesh(kernel, xs)
                    if self._kernel_eval_cache is not None:
                        self._kernel_eval_cache[i] = k_vals
                batched_indices.append(i)
                batched_rows.append(k_vals)
            K_matrix = np.stack(batched_rows, axis=0)
            P_matrix = f_vals[np.newaxis, :] * K_matrix
            if method == "simpson":
                batched_data = np.asarray(_simpson(P_matrix, x=xs, axis=1))
            else:  # 'trapz'
                batched_data = np.asarray(_trapz(P_matrix, x=xs, axis=1))
            for index, value in zip(batched_indices, batched_data):
                results[index] = value

        # ── Grouped support-restricted batched integration ────────────────
        # For each unique intersected-support key, build a restricted mesh
        # once for the group, evaluate f and all kernels on it, and integrate
        # all products in a single batched pass.  Multi-interval supports are
        # handled by computing per-subinterval partial batch integrals and
        # summing (the concatenated mesh from _build_support_mesh cannot be
        # used directly with a single simpson/trapz call for
        # non-contiguous intervals because the gap between intervals would be
        # integrated over).
        for support_key, group in support_groups.items():
            support_list = [list(iv) for iv in support_key]
            alloc = self._compute_subinterval_alloc(support_list, n_points)
            group_totals = None
            t_fb = time.perf_counter()
            for n_i, (ai, bi) in zip(alloc, support_list):
                xs_i = np.linspace(float(ai), float(bi), n_i)
                f_vals_i = self._eval_on_mesh(func, xs_i)
                k_rows_i = [self._eval_on_mesh(k, xs_i) for _, k in group]
                K_i = np.stack(k_rows_i, axis=0)  # (len(group), len(xs_i))
                P_i = f_vals_i[np.newaxis, :] * K_i
                if method == "simpson":
                    part_i = np.asarray(_simpson(P_i, x=xs_i, axis=1))
                else:  # 'trapz'
                    part_i = np.asarray(_trapz(P_i, x=xs_i, axis=1))
                group_totals = (
                    part_i
                    if group_totals is None
                    else group_totals + part_i
                )
            self._stats["compact_support_fallback_time_total_s"] += (
                time.perf_counter() - t_fb
            )
            for (orig_idx, _kernel), val in zip(group, group_totals):
                results[orig_idx] = val
            self._stats["compact_support_fallbacks"] += len(group)

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
                self._stats["disjoint_skips"] += 1
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

    def _build_kernel_matrix(
        self,
        xs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the dense kernel table $K[i, :] = k_i(x_s)$."""
        mesh = self._get_or_build_mesh() if xs is None else np.asarray(xs)
        if mesh.ndim != 1:
            raise ValueError("Kernel mesh must be one-dimensional.")

        use_shared_cache = False
        if self._kernel_eval_cache is not None:
            shared_mesh = self._get_or_build_mesh()
            use_shared_cache = np.array_equal(mesh, shared_mesh)

        rows = []
        for i in range(self.N_d):
            if use_shared_cache and i in self._kernel_eval_cache:
                k_vals = self._kernel_eval_cache[i]
            else:
                k_vals = np.asarray(
                    self._eval_on_mesh(self.get_kernel(i), mesh)
                )
                if use_shared_cache and self._kernel_eval_cache is not None:
                    self._kernel_eval_cache[i] = k_vals
            rows.append(k_vals)

        if not rows:
            return np.empty((0, mesh.size))
        return np.stack(rows, axis=0)

    def _build_quadrature_weights(
        self,
        xs: Optional[np.ndarray] = None,
        *,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Build fixed-grid quadrature weights for the shared mesh.

        For odd sample counts this returns the standard composite Simpson or
        trapezoid weights. For even sample counts with Simpson's rule it uses
        the same Cartwright end correction as ``scipy.integrate.simpson`` so
        the reduced path matches the existing slow quadrature exactly.
        """
        mesh = self._get_or_build_mesh() if xs is None else np.asarray(xs)
        if mesh.ndim != 1:
            raise ValueError("Quadrature mesh must be one-dimensional.")
        if mesh.size == 0:
            return np.empty(0, dtype=float)

        quadrature_method = (
            self.integration.method if method is None else method
        )
        spacings = np.diff(mesh)
        if quadrature_method == "trapz":
            weights = np.empty(mesh.size, dtype=float)
            weights[0] = 0.5 * spacings[0]
            weights[-1] = 0.5 * spacings[-1]
            if mesh.size > 2:
                weights[1:-1] = 0.5 * (spacings[:-1] + spacings[1:])
            return weights

        if quadrature_method != "simpson":
            raise ValueError(
                "Quadrature weights are only defined for fixed-grid methods."
            )

        if not np.allclose(spacings, spacings[0], rtol=0.0, atol=1e-12):
            from scipy.integrate import simpson as _simpson

            return np.asarray(_simpson(np.eye(mesh.size), x=mesh, axis=1))

        h = float(spacings[0])
        weights = np.zeros(mesh.size, dtype=float)
        if mesh.size % 2 == 1:
            weights[0] = h / 3.0
            weights[-1] = h / 3.0
            if mesh.size > 2:
                weights[1:-1:2] = 4.0 * h / 3.0
                weights[2:-1:2] = 2.0 * h / 3.0
            return weights

        weights[0] = h / 3.0
        if mesh.size > 3:
            weights[1:mesh.size - 3:2] = 4.0 * h / 3.0
            weights[2:mesh.size - 3:2] = 2.0 * h / 3.0
        weights[-3] = 5.0 * h / 4.0
        weights[-2] = h
        weights[-1] = 5.0 * h / 12.0
        return weights

    def compute_gram_matrix_fast(self) -> np.ndarray:
        """Assemble the Gram matrix from cached kernel values on one mesh."""
        if not (self.cache_kernels and self.integration.is_fixed_grid):
            return self.compute_gram_matrix()

        xs = self._get_or_build_mesh()
        weights = self._build_quadrature_weights(xs)
        kernel_matrix = self._build_kernel_matrix(xs)
        return (kernel_matrix * weights[np.newaxis, :]) @ kernel_matrix.T

    def _compute_cross_gram_matrix_slow(
        self,
        other: 'SOLAOperator',
    ) -> np.ndarray:
        """Assemble a cross-Gram matrix via pairwise quadrature."""
        cross_gram = np.zeros((self.N_d, other.N_d))
        domain = self._domain.function_domain
        method = self.integration.method
        n_points = max(self.integration.n_points, other.integration.n_points)

        for i in range(self.N_d):
            kernel_i = self.get_kernel(i)
            for j in range(other.N_d):
                kernel_j = other.get_kernel(j)

                intersected_support = Function._intersect_supports(
                    kernel_i.support,
                    kernel_j.support,
                )
                if intersected_support == []:
                    continue

                def product_callable(x, _ki=kernel_i, _kj=kernel_j):
                    return _ki.evaluate(x) * _kj.evaluate(x)

                cross_gram[i, j] = domain.integrate(
                    product_callable,
                    method=method,
                    support=intersected_support,
                    n_points=n_points,
                )

        return cross_gram

    def compute_cross_gram_matrix(self, other: 'SOLAOperator') -> np.ndarray:
        """Assemble the reduced cross-Gram matrix $T G^*$ on a common mesh.

        Both operators must use the same fixed-grid integration method and
        n_points for the fast path. If configurations differ, falls back to
        the slow pairwise quadrature path to preserve semantic agreement.
        """
        if not isinstance(other, SOLAOperator):
            raise TypeError(
                "Cross-Gram assembly requires another SOLAOperator instance."
            )
        if self._domain.function_domain != other._domain.function_domain:
            raise ValueError(
                "Cross-Gram assembly requires SOLA operators on the same "
                "domain."
            )
        if not (
            self.integration.is_fixed_grid
            and other.integration.is_fixed_grid
        ):
            return self._compute_cross_gram_matrix_slow(other)

        # Require matching configs for the fast path to preserve semantics
        if (
            self.integration.method != other.integration.method
            or self.integration.n_points != other.integration.n_points
        ):
            return self._compute_cross_gram_matrix_slow(other)

        xs = self._get_or_build_mesh()
        weights = self._build_quadrature_weights(xs, method=self.integration.method)
        left_kernel_matrix = self._build_kernel_matrix(xs)
        right_kernel_matrix = other._build_kernel_matrix(xs)
        return (
            left_kernel_matrix * weights[np.newaxis, :]
        ) @ right_kernel_matrix.T

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

    def clear_mesh_cache(self):
        """Clear the kernel mesh evaluation cache.

        Forces re-evaluation of all kernel functions on the shared mesh at
        the next forward call.  The shared mesh array (xs) itself is **not**
        cleared because it depends only on immutable construction parameters
        (domain bounds and ``n_points``) and never needs rebuilding.

        Use this when kernel callables may have changed since the last call
        while the operator object is reused across different workloads.  Note
        that ``clear_cache()`` also calls this method, so clearing the kernel
        object cache automatically invalidates mesh evaluations too.

        Has no effect when ``cache_kernels=False``.
        """
        if self._kernel_eval_cache is not None:
            self._kernel_eval_cache.clear()

    def clear_cache(self):
        """Clear the kernel object cache and kernel mesh evaluation cache.

        After this call, the next ``G(f)`` invocation re-fetches all kernels
        from the provider and re-evaluates them on the shared mesh, restoring
        a fully fresh state.  The shared mesh array itself is preserved.
        """
        if self.cache_kernels and self._kernels_cache is not None:
            self._kernels_cache.clear()
        self.clear_mesh_cache()

    def get_cache_info(self) -> dict:
        """
        Get information about the kernel and mesh caches.

        Returns
        -------
        dict
            Cache statistics.  Keys present in all cases:

            - ``caching_enabled``: whether ``cache_kernels=True``.
            - ``shared_mesh_built``: whether the shared fixed-grid mesh has
              been constructed (happens on the first fixed-grid forward call).

            Additional keys when ``caching_enabled`` is ``True``:

            - ``cached_functions``: number of kernel ``Function`` objects
              held in the kernel object cache.
            - ``total_functions``: ``N_d`` (total number of kernels).
            - ``cache_coverage``: fraction of kernels cached as objects.
            - ``kernel_eval_cache_entries``: number of kernel mesh evaluations
              currently stored (Phase 5).  Each entry is an ndarray of shape
              ``(n_points,)`` for one full-domain kernel.
        """
        info: dict = {
            "caching_enabled": self.cache_kernels,
            "shared_mesh_built": self._shared_mesh is not None,
        }
        if not self.cache_kernels:
            return info

        assert self._kernels_cache is not None
        assert self._kernel_eval_cache is not None
        info.update({
            "cached_functions": len(self._kernels_cache),
            "total_functions": self.N_d,
            "cache_coverage": len(self._kernels_cache) / self.N_d,
            "kernel_eval_cache_entries": len(self._kernel_eval_cache),
        })
        return info

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
