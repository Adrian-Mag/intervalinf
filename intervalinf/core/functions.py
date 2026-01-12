"""
Functions on interval domains.

This module provides the Function class that represents functions living
in function spaces on IntervalDomain, or as standalone functions on a domain.

Functions can be created in two modes:
1. **Attached to a space**: Traditional mode where the function knows its
   Hilbert space context. Required for coefficient-based representations.
2. **Standalone on domain**: Function defined only on an IntervalDomain,
   without a space. Useful for defining basis functions before a space
   exists, or for mathematical functions that don't need space context.

Standalone functions can be attached to a space later via `attach_to_space()`.
"""

from __future__ import annotations

import numbers
import operator
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from intervalinf.core.domain import IntervalDomain


def _is_interval_domain(obj) -> bool:
    """Check if object is an IntervalDomain (avoids circular import)."""
    return type(obj).__name__ == 'IntervalDomain'


def _get_domain_from_space(space) -> 'IntervalDomain':
    """Extract function_domain from a space object."""
    if hasattr(space, "_function_domain"):
        return space._function_domain
    if hasattr(space, "function_domain"):
        return space.function_domain
    raise AttributeError(
        f"Space {type(space).__name__} has no function_domain attribute"
    )


class Function:
    """
    A function on an interval domain, optionally attached to a function space.

    Functions can be created in two modes:

    1. **Attached mode** (traditional): Pass a function space as the first
       argument. The function is tied to that space and can use coefficient
       representations if the space has a basis.

    2. **Standalone mode** (new): Pass an IntervalDomain as the first argument.
       The function exists independently of any space. Use `attach_to_space()`
       to later bind it to a space.

    Parameters
    ----------
    space_or_domain : object
        Either a function space (with `function_domain` attribute) or an
        IntervalDomain directly. If IntervalDomain, creates a standalone
        function.
    coefficients : ndarray, optional
        Finite-dimensional coefficient representation. Only valid when
        attached to a space with a basis.
    evaluate_callable : callable, optional
        Callable defining the function rule f(x).
    name : str, optional
        Human-readable name for the function.
    support : tuple or list, optional
        Compact support specification:
        - tuple (a, b): single interval where function is nonzero
        - list of tuples: multiple disjoint intervals
        - None: function has support over entire domain

    Notes
    -----
    Exactly one of `coefficients` or `evaluate_callable` must be provided.
    Coefficient-based functions require attachment to a space with a basis.

    Examples
    --------
    >>> from intervalinf.core import IntervalDomain, Function

    # Standalone function on domain (new mode)
    >>> domain = IntervalDomain(0, 1)
    >>> f = Function(domain, evaluate_callable=lambda x: x**2)
    >>> f(0.5)
    0.25
    >>> f.is_attached
    False

    # Attach to a space later
    >>> from intervalinf.spaces import Lebesgue
    >>> space = Lebesgue(50, domain, basis='fourier')
    >>> f_attached = f.attach_to_space(space)
    >>> f_attached.is_attached
    True

    # Traditional mode with space
    >>> g = Function(space, evaluate_callable=lambda x: np.sin(np.pi * x))
    >>> g.is_attached
    True
    """

    def __init__(
        self,
        space_or_domain,
        *,
        coefficients: Optional[np.ndarray] = None,
        evaluate_callable: Optional[Callable] = None,
        name: Optional[str] = None,
        support: Optional[Union[tuple, list]] = None,
    ):
        # Validate: exactly one of coefficients or evaluate_callable
        if (coefficients is None and evaluate_callable is None) or (
            coefficients is not None and evaluate_callable is not None
        ):
            raise ValueError(
                "Exactly one of 'coefficients' or 'evaluate_callable' "
                "must be provided."
            )

        # Determine if we have a space or just a domain
        if _is_interval_domain(space_or_domain):
            # Standalone mode: function on domain only
            self._space = None
            self._domain = space_or_domain
            if coefficients is not None:
                raise ValueError(
                    "Coefficient-based functions require a space with a "
                    "basis. Pass a space instead of a domain, or use "
                    "evaluate_callable for standalone functions."
                )
        else:
            # Attached mode: function belongs to a space
            self._space = space_or_domain
            self._domain = None  # Will be derived from space

        self.name = name

        # Support specification - list of disjoint intervals
        self.support = self._check_support(support)

        # Function representation
        self.coefficients = (
            coefficients.copy() if coefficients is not None else None
        )
        self.evaluate_callable = evaluate_callable

    # ================================================================
    # Space/Domain Properties
    # ================================================================

    @property
    def space(self):
        """
        The function space this function belongs to (None if standalone).

        For backward compatibility, this property is readable. Functions
        created on a domain only will return None.
        """
        return self._space

    @space.setter
    def space(self, value):
        """Set space (for backward compatibility during transition)."""
        if _is_interval_domain(value):
            self._space = None
            self._domain = value
        else:
            self._space = value
            self._domain = None

    @property
    def function_domain(self) -> "IntervalDomain":
        """
        The IntervalDomain on which this function is defined.

        Always available, whether function is attached to a space or not.
        """
        if self._domain is not None:
            return self._domain
        if self._space is not None:
            return _get_domain_from_space(self._space)
        raise RuntimeError("Function has neither space nor domain set")

    @property
    def is_attached(self) -> bool:
        """
        Whether this function is attached to a function space.

        Returns True if the function was created with a space, False if
        created with just a domain (standalone mode).
        """
        return self._space is not None

    @property
    def has_compact_support(self) -> bool:
        """Check if function has compact support specified."""
        return self.support is not None

    # ================================================================
    # Attachment Methods
    # ================================================================

    def attach_to_space(self, space, *, copy: bool = True) -> 'Function':
        """
        Create a copy of this function attached to a function space.

        This allows standalone functions (created on a domain) to be
        associated with a Hilbert space for operations that require
        space context (e.g., computing coefficients).

        Parameters
        ----------
        space : HilbertSpace
            The function space to attach to. Must have a compatible
            function_domain.
        copy : bool, default True
            If True, returns a new Function. If False, modifies this
            function in-place (use with caution).

        Returns
        -------
        Function
            A function attached to the given space.

        Raises
        ------
        ValueError
            If the space's domain is incompatible with this function's
            domain.

        Examples
        --------
        >>> domain = IntervalDomain(0, 1)
        >>> f = Function(domain, evaluate_callable=lambda x: x**2)
        >>> space = Lebesgue(50, domain, basis='fourier')
        >>> f_in_space = f.attach_to_space(space)
        >>> f_in_space.is_attached
        True
        """
        space_domain = _get_domain_from_space(space)

        # Check domain compatibility
        my_domain = self.function_domain
        if not (space_domain.a == my_domain.a and space_domain.b == my_domain.b):
            raise ValueError(
                f"Domain mismatch: function domain [{my_domain.a}, {my_domain.b}] "
                f"!= space domain [{space_domain.a}, {space_domain.b}]"
            )

        if copy:
            return Function(
                space,
                coefficients=self.coefficients,
                evaluate_callable=self.evaluate_callable,
                name=self.name,
                support=self.support,
            )
        else:
            self._space = space
            self._domain = None
            return self

    def detach(self, *, copy: bool = True) -> 'Function':
        """
        Create a standalone copy of this function (detached from space).

        Useful when you want to pass a function to code that doesn't
        need space context, or to break the reference to a space.

        Parameters
        ----------
        copy : bool, default True
            If True, returns a new Function. If False, modifies this
            function in-place.

        Returns
        -------
        Function
            A standalone function on the domain.

        Raises
        ------
        ValueError
            If function has coefficients (cannot detach coefficient-based
            functions as they require a space with basis).
        """
        if self.coefficients is not None:
            raise ValueError(
                "Cannot detach coefficient-based function. Convert to "
                "callable first using to_callable()."
            )

        domain = self.function_domain

        if copy:
            return Function(
                domain,
                evaluate_callable=self.evaluate_callable,
                name=self.name,
                support=self.support,
            )
        else:
            self._domain = domain
            self._space = None
            return self

    def evaluate(
        self, x: Union[float, np.ndarray], check_domain: Optional[bool] = None
    ) -> Union[float, np.ndarray]:
        """
        Evaluate the function at point(s) x.

        Parameters
        ----------
        x : float or array-like
            Point(s) at which to evaluate.
        check_domain : bool, optional
            Whether to check domain membership.
            Defaults to True if space exists.

        Returns
        -------
        float or ndarray
            Function value(s) at x.

        Raises
        ------
        ValueError
            If check_domain is True and some points are outside the domain.
        """
        if check_domain is None:
            check_domain = self.space is not None

        if check_domain and self.space is not None:
            x_array = np.asarray(x)
            if not np.all(self.function_domain.contains(x_array)):
                msg = f"Some points not in domain {self.function_domain}"
                raise ValueError(msg)

        # Handle compact support
        if self.has_compact_support:
            is_zero = self._is_zero_at(x)
            if np.any(is_zero):
                x_array = np.asarray(x)
                zeros = np.zeros_like(x_array, dtype=float)

                if np.all(is_zero):
                    return zeros.item() if x_array.ndim == 0 else zeros

                # Evaluate only inside support
                if x_array.ndim > 0:
                    result = zeros.copy()
                    inside_support = ~np.asarray(is_zero)
                    if np.any(inside_support):
                        x_inside = x_array[inside_support]
                        if self.evaluate_callable is not None:
                            tmp = self.evaluate_callable(x_inside)
                            result[inside_support] = tmp
                        elif self.coefficients is not None:
                            tmp = self._evaluate_from_coefficients(x_inside)
                            result[inside_support] = tmp
                    return result

        # Standard evaluation
        if self.evaluate_callable is not None:
            return self.evaluate_callable(x)
        elif self.coefficients is not None:
            return self._evaluate_from_coefficients(x)
        else:
            raise RuntimeError("No evaluation method available")

    def integrate(
        self,
        weight: Optional[Callable] = None,
        method: str = "simpson",
        n_points: int = 1000,
        *,
        vectorized: Optional[bool] = None,
    ) -> float:
        """
        Integrate function over its domain: ∫[a,b] f(x) w(x) dx.

        Parameters
        ----------
        weight : callable, optional
            Weight function w(x).
        method : str, optional
            Integration method ('simpson', 'trapz', 'adaptive').
        n_points : int, optional
            Number of quadrature points.
        vectorized : bool, optional
            Whether the callable is vectorized.

        Returns
        -------
        float
            Integral value.
        """
        domain = self.function_domain

        if self.has_compact_support:
            support = self.support
        else:
            support = None

        if weight is None:
            if self.evaluate_callable is not None:
                return domain.integrate(
                    self.evaluate_callable,
                    method=method,
                    support=support,
                    n_points=n_points,
                    vectorized=vectorized,
                )
            elif self.coefficients is not None:

                def integrand(x):
                    return self.evaluate(x, check_domain=False)

                return domain.integrate(
                    integrand,
                    method=method,
                    support=support,
                    n_points=n_points,
                    vectorized=vectorized,
                )
            else:
                return 0.0
        else:

            def weighted_integrand(x):
                return self.evaluate(x, check_domain=False) * weight(x)

            return domain.integrate(
                weighted_integrand,
                method=method,
                support=support,
                n_points=n_points,
                vectorized=vectorized,
            )

    def plot(
        self,
        n_points: int = 1000,
        figsize: tuple = (10, 6),
        use_seaborn: bool = True,
        **kwargs,
    ):
        """
        Plot the function.

        Parameters
        ----------
        n_points : int, optional
            Number of plot points.
        figsize : tuple, optional
            Figure size as (width, height).
        use_seaborn : bool, optional
            Whether to use seaborn styling.
        **kwargs
            Additional matplotlib plot arguments.

        Returns
        -------
        Axes
            Matplotlib axes object.
        """
        import matplotlib.pyplot as plt

        if use_seaborn:
            try:
                import seaborn as sns

                if "color" not in kwargs:
                    palette = sns.color_palette("muted", 1)
                    kwargs["color"] = palette[0]
                if "linewidth" not in kwargs:
                    kwargs["linewidth"] = 2
            except ImportError:
                pass

        x = self.function_domain.uniform_mesh(n_points)
        y = self.evaluate(x)

        plt.figure(figsize=figsize)
        plt.plot(x, y, label=self.name or "Function", **kwargs)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"Function on {self.function_domain}")
        if self.name:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()

    def copy(self) -> "Function":
        """Create a copy of this function, preserving space/domain context."""
        # Determine what to pass as first arg: space or domain
        space_or_domain = self._space if self._space is not None else self._domain

        if self.coefficients is not None:
            return self.__class__(
                space_or_domain,
                coefficients=self.coefficients.copy(),
                name=self.name,
                support=self.support,
            )
        else:
            return self.__class__(
                space_or_domain,
                evaluate_callable=self.evaluate_callable,
                name=self.name,
                support=self.support,
            )

    def _evaluate_from_coefficients(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate using basis expansion: f(x) = Σ c_k φ_k(x)."""
        basis_functions = getattr(self.space, "basis_functions", None)
        coeffs = self.coefficients

        if coeffs is None or basis_functions is None:
            msg = (
                "Coefficients or basis functions not available "
                "for evaluation. The space may not have a basis defined."
            )
            raise RuntimeError(msg)
        if len(coeffs) != len(basis_functions):
            raise ValueError(
                f"Coefficient length {len(coeffs)} does not match "
                f"number of basis functions {len(basis_functions)}."
            )

        x_array = np.asarray(x)
        is_scalar = x_array.ndim == 0
        if is_scalar:
            x_array = x_array.reshape(1)

        # Evaluate each basis function
        basis_evals = np.array(
            [
                bf.evaluate(x_array, check_domain=False)
                for bf in basis_functions
            ]
        )
        # Linear combination
        result = np.tensordot(coeffs, basis_evals, axes=([0], [0]))
        return result[0] if is_scalar else result

    def _is_zero_at(
        self, x: Union[float, np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """Check if function is zero at point(s) due to compact support."""
        # If no compact support is specified, function is not identically zero
        # anywhere
        if not self.has_compact_support or self.support is None:
            return False

        x_array = np.asarray(x)
        is_scalar = x_array.ndim == 0
        if is_scalar:
            # normalize scalar to 1D array for consistent boolean operations
            x_array = x_array.reshape(1)

        outside_support = np.ones_like(x_array, dtype=bool)
        for support_a, support_b in self.support:
            inside_interval = (x_array >= support_a) & (x_array <= support_b)
            outside_support = outside_support & (~inside_interval)

        return outside_support.item() if is_scalar else outside_support

    @staticmethod
    def _union_supports(support1, support2):
        """Compute union of two support specifications."""
        # `None` denotes "no compact support" i.e. the full domain.
        # The union with the full domain should be the full domain.
        if support1 is None or support2 is None:
            return None

        all_intervals = support1 + support2
        all_intervals.sort(key=lambda x: x[0])

        merged = []
        for start, end in all_intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    @staticmethod
    def _intersect_supports(support1, support2):
        """Compute intersection of two support specifications."""
        if support1 is None or support2 is None:
            return None

        intersections = []
        for a1, b1 in support1:
            for a2, b2 in support2:
                intersect_a = max(a1, a2)
                intersect_b = min(b1, b2)
                if intersect_a < intersect_b:
                    intersections.append((intersect_a, intersect_b))

        if not intersections:
            return []

        intersections.sort(key=lambda x: x[0])
        return intersections

    def _check_support(self, support):
        """Validate and normalize support specification."""
        if support is None:
            return None

        if isinstance(support, tuple):
            if len(support) != 2 or support[0] >= support[1]:
                raise ValueError("Support tuple must be (a, b) with a < b")
            support = [support]
        elif isinstance(support, list):
            for i, interval in enumerate(support):
                if not isinstance(interval, tuple) or len(interval) != 2:
                    msg = f"Support interval {i} must be a tuple (a, b)"
                    raise ValueError(msg)
                if interval[0] >= interval[1]:
                    msg = (
                        f"Support interval {i}: a={interval[0]} must be < "
                        f"b={interval[1]}"
                    )
                    raise ValueError(msg)
            support = sorted(support, key=lambda x: x[0])
            for i in range(len(support) - 1):
                if support[i][1] > support[i + 1][0]:
                    msg = (
                        f"Support intervals {support[i]} and "
                        f"{support[i+1]} overlap"
                    )
                    raise ValueError(msg)
        else:
            msg = (
                "Support must be a tuple (a, b) or list of tuples "
                "[(a1, b1), ...]"
            )
            raise ValueError(msg)

        # Validate against domain
        domain = self.function_domain
        for i, (a, b) in enumerate(support):
            if not (domain.a <= a < b <= domain.b):
                msg = (
                    f"Support interval {i}: ({a}, {b}) must be within "
                    f"domain [{domain.a}, {domain.b}]"
                )
                raise ValueError(msg)

        return support

    def __call__(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Allow f(x) syntax."""
        return self.evaluate(x, check_domain=None)

    def _binary_op(
        self,
        other,
        op,
        op_name: str,
        scalar_allowed: bool = True,
        support_strategy: str = "union",
        is_linear: bool = True,
    ) -> "Function":
        """General helper for binary operations."""
        if isinstance(other, Function):
            if support_strategy == "union":
                new_support = self._union_supports(self.support, other.support)
            elif support_strategy == "intersect":
                new_support = self._intersect_supports(
                    self.support, other.support
                )
            else:
                new_support = None

            # Determine result's space/domain context:
            # - If both have same space, use that space
            # - If one has a space, prefer that (result is attached)
            # - If neither has space, use domain from self
            if self._space is not None and other._space is not None:
                if self._space is other._space:
                    result_context = self._space
                else:
                    # Different spaces - fall back to domain
                    result_context = self.function_domain
            elif self._space is not None:
                result_context = self._space
            elif other._space is not None:
                result_context = other._space
            else:
                # Both standalone - use self's domain
                result_context = self.function_domain

            # Use coefficient arithmetic only for linear ops with
            # matching coefficients (requires same space)
            if (
                is_linear
                and self.coefficients is not None
                and other.coefficients is not None
                and len(self.coefficients) == len(other.coefficients)
                and self._space is not None
                and self._space is other._space
            ):
                new_coeffs = op(self.coefficients, other.coefficients)
                return self.__class__(
                    result_context, coefficients=new_coeffs, support=new_support
                )
            else:
                # Create callable-based result
                def _get_eval_callable(fn):
                    call = getattr(fn, "evaluate_callable", None)
                    if callable(call):
                        return call
                    coeffs = getattr(fn, "coefficients", None)
                    if coeffs is not None:
                        return lambda x: fn._evaluate_from_coefficients(x)
                    if hasattr(fn, "evaluate"):
                        return lambda x: fn.evaluate(x, check_domain=False)
                    raise RuntimeError(
                        "Cannot obtain a callable to evaluate the operand"
                    )

                eval_self = _get_eval_callable(self)
                eval_other = _get_eval_callable(other)

                def op_callable(x):
                    return op(eval_self(x), eval_other(x))

                return self.__class__(
                    result_context,
                    evaluate_callable=op_callable,
                    support=new_support,
                )

        elif scalar_allowed and isinstance(other, numbers.Number):
            # Preserve self's context (space or domain)
            result_context = self._space if self._space is not None else self._domain

            def scalar_op_callable(x):
                return op(self.evaluate(x), other)

            return self.__class__(
                result_context,
                evaluate_callable=scalar_op_callable,
                support=self.support,
            )
        else:
            msg = (
                f"Cannot {op_name} Function with {type(other).__name__}. "
                "Expected Function or scalar."
            )
            raise TypeError(msg)

    def __add__(self, other):
        return self._binary_op(
            other, operator.add, "add", support_strategy="union"
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(
            other, operator.sub, "subtract", support_strategy="union"
        )

    def __rsub__(self, other):
        # other - self
        if isinstance(other, numbers.Number):
            result_context = self._space if self._space is not None else self._domain

            def rsub_callable(x):
                # ensure numeric/array-compatible subtraction
                return np.asarray(other) - np.asarray(self.evaluate(x))

            return self.__class__(
                result_context,
                evaluate_callable=rsub_callable,
                support=self.support,
            )
        return NotImplemented

    def __mul__(self, other):
        # Special case: disjoint supports → zero function
        if (
            isinstance(other, Function)
            and self.has_compact_support
            and other.has_compact_support
        ):
            new_support = self._intersect_supports(self.support, other.support)
            if new_support is None or len(new_support) == 0:
                result_context = self._space if self._space is not None else self._domain

                def zero_callable(x):
                    return np.zeros_like(np.asarray(x), dtype=float)

                return Function(result_context, evaluate_callable=zero_callable)

        return self._binary_op(
            other,
            operator.mul,
            "multiply",
            support_strategy="intersect",
            is_linear=False,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        """Negation: -f."""
        result_context = self._space if self._space is not None else self._domain

        if self.coefficients is not None:
            return self.__class__(
                result_context,
                coefficients=-self.coefficients,
                name=f"-{self.name}" if self.name else None,
                support=self.support,
            )
        else:

            def neg_callable(x):
                return -self.evaluate(x, check_domain=False)

            return self.__class__(
                result_context,
                evaluate_callable=neg_callable,
                name=f"-{self.name}" if self.name else None,
                support=self.support,
            )

    def restrict(self, restricted_space) -> "Function":
        """
        Restrict this function to a subspace.

        Parameters
        ----------
        restricted_space : object
            Target space with a function_domain that is a subset of this
            function's domain.

        Returns
        -------
        Function
            New function on the restricted space.

        Raises
        ------
        ValueError
            If function is not evaluable or restricted domain is not a subset.
        """
        if self.evaluate_callable is None:
            raise ValueError(
                "Can only restrict functions with evaluate_callable. "
                "Coefficient-based functions cannot be easily restricted."
            )

        orig_domain = self.function_domain
        rest_domain = (
            restricted_space._function_domain
            if hasattr(restricted_space, "_function_domain")
            else restricted_space.function_domain
        )

        if not (
            orig_domain.a <= rest_domain.a and rest_domain.b <= orig_domain.b
        ):
            msg = (
                f"Restricted domain {rest_domain} is not a subset of "
                f"original domain {orig_domain}"
            )
            raise ValueError(msg)

        return Function(
            restricted_space,
            evaluate_callable=self.evaluate_callable,
            name=f"{self.name}_restricted" if self.name else None,
            support=None,
        )

    def __repr__(self) -> str:
        return f"Function(domain={self.function_domain}, name={self.name})"
