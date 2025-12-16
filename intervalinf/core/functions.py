"""
Functions on interval domains.

This module provides the Function class that represents functions living
in function spaces on IntervalDomain.
"""

from __future__ import annotations

import numbers
import operator
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from intervalinf.core.domain import IntervalDomain

# Protocol for space objects - they need a function_domain property
# This allows Function to work with any space that has this interface


class Function:
    """
    A function in a function space with evaluation and arithmetic support.

    This class represents a function that knows about the space it belongs to.
    Functions can be defined via callable rules or basis representations
    (coefficients).

    Parameters
    ----------
    space : object
        The function space this function belongs to. Must have a `function_domain`
        or `_function_domain` attribute returning an IntervalDomain.
    coefficients : ndarray, optional
        Finite-dimensional coefficient representation.
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

    Examples
    --------
    >>> from intervalinf.core import IntervalDomain, Function
    >>> # Need a space-like object (or use a real space from intervalinf.spaces)
    >>> domain = IntervalDomain(0, 1)
    >>> class SimpleSpace:
    ...     def __init__(self, domain):
    ...         self._function_domain = domain
    ...         self.basis_functions = None
    >>> space = SimpleSpace(domain)
    >>> f = Function(space, evaluate_callable=lambda x: x**2)
    >>> f(0.5)
    0.25
    """

    def __init__(
        self,
        space,
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
                (
                    "Exactly one of 'coefficients' or 'evaluate_callable' "
                    "must be provided."
                )
            )

        self.space = space
        self.name = name

        # Support specification - list of disjoint intervals
        self.support = self._check_support(support)

        # Function representation
        self.coefficients = (
            coefficients.copy() if coefficients is not None else None
        )
        self.evaluate_callable = evaluate_callable

    @property
    def function_domain(self) -> "IntervalDomain":
        """Get the IntervalDomain from the space."""
        # Support both _function_domain and function_domain attributes
        if hasattr(self.space, "_function_domain"):
            return self.space._function_domain
        return self.space.function_domain

    @property
    def has_compact_support(self) -> bool:
        """Check if function has compact support specified."""
        return self.support is not None

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
            Whether to check domain membership. Defaults to True if space exists.

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
                raise ValueError(f"Some points not in domain {self.function_domain}")

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
                            result[inside_support] = self.evaluate_callable(x_inside)
                        elif self.coefficients is not None:
                            result[inside_support] = self._evaluate_from_coefficients(
                                x_inside
                            )
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
        """Create a copy of this function."""
        if self.coefficients is not None:
            return self.__class__(
                self.space,
                coefficients=self.coefficients.copy(),
                name=self.name,
                support=self.support,
            )
        else:
            return self.__class__(
                self.space,
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
            raise RuntimeError(
                "Coefficients or basis functions not available for evaluation. "
                "The space may not have a basis defined."
            )
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
            [bf.evaluate(x_array, check_domain=False) for bf in basis_functions]
        )
        # Linear combination
        result = np.tensordot(coeffs, basis_evals, axes=([0], [0]))
        return result[0] if is_scalar else result

    def _is_zero_at(self, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Check if function is zero at point(s) due to compact support."""
        if not self.has_compact_support:
            return False

        x_array = np.asarray(x)
        is_scalar = x_array.ndim == 0

        outside_support = np.ones_like(x_array, dtype=bool)
        for support_a, support_b in self.support:
            inside_interval = (x_array >= support_a) & (x_array <= support_b)
            outside_support = outside_support & (~inside_interval)

        return outside_support.item() if is_scalar else outside_support

    @staticmethod
    def _union_supports(support1, support2):
        """Compute union of two support specifications."""
        if support1 is None and support2 is None:
            return None
        if support1 is None:
            return support2
        if support2 is None:
            return support1

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
                    raise ValueError(f"Support interval {i} must be a tuple (a, b)")
                if interval[0] >= interval[1]:
                    raise ValueError(
                        f"Support interval {i}: a={interval[0]} must be < b={interval[1]}"
                    )
            support = sorted(support, key=lambda x: x[0])
            for i in range(len(support) - 1):
                if support[i][1] > support[i + 1][0]:
                    raise ValueError(
                        f"Support intervals {support[i]} and {support[i+1]} overlap"
                    )
        else:
            raise ValueError(
                "Support must be a tuple (a, b) or list of tuples [(a1, b1), ...]"
            )

        # Validate against domain
        domain = self.function_domain
        for i, (a, b) in enumerate(support):
            if not (domain.a <= a < b <= domain.b):
                raise ValueError(
                    f"Support interval {i}: ({a}, {b}) must be within "
                    f"domain [{domain.a}, {domain.b}]"
                )

        return support

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
                new_support = self._intersect_supports(self.support, other.support)
            else:
                new_support = None

            # Use coefficient arithmetic only for linear ops with matching coefficients
            if (
                is_linear
                and self.coefficients is not None
                and other.coefficients is not None
                and len(self.coefficients) == len(other.coefficients)
            ):
                new_coeffs = op(self.coefficients, other.coefficients)
                return self.__class__(
                    self.space, coefficients=new_coeffs, support=new_support
                )
            else:
                # Create callable-based result
                def _get_eval_callable(fn):
                    if getattr(fn, "evaluate_callable", None) is not None:
                        return fn.evaluate_callable
                    if getattr(fn, "coefficients", None) is not None:
                        return lambda x: fn._evaluate_from_coefficients(x)
                    return lambda x: fn.evaluate(x, check_domain=False)

                eval_self = _get_eval_callable(self)
                eval_other = _get_eval_callable(other)

                def op_callable(x):
                    return op(eval_self(x), eval_other(x))

                return self.__class__(
                    self.space, evaluate_callable=op_callable, support=new_support
                )

        elif scalar_allowed and isinstance(other, numbers.Number):

            def scalar_op_callable(x):
                return op(self.evaluate(x), other)

            return self.__class__(
                self.space, evaluate_callable=scalar_op_callable, support=self.support
            )
        else:
            raise TypeError(
                f"Cannot {op_name} Function with {type(other).__name__}. "
                f"Expected Function or scalar."
            )

    def __add__(self, other):
        return self._binary_op(other, operator.add, "add", support_strategy="union")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub, "subtract", support_strategy="union")

    def __rsub__(self, other):
        # other - self
        if isinstance(other, numbers.Number):

            def rsub_callable(x):
                return other - self.evaluate(x)

            return self.__class__(
                self.space, evaluate_callable=rsub_callable, support=self.support
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

                def zero_callable(x):
                    return np.zeros_like(np.asarray(x), dtype=float)

                return Function(self.space, evaluate_callable=zero_callable)

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
        if self.coefficients is not None:
            return self.__class__(
                self.space,
                coefficients=-self.coefficients,
                name=f"-{self.name}" if self.name else None,
                support=self.support,
            )
        else:

            def neg_callable(x):
                return -self.evaluate_callable(x)

            return self.__class__(
                self.space,
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

        if not (orig_domain.a <= rest_domain.a and rest_domain.b <= orig_domain.b):
            raise ValueError(
                f"Restricted domain {rest_domain} is not a subset of "
                f"original domain {orig_domain}"
            )

        return Function(
            restricted_space,
            evaluate_callable=self.evaluate_callable,
            name=f"{self.name}_restricted" if self.name else None,
            support=None,
        )

    def __repr__(self) -> str:
        return f"Function(domain={self.function_domain}, name={self.name})"
