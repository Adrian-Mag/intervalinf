"""
Interval domain specification for function spaces.

This module provides a minimal IntervalDomain class with meshing and
integration capabilities.
"""

from typing import Callable, Optional, Tuple, Union
import math
import numpy as np


class IntervalDomain:
    """
    A 1D interval domain with meshing and integration support.

    Represents an interval [a, b] with optional boundary types (open, closed,
    semi-open). Provides methods for meshing, integration, and domain operations.

    Parameters
    ----------
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval.
    boundary_type : str, optional
        Type of interval boundaries. One of:
        - 'closed' (default): [a, b]
        - 'open': (a, b)
        - 'left_open': (a, b]
        - 'right_open': [a, b)
    name : str, optional
        Human-readable name for the domain.
    open_epsilon : float, optional
        For open/semi-open intervals, the distance from boundaries for mesh
        points. Defaults to 0.1% of interval length.

    Examples
    --------
    >>> domain = IntervalDomain(0, 1)
    >>> domain.length
    1.0
    >>> domain.uniform_mesh(5)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """

    def __init__(
        self,
        a: float,
        b: float,
        *,
        boundary_type: str = "closed",
        name: Optional[str] = None,
        open_epsilon: Optional[float] = None,
    ):
        if a >= b:
            raise ValueError(f"Left endpoint a={a} must be less than right endpoint b={b}")
        self.a = float(a)
        self.b = float(b)
        self.boundary_type = boundary_type

        # open_epsilon for open/semi-open intervals: shifts mesh points inward
        if open_epsilon is None:
            # Default: shift by 0.1% of interval length from each boundary
            self.open_epsilon = 0.001 * (b - a)
        else:
            self.open_epsilon = float(open_epsilon)
            if self.open_epsilon < 0:
                raise ValueError("open_epsilon must be non-negative")
            if self.open_epsilon >= 0.5 * (b - a):
                raise ValueError(
                    f"open_epsilon={open_epsilon} too large for interval of length {b - a}"
                )

        # name defaults to the string representation
        if name is None:
            self.name = self._format_name()
        else:
            self.name = name

    @property
    def length(self) -> float:
        """Length of the interval."""
        return self.b - self.a

    @property
    def center(self) -> float:
        """Center point of the interval."""
        return 0.5 * (self.a + self.b)

    @property
    def radius(self) -> float:
        """Half-length of the interval."""
        return 0.5 * self.length

    def contains(self, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Check if point(s) are in the domain.

        Parameters
        ----------
        x : float or array-like
            Point(s) to check.

        Returns
        -------
        bool or ndarray
            True where points are in the domain.
        """
        if self.boundary_type == "closed":
            return (x >= self.a) & (x <= self.b)
        if self.boundary_type == "open":
            return (x > self.a) & (x < self.b)
        if self.boundary_type == "left_open":
            return (x > self.a) & (x <= self.b)
        if self.boundary_type == "right_open":
            return (x >= self.a) & (x < self.b)
        raise ValueError(f"Unknown boundary_type: {self.boundary_type}")

    def uniform_mesh(self, n: int) -> np.ndarray:
        """
        Generate uniform mesh respecting boundary type.

        For open/semi-open intervals, generates n points on a slightly
        contracted interval [a+ε, b-ε] or variants, where ε = self.open_epsilon.

        Parameters
        ----------
        n : int
            Number of mesh points.

        Returns
        -------
        ndarray
            Array of n uniformly spaced points.
        """
        if self.boundary_type == "closed":
            return np.linspace(self.a, self.b, n, endpoint=True)
        if self.boundary_type == "open":
            return np.linspace(self.a + self.open_epsilon, self.b - self.open_epsilon, n)
        if self.boundary_type == "left_open":
            return np.linspace(self.a + self.open_epsilon, self.b, n)
        if self.boundary_type == "right_open":
            return np.linspace(self.a, self.b - self.open_epsilon, n)
        raise ValueError(f"Unknown boundary_type: {self.boundary_type}")

    def interior(self) -> "IntervalDomain":
        """Return the interior (open) version of this domain."""
        return IntervalDomain(
            self.a, self.b, boundary_type="open", open_epsilon=self.open_epsilon
        )

    def closure(self) -> "IntervalDomain":
        """Return the closure (closed) version of this domain."""
        return IntervalDomain(
            self.a, self.b, boundary_type="closed", open_epsilon=self.open_epsilon
        )

    def boundary_points(self) -> Tuple[float, float]:
        """Return the boundary points (a, b)."""
        return (self.a, self.b)

    def integrate(
        self,
        f: Callable,
        method: str = "simpson",
        support: Optional[Union[Tuple[float, float], list]] = None,
        n_points: int = 100,
        *,
        vectorized: Optional[bool] = None,
        **kwargs,
    ) -> float:
        """
        Integrate a function over the domain or a subdomain.

        Parameters
        ----------
        f : callable
            Function to integrate.
        method : str, optional
            Integration method: 'simpson', 'trapz', or 'adaptive'.
        support : tuple or list of tuples, optional
            Subdomain(s) for integration. If None, integrates over entire domain.
        n_points : int, optional
            Number of quadrature points.
        vectorized : bool, optional
            Whether f is vectorized. If None, auto-detects.
        **kwargs
            Additional arguments passed to scipy.integrate.quad for 'adaptive'.

        Returns
        -------
        float
            Integral value.
        """
        # Handle multiple support intervals
        if (
            support is not None
            and hasattr(support, "__iter__")
            and not (
                isinstance(support, (tuple, list))
                and len(support) == 2
                and all(isinstance(x, (int, float)) for x in support)
            )
        ):
            subintervals = list(support)
            if len(subintervals) == 0:
                return 0.0

            # Validate and compute lengths
            lengths = []
            for sub in subintervals:
                if not (isinstance(sub, (tuple, list)) and len(sub) == 2):
                    raise ValueError("Each support entry must be a (a, b) pair")
                a_sub, b_sub = float(sub[0]), float(sub[1])
                if not (self.a <= a_sub < b_sub <= self.b):
                    raise ValueError(
                        f"Support interval ({a_sub}, {b_sub}) outside domain [{self.a}, {self.b}]"
                    )
                lengths.append(b_sub - a_sub)

            total_length = sum(lengths)
            if total_length <= 0:
                return 0.0

            # Allocate points proportionally
            n_sub = len(subintervals)
            effective_total = max(n_points, 3 * n_sub)
            raw = [effective_total * (L / total_length) for L in lengths]
            alloc = [max(3, int(math.floor(r))) for r in raw]
            allocated = sum(alloc)

            # Distribute remaining points
            remainder = effective_total - allocated
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

            # Integrate over each subinterval
            total = 0.0
            for sub, n_i in zip(subintervals, alloc):
                total += self.integrate(
                    f, method=method, support=sub, n_points=n_i,
                    vectorized=vectorized, **kwargs
                )
            return float(total)

        # Single interval integration
        if support is None:
            xs = self.uniform_mesh(max(3, n_points))
        else:
            a, b = support
            if not (self.a <= a < b <= self.b):
                raise ValueError(
                    f"Support ({a}, {b}) outside domain [{self.a}, {self.b}]"
                )
            xs = np.linspace(a, b, max(3, n_points))

        def eval_mesh(xs_vals: np.ndarray) -> np.ndarray:
            if vectorized is True:
                return np.asarray(f(xs_vals))
            if vectorized is False:
                return np.fromiter(
                    (f(x) for x in xs_vals), dtype=float, count=xs_vals.size
                )
            try:
                out = f(xs_vals)
                arr = np.asarray(out)
                if arr.shape == ():
                    raise ValueError("Function returned scalar for array input")
                return arr
            except Exception:
                return np.fromiter(
                    (f(x) for x in xs_vals), dtype=float, count=xs_vals.size
                )

        ys = eval_mesh(xs)

        if method == "simpson":
            from scipy.integrate import simpson
            return float(simpson(ys, x=xs))

        if method == "trapz":
            try:
                from scipy.integrate import trapezoid as trapz
            except ImportError:
                from scipy.integrate import trapz
            return float(trapz(ys, x=xs))

        if method == "adaptive":
            from scipy.integrate import quad
            a_int = self.a if support is None else support[0]
            b_int = self.b if support is None else support[1]
            return float(quad(f, a_int, b_int, **kwargs)[0])

        raise ValueError(f"Unknown integration method: {method}")

    def restriction_to_subinterval(self, a: float, b: float) -> "IntervalDomain":
        """
        Create a subdomain restricted to [a, b].

        Parameters
        ----------
        a, b : float
            Subinterval endpoints.

        Returns
        -------
        IntervalDomain
            New domain restricted to [a, b].
        """
        if a >= b:
            raise ValueError(f"Invalid subinterval: a={a} >= b={b}")
        if not (self.a <= a and b <= self.b):
            raise ValueError(
                f"Subinterval [{a}, {b}] outside domain [{self.a}, {self.b}]"
            )
        return IntervalDomain(
            a, b, boundary_type=self.boundary_type, open_epsilon=self.open_epsilon
        )

    def split_at_discontinuities(self, discontinuity_points: list) -> list:
        """
        Split the domain at specified discontinuity points.

        Parameters
        ----------
        discontinuity_points : list
            Points where discontinuities occur. Must be strictly increasing
            and lie within (a, b).

        Returns
        -------
        list of IntervalDomain
            Subdomains separated by discontinuity points.
        """
        if not discontinuity_points:
            return [self]

        disc_points = sorted(discontinuity_points)

        # Validate points are in interior
        for point in disc_points:
            if not (self.a < point < self.b):
                raise ValueError(
                    f"Discontinuity point {point} must be in interior ({self.a}, {self.b})"
                )

        # Check for duplicates
        if len(disc_points) != len(set(disc_points)):
            raise ValueError("Discontinuity points must be unique")

        # Build subdomains
        subdomains = []
        boundaries = [self.a] + disc_points + [self.b]

        for i in range(len(boundaries) - 1):
            a_sub = boundaries[i]
            b_sub = boundaries[i + 1]

            # Determine boundary types
            if i == 0:
                if self.boundary_type in ["closed", "left_open"]:
                    bt = "left_open" if self.boundary_type == "left_open" else "right_open"
                else:
                    bt = "open"
            elif i == len(boundaries) - 2:
                bt = "left_open"
            else:
                bt = "open"

            subdomains.append(
                IntervalDomain(a_sub, b_sub, boundary_type=bt, open_epsilon=self.open_epsilon)
            )

        return subdomains

    def _format_name(self) -> str:
        """Format interval as string."""
        if self.boundary_type == "closed":
            return f"[{self.a}, {self.b}]"
        if self.boundary_type == "open":
            return f"({self.a}, {self.b})"
        if self.boundary_type == "left_open":
            return f"({self.a}, {self.b}]"
        if self.boundary_type == "right_open":
            return f"[{self.a}, {self.b})"
        return f"[{self.a}, {self.b}]"

    def __repr__(self) -> str:
        return self._format_name()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalDomain):
            return False
        return (
            self.a == other.a
            and self.b == other.b
            and self.boundary_type == other.boundary_type
        )
