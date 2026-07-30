"""Explicit quadrature rules for interval-domain calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from intervalinf.core.domain import IntervalDomain


def canonicalize_breakpoints(
    domain: IntervalDomain,
    breakpoints: Iterable[float] | None,
) -> tuple[float, ...]:
    """Validate and canonically order interior quadrature split points.

    The returned tuple contains conservative integration metadata. Its members
    need not all be actual discontinuities, but each must lie strictly inside
    the domain so it can define an open quadrature panel on either side.
    """
    if breakpoints is None:
        return ()

    points: list[float] = []
    for point in breakpoints:
        if not isinstance(point, Real):
            raise TypeError("Breakpoints must be real numbers")
        value = float(point)
        if not np.isfinite(value):
            raise ValueError("Breakpoints must be finite")
        if not domain.a < value < domain.b:
            raise ValueError("Breakpoints must lie strictly inside the function domain")
        points.append(value)

    points.sort()
    if any(left == right for left, right in zip(points, points[1:], strict=False)):
        raise ValueError("Breakpoints must be unique")
    return tuple(points)


@dataclass(frozen=True)
class QuadratureRule:
    r"""Immutable nodes and weights on a partitioned interval domain.

    A rule represents

    .. math::

       \int_a^b f(x)\,dx \approx \sum_j w_j f(x_j).

    ``panel_bounds`` starts with ``a``, ends with ``b``, and contains any
    declared interior breakpoints. Nodes are strictly interior to one of those
    panels, so evaluating a rule never requires a callable to choose a value
    at a jump.
    """

    nodes: np.ndarray
    weights: np.ndarray
    panel_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float).copy()
        weights = np.asarray(self.weights, dtype=float).copy()
        bounds = tuple(float(value) for value in self.panel_bounds)

        if nodes.ndim != 1 or weights.ndim != 1:
            raise ValueError("Quadrature nodes and weights must be one-dimensional")
        if nodes.size == 0 or nodes.size != weights.size:
            raise ValueError("Quadrature nodes and weights must be non-empty and aligned")
        if len(bounds) < 2 or not np.all(np.isfinite(bounds)):
            raise ValueError("Panel bounds must contain at least two finite values")
        if any(left >= right for left, right in zip(bounds, bounds[1:], strict=False)):
            raise ValueError("Panel bounds must be strictly increasing")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
            raise ValueError("Quadrature nodes and weights must be finite")
        if not np.all(weights > 0.0):
            raise ValueError("Quadrature weights must be positive")
        if not np.all(np.diff(nodes) > 0.0):
            raise ValueError("Quadrature nodes must be strictly increasing")

        panel_indices = np.searchsorted(bounds, nodes, side="right") - 1
        if np.any(panel_indices < 0) or np.any(panel_indices >= len(bounds) - 1):
            raise ValueError("Quadrature nodes must lie inside the outer panel bounds")
        left_bounds = np.asarray(bounds[:-1])[panel_indices]
        right_bounds = np.asarray(bounds[1:])[panel_indices]
        if not np.all((nodes > left_bounds) & (nodes < right_bounds)):
            raise ValueError("Quadrature nodes must lie strictly inside a panel")

        nodes.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "panel_bounds", bounds)

    @property
    def domain_bounds(self) -> tuple[float, float]:
        """Return the closed outer interval occupied by this rule."""
        return self.panel_bounds[0], self.panel_bounds[-1]

    def is_for_domain(self, domain: IntervalDomain) -> bool:
        """Return whether this rule has exactly the given outer bounds."""
        return self.domain_bounds == (domain.a, domain.b)

    @classmethod
    def split_gauss_legendre(
        cls,
        domain: IntervalDomain,
        *,
        breakpoints: Iterable[float] | None = None,
        n_points: int = 32,
    ) -> QuadratureRule:
        """Construct a split Gauss--Legendre rule on ``domain``.

        Each panel receives at least two interior nodes. Any remaining nodes
        are allocated proportionally to panel length with deterministic
        largest-fraction remainder assignment.
        """
        if isinstance(n_points, bool) or int(n_points) != n_points:
            raise ValueError("n_points must be an integer")
        requested = int(n_points)
        if requested <= 0:
            raise ValueError("n_points must be positive")

        points = canonicalize_breakpoints(domain, breakpoints)
        bounds = (float(domain.a), *points, float(domain.b))
        lengths = np.diff(np.asarray(bounds, dtype=float))
        n_panels = len(lengths)
        total = max(requested, 2 * n_panels)
        remaining = total - 2 * n_panels
        raw = remaining * lengths / np.sum(lengths)
        allocations = 2 + np.floor(raw).astype(int)
        remainder = remaining - int(np.sum(allocations - 2))
        if remainder > 0:
            fractions = raw - np.floor(raw)
            order = np.argsort(-fractions, kind="stable")
            for index in order[:remainder]:
                allocations[index] += 1

        reference_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        nodes_by_panel: list[np.ndarray] = []
        weights_by_panel: list[np.ndarray] = []
        for count, left, right in zip(allocations, bounds[:-1], bounds[1:], strict=True):
            if int(count) not in reference_nodes:
                reference_nodes[int(count)] = np.polynomial.legendre.leggauss(int(count))
            legendre_nodes, legendre_weights = reference_nodes[int(count)]
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            nodes_by_panel.append(midpoint + half_width * legendre_nodes)
            weights_by_panel.append(half_width * legendre_weights)

        return cls(
            nodes=np.concatenate(nodes_by_panel),
            weights=np.concatenate(weights_by_panel),
            panel_bounds=bounds,
        )
