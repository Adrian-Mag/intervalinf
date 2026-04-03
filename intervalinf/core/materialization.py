"""Hidden cached function materializations for repeated grid evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class RepresentationSpec:
    """Hashable specification for a concrete function representation."""

    kind: str
    n_points: int
    interval: tuple[float, float]
    method: str

    def __post_init__(self) -> None:
        interval = (float(self.interval[0]), float(self.interval[1]))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "n_points", int(self.n_points))

        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if interval[0] >= interval[1]:
            raise ValueError("interval must satisfy a < b")
        if self.kind not in {"fixed_grid", "spectral"}:
            raise ValueError(
                "kind must be either 'fixed_grid' or 'spectral'"
            )

    def __hash__(self) -> int:
        return hash((self.kind, self.n_points, self.interval, self.method))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RepresentationSpec):
            return False
        return (
            self.kind == other.kind
            and self.n_points == other.n_points
            and self.interval == other.interval
            and self.method == other.method
        )


@dataclass
class Materialization:
    """Cached values of a function on a specific representation grid."""

    spec: RepresentationSpec
    grid: np.ndarray
    values: np.ndarray
