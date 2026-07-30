"""
intervalinf - Function spaces and operators on 1D intervals.

Built on top of pygeoinf, this package provides concrete implementations
of Hilbert spaces for functions defined on 1D intervals with a
continuous-first philosophy: discretization is optional, not forced.
"""

__version__ = "0.1.0"

# Core components (no pygeoinf dependency)
from intervalinf.core import (
    BoundaryConditions,
    Function,
    IntegrationConfig,
    IntervalDomain,
    ParallelConfig,
    QuadratureRule,
)

# Spaces (depend on pygeoinf for base classes)
from intervalinf.spaces import (
    KnownRegion,
    Lebesgue,
    LebesgueIntegrationConfig,
    LebesgueParallelConfig,
    LebesgueSpaceDirectSum,
    LinearFormKernel,
    PartitionedLebesgueSpace,
    Sobolev,
    SobolevSpaceDirectSum,
    WeightedLebesgue,
)

__all__ = [
    # Version
    "__version__",
    # Core (Level 1)
    "IntervalDomain",
    "BoundaryConditions",
    "Function",
    "IntegrationConfig",
    "ParallelConfig",
    "QuadratureRule",
    # Spaces (Level 2)
    "Lebesgue",
    "LebesgueSpaceDirectSum",
    "LebesgueIntegrationConfig",
    "LebesgueParallelConfig",
    "WeightedLebesgue",
    "Sobolev",
    "SobolevSpaceDirectSum",
    "LinearFormKernel",
    "KnownRegion",
    "PartitionedLebesgueSpace",
]
