"""
intervalinf - Function spaces and operators on 1D intervals.

Built on top of pygeoinf, this package provides concrete implementations
of Hilbert spaces for functions defined on 1D intervals with a
continuous-first philosophy: discretization is optional, not forced.
"""

__version__ = "0.1.0"

# Core components (no pygeoinf dependency)
from intervalinf.core import IntervalDomain, BoundaryConditions, Function
from intervalinf.core import IntegrationConfig, ParallelConfig

# Spaces (depend on pygeoinf for base classes)
from intervalinf.spaces import (
    Lebesgue,
    LebesgueSpaceDirectSum,
    LebesgueIntegrationConfig,
    LebesgueParallelConfig,
    Sobolev,
    SobolevSpaceDirectSum,
    LinearFormKernel,
    KnownRegion,
    PartitionedLebesgueSpace,
)

# Operators (depend on pygeoinf) - imported lazily
# from intervalinf.operators import Laplacian, InverseLaplacian, Gradient

__all__ = [
    # Version
    "__version__",
    # Core (Level 1)
    "IntervalDomain",
    "BoundaryConditions",
    "Function",
    "IntegrationConfig",
    "ParallelConfig",
    # Spaces (Level 2)
    "Lebesgue",
    "LebesgueSpaceDirectSum",
    "LebesgueIntegrationConfig",
    "LebesgueParallelConfig",
    "Sobolev",
    "SobolevSpaceDirectSum",
    "LinearFormKernel",
    "KnownRegion",
    "PartitionedLebesgueSpace",
    # Operators (Level 3) - uncomment when operators module is ready
    # "Laplacian",
    # "InverseLaplacian",
    # "Gradient",
]
