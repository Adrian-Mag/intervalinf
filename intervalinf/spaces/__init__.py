"""
Spaces module - Lebesgue and Sobolev function spaces.

This module provides finite-dimensional function space implementations
for intervals:

- `Lebesgue`: L² space with standard inner product
- `LebesgueSpaceDirectSum`: Direct sums of L² spaces
- `Sobolev`: H^s Sobolev spaces with mass-weighted inner product
- `SobolevSpaceDirectSum`: Direct sums of Sobolev spaces
- `LinearFormKernel`: Kernel-based linear forms for dual representations
- `KnownRegion`, `PartitionedLebesgueSpace`: Spaces with known regions

The spaces inherit from `pygeoinf` abstract base classes:
- `Lebesgue` inherits from `pygeoinf.HilbertSpace`
- `Sobolev` inherits from `pygeoinf.MassWeightedHilbertSpace`

Note:
    Some functionality (basis providers, Sobolev mass operators) requires
    Phase 3-4 migrations or pygeoinf for interim support.
"""

from intervalinf.spaces.forms import LinearFormKernel
from intervalinf.spaces.lebesgue import (
    Lebesgue,
    LebesgueSpaceDirectSum,
    LebesgueIntegrationConfig,
    LebesgueParallelConfig,
    KnownRegion,
    PartitionedLebesgueSpace,
)
from intervalinf.spaces.sobolev import (
    Sobolev,
    SobolevSpaceDirectSum,
)

__all__ = [
    # Forms
    "LinearFormKernel",
    # Lebesgue
    "Lebesgue",
    "LebesgueSpaceDirectSum",
    "LebesgueIntegrationConfig",
    "LebesgueParallelConfig",
    "KnownRegion",
    "PartitionedLebesgueSpace",
    # Sobolev
    "Sobolev",
    "SobolevSpaceDirectSum",
]
