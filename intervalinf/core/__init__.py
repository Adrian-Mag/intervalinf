"""
Core module - foundational components with no pygeoinf dependency.

This module provides:
- IntervalDomain: Domain specification and meshing
- BoundaryConditions: Boundary condition specifications
- Function: Function class with arithmetic and evaluation
- Configuration dataclasses for integration and parallelization
"""

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.functions import Function
from intervalinf.core.config import (
    IntegrationConfig,
    ParallelConfig,
    LebesgueIntegrationConfig,
    LebesgueParallelConfig,
)
from intervalinf.core.quadrature import QuadratureRule

__all__ = [
    "IntervalDomain",
    "BoundaryConditions",
    "Function",
    "IntegrationConfig",
    "ParallelConfig",
    "LebesgueIntegrationConfig",
    "LebesgueParallelConfig",
    "QuadratureRule",
]
