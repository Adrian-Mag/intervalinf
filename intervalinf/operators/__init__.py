"""Operators module - Linear operators on function spaces."""

from intervalinf.operators.base import SpectralOperator

from intervalinf.operators.laplacian import Laplacian, InverseLaplacian
from intervalinf.operators.gradient import Gradient
from intervalinf.operators.bessel import BesselSobolev, BesselSobolevInverse
from intervalinf.operators.sola import SOLAOperator
from intervalinf.operators.reduced import (
    ReducedCovarianceOperator,
    ReducedCrossGramOperator,
    ReducedGramOperator,
    compute_reduced_covariance,
)
from intervalinf.operators.radial import (
    RadialLaplacian,
    InverseRadialLaplacian,
    RadialLaplacianEigenvalueProvider,
    RadialLaplacianSpectrumProvider,
)

from intervalinf.operators.spectral_helpers import (
    build_eigenfunction_expansion,
    compute_spectral_coefficients_fast,
    compute_spectral_coefficients_slow,
    validate_eigenvalue,
)

from intervalinf.operators._impl import (
    GeneralFEMSolver,
    fast_spectral_coefficients,
    create_uniform_samples,
)

__all__ = [
    # Base classes
    'SpectralOperator',
    # Laplacian operators
    'Laplacian',
    'InverseLaplacian',
    # Gradient operator
    'Gradient',
    # Bessel operators
    'BesselSobolev',
    'BesselSobolevInverse',
    # SOLA operator
    'SOLAOperator',
    'ReducedGramOperator',
    'ReducedCrossGramOperator',
    'ReducedCovarianceOperator',
    'compute_reduced_covariance',
    # Radial operators
    'RadialLaplacian',
    'InverseRadialLaplacian',
    'RadialLaplacianEigenvalueProvider',
    'RadialLaplacianSpectrumProvider',
    # Spectral helpers
    'build_eigenfunction_expansion',
    'compute_spectral_coefficients_fast',
    'compute_spectral_coefficients_slow',
    'validate_eigenvalue',
    # Implementation utilities
    'GeneralFEMSolver',
    'fast_spectral_coefficients',
    'create_uniform_samples',
]
