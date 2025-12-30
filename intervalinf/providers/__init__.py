"""
Providers package - Basis function, eigenvalue, and spectrum providers.

This package provides a hierarchical system of providers for generating
functions, eigenvalues, and spectral decompositions on interval domains.

Structure:
---------
- base: Abstract base classes and simple wrappers
- eigenvalues: Simple eigenvalue providers (Sine, Cosine, Fourier, etc.)
- functions/: Concrete function providers organized by type
    - trigonometric: Sine, Cosine, Fourier, Mixed, Robin
    - fem: Hat functions, B-splines
    - smooth: Bump functions
    - wavelets: Haar wavelets
    - step: Box-car, discontinuous
    - data: Kernel files, normal modes
- laplacian: Composite Laplacian providers
- radial: Radial Laplacian providers

Usage:
------
>>> from intervalinf.providers import (
...     SineFunctionProvider, LaplacianSpectrumProvider
... )
>>> provider = SineFunctionProvider(space)
>>> phi_0 = provider.get_function_by_index(0)
"""

# =============================================================================
# Base classes (Level 0)
# =============================================================================
from intervalinf.providers.base import (
    # Function providers
    FunctionProvider,
    IndexedFunctionProvider,
    RestrictedFunctionProvider,
    ParametricFunctionProvider,
    RandomFunctionProvider,
    NullFunctionProvider,
    # Eigenvalue providers
    EigenvalueProvider,
    CustomEigenvalueProvider,
    # Basis providers
    BasisProvider,
    CustomBasisProvider,
    # Spectrum providers
    SpectrumProvider,
    CustomSpectrumProvider,
)

# =============================================================================
# Simple eigenvalue providers (Level 1)
# =============================================================================
from intervalinf.providers.eigenvalues import (
    SineEigenvalueProvider,
    CosineEigenvalueProvider,
    FourierEigenvalueProvider,
    ZeroEigenvalueProvider,
    MixedDNEigenvalueProvider,
    MixedNDEigenvalueProvider,
)

# =============================================================================
# Function providers (Level 2)
# =============================================================================
from intervalinf.providers.functions import (
    # Trigonometric
    SineFunctionProvider,
    CosineFunctionProvider,
    FourierFunctionProvider,
    MixedDNFunctionProvider,
    MixedNDFunctionProvider,
    RobinFunctionProvider,
    # FEM
    HatFunctionProvider,
    SplineFunctionProvider,
    # Smooth
    BumpFunctionProvider,
    BumpFunctionGradientProvider,
    # Wavelets
    WaveletFunctionProvider,
    # Step
    BoxCarFunctionProvider,
    DiscontinuousFunctionProvider,
    # Data
    KernelProvider,
    NormalModesProvider,
)

# =============================================================================
# Laplacian providers (Level 3-4)
# =============================================================================
from intervalinf.providers.laplacian import (
    LaplacianEigenvalueProvider,
    LaplacianSpectrumProvider,
)

# =============================================================================
# Radial providers (Level 3)
# =============================================================================
from intervalinf.providers.radial import (
    RadialLaplacianDirichletProvider,
    RadialLaplacianNeumannProvider,
    RadialLaplacianDDProvider,
    RadialLaplacianDNProvider,
    RadialLaplacianNDProvider,
    RadialLaplacianNNProvider,
)

__all__ = [
    # Base classes - Function providers
    'FunctionProvider',
    'IndexedFunctionProvider',
    'RestrictedFunctionProvider',
    'ParametricFunctionProvider',
    'RandomFunctionProvider',
    'NullFunctionProvider',
    # Base classes - Eigenvalue providers
    'EigenvalueProvider',
    'CustomEigenvalueProvider',
    # Base classes - Basis providers
    'BasisProvider',
    'CustomBasisProvider',
    # Base classes - Spectrum providers
    'SpectrumProvider',
    'CustomSpectrumProvider',
    # Simple eigenvalue providers
    'SineEigenvalueProvider',
    'CosineEigenvalueProvider',
    'FourierEigenvalueProvider',
    'ZeroEigenvalueProvider',
    'MixedDNEigenvalueProvider',
    'MixedNDEigenvalueProvider',
    # Function providers - Trigonometric
    'SineFunctionProvider',
    'CosineFunctionProvider',
    'FourierFunctionProvider',
    'MixedDNFunctionProvider',
    'MixedNDFunctionProvider',
    'RobinFunctionProvider',
    # Function providers - FEM
    'HatFunctionProvider',
    'SplineFunctionProvider',
    # Function providers - Smooth
    'BumpFunctionProvider',
    'BumpFunctionGradientProvider',
    # Function providers - Wavelets
    'WaveletFunctionProvider',
    # Function providers - Step
    'BoxCarFunctionProvider',
    'DiscontinuousFunctionProvider',
    # Function providers - Data
    'KernelProvider',
    'NormalModesProvider',
    # Laplacian providers
    'LaplacianEigenvalueProvider',
    'LaplacianSpectrumProvider',
    # Radial providers
    'RadialLaplacianDirichletProvider',
    'RadialLaplacianNeumannProvider',
    'RadialLaplacianDDProvider',
    'RadialLaplacianDNProvider',
    'RadialLaplacianNDProvider',
    'RadialLaplacianNNProvider',
]
