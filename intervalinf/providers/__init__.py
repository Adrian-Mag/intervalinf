"""Providers module - Basis function and spectrum providers."""

from intervalinf.providers.base import (
    FunctionProvider,
    IndexedFunctionProvider,
    RestrictedFunctionProvider,
    ParametricFunctionProvider,
    RandomFunctionProvider,
    NullFunctionProvider,
)

from intervalinf.providers.functions import (
    SineFunctionProvider,
    CosineFunctionProvider,
    FourierFunctionProvider,
    MixedDNFunctionProvider,
    MixedNDFunctionProvider,
    RobinFunctionProvider,
    HatFunctionProvider,
)

from intervalinf.providers.spectrum import (
    EigenvalueProvider,
    SineEigenvalueProvider,
    CosineEigenvalueProvider,
    FourierEigenvalueProvider,
    ZeroEigenvalueProvider,
    CustomEigenvalueProvider,
    LaplacianEigenvalueProvider,
    SpectrumProvider,
    LaplacianSpectrumProvider,
)

from intervalinf.providers.radial import (
    RadialLaplacianDirichletProvider,
    RadialLaplacianNeumannProvider,
    RadialLaplacianDDProvider,
    RadialLaplacianDNProvider,
    RadialLaplacianNDProvider,
    RadialLaplacianNNProvider,
)

__all__ = [
    # Base classes
    'FunctionProvider',
    'IndexedFunctionProvider',
    'RestrictedFunctionProvider',
    'ParametricFunctionProvider',
    'RandomFunctionProvider',
    'NullFunctionProvider',
    # Function providers
    'SineFunctionProvider',
    'CosineFunctionProvider',
    'FourierFunctionProvider',
    'MixedDNFunctionProvider',
    'MixedNDFunctionProvider',
    'RobinFunctionProvider',
    'HatFunctionProvider',
    # Eigenvalue providers
    'EigenvalueProvider',
    'SineEigenvalueProvider',
    'CosineEigenvalueProvider',
    'FourierEigenvalueProvider',
    'ZeroEigenvalueProvider',
    'CustomEigenvalueProvider',
    'LaplacianEigenvalueProvider',
    # Spectrum providers
    'SpectrumProvider',
    'LaplacianSpectrumProvider',
    # Radial providers
    'RadialLaplacianDirichletProvider',
    'RadialLaplacianNeumannProvider',
    'RadialLaplacianDDProvider',
    'RadialLaplacianDNProvider',
    'RadialLaplacianNDProvider',
    'RadialLaplacianNNProvider',
]
