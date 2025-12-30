"""
Function providers package.

This package contains concrete function providers organized by type:

- trigonometric: Sine, Cosine, Fourier, Mixed, Robin
- fem: Hat functions, B-splines
- smooth: Bump functions and their gradients
- wavelets: Haar and other wavelet bases
- step: Box-car and discontinuous functions
- data: Kernel files and normal modes
"""

# Trigonometric providers
from intervalinf.providers.functions.trigonometric import (
    SineFunctionProvider,
    CosineFunctionProvider,
    FourierFunctionProvider,
    MixedDNFunctionProvider,
    MixedNDFunctionProvider,
    RobinFunctionProvider,
)

# FEM providers
from intervalinf.providers.functions.fem import (
    HatFunctionProvider,
    SplineFunctionProvider,
)

# Smooth function providers
from intervalinf.providers.functions.smooth import (
    BumpFunctionProvider,
    BumpFunctionGradientProvider,
)

# Wavelet providers
from intervalinf.providers.functions.wavelets import (
    WaveletFunctionProvider,
)

# Step function providers
from intervalinf.providers.functions.step import (
    BoxCarFunctionProvider,
    DiscontinuousFunctionProvider,
)

# Data-based providers
from intervalinf.providers.functions.data import (
    KernelProvider,
    NormalModesProvider,
)

__all__ = [
    # Trigonometric
    'SineFunctionProvider',
    'CosineFunctionProvider',
    'FourierFunctionProvider',
    'MixedDNFunctionProvider',
    'MixedNDFunctionProvider',
    'RobinFunctionProvider',
    # FEM
    'HatFunctionProvider',
    'SplineFunctionProvider',
    # Smooth
    'BumpFunctionProvider',
    'BumpFunctionGradientProvider',
    # Wavelets
    'WaveletFunctionProvider',
    # Step
    'BoxCarFunctionProvider',
    'DiscontinuousFunctionProvider',
    # Data
    'KernelProvider',
    'NormalModesProvider',
]
