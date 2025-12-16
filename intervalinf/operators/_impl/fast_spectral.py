"""
Fast transform methods for spectral integration in operators.

This module provides optimized coefficient computation using fast transforms
instead of expensive numerical integration:

- DST for Dirichlet boundary conditions (sine eigenfunctions)
- DCT for Neumann boundary conditions (cosine eigenfunctions)
- DFT for periodic boundary conditions (Fourier eigenfunctions)

The key insight is that computing coefficients ∫ f(x) φₖ(x) dx for
orthogonal eigenfunctions can be done via fast transforms when f(x)
is sampled uniformly on the domain.
"""

import numpy as np
from scipy.fft import dst, dct, fft
from typing import Literal, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def fast_spectral_coefficients(
    f_samples: np.ndarray,
    boundary_condition: Literal[
        'dirichlet', 'neumann', 'periodic',
        'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
    ],
    domain_length: float = 1.0,
    n_coeffs: Optional[int] = None,
    left_bc_type: Optional[str] = None,
    right_bc_type: Optional[str] = None
) -> np.ndarray:
    """
    Compute spectral coefficients using fast transforms.

    This replaces expensive numerical integration ∫ f(x) φₖ(x) dx with
    fast transform methods that exploit the orthogonality of eigenfunctions.

    Args:
        f_samples: Uniform samples of f(x) on the domain
        boundary_condition: Type of boundary conditions
        domain_length: Length of the domain (b - a)
        n_coeffs: Number of coefficients to return (default: len(f_samples))
        left_bc_type: For mixed BCs, type at left boundary (DEPRECATED)
        right_bc_type: For mixed BCs, type at right boundary (DEPRECATED)

    Returns:
        np.ndarray: Spectral coefficients for the given basis

    Notes:
        - For Dirichlet: Uses DST-I (Discrete Sine Transform Type I)
        - For Neumann: Uses DCT-I (Discrete Cosine Transform Type I)
        - For Periodic: Uses DFT (Discrete Fourier Transform)
        - For Mixed D-N: Uses DST-II (shifted sine basis)
        - For Mixed N-D: Uses DCT-II (shifted cosine basis)
    """
    n_samples = len(f_samples)
    if n_coeffs is None:
        n_coeffs = n_samples

    if boundary_condition == 'dirichlet':
        return _fast_dirichlet_coefficients(f_samples, domain_length, n_coeffs)
    elif boundary_condition == 'neumann':
        return _fast_neumann_coefficients(f_samples, domain_length, n_coeffs)
    elif boundary_condition == 'periodic':
        return _fast_periodic_coefficients(f_samples, domain_length, n_coeffs)
    elif boundary_condition == 'mixed_dirichlet_neumann':
        return _fast_mixed_coefficients(
            f_samples, domain_length, n_coeffs, 'dirichlet', 'neumann'
        )
    elif boundary_condition == 'mixed_neumann_dirichlet':
        return _fast_mixed_coefficients(
            f_samples, domain_length, n_coeffs, 'neumann', 'dirichlet'
        )
    else:
        raise ValueError(
            f"Unsupported boundary condition: {boundary_condition}"
        )


def _fast_dirichlet_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int
) -> np.ndarray:
    """
    Fast computation of Dirichlet coefficients using DST.

    For Dirichlet BCs, eigenfunctions are: φₖ(x) = √(2/L) sin(kπx/L)
    The DST directly computes coefficients: ∫ f(x) sin(kπx/L) dx
    """
    n_samples = len(f_samples)

    # Use DST Type I which matches our sine basis
    coeffs_raw = dst(f_samples, type=1)

    # Apply correct normalization factors
    scaling = np.sqrt(2.0 * domain_length) / (2.0 * (n_samples + 1))
    coeffs_normalized = coeffs_raw * scaling

    # Return only the requested number of coefficients
    return coeffs_normalized[:n_coeffs]


def _fast_neumann_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int
) -> np.ndarray:
    """
    Fast computation of Neumann coefficients using DCT.

    For Neumann BCs, eigenfunctions are:
    - φ₀(x) = 1/√L (constant mode)
    - φₖ(x) = √(2/L) cos(kπx/L) for k ≥ 1
    """
    n_samples = len(f_samples)

    # Use DCT Type I which matches our cosine basis
    coeffs_raw = dct(f_samples, type=1)

    # Apply correct normalization factors
    coeffs_normalized = np.zeros(n_coeffs)

    # For Neumann, we use N+1 sample points, so N = n_samples - 1
    N = n_samples - 1

    # Constant mode (k=0): c₀ = (1/√L) * ∫₀ᴸ f(x) dx
    if n_coeffs > 0:
        integral_0 = coeffs_raw[0] * domain_length / (2.0 * N)
        coeffs_normalized[0] = (1.0 / np.sqrt(domain_length)) * integral_0

    # Cosine modes (k≥1): cₖ = √(2/L) * ∫₀ᴸ f(x) cos(kπx/L) dx
    for k in range(1, min(n_coeffs, len(coeffs_raw))):
        integral_k = coeffs_raw[k] * domain_length / (2.0 * N)
        coeffs_normalized[k] = np.sqrt(2.0 / domain_length) * integral_k

    return coeffs_normalized


def _fast_periodic_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int
) -> np.ndarray:
    """
    Fast computation of periodic coefficients using DFT.

    For periodic BCs, eigenfunctions are: φₖ(x) = e^(2πikx/L) / √L
    The DFT directly computes these Fourier coefficients.
    """
    n_samples = len(f_samples)

    # Compute DFT
    coeffs_fft = np.asarray(fft(f_samples), dtype=np.complex128)

    # Apply normalization
    dx_over_sqrtL = (domain_length / n_samples) / np.sqrt(domain_length)

    # Convert complex coefficients to real cos/sin coefficients
    sqrt2 = np.sqrt(2.0)
    coeffs_real = np.zeros(n_coeffs)

    # DC component (k=0)
    if n_coeffs > 0:
        coeffs_real[0] = coeffs_fft[0].real * dx_over_sqrtL

    # Alternating cos/sin modes with correct √2 scaling
    max_k = min(n_coeffs, n_samples // 2 + 1)
    for k in range(1, max_k):
        if 2 * k - 1 < n_coeffs:
            # Cosine coefficient: √2 · Re(a_k)
            coeffs_real[2 * k - 1] = (
                sqrt2 * coeffs_fft[k].real * dx_over_sqrtL
            )
        if 2 * k < n_coeffs:
            # Sine coefficient: √2 · Im(a_k) with FFT sign adjustment
            coeffs_real[2 * k] = (
                -sqrt2 * coeffs_fft[k].imag * dx_over_sqrtL
            )

    return coeffs_real


def _fast_mixed_coefficients(
    f_samples: np.ndarray,
    domain_length: float,
    n_coeffs: int,
    left_bc_type: str,
    right_bc_type: str
) -> np.ndarray:
    """
    Fast computation of mixed BC coefficients using DST-II or DCT-II.

    For mixed BCs, the eigenfunctions are shifted sines or cosines:
    - Dirichlet-Neumann: φₖ(x) = √(2/L) sin((k+½)πx/L) → Use DST-IV
    - Neumann-Dirichlet: φₖ(x) = √(2/L) cos((k+½)πx/L) → Use DCT-IV
    """
    n_samples = len(f_samples)

    if left_bc_type == 'dirichlet' and right_bc_type == 'neumann':
        # Eigenfunctions: φₖ(x) = √(2/L) sin((k+½)πx/L)
        # Use DST-IV (Type 4)
        coeffs_raw = dst(f_samples, type=4)

        # Scaling: (L/N) * √(2/L) / 2 = √(L/2) / N
        scaling = np.sqrt(domain_length / 2.0) / n_samples
        coeffs_normalized = coeffs_raw * scaling

    elif left_bc_type == 'neumann' and right_bc_type == 'dirichlet':
        # Eigenfunctions: φₖ(x) = √(2/L) cos((k+½)πx/L)
        # Use DCT-IV (Type 4)
        coeffs_raw = dct(f_samples, type=4)

        # Same scaling as DST-IV
        scaling = np.sqrt(domain_length / 2.0) / n_samples
        coeffs_normalized = coeffs_raw * scaling

    else:
        raise ValueError(
            f"Unsupported mixed BC combination: "
            f"left={left_bc_type}, right={right_bc_type}. "
            "Only 'dirichlet-neumann' and 'neumann-dirichlet' are supported."
        )

    return coeffs_normalized[:n_coeffs]


def create_uniform_samples(
    func,
    domain: Tuple[float, float],
    n_samples: int,
    boundary_condition: Literal[
        'dirichlet', 'neumann', 'periodic',
        'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
    ],
    left_bc_type: Optional[str] = None,
    right_bc_type: Optional[str] = None
) -> np.ndarray:
    """
    Create uniform samples of a function on the domain.

    Args:
        func: Function to sample (callable)
        domain: (a, b) domain endpoints
        n_samples: Number of sample points
        boundary_condition: Affects sampling strategy
        left_bc_type: DEPRECATED - inferred from boundary_condition
        right_bc_type: DEPRECATED - inferred from boundary_condition

    Returns:
        np.ndarray: Function samples at appropriate points
    """
    a, b = domain

    if boundary_condition == 'dirichlet':
        # For Dirichlet, exclude boundary points
        if hasattr(func, 'space') and hasattr(func.space, 'function_domain'):
            domain_obj = func.space.function_domain
            if hasattr(domain_obj, 'uniform_mesh'):
                x = domain_obj.uniform_mesh(n_samples + 2)[1:-1]
            else:
                x = np.linspace(a, b, n_samples + 2)[1:-1]
        else:
            x = np.linspace(a, b, n_samples + 2)[1:-1]

    elif boundary_condition == 'neumann':
        # For Neumann, use domain's uniform_mesh
        if hasattr(func, 'space') and hasattr(func.space, 'function_domain'):
            domain_obj = func.space.function_domain
            if hasattr(domain_obj, 'uniform_mesh'):
                x = domain_obj.uniform_mesh(n_samples)
            else:
                x = np.linspace(a, b, n_samples)
        else:
            x = np.linspace(a, b, n_samples)

    elif boundary_condition == 'periodic':
        # For periodic, exclude the right endpoint
        if hasattr(func, 'space') and hasattr(func.space, 'function_domain'):
            domain_obj = func.space.function_domain
            if hasattr(domain_obj, 'uniform_mesh'):
                x = domain_obj.uniform_mesh(n_samples + 1)[:-1]
            else:
                x = np.linspace(a, b, n_samples + 1)[:-1]
        else:
            x = np.linspace(a, b, n_samples + 1)[:-1]

    elif boundary_condition == 'mixed_dirichlet_neumann':
        # DST-II: sample at (k+½)Δx for k=0..N-1
        dx = (b - a) / n_samples
        x = a + (np.arange(n_samples) + 0.5) * dx

    elif boundary_condition == 'mixed_neumann_dirichlet':
        # DCT-II: sample at (k+½)Δx for k=0..N-1
        dx = (b - a) / n_samples
        x = a + (np.arange(n_samples) + 0.5) * dx

    else:
        raise ValueError(
            f"Unsupported boundary condition: {boundary_condition}"
        )

    return func(x)


def benchmark_integration_methods(
    func,
    domain: Tuple[float, float],
    boundary_condition: Literal['dirichlet', 'neumann', 'periodic'],
    n_coeffs: int = 100,
    n_samples: int = 1024
) -> dict:
    """
    Benchmark fast transforms vs numerical integration.

    Returns:
        dict: Timing and accuracy results
    """
    import time

    a, b = domain
    domain_length = b - a

    # Create function samples
    f_samples = create_uniform_samples(
        func, domain, n_samples, boundary_condition
    )

    # Fast transform method
    start_time = time.time()
    coeffs_fast = fast_spectral_coefficients(
        f_samples, boundary_condition, domain_length, n_coeffs
    )
    fast_time = time.time() - start_time

    results = {
        'fast_transform_time': fast_time,
        'n_coeffs': n_coeffs,
        'n_samples': n_samples,
        'coefficients_computed': len(coeffs_fast),
        'method': boundary_condition,
    }

    logger.info(
        f"Fast transform ({boundary_condition}): {fast_time:.6f}s for "
        f"{n_coeffs} coefficients"
    )

    return results
