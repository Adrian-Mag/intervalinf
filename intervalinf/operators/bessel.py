"""Bessel-Sobolev operators for interval domains."""

from typing import Optional, Literal

import numpy as np

from pygeoinf.linear_operators import LinearOperator
from pygeoinf import MassWeightedHilbertSpace

from intervalinf.operators.base import SpectralOperator
from intervalinf.operators.spectral_helpers import (
    build_eigenfunction_expansion,
    compute_spectral_coefficients_fast,
    compute_spectral_coefficients_slow,
    validate_eigenvalue,
)
from intervalinf.operators._impl.fast_spectral import (
    fast_spectral_coefficients,
    create_uniform_samples,
)
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.core.functions import Function
from intervalinf.core.config import IntegrationConfig


def _radial_dirichlet_fast_eligible(L) -> bool:
    """True if ``L`` is a spectral ``RadialLaplacian`` with Dirichlet BCs.

    For the radial Dirichlet problem on $[a,b]$ the eigenfunctions are
    $\\phi_n(r) = \\sqrt{2/L}\\,\\sin(n\\pi(r-a)/L)/r$ with $\\lambda_n=(n\\pi/L)^2$,
    identical to the *flat* Dirichlet sine spectrum after the substitution
    $u(r)=r\\,f(r)$ (since $r\\,\\phi_n = \\sqrt{2/L}\\sin(n\\pi(r-a)/L)$). The
    weighted radial projection $\\int \\phi_n f\\,r^2\\,dr$ then equals the flat
    sine projection of $u$, so the existing DST fast path applies to $u=r f$ and
    the result is reconstructed with the radial eigenfunctions. Neumann/mixed/
    Robin radial spectra are transcendental and have no such DST.
    """
    if L is None or L.__class__.__name__ != "RadialLaplacian":
        return False
    if getattr(L, "_method", "spectral") != "spectral":
        return False
    bc = getattr(L, "_boundary_conditions", None)
    return bc is not None and getattr(bc, "type", None) == "dirichlet"


def _apply_radial_fast_impl(op, f: Function, scale_func) -> Function:
    """Shared radial-Dirichlet fast apply for BesselSobolev{,Inverse}.

    Substitutes ``u(r) = r·f(r)`` and runs the flat sine (Dirichlet) DST on
    ``u``: the resulting coefficients equal the weighted radial projections
    ⟨φ_n, f⟩_{r²} (because ``r·φ_n = √(2/L) sin(nπ(r-a)/L)`` and the flat sine
    projection of ``u`` is ``√(2/L) ∫ sin(nπ(r-a)/L) u dr``). The terms are then
    reconstructed with the *radial* eigenfunctions of ``op._L`` (and the radial
    eigenvalues, which equal the flat Dirichlet ones), so the output is the
    correct radial Bessel image of ``f``.
    """
    interval = op._domain.function_domain
    a, b = interval.a, interval.b
    length = b - a

    def _u(x, _f=f):
        xr = np.asarray(x, dtype=float)
        return xr * np.asarray(_f(xr), dtype=float)

    u = Function(interval, evaluate_callable=_u)
    u_samples = create_uniform_samples(
        u, (a, b), op._n_samples, "dirichlet", domain_obj=interval
    )
    coefficients = fast_spectral_coefficients(
        u_samples, "dirichlet", length, op._dofs
    )
    terms = compute_spectral_coefficients_fast(
        op._L, f, coefficients, scale_func=scale_func
    )
    return build_eigenfunction_expansion(terms, op._domain, op._codomain)


class BesselSobolev(LinearOperator):
    """
    Fast Bessel potential operator using fast transforms for coefficient
    computation.

    This is a drop-in replacement for BesselSobolev that uses fast transforms
    (DST/DCT/DFT) instead of numerical integration when the underlying spectral
    operator L uses Laplacian eigenfunctions with homogeneous boundary
    conditions.

    Supports:
    - Dirichlet BC: Uses DST (Discrete Sine Transform)
    - Neumann BC: Uses DCT (Discrete Cosine Transform)
    - Periodic BC: Uses DFT (Discrete Fourier Transform)

    For non-Laplacian operators, falls back to the original slow method.
    """

    def __init__(
        self,
        domain: Lebesgue,
        codomain: Lebesgue,
        k: float,
        s: float,
        L: SpectralOperator,
        dofs: Optional[int] = None,
        n_samples: int = 1024,
        use_fast_transforms: bool = True,
        integration_config: IntegrationConfig = IntegrationConfig(
            method='simpson', n_points=100
        )
    ):
        """
        Initialize the fast Bessel potential operator.

        Args:
            domain: Lebesgue space (input)
            codomain: Lebesgue space (output)
            k: Bessel parameter k²
            s: Sobolev order s
            L: Spectral operator (should be Laplacian for fast transforms)
            dofs: Number of degrees of freedom
            n_samples: Number of samples for fast transform
            use_fast_transforms: If False, fall back to slow integration
            integration_config: Integration configuration
        """
        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs if dofs is not None else domain.dim
        self._n_samples = max(n_samples, self._dofs)
        self._use_fast_transforms = use_fast_transforms

        self.integration = integration_config

        self._boundary_condition = self._detect_boundary_condition()
        # The DST/DCT/DFT fast path computes *unweighted* spectral coefficients,
        # so it is only valid on a plain-L² domain. A mass-weighted domain (e.g.
        # a WeightedLebesgue with the radial r² weight) must use the weight-aware
        # slow projection through the domain's own inner product instead.
        self._is_mass_weighted = isinstance(domain, MassWeightedHilbertSpace)
        self._projection_inner_product = (
            domain.inner_product if self._is_mass_weighted else None
        )
        self._can_use_fast_transforms = (
            self._use_fast_transforms and
            self._boundary_condition is not None and
            not self._is_mass_weighted
        )
        # Radial Dirichlet fast path (via the u = r·f substitution → flat DST).
        # Applies on a weighted radial domain where the generic fast path is off.
        self._radial_dirichlet_fast = (
            self._use_fast_transforms and _radial_dirichlet_fast_eligible(L)
        )

        # BesselSobolev is self-adjoint: (k²I - Δ)^s = ((k²I - Δ)^s)^*
        super().__init__(domain, codomain, self._apply, adjoint_mapping=self._apply)

    def _detect_boundary_condition(self) -> Optional[Literal[
        'dirichlet', 'neumann', 'periodic',
        'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
    ]]:
        """Detect boundary condition type from the spectral operator."""
        if hasattr(self._L, '_spectrum_provider'):
            provider = self._L._spectrum_provider
            if hasattr(provider, 'type'):
                if provider.type == 'sine_dirichlet':
                    return 'dirichlet'
                elif provider.type == 'cosine_neumann':
                    return 'neumann'
                elif provider.type == 'fourier_periodic':
                    return 'periodic'

        if hasattr(self._L, '_boundary_conditions'):
            bc = self._L._boundary_conditions
            if hasattr(bc, 'type'):
                if bc.type in ['dirichlet', 'neumann', 'periodic',
                               'mixed_dirichlet_neumann',
                               'mixed_neumann_dirichlet']:
                    return bc.type

        return None

    def _apply(self, f: Function) -> Function:
        """Apply the Bessel potential operator to a function."""
        if self._radial_dirichlet_fast:
            return self._apply_radial_fast(f)
        elif self._can_use_fast_transforms:
            return self._apply_fast(f)
        else:
            return self._apply_slow(f)

    def _apply_fast(self, f: Function) -> Function:
        """Apply using fast transforms."""
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        f_samples = create_uniform_samples(
            f, domain_tuple, self._n_samples, self._boundary_condition,
            domain_obj=domain_interval
        )

        coefficients = fast_spectral_coefficients(
            f_samples, self._boundary_condition, domain_length, self._dofs
        )

        terms = compute_spectral_coefficients_fast(
            self._L, f, coefficients,
            scale_func=lambda i, ev: (self._k**2 + ev)**(self._s / 2)
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def _apply_radial_fast(self, f: Function) -> Function:
        """Radial Dirichlet fast path via the substitution u(r)=r·f(r).

        The flat sine DST of ``u`` yields exactly the weighted radial spectral
        coefficients ⟨φ_n, f⟩_{r²}; reconstruction uses the radial eigenfunctions.
        """
        return _apply_radial_fast_impl(
            self, f, lambda i, ev: (self._k**2 + ev)**(self._s / 2)
        )

    def _apply_slow(self, f: Function) -> Function:
        """Apply using slow numerical integration (fallback)."""
        terms = compute_spectral_coefficients_slow(
            self._L, f, self._dofs,
            self.integration.method,
            self.integration.n_points,
            scale_func=lambda i, ev: (self._k**2 + ev)**(self._s / 2),
            inner_product=self._projection_inner_product,
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._L.get_eigenfunction(index)

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        eigval = self._L.get_eigenvalue(index)
        validate_eigenvalue(eigval, index, allow_negative=False)
        return (self._k**2 + eigval)**(self._s / 2)


class BesselSobolevInverse(LinearOperator):
    """
    Fast inverse Bessel potential operator using fast transforms.

    This is a drop-in replacement for BesselSobolevInverse that uses fast
    transforms instead of numerical integration for coefficient computation.
    """

    def __init__(
        self,
        domain: Lebesgue,
        codomain: Lebesgue,
        k: float,
        s: float,
        L: SpectralOperator,
        dofs: Optional[int] = None,
        n_samples: int = 1024,
        use_fast_transforms: bool = True,
        integration_config: IntegrationConfig = IntegrationConfig(
            method='simpson', n_points=10000
        )
    ):
        """
        Initialize the fast inverse Bessel potential operator.

        Args:
            domain: Lebesgue space (input)
            codomain: Lebesgue space (output)
            k: Bessel parameter k²
            s: Sobolev order s
            L: Spectral operator (should be Laplacian for fast transforms)
            dofs: Number of degrees of freedom
            n_samples: Number of samples for fast transform
            use_fast_transforms: If False, fall back to slow method
            integration_config: Integration configuration for slow path
        """
        self._domain = domain
        self._codomain = codomain
        self._L = L
        self._k = k
        self._s = s
        self._dofs = dofs if dofs is not None else domain.dim
        self._n_samples = max(n_samples, self._dofs)
        self._use_fast_transforms = use_fast_transforms

        self.integration = integration_config

        self._boundary_condition = self._detect_boundary_condition()
        # The DST/DCT/DFT fast path computes *unweighted* spectral coefficients,
        # so it is only valid on a plain-L² domain. A mass-weighted domain (e.g.
        # a WeightedLebesgue with the radial r² weight) must use the weight-aware
        # slow projection through the domain's own inner product instead.
        self._is_mass_weighted = isinstance(domain, MassWeightedHilbertSpace)
        self._projection_inner_product = (
            domain.inner_product if self._is_mass_weighted else None
        )
        self._can_use_fast_transforms = (
            self._use_fast_transforms and
            self._boundary_condition is not None and
            not self._is_mass_weighted
        )
        # Radial Dirichlet fast path (via the u = r·f substitution → flat DST).
        # Applies on a weighted radial domain where the generic fast path is off.
        self._radial_dirichlet_fast = (
            self._use_fast_transforms and _radial_dirichlet_fast_eligible(L)
        )

        # BesselSobolevInverse is self-adjoint: (k²I - Δ)^{-s} = ((k²I - Δ)^{-s})^*
        super().__init__(domain, codomain, self._apply, adjoint_mapping=self._apply)

    def _detect_boundary_condition(self) -> Optional[Literal[
        'dirichlet', 'neumann', 'periodic',
        'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
    ]]:
        """Detect boundary condition type from the spectral operator."""
        if hasattr(self._L, '_spectrum_provider'):
            provider = self._L._spectrum_provider
            if hasattr(provider, 'type'):
                if provider.type == 'sine_dirichlet':
                    return 'dirichlet'
                elif provider.type == 'cosine_neumann':
                    return 'neumann'
                elif provider.type == 'fourier_periodic':
                    return 'periodic'

        if hasattr(self._L, '_boundary_conditions'):
            bc = self._L._boundary_conditions
            if hasattr(bc, 'type'):
                if bc.type in ['dirichlet', 'neumann', 'periodic',
                               'mixed_dirichlet_neumann',
                               'mixed_neumann_dirichlet']:
                    return bc.type

        return None

    def _apply(self, f: Function) -> Function:
        """Apply the inverse Bessel potential operator to a function."""
        if self._radial_dirichlet_fast:
            return _apply_radial_fast_impl(
                self, f, lambda i, ev: (self._k**2 + ev)**(-self._s / 2)
            )
        if self._can_use_fast_transforms:
            return self._apply_fast(f)
        else:
            return self._apply_slow(f)

    def _apply_fast(self, f: Function) -> Function:
        """Apply using fast transforms."""
        domain_interval = self._domain.function_domain
        domain_tuple = (domain_interval.a, domain_interval.b)
        domain_length = domain_interval.b - domain_interval.a

        f_samples = create_uniform_samples(
            f, domain_tuple, self._n_samples, self._boundary_condition,
            domain_obj=domain_interval
        )

        coefficients = fast_spectral_coefficients(
            f_samples, self._boundary_condition, domain_length, self._dofs
        )

        terms = compute_spectral_coefficients_fast(
            self._L, f, coefficients,
            scale_func=lambda i, ev: (self._k**2 + ev)**(-self._s / 2)
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def _apply_slow(self, f: Function) -> Function:
        """Apply using slow numerical integration (fallback)."""
        terms = compute_spectral_coefficients_slow(
            self._L, f, self._dofs,
            self.integration.method,
            self.integration.n_points,
            scale_func=lambda i, ev: (self._k**2 + ev)**(-self._s / 2),
            inner_product=self._projection_inner_product,
        )
        return build_eigenfunction_expansion(
            terms, self._domain, self._codomain
        )

    def get_eigenfunction(self, index: int) -> Function:
        """Get the eigenfunction at a specific index."""
        return self._L.get_eigenfunction(index)

    def get_eigenvalue(self, index: int) -> float:
        """Get the eigenvalue at a specific index."""
        eigval = self._L.get_eigenvalue(index)
        validate_eigenvalue(eigval, index, allow_negative=False)
        return (self._k**2 + eigval)**(-self._s / 2)
