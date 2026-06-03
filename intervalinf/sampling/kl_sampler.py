"""Spectral (KL) sampling utilities for interval operators.

Provides `KLSampler` for Gaussian measures with covariance admitting
eigenpairs (λ_i, φ_i). Generates samples using truncated expansion:

    u = m + Σ_{i<k} sqrt(λ_i) ξ_i φ_i,  ξ_i ~ N(0,1).

Also returns a covariance factor L so that C ≈ L L*.

Supports both Lebesgue (unweighted) and Sobolev (mass-weighted) spaces:
- In L² spaces: Uses eigenvalues λ_i directly from the covariance operator
- In H^s spaces: Adjusts eigenvalues to λ_i' = λ_i / μ_i where μ_i are the
  mass operator eigenvalues, automatically handling the weighted inner product
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, cast, Protocol
import numpy as np

from pygeoinf import EuclideanSpace, LinearOperator
from pygeoinf import MassWeightedHilbertSpace

if TYPE_CHECKING:
    from intervalinf.operators import SpectralOperator
    from intervalinf.core.functions import Function


class _MassOperatorProtocol(Protocol):
    """Protocol for mass operators with eigenvalue access."""
    def get_eigenvalue(self, index: int) -> float: ...


@dataclass
class TruncationInfo:
    """Information about truncation of the KL expansion.

    Attributes
    ----------
    n_modes : int
        Number of modes retained in the expansion.
    reason : str
        Reason for truncation ('explicit', 'energy_tol-heuristic', 'default').
    energy_fraction : float, optional
        Fraction of energy captured (if computed).
    """
    n_modes: int
    reason: str
    energy_fraction: Optional[float] = None


class KLSampler:
    """Sampler for Gaussian measures using a spectral (KL) expansion.

    Automatically handles both Lebesgue (unweighted) and Sobolev
    (mass-weighted) spaces. For Sobolev spaces, eigenvalues are adjusted
    to account for the mass operator: λ_i' = λ_i / μ_i² where μ_i² are
    the squared mass eigenvalues.

    Parameters
    ----------
    operator : SpectralOperator
        Operator implementing ``get_eigenvalue(i)`` & ``get_eigenfunction(i)``.
        Typically ``InverseLaplacian`` or another covariance operator.
    mean : Function, optional
        Mean function (defaults to zero element of the domain).
    n_modes : int, optional
        Explicit number of modes to retain. Mutually exclusive with
        ``energy_tol``.
    energy_tol : float in (0,1], optional
        Retain the minimal number of leading modes whose cumulative
        eigenvalue sum fraction >= energy_tol (approximate if tail unknown).
    max_modes : int, optional
        Hard cap on modes (used with energy_tol to avoid runaway growth).
    rng : np.random.Generator, optional
        Random generator for reproducibility.
    cache : bool, default=True
        Cache fetched eigenpairs.

    Examples
    --------
    >>> from intervalinf.core import IntervalDomain
    >>> from intervalinf.spaces import Lebesgue
    >>> from intervalinf.operators import InverseLaplacian
    >>> domain = IntervalDomain(0, 1)
    >>> space = Lebesgue(50, domain, basis='sine')
    >>> cov_op = InverseLaplacian(space)
    >>> sampler = KLSampler(cov_op, n_modes=20)
    >>> sample = sampler.sample()
    """

    def __init__(
        self,
        operator: 'SpectralOperator',
        *,
        mean: Optional['Function'] = None,
        n_modes: Optional[int] = None,
        energy_tol: Optional[float] = None,
        max_modes: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        cache: bool = True,
    ) -> None:

        if n_modes is not None and n_modes < 1:
            raise ValueError("n_modes must be positive")
        if energy_tol is not None:
            if not (0 < energy_tol <= 1):
                raise ValueError("energy_tol must be in (0,1]")
            if n_modes is not None:
                raise ValueError(
                    "Specify either n_modes or energy_tol, not both"
                )

        self._op = operator
        self._domain = operator.domain
        self._mean = mean if mean is not None else self._domain.zero
        self._explicit_n_modes = n_modes
        self._energy_tol = energy_tol
        self._max_modes = max_modes
        self._rng = rng if rng is not None else np.random.default_rng()
        self._cache_enabled = cache

        self._eigenvalues: List[float] = []
        self._eigenfunctions: List = []
        self._truncation: Optional[TruncationInfo] = None

        # Detect and configure mass weighting
        self._setup_mass_weighting()

    # ------------------------------------------------------------------
    # Mass weighting detection and configuration
    # ------------------------------------------------------------------
    def _setup_mass_weighting(self) -> None:
        """
        Detect if domain is mass-weighted and configure eigenvalue adjustment.

        For Sobolev spaces with a BesselSobolev mass operator, the covariance
        eigenvalues must be adjusted: λ_i' = λ_i / μ_i² where μ_i² are the
        squared mass operator eigenvalues.

        For spaces with a multiplicative mass operator (e.g. WeightedLebesgue),
        the operator's eigenfunctions are already orthonormal w.r.t. the
        weighted inner product, so no eigenvalue adjustment is needed.
        """
        self._is_weighted = isinstance(self._domain, MassWeightedHilbertSpace)

        if not self._is_weighted:
            self._mass_operator = None
            return

        weighted_domain = cast(MassWeightedHilbertSpace, self._domain)
        self._mass_operator = weighted_domain.mass_operator

        self._is_bessel_sobolev = self._detect_bessel_sobolev()
        # For multiplicative-weight spaces (e.g. WeightedLebesgue), eigenfunctions
        # are already w-orthonormal — treat the same as unweighted for sampling.
        if not self._is_bessel_sobolev:
            self._is_weighted = False
            self._mass_operator = None

    def _detect_bessel_sobolev(self) -> bool:
        """Check if mass operator is a BesselSobolev operator."""
        from intervalinf.operators import BesselSobolev
        return isinstance(self._mass_operator, BesselSobolev)

    def _get_mass_eigenvalue(self, i: int) -> float:
        """
        Get the i-th eigenvalue of the mass operator squared.

        For BesselSobolev mass operator M with parameters (k, s):
        - M has eigenvalues μ_i = (k² + λ_i^L)^(s/2)
        - The squared mass operator M² has eigenvalues μ_i² = (k² + λ_i^L)^s
        - We need μ_i² for the covariance adjustment: C' = M^{-2} C

        Returns 1.0 for unweighted spaces (backward compatible).

        Parameters
        ----------
        i : int
            Eigenvalue index

        Returns
        -------
        float
            The squared mass eigenvalue μ_i²
        """
        if not self._is_weighted:
            return 1.0

        # For BesselSobolev: eigenvalue is (k² + λ_i^L)^(s/2)
        # We need the square: μ_i² = (k² + λ_i^L)^s
        if self._mass_operator is None:
            raise RuntimeError(
                "Mass operator is None but space is weighted - "
                "this should not happen"
            )

        # BesselSobolev has get_eigenvalue method
        if not hasattr(self._mass_operator, 'get_eigenvalue'):
            raise NotImplementedError(
                "Mass operator does not have get_eigenvalue method"
            )

        # Narrow type to our protocol
        mass_op = cast(_MassOperatorProtocol, self._mass_operator)
        return mass_op.get_eigenvalue(i)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_eigenpair(self, i: int):
        """
        Fetch eigenvalue and eigenfunction, adjusting for mass weighting.

        For mass-weighted spaces (Sobolev), returns adjusted eigenvalue:
            λ_i' = λ_i / μ_i²
        where λ_i is the raw eigenvalue from the covariance operator and
        μ_i² is the squared mass operator eigenvalue.

        Eigenfunctions remain unchanged (same basis in L² and H^s).
        """
        if i < len(self._eigenvalues):
            return self._eigenvalues[i], self._eigenfunctions[i]

        # Get raw eigenvalue from covariance operator (in L² sense)
        lam = float(self._op.get_eigenvalue(i))

        # Adjust for mass weighting: λ' = λ / μ²
        mu = self._get_mass_eigenvalue(i)

        # Eigenfunction remains unchanged (same basis in both spaces)
        func = self._op.get_eigenfunction(i)
        func_adjusted = func * (np.power(np.sqrt(mu), -1.0))

        if self._cache_enabled:
            self._eigenvalues.append(lam)
            self._eigenfunctions.append(func_adjusted)

        return lam, func_adjusted

    def _determine_truncation(self) -> TruncationInfo:
        if self._truncation is not None:
            return self._truncation
        # Case 1: explicit number of modes
        if self._explicit_n_modes is not None:
            info = TruncationInfo(
                n_modes=self._explicit_n_modes, reason="explicit"
            )
            self._truncation = info
            # Ensure eigenpairs loaded
            for i in range(info.n_modes):
                self._fetch_eigenpair(i)
            return info
        # Case 2: energy-based
        if self._energy_tol is not None:
            # Heuristic: keep taking modes until relative drop large
            i = 0
            prev = None
            while True:
                lam, _ = self._fetch_eigenpair(i)
                i += 1
                if prev is not None and lam <= prev * (1 - self._energy_tol):
                    break
                prev = lam
                if self._max_modes is not None and i >= self._max_modes:
                    break
            info = TruncationInfo(
                n_modes=i, reason="energy_tol-heuristic", energy_fraction=None
            )
            self._truncation = info
            return info
        # Fallback: single mode
        info = TruncationInfo(n_modes=1, reason="default")
        self._truncation = info
        self._fetch_eigenpair(0)
        return info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def truncation(self) -> TruncationInfo:
        """Get truncation information for the KL expansion."""
        return self._determine_truncation()

    @property
    def n_modes(self) -> int:
        """Number of modes in the truncated expansion."""
        return self.truncation.n_modes

    def eigenvalue(self, i: int) -> float:
        """Get the i-th (possibly adjusted) eigenvalue."""
        return self._fetch_eigenpair(i)[0]

    def eigenfunction(self, i: int):
        """Get the i-th eigenfunction."""
        return self._fetch_eigenpair(i)[1]

    def covariance_factor(self) -> 'LinearOperator':
        """Return a covariance factor L : R^k → H with C ≈ L L*.

        For mass-weighted spaces, this returns the adjusted covariance factor
        L' = M^{-1} L where eigenvalues are already scaled by 1/μ_i².

        Adjoint mapping implements L*: H → R^k via inner products with
        eigenfunctions multiplied by sqrt(λ_i) (already adjusted).

        Returns
        -------
        LinearOperator
            Covariance factor operator.
        """
        k = self.n_modes
        # Eigenvalues already adjusted for mass weighting in _fetch_eigenpair
        eigvals = np.array([self.eigenvalue(i) for i in range(k)])
        eigfuncs = [self.eigenfunction(i) for i in range(k)]
        sqrt_lam = np.sqrt(eigvals)
        domain = EuclideanSpace(k)
        codomain = self._domain

        def mapping(w):  # w in R^k
            # Avoid deep recursion by accumulating via direct evaluation
            # instead of nested Function operations
            from intervalinf.core.functions import Function

            def evaluate_sum(x):
                # Evaluate eigenfunctions at x, sum weighted contributions
                result = np.zeros_like(x) if isinstance(x, np.ndarray) \
                    else 0.0
                for coeff, lam_sqrt, phi in zip(w, sqrt_lam, eigfuncs):
                    if coeff != 0.0:
                        result = result + (lam_sqrt * coeff) * phi.evaluate(x)
                return result

            return Function(codomain, evaluate_callable=evaluate_sum)

        def adjoint_mapping(v):  # v in H
            comps = np.zeros(k)
            for i, (lam_sqrt, phi) in enumerate(zip(sqrt_lam, eigfuncs)):
                inner = self._domain.inner_product(phi, v)
                comps[i] = lam_sqrt * inner if lam_sqrt > 0 else 0.0
            return comps

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    @property
    def variance_function(self):
        """Return a function representing the pointwise variance.

        Computes the pointwise variance of the truncated KL expansion:

            Var[u](x) = Σ_{i<k} λ_i φ_i(x)²

        The result is returned in the sampler's codomain (an element of the
        Hilbert space). For function spaces this is a ``Function``; for
        Euclidean spaces this is a NumPy array.

        Returns
        -------
        Function or ndarray
            Pointwise variance function/vector.
        """
        k = self.n_modes
        codomain = self._domain

        # Special-case Euclidean spaces (component-wise variance)
        if isinstance(codomain, EuclideanSpace):
            # Accumulate variance in component form
            var = np.zeros(codomain.dim)
            for i in range(k):
                lam = self.eigenvalue(i)
                phi = self.eigenfunction(i)
                # phi is expected to be a numpy array in EuclideanSpace
                phi_arr = np.asarray(phi)
                var += lam * (phi_arr ** 2)
            # Return as element of the space
            return codomain.from_components(var)

        # General function-space case: Avoid deep recursion
        from intervalinf.core.functions import Function
        eigvals = [self.eigenvalue(i) for i in range(k)]
        eigfuncs = [self.eigenfunction(i) for i in range(k)]

        def evaluate_variance(x):
            var = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            for lam, phi in zip(eigvals, eigfuncs):
                phi_val = phi.evaluate(x)
                var = var + lam * (phi_val ** 2)
            return var

        return Function(codomain, evaluate_callable=evaluate_variance)

    def sample(self):
        """Draw a single sample from the Gaussian measure.

        For mass-weighted spaces, uses adjusted eigenvalues automatically:
            u = m + Σ sqrt(λ_i / μ_i²) ξ_i φ_i
        where eigenvalues are already adjusted in _fetch_eigenpair.

        Returns
        -------
        Function
            A sample function from the Gaussian measure.
        """
        k = self.n_modes
        z = self._rng.standard_normal(k)
        # Eigenvalues already adjusted for mass weighting
        eigvals = [self.eigenvalue(i) for i in range(k)]
        eigfuncs = [self.eigenfunction(i) for i in range(k)]
        sqrt_lam = np.sqrt(np.maximum(eigvals, 0))

        # Avoid deep recursion by direct evaluation
        from intervalinf.core.functions import Function

        def evaluate_sample(x):
            # Start with mean
            result = self._mean.evaluate(x)
            # Add KL contributions
            for coeff, lam_sqrt, phi in zip(z, sqrt_lam, eigfuncs):
                if lam_sqrt > 0 and coeff != 0:
                    result = result + lam_sqrt * coeff * phi.evaluate(x)
            return result

        return Function(self._domain, evaluate_callable=evaluate_sample)

    def samples(self, n: int):
        """Draw n samples from the Gaussian measure.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        list of Function
            List of n sample functions.
        """
        return [self.sample() for _ in range(n)]


__all__ = ["KLSampler", "TruncationInfo"]
