"""
prior_viz.py
============

Prior visualisation helpers for the full-spectrum splitting-function PLI example.

Phase 1 — PriorViewer
~~~~~~~~~~~~~~~~~~~~~
:class:`PriorViewer` precomputes reference std arrays and sample curves for
each of the four radial model parameters (``'vp'``, ``'vs_IC'``, ``'vs_M'``,
``'rho'``) using the eigendecomposition already stored inside the shared
:class:`~intervalinf.operators.BesselSobolevInverse` operators.

All quantities are cached at τ=1.  Interactive updates (τ slider moves,
block switches) are then purely arithmetic — no operator calls are made after
:meth:`PriorViewer.precompute` returns.

Phase 2 — make_prior_widget (ipywidgets interactive viewer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
See :func:`make_prior_widget` (added in Phase 2).

Usage example
-------------
::

    from full_spectrum_utils import build_shared_bessel_blocks, RadialSpecs
    from prior_viz import PriorViewer

    specs = RadialSpecs(n_basis=100)
    shared = build_shared_bessel_blocks(specs)   # pre-built from notebook Phase 3

    viewer = PriorViewer(shared, specs, n_grid=300, n_samples=5)
    viewer.precompute()   # ~10-30 s depending on n_basis

    data = viewer.get_display_data(block, tau_vp=1.5, sigma_var=100.0)
    # data['vp']['std'], data['vp']['samples'], etc. — instant from cache

Mathematical basis
------------------
The covariance operator for parameter *p* is

    C_p = τ²  C_ref_p

where C_ref_p = :class:`~intervalinf.operators.BesselSobolevInverse` has the
KL decomposition

    C_ref_p  f  =  Σ_k  λ_k  ⟨f, φ_k⟩  φ_k

with (λ_k, φ_k) the eigenpairs of the Laplacian-based spectral operator.
Pointwise variance at τ=1:

    σ²_ref(r)  =  Σ_k  λ_k  φ_k(r)²

A sample from N(0, C_ref_p) is:

    f_ref(r)  =  Σ_k  √λ_k  z_k  φ_k(r),   z_k ~ N(0, 1)

At arbitrary τ: std = τ · std_ref,  sample = τ · sample_ref.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from full_spectrum_utils import BlockIndex, RadialSpecs
    from intervalinf.operators import BesselSobolevInverse


# =============================================================================
# Internal data containers
# =============================================================================

@dataclasses.dataclass
class _PriorComponentData:
    """Reference data at τ=1 for one radial parameter.

    All arrays are at τ=1; :meth:`PriorViewer.get_display_data` scales them.

    Attributes
    ----------
    r_grid : np.ndarray, shape (n_grid,)
        Radial evaluation grid in km.
    ref_std : np.ndarray, shape (n_grid,)
        Pointwise standard deviation at τ=1.
    ref_samples : np.ndarray, shape (n_samples, n_grid)
        Random sample curves at τ=1.
    _eigenvalues : np.ndarray, shape (n_basis,)
        KL eigenvalues (stored to allow :meth:`PriorViewer.resample` without
        re-fetching eigenpairs).
    _phi_matrix : np.ndarray, shape (n_basis, n_grid)
        Eigenfunction values on ``r_grid`` (stored for :meth:`PriorViewer.resample`).
    """

    r_grid: np.ndarray
    ref_std: np.ndarray
    ref_samples: np.ndarray
    _eigenvalues: np.ndarray
    _phi_matrix: np.ndarray


# Domain bounds per parameter — expressed as (a, b) lambdas over RadialSpecs.
_PARAM_DOMAINS: Dict[str, Callable] = {
    "vp":    lambda s: (0.0, s.earth_radius_km),
    "vs_IC": lambda s: (0.0, s.icb_radius_km),
    "vs_M":  lambda s: (s.cmb_radius_km, s.earth_radius_km),
    "rho":   lambda s: (0.0, s.earth_radius_km),
}

_RADIAL_PARAMS = ("vp", "vs_IC", "vs_M", "rho")


# =============================================================================
# PriorViewer
# =============================================================================

class PriorViewer:
    """Precompute and serve display data for block prior visualisation.

    All expensive work is done once in :meth:`precompute`.  After that,
    :meth:`get_display_data` performs only arithmetic (τ scaling), so
    interactive updates are instant regardless of how many blocks exist.

    Parameters
    ----------
    shared_bessel : dict
        Output of :func:`~full_spectrum_utils.build_shared_bessel_blocks`.
        Keys: ``'vp'``, ``'vs_IC'``, ``'vs_M'``, ``'rho'``.
    specs : RadialSpecs
        Shared domain / configuration (domain bounds, ``n_basis``).
    n_grid : int
        Number of radial evaluation points per component.  Default 300.
    n_samples : int
        Number of sample curves to precompute per component.  Default 5.
    seed : int
        Random seed for reproducible samples.  Default 42.
    """

    def __init__(
        self,
        shared_bessel: Dict[str, "BesselSobolevInverse"],
        specs: "RadialSpecs",
        *,
        n_grid: int = 300,
        n_samples: int = 5,
        seed: int = 42,
    ) -> None:
        self._shared_bessel = shared_bessel
        self._specs = specs
        self._n_grid = n_grid
        self._n_samples = n_samples
        self._seed = seed
        self._ref_data: Dict[str, _PriorComponentData] = {}
        self._scalar_z: Optional[np.ndarray] = None  # shape (n_samples,)

    # ── Precomputation ───────────────────────────────────────────────────────

    def precompute(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Precompute reference std and sample curves for all four parameters.

        Iterates over ``('vp', 'vs_IC', 'vs_M', 'rho')``.  For each parameter
        it evaluates all ``specs.n_basis`` eigenpairs of the corresponding
        :class:`~intervalinf.operators.BesselSobolevInverse` on a dense grid,
        then constructs the pointwise-variance std and ``n_samples`` KL samples.

        After this call, :attr:`is_precomputed` is ``True``.

        Parameters
        ----------
        progress_cb : callable, optional
            Invoked with each parameter name string after it completes.
            Useful for notebook progress bars.

        Notes
        -----
        Eigenfunction evaluation uses array-valued calls (``phi_k(r_grid)``)
        when the underlying callable is vectorised (the default for spectral
        providers).  If a particular provider is not vectorised, the code
        falls back to a scalar loop automatically.
        """
        rng = np.random.default_rng(self._seed)
        n_basis = self._specs.n_basis

        for p in _RADIAL_PARAMS:
            a, b = _PARAM_DOMAINS[p](self._specs)
            r_grid = np.linspace(a, b, self._n_grid)
            bessel = self._shared_bessel[p]

            eigenvalues = np.empty(n_basis)
            phi_matrix = np.empty((n_basis, self._n_grid))

            for k in range(n_basis):
                eigenvalues[k] = bessel.get_eigenvalue(k)
                phi_k = bessel.get_eigenfunction(k)
                # Try vectorised evaluation; fall back to scalar loop.
                try:
                    vals = phi_k(r_grid)
                    if not isinstance(vals, np.ndarray) or vals.shape != (self._n_grid,):
                        raise ValueError
                    phi_matrix[k] = vals
                except Exception:
                    phi_matrix[k] = np.array([phi_k(r) for r in r_grid])

            # Reference pointwise std: sqrt(Σ_k λ_k φ_k(r)²)
            ref_var = eigenvalues @ (phi_matrix ** 2)  # (n_grid,)
            ref_std = np.sqrt(np.maximum(0.0, ref_var))

            # Reference samples: shape (n_samples, n_grid)
            # f_ref(r) = Σ_k sqrt(λ_k) z_k φ_k(r),  z ~ N(0,1)
            z = rng.standard_normal((self._n_samples, n_basis))  # (n_samples, n_basis)
            sqrt_lam = np.sqrt(np.maximum(0.0, eigenvalues))      # (n_basis,)
            ref_samples = z @ (sqrt_lam[:, None] * phi_matrix)    # (n_samples, n_grid)

            self._ref_data[p] = _PriorComponentData(
                r_grid=r_grid,
                ref_std=ref_std,
                ref_samples=ref_samples,
                _eigenvalues=eigenvalues,
                _phi_matrix=phi_matrix,
            )

            if progress_cb is not None:
                progress_cb(p)

        # Scalar component z-values (scaled by sqrt(sigma_var) in get_display_data)
        self._scalar_z = rng.standard_normal(self._n_samples)

    def resample(self, seed: Optional[int] = None) -> None:
        """Redraw sample curves without rebuilding eigenpairs.

        The pointwise std (:attr:`~_PriorComponentData.ref_std`) is
        unchanged — it depends only on eigenvalues, not on the random draw.

        Parameters
        ----------
        seed : int, optional
            New random seed.  If ``None``, uses a fresh random integer.

        Raises
        ------
        RuntimeError
            If :meth:`precompute` has not been called yet.
        """
        if not self._ref_data:
            raise RuntimeError("Call precompute() before resample().")
        rng = np.random.default_rng(
            seed if seed is not None else int(np.random.randint(0, 2**31))
        )
        for data in self._ref_data.values():
            n_basis = len(data._eigenvalues)
            z = rng.standard_normal((self._n_samples, n_basis))
            sqrt_lam = np.sqrt(np.maximum(0.0, data._eigenvalues))
            data.ref_samples = z @ (sqrt_lam[:, None] * data._phi_matrix)

        self._scalar_z = rng.standard_normal(self._n_samples)

    def invalidate_bessel(
        self,
        new_shared_bessel: Dict[str, "BesselSobolevInverse"],
    ) -> None:
        """Replace the underlying Bessel operators and clear the cache.

        Call this when ``specs.n_basis`` or the Bessel hyperparameters change
        (e.g., after rebuilding
        :func:`~full_spectrum_utils.build_shared_bessel_blocks`), then call
        :meth:`precompute` again.

        Parameters
        ----------
        new_shared_bessel : dict
            Fresh output of
            :func:`~full_spectrum_utils.build_shared_bessel_blocks`.
        """
        self._shared_bessel = new_shared_bessel
        self._ref_data.clear()
        self._scalar_z = None

    # ── Display data ─────────────────────────────────────────────────────────

    def get_display_data(
        self,
        block: "BlockIndex",
        *,
        tau_vp: float = 1.0,
        tau_vs_IC: float = 1.0,
        tau_vs_M: float = 1.0,
        tau_rho: float = 1.0,
        sigma_var: float = 100.0,
    ) -> Dict:
        """Return τ-scaled display data for *block*.

        This method performs only arithmetic (multiplication by τ); all
        operator calls were completed during :meth:`precompute`.

        Parameters
        ----------
        block : BlockIndex
            The (s, t) spectral block to display.  Not used computationally;
            carried through for labelling and future per-s τ support.
        tau_vp, tau_vs_IC, tau_vs_M, tau_rho : float
            Amplitude scaling applied uniformly across all s.  The prior
            covariance is $τ^2 C_\\mathrm{ref}$, so std scales by τ and
            samples scale by τ.
        sigma_var : float
            Variance for the scalar topography components σ₀ and σ₁.

        Returns
        -------
        dict
            Keys ``'vp'``, ``'vs_IC'``, ``'vs_M'``, ``'rho'`` each map to::

                {
                    'r':       np.ndarray of shape (n_grid,),   # km
                    'mean':    np.ndarray of shape (n_grid,),   # zeros
                    'std':     np.ndarray of shape (n_grid,),   # τ · std_ref
                    'samples': np.ndarray of shape (n_samples, n_grid),
                }

            Key ``'sigma_1'`` maps to::

                {
                    'mean':    0.0,
                    'std':     float  (= sqrt(sigma_var)),
                    'samples': np.ndarray of shape (n_samples,),
                }

        Raises
        ------
        RuntimeError
            If :meth:`precompute` has not been called yet.
        """
        if not self._ref_data:
            raise RuntimeError("Call precompute() before get_display_data().")

        taus = {
            "vp":    tau_vp,
            "vs_IC": tau_vs_IC,
            "vs_M":  tau_vs_M,
            "rho":   tau_rho,
        }

        result: Dict = {}
        for p, tau in taus.items():
            data = self._ref_data[p]
            result[p] = {
                "r":       data.r_grid,
                "mean":    np.zeros(self._n_grid),
                "std":     tau * data.ref_std,
                "samples": tau * data.ref_samples,  # (n_samples, n_grid)
            }

        sigma_std = float(np.sqrt(sigma_var))
        scalar_samples = (
            sigma_std * self._scalar_z
            if self._scalar_z is not None
            else np.zeros(self._n_samples)
        )
        result["sigma_1"] = {
            "mean":    0.0,
            "std":     sigma_std,
            "samples": scalar_samples,
        }

        return result

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_precomputed(self) -> bool:
        """``True`` if :meth:`precompute` has been called and data is available."""
        return bool(self._ref_data)

    @property
    def n_grid(self) -> int:
        """Number of radial evaluation points per component."""
        return self._n_grid

    @property
    def n_samples(self) -> int:
        """Number of precomputed sample curves per component."""
        return self._n_samples
