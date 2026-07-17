"""Weighted Lebesgue space L²([a,b]; w) as a mass-weighted Hilbert space.

The weighted inner product

    ⟨u, v⟩_w = ∫_a^b u(x) v(x) w(x) dx

is realized through pygeoinf's :class:`~pygeoinf.MassWeightedHilbertSpace`: a
plain (unweighted) :class:`~intervalinf.spaces.lebesgue.Lebesgue` underlying
space together with the **mass operator** ``M = (× w)`` (pointwise multiplication
by the weight) and its inverse ``M⁻¹ = (× 1/w)``.  Then

    ⟨u, v⟩_w = ⟨M u, v⟩_{L²} = ∫ (w u) v dx,

and, crucially, the Riesz maps are weight-consistent by construction
(``to_dual = R∘M``, ``from_dual = M⁻¹∘R⁻¹``).  This is what makes downstream
operator adjoints (SOLA, Bessel-Sobolev) correct on weighted domains — see the
``weighting-via-massweighted-design`` decision.

This is the single, unified weighting mechanism; it replaces the former
``Lebesgue(weight=...)`` constructor argument, whose dual maps did not thread the
weight through.

The multiplicative mass is the diagonal special case of a general
self-adjoint positive mass operator (e.g. :class:`~intervalinf.spaces.sobolev.Sobolev`
uses the differential mass ``(k²I + Δ)^s``).

Note on the ``r=0`` singularity: for ``w(r)=r²`` the inverse weight ``1/w`` is
singular at the origin (the mass is only positive *semi*-definite there).  Use a
domain ``[ε, R]`` with ``ε>0``, or pass a regularized ``inverse_weight``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from pygeoinf import MassWeightedHilbertSpace, LinearOperator

from intervalinf.core.config import IntegrationConfig, ParallelConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.forms import LinearFormKernel
from intervalinf.spaces.lebesgue import Lebesgue


class WeightedLebesgue(MassWeightedHilbertSpace):
    """``L²([a,b]; w)`` realized as a mass-weighted plain ``Lebesgue`` space.

    Parameters
    ----------
    dim : int
        Basis count for basis-backed operations (``0`` for basis-free use).
    function_domain : IntervalDomain
        The interval ``[a, b]``.
    weight : callable
        The inner-product weight ``w(x) > 0``.  Must accept a numpy array and
        return an array of the same shape.
    inverse_weight : callable, optional
        ``1/w(x)``.  Defaults to ``lambda x: 1.0 / weight(x)``.  Provide an
        explicit (e.g. regularized) inverse when ``w`` vanishes on the domain.
    basis : str or list, optional
        Basis for the underlying ``Lebesgue`` space (default basis-free).
    integration_config, parallel_config :
        Forwarded to the underlying ``Lebesgue`` space.
    """

    def __init__(
        self,
        dim: int,
        function_domain,
        weight: Callable,
        *,
        inverse_weight: Optional[Callable] = None,
        basis=None,
        integration_config: Optional[IntegrationConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        underlying = Lebesgue(
            dim,
            function_domain,
            basis=basis,
            integration_config=integration_config,
            parallel_config=parallel_config,
        )

        if inverse_weight is None:
            def inverse_weight(x, _w=weight):
                return 1.0 / np.asarray(_w(x), dtype=float)

        # NB: deliberately not named ``_weight`` — ``LinearFormKernel`` keys off
        # ``getattr(domain, '_weight', None)`` (the legacy Lebesgue(weight=)
        # mechanism) and would otherwise divide our already-mass-folded kernel
        # by w. Here the weight lives entirely in the mass operator.
        self._weight_fn = weight
        self._inverse_weight = inverse_weight
        self._function_domain = underlying.function_domain

        w_func = Function(self._function_domain, evaluate_callable=weight)
        w_inv_func = Function(self._function_domain, evaluate_callable=inverse_weight)

        def _mul_w(f, _w=w_func):
            return _w * f

        def _mul_w_inv(f, _wi=w_inv_func):
            return _wi * f

        # Multiplication by w is self-adjoint w.r.t. the plain L² inner product:
        # ⟨w f, g⟩ = ∫ w f g = ⟨f, w g⟩.
        mass = LinearOperator(
            underlying, underlying, _mul_w, adjoint_mapping=_mul_w
        )
        inverse_mass = LinearOperator(
            underlying, underlying, _mul_w_inv, adjoint_mapping=_mul_w_inv
        )

        super().__init__(underlying, mass, inverse_mass)

    @property
    def function_domain(self):
        """The underlying ``IntervalDomain``."""
        return self._function_domain

    @property
    def weight(self) -> Callable:
        """The inner-product weight ``w(x)``."""
        return self._weight_fn

    @property
    def integration(self):
        """Integration configuration of the underlying space."""
        return self.underlying_space.integration

    @property
    def parallel(self):
        """Parallel configuration of the underlying space."""
        return self.underlying_space.parallel

    def to_dual(self, x: Function) -> LinearFormKernel:
        """Riesz map ``φ_x(y) = ⟨x, y⟩_w = ⟨M x, y⟩_{L²}``.

        Returns a kernel form whose kernel is ``M x = w·x``, so the (plain)
        kernel pairing ``∫ (w x) y dx`` reproduces the weighted inner product.
        """
        if not isinstance(x, Function):
            raise TypeError("Expected Function for primal element")
        kernel = self._mass_operator(x)
        return LinearFormKernel(
            self,
            kernel=kernel,
            integration_config=self.underlying_space.integration.dual,
            parallel_config=self.underlying_space.parallel.dual,
        )

    def from_dual(self, xp) -> Function:
        """Inverse Riesz map ``x = M⁻¹ (kernel)``."""
        if isinstance(xp, LinearFormKernel):
            return self._inverse_mass_operator(xp.kernel)
        kernel = self.underlying_space.from_dual(xp)
        return self._inverse_mass_operator(kernel)
