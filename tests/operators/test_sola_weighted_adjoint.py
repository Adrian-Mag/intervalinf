"""SOLAOperator correctness on mass-weighted (WeightedLebesgue) domains.

The SOLA forward is the (inner-product-independent) data functional
``(G f)_i = ∫ k_i(x) f(x) dx``.  The adjoint is defined w.r.t. the *model* inner
product: ``⟨G f, y⟩_{R^Nd} = ⟨f, G* y⟩_M``.  For a weighted model space
``⟨·,·⟩_w`` this forces ``G* y = M⁻¹(Σ_i y_i k_i)`` (M = ×w).

These tests check:
1. the adjoint identity holds on a WeightedLebesgue domain (generic adjoint
   path routes through ``from_dual`` = M⁻¹), and
2. the cached/fast Gram assembly equals the correct ``∫ k_i M⁻¹(k_j) dx``
   (this is the path historically flagged ``TODO(mass-weighted)``).
"""

import numpy as np
from numpy.testing import assert_allclose

from pygeoinf import EuclideanSpace
from intervalinf.core.domain import IntervalDomain
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.spaces.weighted_lebesgue import WeightedLebesgue
from intervalinf.operators import SOLAOperator


A, B = 0.2, 1.0
INTEG = IntegrationConfig(method="simpson", n_points=4000)


def _w(r):
    return np.asarray(r, dtype=float) ** 2


_KERNELS = [
    lambda x: np.sin(np.pi * (np.asarray(x) - A) / (B - A)),
    lambda x: np.cos(np.asarray(x)),
    lambda x: np.asarray(x, dtype=float),
]


def _make(domain_space, cache=False):
    cod = EuclideanSpace(len(_KERNELS))
    return SOLAOperator(domain_space, cod, _KERNELS,
                        cache_kernels=cache, integration_config=INTEG)


def _wspace():
    return WeightedLebesgue(0, IntervalDomain(A, B), _w, integration_config=INTEG)


def _fn(callable_):
    return Function(IntervalDomain(A, B), evaluate_callable=callable_)


def test_forward_is_plain_data_functional():
    """Forward is ∫ k_i f dx, independent of the model inner product weight."""
    G = _make(_wspace())
    f = _fn(lambda x: np.asarray(x) ** 2)
    got = G(f)
    r = np.linspace(A, B, 4001)
    expected = np.array([np.trapezoid(k(r) * f(r), r) for k in _KERNELS])
    assert_allclose(got, expected, rtol=1e-3, atol=1e-4)


def test_adjoint_identity_weighted():
    """⟨G f, y⟩ == ⟨f, G* y⟩_w on a WeightedLebesgue domain."""
    space = _wspace()
    G = _make(space)
    f = _fn(lambda x: np.sin(2 * np.asarray(x)))
    y = np.array([0.7, -1.3, 0.4])

    lhs = float(np.dot(G(f), y))           # ⟨G f, y⟩_{R^Nd}
    gstar_y = G.adjoint(y)                  # element of the weighted model space
    rhs = space.inner_product(f, gstar_y)   # ⟨f, G* y⟩_w
    assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-6)


def test_adjoint_equals_Minv_kernel_sum_weighted():
    """G* y == M⁻¹(Σ y_i k_i) = (Σ y_i k_i)/w  for the r² weight."""
    space = _wspace()
    G = _make(space)
    y = np.array([1.0, 0.5, -2.0])
    gstar_y = G.adjoint(y)
    r = np.linspace(A, B, 500)
    ksum = sum(y[i] * _KERNELS[i](r) for i in range(len(_KERNELS)))
    expected = ksum / _w(r)
    assert_allclose(gstar_y(r), expected, rtol=1e-6, atol=1e-8)


def test_adjoint_identity_plain_unchanged():
    """Plain L² domain: adjoint identity with the standard inner product."""
    space = Lebesgue(0, IntervalDomain(A, B), basis=None, integration_config=INTEG)
    G = _make(space)
    f = _fn(lambda x: np.sin(2 * np.asarray(x)))
    y = np.array([0.7, -1.3, 0.4])
    lhs = float(np.dot(G(f), y))
    rhs = space.inner_product(f, G.adjoint(y))
    assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-6)


def test_fast_gram_weighted_matches_Minv_pairwise():
    """Cached fast Gram (G G*)_{ij} must equal ∫ k_i M⁻¹(k_j) dx on a weighted
    domain — i.e. ∫ k_i k_j / w dx, NOT the unweighted ∫ k_i k_j dx."""
    space = _wspace()
    G = _make(space, cache=True)
    # Trigger kernel-eval cache population via a forward apply on the shared mesh.
    G(_fn(lambda x: np.ones_like(np.asarray(x, dtype=float))))

    fast = G.compute_gram_matrix_fast()
    r = np.linspace(A, B, 4001)
    n = len(_KERNELS)
    expected = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            expected[i, j] = np.trapezoid(
                _KERNELS[i](r) * _KERNELS[j](r) / _w(r), r
            )
    assert_allclose(fast, expected, rtol=1e-2, atol=1e-3)


def test_fast_gram_plain_unchanged():
    """Plain L² domain: fast Gram == ∫ k_i k_j dx (regression)."""
    space = Lebesgue(0, IntervalDomain(A, B), basis=None, integration_config=INTEG)
    G = _make(space, cache=True)
    G(_fn(lambda x: np.ones_like(np.asarray(x, dtype=float))))
    fast = G.compute_gram_matrix_fast()
    r = np.linspace(A, B, 4001)
    n = len(_KERNELS)
    expected = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            expected[i, j] = np.trapezoid(_KERNELS[i](r) * _KERNELS[j](r), r)
    assert_allclose(fast, expected, rtol=1e-2, atol=1e-3)
