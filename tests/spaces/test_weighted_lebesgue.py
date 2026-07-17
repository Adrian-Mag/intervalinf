"""Tests for WeightedLebesgue — L²([a,b]; w) as a MassWeightedHilbertSpace.

The weighted inner product ⟨u,v⟩ = ∫ u v w dx is realized through pygeoinf's
`MassWeightedHilbertSpace` with mass operator M = (×w) over a plain `Lebesgue`
underlying space. This is the single, unified weighting mechanism (replacing the
old `Lebesgue(weight=...)`): because the mass operator carries the weight, the
Riesz maps (`to_dual`/`from_dual`) are automatically weight-consistent, which is
what makes downstream operator adjoints (SOLA, Bessel) correct.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from pygeoinf import MassWeightedHilbertSpace
from intervalinf.core.domain import IntervalDomain
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.weighted_lebesgue import WeightedLebesgue


def _w(r):
    return np.asarray(r, dtype=float) ** 2


def _make_space(a=0.1, b=1.0, n_points=6000):
    domain = IntervalDomain(a, b)
    integration = IntegrationConfig(method="simpson", n_points=n_points)
    return WeightedLebesgue(0, domain, _w, integration_config=integration)


def _fn(space, callable_):
    return Function(space.function_domain, evaluate_callable=callable_)


def test_is_mass_weighted_subclass():
    space = _make_space()
    assert isinstance(space, MassWeightedHilbertSpace)
    # The mass operator and its inverse are exposed by the base class.
    assert space.mass_operator is not None
    assert space.inverse_mass_operator is not None


def test_inner_product_matches_weighted_integral():
    space = _make_space()
    u = _fn(space, lambda r: np.sin(np.pi * np.asarray(r)))
    v = _fn(space, lambda r: np.asarray(r) ** 1.5)

    got = space.inner_product(u, v)
    expected, _ = quad(lambda r: np.sin(np.pi * r) * (r ** 1.5) * (r ** 2), 0.1, 1.0)
    assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


def test_norm_squared_matches_weighted_integral():
    space = _make_space()
    u = _fn(space, lambda r: np.cos(np.asarray(r)))
    got = space.inner_product(u, u)
    expected, _ = quad(lambda r: np.cos(r) ** 2 * r ** 2, 0.1, 1.0)
    assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


def test_mass_operator_multiplies_by_weight():
    space = _make_space()
    f = _fn(space, lambda r: np.ones_like(np.asarray(r, dtype=float)))
    Mf = space.mass_operator(f)
    r = np.linspace(0.1, 1.0, 50)
    assert_allclose(Mf(r), _w(r), rtol=1e-12, atol=1e-12)


def test_from_dual_to_dual_roundtrip_applies_inverse_mass():
    """from_dual(to_dual(x)) == x, i.e. M⁻¹(M x) == x — the Riesz round trip."""
    space = _make_space()
    x = _fn(space, lambda r: np.sin(2 * np.pi * np.asarray(r)) + 0.5)
    xp = space.to_dual(x)        # kernel = M x = w·x
    x_back = space.from_dual(xp)  # M⁻¹ kernel = x
    r = np.linspace(0.1, 1.0, 100)
    assert_allclose(x_back(r), x(r), rtol=1e-10, atol=1e-12)


def test_to_dual_kernel_is_weighted():
    """to_dual(x) carries kernel = w·x so the (plain) form pairing reproduces
    the weighted inner product."""
    space = _make_space()
    x = _fn(space, lambda r: np.asarray(r, dtype=float))
    xp = space.to_dual(x)
    r = np.linspace(0.1, 1.0, 50)
    assert_allclose(xp.kernel(r), _w(r) * r, rtol=1e-10, atol=1e-12)


def test_function_domain_and_dim_delegate():
    space = _make_space()
    assert space.function_domain.a == 0.1
    assert space.function_domain.b == 1.0
    assert space.dim == 0


def test_custom_inverse_weight():
    """An explicit inverse_weight is used for M⁻¹ (lets callers regularize)."""
    domain = IntervalDomain(0.1, 1.0)
    integration = IntegrationConfig(method="simpson", n_points=2000)
    space = WeightedLebesgue(
        0, domain, _w, inverse_weight=lambda r: 1.0 / (np.asarray(r) ** 2),
        integration_config=integration,
    )
    f = _fn(space, lambda r: np.asarray(r, dtype=float) ** 2)
    Minv_f = space.inverse_mass_operator(f)
    r = np.linspace(0.1, 1.0, 50)
    assert_allclose(Minv_f(r), np.ones_like(r), rtol=1e-10, atol=1e-12)
