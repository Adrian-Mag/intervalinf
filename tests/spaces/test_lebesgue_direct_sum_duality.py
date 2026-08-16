"""Regression tests for LebesgueSpaceDirectSum duality with weighted members.

Found 2026-08-16 (thesis ch7 model-space checks): `to_dual` used the raw
member functions as the kernel, so mass-weighted members lost their
weights -- the direct-sum inner product of `WeightedLebesgue` members
evaluated the UNWEIGHTED pairing (orders of magnitude wrong for small
weights), while plain-Lebesgue sums were correct by accident of the
identity Riesz map. The fix delegates to each member's own `to_dual`
(collecting the member Riesz kernels) and inverts memberwise in
`from_dual`.

These tests pin:
  1. direct-sum inner product == sum of member inner products, for
     weighted members, plain members, and nested sums;
  2. to_dual/from_dual is a memberwise Riesz round trip;
  3. plain-member kernels are still the functions themselves (the old
     behaviour, correct there, is preserved).
"""

import numpy as np
from numpy.testing import assert_allclose

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.lebesgue import Lebesgue, LebesgueSpaceDirectSum
from intervalinf.spaces.weighted_lebesgue import WeightedLebesgue

INTEGRATION = IntegrationConfig(method="simpson", n_points=4000)


def _weighted(a, b, scale):
    def w(r, scale=scale):
        return scale * np.asarray(r, dtype=float) ** 2

    return WeightedLebesgue(
        0, IntervalDomain(a, b), w, integration_config=INTEGRATION
    )


def _plain(a, b):
    return Lebesgue(0, IntervalDomain(a, b), integration_config=INTEGRATION)


def _fn(space, amp, k):
    r_b = space.function_domain.b

    def evaluate(r, amp=amp, k=k, r_b=r_b):
        return amp * np.sin(k * np.asarray(r, dtype=float) / r_b + 0.3)

    return Function(space.function_domain, evaluate_callable=evaluate)


def test_weighted_pair_inner_product_is_sum_of_members():
    # Mimics the thesis vs inner-core/mantle pair: disjoint domains,
    # small shared weight scale.
    s1 = _weighted(0.0, 1221.0, 1.0e-12)
    s2 = _weighted(3480.0, 6371.0, 1.0e-12)
    pair = LebesgueSpaceDirectSum([s1, s2])

    x = [_fn(s1, 0.5, 1.2), _fn(s2, 0.6, 1.9)]
    y = [_fn(s1, 0.4, 2.7), _fn(s2, 0.3, 0.8)]

    expected = float(s1.inner_product(x[0], y[0])) + float(
        s2.inner_product(x[1], y[1])
    )
    got = float(pair.inner_product(x, y))
    assert_allclose(got, expected, rtol=1e-10)
    # The regression this guards against was catastrophic, not subtle:
    # the unweighted pairing is ~5e4x larger for these weights.
    unweighted = float((x[0] * y[0]).integrate()) + float(
        (x[1] * y[1]).integrate()
    )
    assert abs(got - unweighted) > 1.0e3 * abs(got)


def test_nested_mixed_sum_inner_product_is_sum_of_members():
    plain = _plain(0.0, 1.0)
    w1 = _weighted(0.0, 1.0, 3.0)
    w2 = _weighted(0.5, 2.0, 0.25)
    inner_pair = LebesgueSpaceDirectSum([w1, w2])
    outer = LebesgueSpaceDirectSum([plain, inner_pair])

    x = [_fn(plain, 1.0, 2.0), [_fn(w1, 0.5, 1.2), _fn(w2, 0.6, 1.9)]]
    y = [_fn(plain, 0.7, 3.1), [_fn(w1, 0.4, 2.7), _fn(w2, 0.3, 0.8)]]

    expected = (
        float(plain.inner_product(x[0], y[0]))
        + float(w1.inner_product(x[1][0], y[1][0]))
        + float(w2.inner_product(x[1][1], y[1][1]))
    )
    assert_allclose(float(outer.inner_product(x, y)), expected, rtol=1e-10)


def test_to_dual_from_dual_round_trip_weighted():
    w1 = _weighted(0.1, 1.0, 2.0)
    w2 = _weighted(1.0, 3.0, 5.0)
    pair = LebesgueSpaceDirectSum([w1, w2])
    x = [_fn(w1, 0.5, 1.2), _fn(w2, 0.6, 1.9)]

    back = pair.from_dual(pair.to_dual(x))
    r1 = np.linspace(0.1, 1.0, 101)
    r2 = np.linspace(1.0, 3.0, 101)
    assert_allclose(back[0].evaluate(r1), x[0].evaluate(r1), rtol=1e-8)
    assert_allclose(back[1].evaluate(r2), x[1].evaluate(r2), rtol=1e-8)


def test_plain_member_kernels_unchanged():
    p1 = _plain(0.0, 1.0)
    p2 = _plain(1.0, 2.0)
    pair = LebesgueSpaceDirectSum([p1, p2])
    x = [_fn(p1, 0.9, 1.1), _fn(p2, 0.2, 4.1)]

    form = pair.to_dual(x)
    r1 = np.linspace(0.0, 1.0, 51)
    r2 = np.linspace(1.0, 2.0, 51)
    # Identity Riesz map for plain L2 members: kernel is the function.
    assert_allclose(form.kernel[0].evaluate(r1), x[0].evaluate(r1), rtol=1e-12)
    assert_allclose(form.kernel[1].evaluate(r2), x[1].evaluate(r2), rtol=1e-12)

    expected = float(p1.inner_product(x[0], x[0])) + float(
        p2.inner_product(x[1], x[1])
    )
    assert_allclose(float(pair.inner_product(x, x)), expected, rtol=1e-10)
