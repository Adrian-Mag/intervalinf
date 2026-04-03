"""Tests for hidden function materialization caches."""

import numpy as np

from intervalinf.core import Function, IntervalDomain
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.materialization import RepresentationSpec
from intervalinf.spaces.lebesgue import Lebesgue


def _make_spec(
    domain: IntervalDomain,
    *,
    n_points: int = 11,
) -> RepresentationSpec:
    return RepresentationSpec(
        kind="fixed_grid",
        n_points=n_points,
        interval=(domain.a, domain.b),
        method="uniform",
    )


def test_representation_spec_hashable():
    spec = RepresentationSpec(
        kind="fixed_grid",
        n_points=17,
        interval=(0.0, 1.0),
        method="uniform",
    )

    cache = {spec: "cached"}

    assert cache[spec] == "cached"


def test_representation_spec_equality():
    spec_a = RepresentationSpec(
        kind="fixed_grid",
        n_points=17,
        interval=(0.0, 1.0),
        method="uniform",
    )
    spec_b = RepresentationSpec(
        kind="fixed_grid",
        n_points=17,
        interval=(0.0, 1.0),
        method="uniform",
    )
    spec_c = RepresentationSpec(
        kind="fixed_grid",
        n_points=33,
        interval=(0.0, 1.0),
        method="uniform",
    )

    assert spec_a == spec_b
    assert hash(spec_a) == hash(spec_b)
    assert spec_a != spec_c


def test_materialize_creates_grid_and_values():
    domain = IntervalDomain(0.0, 1.0)
    f = Function(domain, evaluate_callable=lambda x: np.asarray(x) ** 2)
    spec = _make_spec(domain, n_points=11)

    materialized = f.materialize(spec)

    assert materialized.spec == spec
    assert materialized.grid.shape == (11,)
    assert materialized.values.shape == (11,)
    np.testing.assert_allclose(
        materialized.grid,
        np.linspace(0.0, 1.0, 11),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        materialized.values,
        materialized.grid ** 2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_materialize_caches_result():
    domain = IntervalDomain(0.0, 1.0)
    spec = _make_spec(domain, n_points=21)
    call_counter = {"count": 0}

    def counting_callable(x):
        call_counter["count"] += 1
        x = np.asarray(x)
        return np.sin(np.pi * x)

    f = Function(domain, evaluate_callable=counting_callable)

    first = f.materialize(spec)
    second = f.materialize(spec)

    assert first is second
    assert call_counter["count"] == 1


def test_materialize_evaluates_correctly():
    domain = IntervalDomain(0.0, 1.0)
    f = Function(domain, evaluate_callable=lambda x: np.exp(np.asarray(x)))
    spec = _make_spec(domain, n_points=19)

    materialized = f.materialize(spec)
    expected = np.exp(materialized.grid)

    np.testing.assert_allclose(
        materialized.values,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_clear_materializations():
    domain = IntervalDomain(0.0, 1.0)
    spec = _make_spec(domain, n_points=15)
    call_counter = {"count": 0}

    def counting_callable(x):
        call_counter["count"] += 1
        x = np.asarray(x)
        return x + 1.0

    f = Function(domain, evaluate_callable=counting_callable)

    first = f.materialize(spec)
    f.clear_materializations()
    second = f.materialize(spec)

    assert first is not second
    assert call_counter["count"] == 2


def test_inner_product_with_materialization_matches_original():
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(
        0,
        domain,
        integration_config=IntegrationConfig(method="trapz", n_points=501),
        weight=lambda x: 1.0 + np.asarray(x),
    )
    u = Function(
        space,
        evaluate_callable=lambda x: (
            np.sin(np.pi * np.asarray(x)) + np.asarray(x)
        ),
    )
    v = Function(
        space,
        evaluate_callable=lambda x: (
            np.cos(np.pi * np.asarray(x)) + np.asarray(x) ** 2
        ),
    )

    expected = (u * v).integrate(
        method=space.integration_method,
        n_points=space.integration_npoints,
        weight=space._weight,
    )
    actual = space.inner_product(u, v)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_inner_product_caches_materializations():
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(
        0,
        domain,
        integration_config=IntegrationConfig(method="simpson", n_points=401),
    )
    u = Function(
        space,
        evaluate_callable=lambda x: np.sin(np.pi * np.asarray(x)),
    )
    v = Function(
        space,
        evaluate_callable=lambda x: np.cos(np.pi * np.asarray(x)),
    )
    spec = _make_spec(domain, n_points=space.integration_npoints)

    _ = space.inner_product(u, v)

    assert u.get_materialized(spec) is not None
    assert v.get_materialized(spec) is not None


def test_materialization_survives_arithmetic():
    domain = IntervalDomain(0.0, 1.0)
    spec = _make_spec(domain, n_points=25)
    u = Function(domain, evaluate_callable=lambda x: np.asarray(x))
    v = Function(domain, evaluate_callable=lambda x: np.asarray(x) ** 2)

    summed = (u + v).materialize(spec)
    product = (u * v).materialize(spec)

    np.testing.assert_allclose(
        summed.values,
        summed.grid + summed.grid ** 2,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        product.values,
        summed.grid ** 3,
        rtol=1e-12,
        atol=1e-12,
    )


def test_different_specs_independent():
    domain = IntervalDomain(0.0, 1.0)
    spec_1 = _make_spec(domain, n_points=9)
    spec_2 = _make_spec(domain, n_points=27)
    call_counter = {"count": 0}

    def counting_callable(x):
        call_counter["count"] += 1
        x = np.asarray(x)
        return x ** 2 + 1.0

    f = Function(domain, evaluate_callable=counting_callable)

    materialized_1 = f.materialize(spec_1)
    materialized_2 = f.materialize(spec_2)

    assert materialized_1 is not materialized_2
    assert call_counter["count"] == 2
    assert f.get_materialized(spec_1) is materialized_1
    assert f.get_materialized(spec_2) is materialized_2
