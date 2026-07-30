"""Regression tests for breakpoint-aware split quadrature.

The tests state the public contracts for declared discontinuities, not an
attempt to infer them from arbitrary callables.
"""

from __future__ import annotations

import numpy as np
import pytest
from pygeoinf import EuclideanSpace

from intervalinf import (
    Function,
    IntegrationConfig,
    IntervalDomain,
    Lebesgue,
    QuadratureRule,
)
from intervalinf.operators import SOLAOperator
from intervalinf.providers import BoxCarFunctionProvider, DiscontinuousFunctionProvider


@pytest.fixture
def domain() -> IntervalDomain:
    """Return the common closed unit interval."""
    return IntervalDomain(0.0, 1.0, boundary_type="closed")


@pytest.fixture
def space(domain: IntervalDomain) -> Lebesgue:
    """Return a basis-free space with enough nodes for the test rule."""
    return Lebesgue(
        0,
        domain,
        basis=None,
        integration_config=IntegrationConfig(method="split_gauss_legendre", n_points=32),
    )


def test_breakpoints_are_canonical_immutable_interior_metadata(
    domain: IntervalDomain,
) -> None:
    """Functions store declared integration splits, not a smoothness claim."""
    function = Function(
        domain,
        evaluate_callable=lambda x: np.asarray(x),
        breakpoints=[0.75, 0.25],
    )

    assert function.breakpoints == (0.25, 0.75)
    assert isinstance(function.breakpoints, tuple)

    with pytest.raises(ValueError, match="unique"):
        Function(
            domain,
            evaluate_callable=lambda x: np.asarray(x),
            breakpoints=[0.25, 0.25],
        )
    with pytest.raises(ValueError, match="strictly inside"):
        Function(
            domain,
            evaluate_callable=lambda x: np.asarray(x),
            breakpoints=[0.0],
        )
    with pytest.raises(ValueError, match="finite"):
        Function(
            domain,
            evaluate_callable=lambda x: np.asarray(x),
            breakpoints=[np.nan],
        )


def test_breakpoints_propagate_through_arithmetic_and_restriction(
    domain: IntervalDomain,
    space: Lebesgue,
) -> None:
    """Derived functions retain a conservative union of relevant splits."""
    left = Function(
        space,
        evaluate_callable=lambda x: np.asarray(x),
        breakpoints=(0.25,),
    )
    right = Function(
        space,
        evaluate_callable=lambda x: np.asarray(x) ** 2,
        breakpoints=(0.5, 0.75),
    )

    assert (left + right).breakpoints == (0.25, 0.5, 0.75)
    assert (left * right).breakpoints == (0.25, 0.5, 0.75)
    assert (-left).breakpoints == (0.25,)
    assert (0.0 * left).breakpoints == ()

    restricted = left.restrict(Lebesgue(0, IntervalDomain(0.0, 0.5), basis=None))
    assert restricted.breakpoints == (0.25,)


def test_step_provider_and_sola_reconstruction_keep_breakpoints(
    domain: IntervalDomain,
    space: Lebesgue,
) -> None:
    """Adjacent boxes remain full-support yet retain their internal joins."""
    centers = (np.arange(4) + 0.5) / 4.0
    provider = BoxCarFunctionProvider(
        domain,
        default_width=0.25,
        centers=centers,
        normalize=False,
        default_height=4.0,
    )
    operator = SOLAOperator(
        space,
        EuclideanSpace(4),
        kernels=provider,
        integration_config=IntegrationConfig(method="simpson", n_points=257),
    )

    assert provider.get_function_by_index(0).breakpoints == (0.25,)
    sparse = operator.adjoint(np.array([1.0, 0.0, 0.0, 0.0]))
    dense = operator.adjoint(np.ones(4))

    assert sparse.support == [(0.0, 0.25)]
    assert sparse.breakpoints == (0.25,)
    assert dense.support is None
    assert dense.breakpoints == (0.25, 0.5, 0.75)

    discontinuous = DiscontinuousFunctionProvider(
        domain, random_state=np.random.default_rng(42)
    ).get_random_function(n_discontinuities=3)
    assert len(discontinuous.breakpoints) == 3
    assert all(0.0 < point < 1.0 for point in discontinuous.breakpoints)


def test_split_gauss_legendre_uses_only_panel_interiors(
    domain: IntervalDomain,
) -> None:
    """Split rules must never query a pointwise convention at a jump."""
    rule = QuadratureRule.split_gauss_legendre(domain, breakpoints=(0.5,), n_points=8)
    assert np.all(rule.nodes > 0.0)
    assert np.all(rule.nodes < 1.0)
    assert not np.any(np.isclose(rule.nodes, 0.5))
    np.testing.assert_allclose(
        np.dot(rule.weights, rule.nodes**3),
        0.25,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    def endpoint_sensitive(x):
        values = np.asarray(x)
        if np.any(np.isclose(values, 0.5)):
            raise AssertionError("quadrature sampled the declared breakpoint")
        return values**2

    function = Function(
        domain,
        evaluate_callable=endpoint_sensitive,
        breakpoints=(0.5,),
    )
    np.testing.assert_allclose(
        function.integrate(method="split_gauss_legendre", n_points=8),
        1.0 / 3.0,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    with pytest.raises(ValueError, match="unique"):
        QuadratureRule.split_gauss_legendre(domain, breakpoints=(0.5, 0.5), n_points=8)
    with pytest.raises(ValueError, match="positive"):
        QuadratureRule.split_gauss_legendre(domain, n_points=0)

    uneven = QuadratureRule.split_gauss_legendre(domain, breakpoints=(0.01,), n_points=4)
    assert uneven.nodes.size == 4
    assert not uneven.nodes.flags.writeable
    with pytest.raises(ValueError):
        uneven.nodes[0] = 0.1


def test_split_method_uses_union_of_pairing_breakpoints(
    domain: IntervalDomain,
    space: Lebesgue,
) -> None:
    """The opt-in Lebesgue path splits at either operand's declared join."""

    def left_value(x):
        values = np.asarray(x)
        if np.any(np.isclose(values, 0.25)):
            raise AssertionError("pairing sampled the left breakpoint")
        return values + 1.0

    def right_value(x):
        values = np.asarray(x)
        if np.any(np.isclose(values, 0.75)):
            raise AssertionError("pairing sampled the right breakpoint")
        return values**2

    left = Function(space, evaluate_callable=left_value, breakpoints=(0.25,))
    right = Function(space, evaluate_callable=right_value, breakpoints=(0.75,))
    np.testing.assert_allclose(
        space.inner_product(left, right),
        7.0 / 12.0,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_shared_rule_gives_a_discrete_boxcar_adjoint_identity(
    domain: IntervalDomain,
    space: Lebesgue,
) -> None:
    """One split rule makes the plain-L2 SOLA adjoint exact discretely."""
    n_targets = 16
    width = 1.0 / n_targets
    centers = (np.arange(n_targets) + 0.5) * width
    provider = BoxCarFunctionProvider(
        domain,
        default_width=width,
        centers=centers,
        normalize=False,
        default_height=1.0 / width,
    )
    operator = SOLAOperator(
        space,
        EuclideanSpace(n_targets),
        kernels=provider,
        cache_kernels=True,
        integration_config=IntegrationConfig(method="simpson", n_points=1025),
    )
    model = Function(
        space,
        evaluate_callable=lambda x: (
            np.sin(2.0 * np.pi * np.asarray(x)) + 0.3 * np.cos(5.0 * np.pi * np.asarray(x))
        ),
    )
    dual = np.random.default_rng(42).normal(size=n_targets)
    rule = QuadratureRule.split_gauss_legendre(
        domain,
        breakpoints=tuple(np.arange(1, n_targets) * width),
        n_points=256,
    )

    lhs = float(np.dot(operator.apply_with_quadrature_rule(model, rule), dual))
    rhs = space.inner_product(model, operator.adjoint(dual), quadrature_rule=rule)

    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-13, atol=1.0e-13)


def test_standard_sola_split_configuration_integrates_declared_boxes(
    domain: IntervalDomain,
) -> None:
    """The opt-in SOLA path uses product breakpoint metadata, not ``quad``."""
    integration = IntegrationConfig(method="split_gauss_legendre", n_points=32)
    space = Lebesgue(0, domain, basis=None, integration_config=integration)
    centers = (np.arange(4) + 0.5) / 4.0
    operator = SOLAOperator(
        space,
        EuclideanSpace(4),
        kernels=BoxCarFunctionProvider(
            domain,
            default_width=0.25,
            centers=centers,
            normalize=False,
            default_height=4.0,
        ),
        integration_config=integration,
    )
    model = Function(
        space, evaluate_callable=lambda x: np.ones_like(np.asarray(x))
    )

    np.testing.assert_allclose(
        operator(model),
        np.ones(4),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
