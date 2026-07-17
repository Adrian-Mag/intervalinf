"""Cross-package tests for functional vector updates on interval functions."""

from __future__ import annotations

import numpy as np
import pytest
from pygeoinf import EuclideanSpace, GaussianMeasure, HilbertSpaceDirectSum
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.linear_solvers import CGSolver

from intervalinf import (
    Function,
    IntegrationConfig,
    IntervalDomain,
    Lebesgue,
    WeightedLebesgue,
)
from intervalinf.spaces import lebesgue as lebesgue_module


def _basis_free_space() -> Lebesgue:
    domain = IntervalDomain(0.0, 1.0)
    return Lebesgue(
        0,
        domain,
        basis=None,
        integration_config=IntegrationConfig(method="trapz", n_points=1001),
    )


def _assert_function_allclose(
    actual: Function,
    expected: np.ndarray,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-10,
) -> None:
    points = np.linspace(0.0, 1.0, 17)
    np.testing.assert_allclose(
        actual.evaluate(points),
        expected,
        rtol=rtol,
        atol=atol,
    )


def test_basis_free_space_rejects_legacy_mutating_vector_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lebesgue_module,
        "_PYGEOINF_FUNCTIONAL_VECTOR_UPDATES",
        False,
    )

    with pytest.raises(RuntimeError, match="functional vector-update contract"):
        _basis_free_space()


def test_basis_free_ax_and_axpy_return_new_functions() -> None:
    space = _basis_free_space()
    x = Function(space, evaluate_callable=lambda points: np.asarray(points))
    y = Function(space, evaluate_callable=lambda points: 1.0 + np.asarray(points))
    points = np.linspace(0.0, 1.0, 17)

    scaled = space.ax(2.0, x)
    updated = space.axpy(-0.5, x, y)

    _assert_function_allclose(scaled, 2.0 * points)
    _assert_function_allclose(updated, 1.0 + 0.5 * points)
    _assert_function_allclose(x, points)
    _assert_function_allclose(y, 1.0 + points)


def test_coefficient_backed_updates_keep_in_place_fast_path() -> None:
    space = Lebesgue(3, IntervalDomain(0.0, 1.0), basis="sine")
    x = space.from_components(np.array([1.0, -2.0, 3.0]))
    y = space.from_components(np.array([4.0, 5.0, -1.0]))

    scaled = space.ax(2.0, x)
    updated = space.axpy(0.5, x, y)

    assert scaled is x
    assert updated is y
    np.testing.assert_allclose(
        x.coefficients,
        [2.0, -4.0, 6.0],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        y.coefficients,
        [5.0, 3.0, 2.0],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_basis_free_weighted_space_retains_functional_update() -> None:
    domain = IntervalDomain(0.0, 1.0)
    space = WeightedLebesgue(
        0,
        domain,
        weight=lambda points: 1.0 + np.asarray(points),
        inverse_weight=lambda points: 1.0 / (1.0 + np.asarray(points)),
        basis=None,
    )
    x = Function(space, evaluate_callable=lambda points: np.asarray(points))
    y = Function(space, evaluate_callable=lambda points: np.ones_like(points))

    updated = space.axpy(2.0, x, y)

    points = np.linspace(0.0, 1.0, 17)
    assert updated is not y
    _assert_function_allclose(updated, 1.0 + 2.0 * points)
    _assert_function_allclose(y, np.ones_like(points))


def test_basis_free_coordinate_paths_fail_clearly() -> None:
    space = _basis_free_space()
    function = Function(
        space,
        evaluate_callable=lambda points: 1.0 + np.asarray(points),
    )
    matrix_operator = LinearOperator.from_matrix(
        space,
        space,
        np.empty((0, 0)),
    )

    with pytest.raises(RuntimeError, match="no finite coordinate representation"):
        space.to_components(function)

    with pytest.raises(RuntimeError, match="no finite coordinate representation"):
        space.from_components(np.zeros(0))

    with pytest.raises(RuntimeError, match="no finite coordinate representation"):
        matrix_operator(function)


def test_direct_sum_retains_returned_basis_free_component() -> None:
    function_space = _basis_free_space()
    space = HilbertSpaceDirectSum([function_space, EuclideanSpace(1)])
    x_function = Function(
        function_space,
        evaluate_callable=lambda points: np.asarray(points),
    )
    y_function = Function(
        function_space,
        evaluate_callable=lambda points: 1.0 + np.asarray(points),
    )
    x = [x_function, np.array([2.0])]
    y = [y_function, np.array([5.0])]
    points = np.linspace(0.0, 1.0, 17)

    result = space.axpy(0.5, x, y)

    assert result is y
    _assert_function_allclose(result[0], 1.0 + 1.5 * points)
    np.testing.assert_allclose(result[1], [6.0], rtol=0.0, atol=1.0e-12)


def test_cg_solves_scaled_identity_on_basis_free_space() -> None:
    space = _basis_free_space()

    def mapping(function: Function) -> Function:
        return space.multiply(2.0, function)

    operator = LinearOperator(
        space,
        space,
        mapping,
        adjoint_mapping=mapping,
    )
    rhs = Function(
        space,
        evaluate_callable=lambda points: 2.0 + np.sin(np.pi * np.asarray(points)),
    )

    solution = CGSolver(rtol=1.0e-12, atol=0.0, maxiter=4)(operator)(rhs)

    points = np.linspace(0.0, 1.0, 17)
    expected = 1.0 + 0.5 * np.sin(np.pi * points)
    _assert_function_allclose(solution, expected, rtol=1.0e-9, atol=1.0e-9)


def test_gaussian_covariance_accumulates_basis_free_functions() -> None:
    space = _basis_free_space()
    positive = Function(
        space,
        evaluate_callable=lambda points: np.ones_like(np.asarray(points), dtype=float),
    )
    negative = space.negative(positive)

    measure = GaussianMeasure.from_samples(space, [negative, positive])
    covariance_action = measure.covariance(positive)

    points = np.linspace(0.0, 1.0, 17)
    _assert_function_allclose(measure.expectation, np.zeros_like(points))
    _assert_function_allclose(covariance_action, 2.0 * np.ones_like(points))
