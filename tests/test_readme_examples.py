"""Executable checks for the public examples in README.md."""

import numpy as np

from intervalinf import (
    BoundaryConditions,
    Function,
    IntervalDomain,
    Lebesgue,
    WeightedLebesgue,
)
from intervalinf.operators import Laplacian, RadialLaplacian


def test_continuous_inner_product_example():
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(0, domain, basis=None)
    f = Function(space, evaluate_callable=lambda x: x**2)
    g = Function(space, evaluate_callable=lambda x: x**3)

    np.testing.assert_allclose(
        space.inner_product(f, g),
        1.0 / 6.0,
        rtol=1e-6,
        atol=1e-10,
    )


def test_laplacian_application_example():
    domain = IntervalDomain(0.0, np.pi)
    space = Lebesgue(32, domain, basis="sine")
    function = Function(space, evaluate_callable=np.sin)
    laplacian = Laplacian(space, BoundaryConditions.dirichlet())

    np.testing.assert_allclose(
        laplacian(function)(np.pi / 2.0),
        1.0,
        rtol=1e-6,
        atol=1e-10,
    )


def test_weighted_radial_space_example():
    domain = IntervalDomain(1.0, 2.0)
    space = WeightedLebesgue(
        0,
        domain,
        weight=lambda r: r**2,
        inverse_weight=lambda r: 1.0 / r**2,
        basis=None,
    )
    radial_laplacian = RadialLaplacian(
        space,
        BoundaryConditions.dirichlet(),
        1.0,
        dofs=16,
        ell=2,
    )

    assert radial_laplacian.get_eigenvalue(0) > 0.0
