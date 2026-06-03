"""Tests for function-space prior-family exploration helpers."""

import os
import sys
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../utils"))

from prior_family_exploration import (  # noqa: E402
    DataSpaceCoverage,
    build_block_prior_from_component_covariances,
    build_radial_component_spaces,
    contraction_spectrum,
    data_space_covariance_operator,
    effective_rank,
    gaussian_prior_from_covariance,
    make_bessel_covariance,
    make_mixture_covariance,
    make_power_law_covariance,
    noise_normalized_information,
)
from pygeoinf import EuclideanSpace, LinearOperator  # noqa: E402

from intervalinf.core.boundary import BoundaryConditions  # noqa: E402
from intervalinf.core.config import IntegrationConfig  # noqa: E402
from intervalinf.core.domain import IntervalDomain  # noqa: E402
from intervalinf.core.functions import Function  # noqa: E402
from intervalinf.spaces.lebesgue import Lebesgue  # noqa: E402


def _space():
    domain = IntervalDomain(0, np.pi)
    integration = IntegrationConfig(method="simpson", n_points=500)
    return (
        Lebesgue(30, domain, basis=None, integration_config=integration),
        integration,
    )


def test_effective_rank_uses_relative_threshold():
    values = np.array([10.0, 1.0, 1e-3, 1e-8])
    assert effective_rank(values, rtol=1e-4) == 3


def test_noise_normalized_information_and_contraction():
    signal = np.array([9.0, 1.0, 0.0])
    noise = np.array([3.0, 1.0, 2.0])

    info = noise_normalized_information(signal, noise)
    contraction = contraction_spectrum(signal, noise)

    assert_allclose(info, np.array([3.0, 1.0, 0.0]), rtol=1e-12, atol=1e-12)
    assert_allclose(
        contraction,
        np.array([0.75, 0.5, 0.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_data_space_coverage_summary_counts_contracted_directions():
    coverage = DataSpaceCoverage.from_eigenvalues(
        signal_eigenvalues=np.array([9.0, 1.0, 0.0]),
        noise_eigenvalues=np.array([1.0, 1.0, 1.0]),
        rank_rtol=1e-8,
    )

    assert coverage.n_data == 3
    assert coverage.effective_rank == 2
    assert coverage.n_directions_contraction_gt_95 == 0
    assert_allclose(
        coverage.median_information_ratio,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_make_bessel_covariance_returns_function_space_operator():
    space, integration = _space()
    covariance = make_bessel_covariance(
        space,
        s_order=2.0,
        length=0.5,
        variance=3.0,
        bc="dirichlet",
        dofs=12,
        integration_config=integration,
    )

    assert covariance.domain == space
    assert covariance.codomain == space
    assert covariance.get_eigenvalue(0) > covariance.get_eigenvalue(4)


def test_make_power_law_covariance_returns_trace_class_operator():
    space, integration = _space()
    covariance = make_power_law_covariance(
        space,
        boundary_conditions=BoundaryConditions.dirichlet(),
        variance=2.0,
        length=0.25,
        decay_order=2.0,
        dofs=10,
        integration_config=integration,
    )

    assert covariance.domain == space
    assert covariance.codomain == space
    assert covariance.trace_estimate() > 0.0
    assert covariance.get_eigenvalue(0) > covariance.get_eigenvalue(5)


def test_covariance_wraps_as_gaussian_measure():
    space, integration = _space()
    covariance = make_power_law_covariance(
        space,
        boundary_conditions=BoundaryConditions.dirichlet(),
        variance=2.0,
        length=0.25,
        decay_order=2.0,
        dofs=10,
        integration_config=integration,
    )

    prior = gaussian_prior_from_covariance(covariance)

    assert prior.domain == space
    assert prior.covariance.domain == space


def test_data_space_covariance_operator_stays_diagnostic():
    space, integration = _space()
    smooth = make_power_law_covariance(
        space,
        boundary_conditions=BoundaryConditions.dirichlet(),
        variance=1.0,
        length=0.5,
        decay_order=2.0,
        dofs=10,
        integration_config=integration,
    )
    rough = make_power_law_covariance(
        space,
        boundary_conditions=BoundaryConditions.dirichlet(),
        variance=0.5,
        length=0.2,
        decay_order=1.2,
        dofs=10,
        integration_config=integration,
    )
    covariance = make_mixture_covariance([(0.75, smooth), (0.25, rough)])
    data_space = EuclideanSpace(1)
    kernel = Function(space, evaluate_callable=lambda x: np.sin(x))

    def mapping(f):
        value = (f * kernel).integrate(method="simpson", n_points=500)
        return np.array([value])

    def adjoint(y):
        coefficient = float(np.asarray(y)[0])
        return Function(
            space,
            evaluate_callable=lambda x: coefficient * kernel(x),
        )

    forward = LinearOperator(
        space,
        data_space,
        mapping,
        adjoint_mapping=adjoint,
    )

    diagnostic = data_space_covariance_operator(forward, covariance)
    value = diagnostic(np.array([1.0]))

    assert diagnostic.domain == data_space
    assert diagnostic.codomain == data_space
    assert value.shape == (1,)
    assert value[0] > 0.0


def test_build_block_prior_from_component_covariances():
    integration = IntegrationConfig(method="simpson", n_points=400)
    specs = SimpleNamespace(
        lebesgue_cfg=SimpleNamespace(inner_product=integration),
        parallel_cfg=None,
        earth_radius_km=np.pi,
        icb_radius_km=np.pi / 3.0,
        cmb_radius_km=2.0 * np.pi / 3.0,
    )
    spaces = build_radial_component_spaces(specs)
    covariances = {
        param: make_power_law_covariance(
            space,
            boundary_conditions=BoundaryConditions.dirichlet(),
            variance=1.0,
            length=0.3,
            decay_order=2.0,
            dofs=8,
            integration_config=integration,
        )
        for param, space in spaces.items()
    }

    prior = build_block_prior_from_component_covariances(
        0,
        covariances,
        specs,
        sigma_var=2.0,
    )

    assert len(prior.domain.subspaces) == 2
    assert len(prior.domain.subspaces[0].subspaces) == 3
    assert len(prior.domain.subspaces[1].subspaces) == 2
