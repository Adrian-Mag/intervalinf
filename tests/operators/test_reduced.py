"""Tests for reduced SOLA Gram and cross-Gram assembly."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from intervalinf import (
    BoundaryConditions,
    Function,
    IntegrationConfig,
    IntervalDomain,
    Lebesgue,
    LebesgueIntegrationConfig,
    ParallelConfig,
)
from intervalinf.operators import (
    BesselSobolev,
    InverseLaplacian,
    Laplacian,
    ReducedCovarianceOperator,
    ReducedCrossGramOperator,
    ReducedGramOperator,
    SOLAOperator,
    compute_reduced_covariance,
)
from pygeoinf import EuclideanSpace, LinearOperator, MatrixLinearOperator


def _slow_cross_gram_matrix(
    left: SOLAOperator,
    right: SOLAOperator,
) -> np.ndarray:
    """Assemble a reference cross-Gram matrix via pairwise quadrature."""
    domain = left.domain.function_domain
    cross_gram = np.zeros((left.N_d, right.N_d), dtype=float)
    n_points = max(left.integration.n_points, right.integration.n_points)

    for i in range(left.N_d):
        kernel_i = left.get_kernel(i)
        for j in range(right.N_d):
            kernel_j = right.get_kernel(j)
            intersected_support = Function._intersect_supports(
                kernel_i.support,
                kernel_j.support,
            )
            if intersected_support == []:
                continue

            def product_callable(x, _ki=kernel_i, _kj=kernel_j):
                return _ki.evaluate(x) * _kj.evaluate(x)

            cross_gram[i, j] = domain.integrate(
                product_callable,
                method=left.integration.method,
                support=intersected_support,
                n_points=n_points,
            )

    return cross_gram


def _semantic_reduced_covariance_matrix(
    operator: SOLAOperator,
    model_operator: LinearOperator,
) -> np.ndarray:
    """Assemble ``G C G*`` column-by-column via operator semantics."""
    reduced = np.zeros((operator.N_d, operator.N_d), dtype=float)
    eye = np.eye(operator.N_d)
    for j in range(operator.N_d):
        reduced[:, j] = operator(model_operator(operator.adjoint(eye[j])))
    return reduced


def _semantic_reduced_covariance_apply(
    operator: SOLAOperator,
    model_operator: LinearOperator,
    data: np.ndarray,
    noise_covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``G C G* + C_d`` using the existing operator semantics."""
    reduced = operator(model_operator(operator.adjoint(data)))
    if noise_covariance is not None:
        reduced = reduced + noise_covariance @ data
    return reduced


@pytest.fixture
def lebesgue_space() -> Lebesgue:
    """Return a basis-free Lebesgue space on the unit interval."""
    domain = IntervalDomain(0.0, 1.0)
    return Lebesgue(
        0,
        domain,
        basis=None,
        integration_config=LebesgueIntegrationConfig.from_single(
            IntegrationConfig(method="simpson", n_points=500)
        ),
        parallel_config=ParallelConfig(enabled=False),
    )


@pytest.fixture
def gram_operator(lebesgue_space: Lebesgue) -> SOLAOperator:
    """Return a cached forward SOLA operator with analytic sine kernels."""
    domain = lebesgue_space.function_domain
    kernels = [
        Function(
            domain,
            evaluate_callable=lambda x, i=i: np.sin((i + 1) * np.pi * x),
        )
        for i in range(5)
    ]
    return SOLAOperator(
        lebesgue_space,
        EuclideanSpace(5),
        kernels=kernels,
        cache_kernels=True,
        integration_config=IntegrationConfig(method="simpson", n_points=1000),
    )


@pytest.fixture
def target_operator(lebesgue_space: Lebesgue) -> SOLAOperator:
    """Return a cached target SOLA operator with polynomial kernels."""
    domain = lebesgue_space.function_domain
    kernels = [
        Function(
            domain,
            evaluate_callable=lambda x, i=i: x ** (i + 1),
        )
        for i in range(3)
    ]
    return SOLAOperator(
        lebesgue_space,
        EuclideanSpace(3),
        kernels=kernels,
        cache_kernels=True,
        integration_config=IntegrationConfig(method="simpson", n_points=1000),
    )


@pytest.fixture
def trapz_operator(lebesgue_space: Lebesgue) -> SOLAOperator:
    """Return a cached forward SOLA operator configured with trapz."""
    domain = lebesgue_space.function_domain
    kernels = [
        Function(
            domain,
            evaluate_callable=lambda x, i=i: np.sin((i + 1) * np.pi * x),
        )
        for i in range(5)
    ]
    return SOLAOperator(
        lebesgue_space,
        EuclideanSpace(5),
        kernels=kernels,
        cache_kernels=True,
        integration_config=IntegrationConfig(method="trapz", n_points=1000),
    )


@pytest.fixture(scope="module")
def covariance_lebesgue_space() -> Lebesgue:
    """Return the Hilbert space used in reduced covariance tests."""
    domain = IntervalDomain(0.0, 1.0)
    return Lebesgue(
        20,
        domain,
        basis="sine",
        integration_config=LebesgueIntegrationConfig.from_single(
            IntegrationConfig(method="simpson", n_points=500)
        ),
        parallel_config=ParallelConfig(enabled=False),
    )


@pytest.fixture(scope="module")
def covariance_boundary_conditions() -> BoundaryConditions:
    """Return homogeneous Dirichlet boundary conditions."""
    return BoundaryConditions.dirichlet(0.0, 0.0)


@pytest.fixture(scope="module")
def covariance_sola_operator(
    covariance_lebesgue_space: Lebesgue,
) -> SOLAOperator:
    """Return a cached SOLA operator with sine kernels on a shared mesh."""
    domain = covariance_lebesgue_space.function_domain
    kernels = [
        Function(
            domain,
            evaluate_callable=lambda x, i=i: np.sin((i + 1) * np.pi * x),
        )
        for i in range(5)
    ]
    return SOLAOperator(
        covariance_lebesgue_space,
        EuclideanSpace(5),
        kernels=kernels,
        cache_kernels=True,
        integration_config=IntegrationConfig(method="simpson", n_points=1000),
    )


@pytest.fixture(scope="module")
def covariance_laplacian(
    covariance_lebesgue_space: Lebesgue,
    covariance_boundary_conditions: BoundaryConditions,
) -> Laplacian:
    """Return the spectral Laplacian used by the Bessel covariance."""
    return Laplacian(
        covariance_lebesgue_space,
        covariance_boundary_conditions,
        method="spectral",
        dofs=20,
    )


@pytest.fixture(scope="module")
def bessel_covariance_operator(
    covariance_lebesgue_space: Lebesgue,
    covariance_laplacian: Laplacian,
) -> BesselSobolev:
    """Return a Bessel covariance operator using the fast spectral path."""
    return BesselSobolev(
        covariance_lebesgue_space,
        covariance_lebesgue_space,
        k=1.0,
        s=1.0,
        L=covariance_laplacian,
        dofs=20,
    )


@pytest.fixture(scope="module")
def fem_covariance_operator(
    covariance_lebesgue_space: Lebesgue,
    covariance_boundary_conditions: BoundaryConditions,
) -> InverseLaplacian:
    """Return an FEM inverse Laplacian covariance operator."""
    return InverseLaplacian(
        covariance_lebesgue_space,
        covariance_boundary_conditions,
        method="fem",
        dofs=50,
    )


@pytest.fixture(scope="module")
def noise_covariance(
    covariance_sola_operator: SOLAOperator,
) -> np.ndarray:
    """Return a diagonal data-side noise covariance."""
    diagonal = np.linspace(0.05, 0.25, covariance_sola_operator.N_d)
    return np.diag(diagonal)


def test_gram_matrix_fast_matches_slow(gram_operator: SOLAOperator) -> None:
    """Fast Gram assembly matches pairwise quadrature."""
    slow = gram_operator.compute_gram_matrix()
    fast = gram_operator.compute_gram_matrix_fast()

    npt.assert_allclose(fast, slow, rtol=1e-8, atol=1e-10)


def test_gram_matrix_fast_symmetry(gram_operator: SOLAOperator) -> None:
    """The reduced Gram matrix is symmetric."""
    fast = gram_operator.compute_gram_matrix_fast()

    npt.assert_allclose(fast, fast.T, rtol=1e-10, atol=1e-12)


def test_gram_matrix_fast_positive_semidefinite(
    gram_operator: SOLAOperator,
) -> None:
    """The reduced Gram matrix is positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(gram_operator.compute_gram_matrix_fast())

    assert np.min(eigenvalues) >= -1e-10


def test_cross_gram_matrix_shape(
    gram_operator: SOLAOperator,
    target_operator: SOLAOperator,
) -> None:
    """Cross-Gram assembly returns a dense target-by-data matrix."""
    cross_gram = target_operator.compute_cross_gram_matrix(gram_operator)

    assert cross_gram.shape == (target_operator.N_d, gram_operator.N_d)


def test_cross_gram_matches_inner_products(
    gram_operator: SOLAOperator,
    target_operator: SOLAOperator,
) -> None:
    """Cross-Gram entries match pairwise kernel inner products."""
    slow = _slow_cross_gram_matrix(target_operator, gram_operator)
    fast = target_operator.compute_cross_gram_matrix(gram_operator)

    npt.assert_allclose(fast, slow, rtol=1e-8, atol=1e-10)


def test_reduced_gram_operator_apply(gram_operator: SOLAOperator) -> None:
    """The reduced Gram operator matches the semantic apply path."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(gram_operator.N_d)

    reduced = ReducedGramOperator.from_sola(gram_operator)
    expected = gram_operator(gram_operator.adjoint(data))

    assert isinstance(reduced, MatrixLinearOperator)
    npt.assert_allclose(reduced(data), expected, rtol=1e-10, atol=1e-12)


def test_reduced_gram_operator_adjoint(gram_operator: SOLAOperator) -> None:
    """The reduced Gram operator is self-adjoint."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(gram_operator.N_d)

    reduced = ReducedGramOperator.from_sola(gram_operator)

    npt.assert_allclose(
        reduced.adjoint(data),
        reduced(data),
        rtol=1e-10,
        atol=1e-12,
    )


def test_quadrature_weights_simpson(gram_operator: SOLAOperator) -> None:
    """Simpson weights sum to the interval length."""
    weights = gram_operator._build_quadrature_weights()

    npt.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_quadrature_weights_trapz(trapz_operator: SOLAOperator) -> None:
    """Trapz weights sum to the interval length."""
    weights = trapz_operator._build_quadrature_weights()

    npt.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_reduced_cross_gram_operator_apply(
    gram_operator: SOLAOperator,
    target_operator: SOLAOperator,
) -> None:
    """The reduced cross-Gram operator matches T(G*(d))."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal(gram_operator.N_d)

    reduced = ReducedCrossGramOperator.from_sola_pair(
        target_operator,
        gram_operator,
    )
    expected = target_operator(gram_operator.adjoint(data))

    assert isinstance(reduced, MatrixLinearOperator)
    npt.assert_allclose(reduced(data), expected, rtol=1e-10, atol=1e-12)


def test_reduced_covariance_bessel_shape(
    covariance_sola_operator: SOLAOperator,
    bessel_covariance_operator: BesselSobolev,
) -> None:
    """Reduced Bessel covariance assembly returns a dense data matrix."""
    reduced = compute_reduced_covariance(
        covariance_sola_operator,
        bessel_covariance_operator,
    )

    assert reduced.shape == (
        covariance_sola_operator.N_d,
        covariance_sola_operator.N_d,
    )


def test_reduced_covariance_bessel_symmetry(
    covariance_sola_operator: SOLAOperator,
    bessel_covariance_operator: BesselSobolev,
) -> None:
    """Reduced Bessel covariance is symmetric up to quadrature error."""
    reduced = compute_reduced_covariance(
        covariance_sola_operator,
        bessel_covariance_operator,
    )

    npt.assert_allclose(reduced, reduced.T, rtol=1e-4, atol=1e-6)


def test_reduced_covariance_bessel_matches_semantic(
    covariance_sola_operator: SOLAOperator,
    bessel_covariance_operator: BesselSobolev,
) -> None:
    """Reduced Bessel covariance matches the semantic operator path."""
    reduced = compute_reduced_covariance(
        covariance_sola_operator,
        bessel_covariance_operator,
    )
    expected = _semantic_reduced_covariance_matrix(
        covariance_sola_operator,
        bessel_covariance_operator,
    )

    npt.assert_allclose(reduced, expected, rtol=1e-4, atol=1e-6)


def test_reduced_covariance_fem_shape(
    covariance_sola_operator: SOLAOperator,
    fem_covariance_operator: InverseLaplacian,
) -> None:
    """Reduced FEM covariance assembly returns a dense data matrix."""
    reduced = compute_reduced_covariance(
        covariance_sola_operator,
        fem_covariance_operator,
    )

    assert reduced.shape == (
        covariance_sola_operator.N_d,
        covariance_sola_operator.N_d,
    )


def test_reduced_covariance_fem_matches_semantic(
    covariance_sola_operator: SOLAOperator,
    fem_covariance_operator: InverseLaplacian,
) -> None:
    """Reduced FEM covariance matches the semantic operator path."""
    reduced = compute_reduced_covariance(
        covariance_sola_operator,
        fem_covariance_operator,
    )
    expected = _semantic_reduced_covariance_matrix(
        covariance_sola_operator,
        fem_covariance_operator,
    )

    npt.assert_allclose(reduced, expected, rtol=1e-3, atol=1e-5)


def test_reduced_covariance_with_C_d(
    covariance_sola_operator: SOLAOperator,
    bessel_covariance_operator: BesselSobolev,
    noise_covariance: np.ndarray,
) -> None:
    """Adding ``C_d`` shifts the reduced covariance by the same matrix."""
    reduced = compute_reduced_covariance(
        covariance_sola_operator,
        bessel_covariance_operator,
    )
    shifted = compute_reduced_covariance(
        covariance_sola_operator,
        bessel_covariance_operator,
        C_d=noise_covariance,
    )

    npt.assert_allclose(shifted, reduced + noise_covariance)


def test_reduced_covariance_operator_apply(
    covariance_sola_operator: SOLAOperator,
    bessel_covariance_operator: BesselSobolev,
    noise_covariance: np.ndarray,
) -> None:
    """Matrix-backed reduced covariance matches the semantic apply path."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(covariance_sola_operator.N_d)

    reduced = ReducedCovarianceOperator.from_sola_and_model(
        covariance_sola_operator,
        bessel_covariance_operator,
        C_d=noise_covariance,
    )
    expected = _semantic_reduced_covariance_apply(
        covariance_sola_operator,
        bessel_covariance_operator,
        data,
        noise_covariance=noise_covariance,
    )

    assert isinstance(reduced, MatrixLinearOperator)
    npt.assert_allclose(reduced(data), expected, rtol=1e-4, atol=1e-6)
