"""Function-space prior helpers for the normal-mode tuner."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from pygeoinf import EuclideanSpace, GaussianMeasure
from pygeoinf.linear_operators import LinearOperator

from intervalinf.core.boundary import BoundaryConditions
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.domain import IntervalDomain
from intervalinf.core.functions import Function
from intervalinf.operators import (
    BesselSobolevInverse,
    Laplacian,
)
from intervalinf.spaces.lebesgue import Lebesgue


def effective_rank(eigenvalues: np.ndarray, *, rtol: float = 1e-8) -> int:
    """Count non-negligible data-space covariance eigenvalues.

    This rank is a finite diagnostic for the reduced operator $G C G^*$; it is
    not the dimension of the underlying function space.

    Args:
        eigenvalues: Eigenvalues of a finite diagnostic covariance matrix.
        rtol: Relative cutoff against the largest absolute eigenvalue.

    Returns:
        Number of eigenvalues larger than ``rtol * max(abs(eigenvalues))``.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.size == 0:
        return 0
    scale = float(np.max(np.abs(values)))
    if scale <= 0.0:
        return 0
    return int(np.count_nonzero(values >= rtol * scale))


def noise_normalized_information(
    signal_eigenvalues: np.ndarray,
    noise_eigenvalues: np.ndarray,
) -> np.ndarray:
    r"""Return elementwise ratios $\lambda_s / \lambda_n$."""
    signal = np.asarray(signal_eigenvalues, dtype=float)
    noise = np.asarray(noise_eigenvalues, dtype=float)
    if signal.shape != noise.shape:
        raise ValueError(
            "signal and noise eigenvalue arrays must have the same shape"
        )
    if np.any(noise <= 0.0):
        raise ValueError("noise eigenvalues must be strictly positive")
    return signal / noise


def contraction_spectrum(
    signal_eigenvalues: np.ndarray,
    noise_eigenvalues: np.ndarray,
) -> np.ndarray:
    r"""Return variance contraction $\lambda_s/(\lambda_s+\lambda_n)$."""
    info = noise_normalized_information(signal_eigenvalues, noise_eigenvalues)
    return info / (1.0 + info)


@dataclass(frozen=True)
class DataSpaceCoverage:
    """Finite diagnostic summary of the data-visible prior covariance."""

    n_data: int
    effective_rank: int
    median_information_ratio: float
    n_directions_contraction_gt_95: int
    n_directions_contraction_gt_99: int
    max_contraction: float

    @classmethod
    def from_eigenvalues(
        cls,
        *,
        signal_eigenvalues: np.ndarray,
        noise_eigenvalues: np.ndarray,
        rank_rtol: float = 1e-8,
    ) -> DataSpaceCoverage:
        """Build a coverage summary from finite $G C G^*$ and noise spectra."""
        signal = np.asarray(signal_eigenvalues, dtype=float)
        noise = np.asarray(noise_eigenvalues, dtype=float)
        if signal.shape != noise.shape:
            raise ValueError(
                "signal and noise eigenvalue arrays must have the same shape"
            )
        contraction = contraction_spectrum(signal, noise)
        info = noise_normalized_information(signal, noise)
        return cls(
            n_data=int(signal.size),
            effective_rank=effective_rank(signal, rtol=rank_rtol),
            median_information_ratio=(
                float(np.median(info)) if info.size else 0.0
            ),
            n_directions_contraction_gt_95=int(
                np.count_nonzero(contraction > 0.95)
            ),
            n_directions_contraction_gt_99=int(
                np.count_nonzero(contraction > 0.99)
            ),
            max_contraction=(
                float(np.max(contraction)) if contraction.size else 0.0
            ),
        )


def _boundary_conditions(
    value: str | BoundaryConditions,
) -> BoundaryConditions:
    if isinstance(value, BoundaryConditions):
        return value
    return BoundaryConditions(bc_type=str(value))


def make_bessel_covariance(
    space: Lebesgue,
    *,
    s_order: float,
    length: float,
    variance: float,
    bc: str | BoundaryConditions,
    dofs: int,
    integration_config: IntegrationConfig | None = None,
    n_samples: int | None = None,
) -> BesselSobolevInverse:
    """Build a Bessel-Sobolev covariance operator on a Lebesgue function space.

    This mirrors the normal-mode tuner parameterization while returning a
    function-space covariance operator suitable for `pygeoinf.GaussianMeasure`.
    """
    if s_order <= 0.0:
        raise ValueError("s_order must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    if dofs <= 0:
        raise ValueError("dofs must be positive")

    integration = integration_config or IntegrationConfig(
        method="simpson", n_points=1000
    )
    bessel_integration = IntegrationConfig(
        method="trapz", n_points=max(2000, dofs * 40)
    )
    k = float(np.power(variance, -0.5 / s_order))
    alpha = float((length**2) * (k**2))
    laplacian = Laplacian(
        space,
        _boundary_conditions(bc),
        alpha,
        method="spectral",
        dofs=dofs,
        integration_config=integration,
        n_samples=n_samples or max(512, dofs * 16),
    )
    return BesselSobolevInverse(
        space,
        space,
        k,
        s_order,
        laplacian,
        dofs=dofs,
        n_samples=n_samples or max(512, dofs * 16),
        use_fast_transforms=True,
        integration_config=bessel_integration,
    )


def gaussian_prior_from_covariance(
    covariance: LinearOperator,
) -> GaussianMeasure:
    """Wrap a covariance operator as a zero-mean Gaussian prior."""
    return GaussianMeasure(covariance=covariance)


def build_radial_component_spaces(specs) -> dict[str, Lebesgue]:
    """Build the radial Lebesgue spaces used by one normal-mode block."""
    cfg = specs.lebesgue_cfg.inner_product
    pcfg = specs.parallel_cfg
    radius = specs.earth_radius_km
    icb = specs.icb_radius_km
    cmb = specs.cmb_radius_km
    full_domain = IntervalDomain(0, radius)
    return {
        "vp": Lebesgue(
            0,
            full_domain,
            basis=None,
            integration_config=cfg,
            parallel_config=pcfg,
        ),
        "vs_IC": Lebesgue(
            0,
            IntervalDomain(0, icb),
            basis=None,
            integration_config=cfg,
            parallel_config=pcfg,
        ),
        "vs_M": Lebesgue(
            0,
            IntervalDomain(cmb, radius),
            basis=None,
            integration_config=cfg,
            parallel_config=pcfg,
        ),
        "rho": Lebesgue(
            0,
            full_domain,
            basis=None,
            integration_config=cfg,
            parallel_config=pcfg,
        ),
    }


def reassociate_covariance(
    covariance: LinearOperator,
    target_space: Lebesgue,
    *,
    scale: float = 1.0,
) -> LinearOperator:
    """Reuse a reference covariance on an equivalent block component space."""
    if scale < 0.0:
        raise ValueError("scale must be non-negative for a covariance")

    def apply(f):
        f_ref = Function(covariance.domain, evaluate_callable=f.__call__)
        out_ref = covariance(f_ref)
        return Function(
            target_space,
            evaluate_callable=lambda x: scale * out_ref(x),
        )

    return LinearOperator(
        target_space,
        target_space,
        apply,
        adjoint_mapping=apply,
    )


def build_block_prior_from_component_covariances(
    s: int,
    component_covariances: dict[str, LinearOperator],
    specs,
    *,
    tau_fn=None,
    sigma_var: float = 100.0,
) -> GaussianMeasure:
    """Assemble a normal-mode block prior from function-space covariances."""
    if tau_fn is None:
        def tau_fn(_param, _s):
            return 1.0

    spaces = build_radial_component_spaces(specs)
    required = {"vp", "vs_IC", "vs_M", "rho"}
    missing = required.difference(component_covariances)
    if missing:
        raise ValueError(f"Missing covariance components: {sorted(missing)}")

    priors = {}
    for param, space in spaces.items():
        tau = float(tau_fn(param, s))
        priors[param] = GaussianMeasure(
            covariance=reassociate_covariance(
                component_covariances[param],
                space,
                scale=tau**2,
            )
        )

    sigma_0 = GaussianMeasure.from_covariance_matrix(
        EuclideanSpace(1), np.array([[sigma_var]]), expectation=np.array([0.0])
    )
    sigma_1 = GaussianMeasure.from_covariance_matrix(
        EuclideanSpace(1), np.array([[sigma_var]]), expectation=np.array([0.0])
    )
    prior_vs = GaussianMeasure.from_direct_sum(
        [priors["vs_IC"], priors["vs_M"]]
    )
    prior_functions = GaussianMeasure.from_direct_sum(
        [priors["vp"], prior_vs, priors["rho"]]
    )
    prior_euclidean = GaussianMeasure.from_direct_sum([sigma_0, sigma_1])
    return GaussianMeasure.from_direct_sum([prior_functions, prior_euclidean])


def data_space_covariance_operator(
    forward_operator: LinearOperator,
    covariance: LinearOperator,
) -> LinearOperator:
    """Return the finite data-space diagnostic operator $G C G^*$."""
    if forward_operator.domain != covariance.domain:
        raise ValueError(
            "forward operator domain must match covariance domain"
        )
    if covariance.domain != covariance.codomain:
        raise ValueError("covariance must map a function space to itself")
    return forward_operator @ covariance @ forward_operator.adjoint


def symmetrized_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetric part of a finite diagnostic matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    return np.linalg.eigvalsh(0.5 * (arr + arr.T))


def data_space_coverage_from_matrices(
    signal_covariance: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    rank_rtol: float = 1e-8,
) -> DataSpaceCoverage:
    """Summarize finite data-space prior coverage from covariance matrices."""
    signal = symmetrized_eigenvalues(signal_covariance)
    noise = symmetrized_eigenvalues(noise_covariance)
    return DataSpaceCoverage.from_eigenvalues(
        signal_eigenvalues=np.maximum(signal, 0.0),
        noise_eigenvalues=noise,
        rank_rtol=rank_rtol,
    )


def candidate_grid(
    *,
    s_orders: Iterable[float],
    length_factors: Iterable[float],
    variance_factors: Iterable[float],
) -> list[dict[str, float]]:
    """Generate a small Cartesian grid of scalar prior-family knobs."""
    return [
        {
            "s_order": float(s_order),
            "length_factor": float(length_factor),
            "variance_factor": float(variance_factor),
        }
        for s_order in s_orders
        for length_factor in length_factors
        for variance_factor in variance_factors
    ]
