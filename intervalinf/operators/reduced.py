"""Matrix-backed reduced operators for SOLA reduced assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from pygeoinf import LinearOperator, MatrixLinearOperator


if TYPE_CHECKING:
    from intervalinf.operators.sola import SOLAOperator


def compute_reduced_covariance(
    G: "SOLAOperator",
    C: LinearOperator,
    C_d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assemble the dense reduced covariance ``G C G* + C_d``.

    This evaluates each SOLA kernel ``g_i`` on the shared fixed-grid mesh,
    applies the model-side operator ``C`` to the adjoint representative
    ``G*(e_j)``, evaluates the transformed functions on the same mesh, and
    contracts the two tables with the fixed-grid quadrature weights. For a
    mass-weighted domain, ``G*(e_j) = M^-1 g_j``.

    Args:
        G: SOLA operator whose data kernels define the reduced basis.
        C: Model-side operator mapping ``G.domain`` to ``G.domain``.
        C_d: Optional dense data-noise covariance to add to the reduced matrix.

    Returns:
        Dense reduced covariance matrix of shape ``(G.N_d, G.N_d)``.

    Raises:
        TypeError: If ``C`` is not a ``LinearOperator``.
        ValueError: If ``G`` does not use a fixed-grid integration method,
            if ``C`` does not map ``G.domain`` to itself, or if ``C_d`` has an
            incompatible shape.
    """
    if not isinstance(C, LinearOperator):
        raise TypeError("Reduced covariance assembly requires a LinearOperator.")
    if not G.integration.is_fixed_grid:
        raise ValueError(
            "Reduced covariance assembly requires a fixed-grid SOLAOperator."
        )
    if C.domain != G.domain or C.codomain != G.domain:
        raise ValueError(
            "Reduced covariance assembly requires a model operator on G.domain."
        )

    xs = G._get_or_build_mesh()
    weights = G._build_quadrature_weights(xs)
    kernel_matrix = G._build_kernel_matrix(xs)

    transformed_rows = []
    for j in range(G.N_d):
        transformed_kernel = C(G._adjoint_kernel(j))
        transformed_rows.append(np.asarray(G._eval_on_mesh(transformed_kernel, xs)))

    transformed_matrix = (
        np.stack(transformed_rows, axis=0)
        if transformed_rows
        else np.empty((0, xs.size), dtype=kernel_matrix.dtype)
    )
    reduced_covariance = (
        kernel_matrix * weights[np.newaxis, :]
    ) @ transformed_matrix.T

    if C_d is None:
        return reduced_covariance

    noise_covariance = np.asarray(C_d)
    expected_shape = (G.N_d, G.N_d)
    if noise_covariance.shape != expected_shape:
        raise ValueError(
            f"C_d must have shape {expected_shape}, got {noise_covariance.shape}."
        )
    return reduced_covariance + noise_covariance


class ReducedGramOperator:
    """Factory for the reduced data-space Gram operator $G G^*$."""

    @staticmethod
    def from_sola(operator: "SOLAOperator") -> MatrixLinearOperator:
        """Return a dense matrix-backed Gram operator for a SOLA map."""
        gram_matrix = operator.compute_gram_matrix_fast()
        return LinearOperator.from_matrix(
            operator.codomain,
            operator.codomain,
            gram_matrix,
            galerkin=True,
        )


class ReducedCrossGramOperator:
    """Factory for the reduced cross-Gram operator $T G^*$."""

    @staticmethod
    def from_sola_pair(
        target: "SOLAOperator",
        forward: "SOLAOperator",
    ) -> MatrixLinearOperator:
        """Return a dense matrix-backed cross-Gram operator for a SOLA pair."""
        cross_gram_matrix = target.compute_cross_gram_matrix(forward)
        return LinearOperator.from_matrix(
            forward.codomain,
            target.codomain,
            cross_gram_matrix,
            galerkin=True,
        )


class ReducedCovarianceOperator:
    """Factory for the reduced data-space covariance operator ``G C G*``."""

    @staticmethod
    def from_sola_and_model(
        operator: "SOLAOperator",
        model_operator: LinearOperator,
        C_d: Optional[np.ndarray] = None,
    ) -> MatrixLinearOperator:
        """Return a dense matrix-backed reduced covariance operator."""
        reduced_covariance = compute_reduced_covariance(
            operator,
            model_operator,
            C_d=C_d,
        )
        return LinearOperator.from_matrix(
            operator.codomain,
            operator.codomain,
            reduced_covariance,
            galerkin=True,
        )
