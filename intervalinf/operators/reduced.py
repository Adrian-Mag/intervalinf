"""Matrix-backed reduced operators for SOLA Gram assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygeoinf import LinearOperator, MatrixLinearOperator


if TYPE_CHECKING:
    from intervalinf.operators.sola import SOLAOperator


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
