"""Tests for the bulk spectral-operator accessors."""

import numpy as np

from intervalinf.core.functions import Function
from intervalinf.operators.base import SpectralOperator
from intervalinf.operators.laplacian import InverseLaplacian
from intervalinf.operators.radial import InverseRadialLaplacian


class _SpectralStub(SpectralOperator):
    """Minimal spectral operator used to exercise base implementations."""

    def __init__(self) -> None:
        pass

    def get_eigenvalue(self, index: int) -> float:
        return float(index) + 0.5

    def get_eigenfunction(self, index: int) -> Function:
        return index  # type: ignore[return-value]

    def _apply(self, f: Function) -> Function:
        return f


def test_bulk_eigenvalue_accessor_returns_float_array() -> None:
    operator = _SpectralStub()

    values = operator.get_eigenvalues(np.array([2, 0, 3]))

    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(
        values,
        np.array([2.5, 0.5, 3.5]),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_bulk_eigenfunction_accessor_returns_ordered_list() -> None:
    operator = _SpectralStub()

    functions = operator.get_eigenfunctions(np.array([2, 0, 3]))

    assert functions == [2, 0, 3]


def test_spectral_subclasses_inherit_bulk_eigenvalue_accessor() -> None:
    assert InverseLaplacian.get_eigenvalues is SpectralOperator.get_eigenvalues
    assert (
        InverseRadialLaplacian.get_eigenvalues
        is SpectralOperator.get_eigenvalues
    )
