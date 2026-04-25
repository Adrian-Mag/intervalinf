"""Phase 3 integration tests: PrimalKKTSolverBasisFree with Lebesgue/SOLAOperator.

Verifies that the basis-free solver works correctly when:
- Model space is Lebesgue([0,1]) — infinite-dimensional (discretised at N basis
  functions, but basis-free in the sense that solve() never calls to_components)
- Forward operator is SOLAOperator (integrates against kernels — no to_components)
- Data space is EuclideanSpace(M)

Key test: monkey-patch Lebesgue.to_components to raise AssertionError and verify
that PrimalKKTSolverBasisFree.solve() completes without triggering it.
"""

from __future__ import annotations

import numpy as np
import pytest

from intervalinf.core.domain import IntervalDomain
from intervalinf.core.config import IntegrationConfig
from intervalinf.core.functions import Function
from intervalinf.spaces.lebesgue import Lebesgue
from intervalinf.operators import SOLAOperator

from pygeoinf import PrimalKKTSolverBasisFree
from pygeoinf import BallSupportFunction, EllipsoidSupportFunction, KKTResult
from pygeoinf.hilbert_space import EuclideanSpace


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_lebesgue_sola_ball_ball(n_basis=20, n_data=3, seed=17):
    """Lebesgue model + SOLAOperator + ball/ball support functions.

    The data ball is tight (radius 0.1) so Branch 2 (both constraints active)
    is triggered for generic c.
    """
    domain = IntervalDomain(0.0, 1.0)
    ms = Lebesgue(n_basis, domain, basis="cosine")
    ds = EuclideanSpace(n_data)

    # M polynomial kernels: k_i(x) = x^i (i=0,...,n_data-1)
    kernels = [
        Function(domain, evaluate_callable=lambda x, i=i: x ** i)
        for i in range(n_data)
    ]
    G = SOLAOperator(
        ms, ds,
        kernels=kernels,
        integration_config=IntegrationConfig(method="simpson", n_points=500),
    )

    d_tilde = ds.zero
    B = BallSupportFunction(ms, ms.zero, 1.0)
    V = BallSupportFunction(ds, ds.zero, 0.1)   # tight → forces Branch 2

    # Build a generic objective c in model space (a simple function)
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal(n_basis)
    coeffs /= np.linalg.norm(coeffs)
    c = ms.from_components(coeffs)

    # Verify Branch 2 will actually be triggered (G(support_point(c)) >> 0.1)
    u_ball = B.support_point(c)
    data_res = np.linalg.norm(ds.to_components(G(u_ball)))
    assert data_res > 0.5 * V._radius, (
        f"Expected Branch 2 fixture, but ||G u_ball|| = {data_res:.4f} "
        f"<= 0.5 * r = {0.5 * V._radius:.4f}"
    )

    return dict(ms=ms, ds=ds, G=G, d_tilde=d_tilde, B=B, V=V, c=c, domain=domain)


# ---------------------------------------------------------------------------
# Phase 3 tests
# ---------------------------------------------------------------------------

class TestLebesgueSolaCorrectness:
    """Correctness: KKT residuals are ≈ 0 at the solution."""

    def test_residuals_zero_ball_ball(self):
        """solve() on Lebesgue+SOLA satisfies both KKT constraints."""
        fx = _make_lebesgue_sola_ball_ball()
        solver = PrimalKKTSolverBasisFree(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        result = solver.solve(fx["c"])

        assert isinstance(result, KKTResult)
        assert result.converged, f"fsolve did not converge: {result}"
        assert result.m is not None

        lam, mu = result.multipliers
        r1, r2 = solver._residuals(lam, mu, fx["c"])

        np.testing.assert_allclose(r1, 0.0, atol=1e-6,
                                   err_msg=f"Prior constraint residual r1={r1:.2e}")
        np.testing.assert_allclose(r2, 0.0, atol=1e-6,
                                   err_msg=f"Data constraint residual r2={r2:.2e}")

    def test_prior_constraint_satisfied(self):
        """||u* - u0||_H ≤ η (prior ball constraint)."""
        fx = _make_lebesgue_sola_ball_ball()
        solver = PrimalKKTSolverBasisFree(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        result = solver.solve(fx["c"])

        ms = fx["ms"]
        u_star = result.m
        u0 = fx["B"]._center
        diff = ms.subtract(u_star, u0)
        norm_diff = float(ms.norm(diff))
        eta = float(fx["B"]._radius)
        assert norm_diff <= eta * (1.0 + 1e-6), (
            f"Prior constraint violated: ||u*-u0|| = {norm_diff:.6f} > η = {eta:.6f}"
        )

    def test_data_constraint_satisfied(self):
        """||G u* - d̃||_D ≤ r (data constraint)."""
        fx = _make_lebesgue_sola_ball_ball()
        solver = PrimalKKTSolverBasisFree(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        result = solver.solve(fx["c"])

        ds = fx["ds"]
        G = fx["G"]
        d_tilde = fx["d_tilde"]
        res = ds.subtract(G(result.m), d_tilde)
        norm_res = float(ds.norm(res))
        r = float(fx["V"]._radius)
        assert norm_res <= r * (1.0 + 1e-6), (
            f"Data constraint violated: ||Gu*-d|| = {norm_res:.6f} > r = {r:.6f}"
        )

    def test_kkt_result_fields(self):
        """KKTResult has the correct structure."""
        fx = _make_lebesgue_sola_ball_ball()
        solver = PrimalKKTSolverBasisFree(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        result = solver.solve(fx["c"])
        assert isinstance(result, KKTResult)
        assert len(result.multipliers) == 2
        lam, mu = result.multipliers
        assert lam > 0, "λ must be positive"
        assert mu > 0, "μ must be positive (Branch 2 triggered)"


class TestLebesgueBasisFreedom:
    """Monkey-patch Lebesgue.to_components: verify solve() never calls it."""

    def test_no_model_to_components_during_solve(self):
        """solve() on Lebesgue+SOLA does NOT call Lebesgue.to_components."""
        fx = _make_lebesgue_sola_ball_ball()
        solver = PrimalKKTSolverBasisFree(fx["B"], fx["V"], fx["G"], fx["d_tilde"])

        ms = fx["ms"]
        original_to = ms.to_components
        original_from = ms.from_components

        def _no_to(v):
            raise AssertionError(
                "PrimalKKTSolverBasisFree.solve() called Lebesgue.to_components! "
                "This breaks basis-freedom."
            )

        def _no_from(coeffs):
            raise AssertionError(
                "PrimalKKTSolverBasisFree.solve() called Lebesgue.from_components! "
                "This breaks basis-freedom."
            )

        ms.to_components = _no_to
        ms.from_components = _no_from
        try:
            result = solver.solve(fx["c"])   # must not raise
        finally:
            ms.to_components = original_to
            ms.from_components = original_from

        assert result.m is not None, "solve() returned None result"
        assert isinstance(result, KKTResult)

    def test_p_mat_shape_is_data_space_only(self):
        """P_mat is M×M (data-space), not N×M or N×N (model-space)."""
        fx = _make_lebesgue_sola_ball_ball(n_basis=50, n_data=4)
        solver = PrimalKKTSolverBasisFree(fx["B"], fx["V"], fx["G"], fx["d_tilde"])
        assert solver._P_mat.shape == (4, 4), (
            f"Expected P_mat shape (4, 4), got {solver._P_mat.shape}"
        )
        assert solver._AV_inv_mat.shape == (4, 4)
