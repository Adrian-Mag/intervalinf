"""Safety tests for skipping redundant checks inside composite functions."""

from __future__ import annotations

import numpy as np
import pytest
from pygeoinf import EuclideanSpace

from intervalinf import Function, IntegrationConfig, IntervalDomain, Lebesgue
from intervalinf.operators import SOLAOperator, build_eigenfunction_expansion
from intervalinf.sampling.kl_sampler import KLSampler


def _space() -> Lebesgue:
    return Lebesgue(0, IntervalDomain(0.0, 1.0), basis=None)


def _record_checks(
    monkeypatch: pytest.MonkeyPatch,
    function: Function,
) -> list[bool | None]:
    calls: list[bool | None] = []
    original = function.evaluate

    def recording_evaluate(points, check_domain=None):
        calls.append(check_domain)
        return original(points, check_domain=check_domain)

    monkeypatch.setattr(function, "evaluate", recording_evaluate)
    return calls


def _assert_outer_check_blocks_nested_evaluation(
    function: Function,
    calls: list[bool | None],
) -> None:
    calls.clear()
    with pytest.raises(ValueError, match="not in domain"):
        function.evaluate(np.array([-0.1, 0.5]))
    assert calls == []


def test_scalar_composition_skips_only_nested_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _space()
    source = Function(space, evaluate_callable=lambda x: np.asarray(x) ** 2)
    calls = _record_checks(monkeypatch, source)
    scaled = 3.0 * source
    points = np.linspace(0.0, 1.0, 11)

    np.testing.assert_allclose(
        scaled.evaluate(points),
        3.0 * points**2,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert calls == [False]
    _assert_outer_check_blocks_nested_evaluation(scaled, calls)


def test_spectral_expansion_skips_only_nested_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _space()
    mode = Function(space, evaluate_callable=lambda x: np.sin(np.pi * np.asarray(x)))
    calls = _record_checks(monkeypatch, mode)
    expansion = build_eigenfunction_expansion(
        [(2.0, mode)],
        space,
        space,
    )
    points = np.linspace(0.0, 1.0, 11)

    np.testing.assert_allclose(
        expansion.evaluate(points),
        2.0 * np.sin(np.pi * points),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert calls == [False]
    _assert_outer_check_blocks_nested_evaluation(expansion, calls)


def test_sola_adjoint_attaches_outer_domain_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _space()
    kernel = Function(space, evaluate_callable=lambda x: 1.0 + np.asarray(x))
    calls = _record_checks(monkeypatch, kernel)
    operator = SOLAOperator(
        space,
        EuclideanSpace(1),
        kernels=[kernel],
        integration_config=IntegrationConfig(method="trapz", n_points=101),
    )
    reconstructed = operator.adjoint(np.array([2.0]))
    points = np.linspace(0.0, 1.0, 11)

    np.testing.assert_allclose(
        reconstructed.evaluate(points),
        2.0 * (1.0 + points),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert calls == [False]
    _assert_outer_check_blocks_nested_evaluation(reconstructed, calls)


def test_kl_outputs_skip_only_nested_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _space()
    mode = Function(space, evaluate_callable=lambda x: 1.0 + np.asarray(x))
    mean = Function(space, evaluate_callable=lambda x: 0.5 * np.ones_like(x))
    mode_calls = _record_checks(monkeypatch, mode)
    mean_calls = _record_checks(monkeypatch, mean)

    class _SpectralStub:
        domain = space

        @staticmethod
        def get_eigenvalue(index: int) -> float:
            assert index == 0
            return 2.0

        @staticmethod
        def get_eigenfunction(index: int) -> Function:
            assert index == 0
            return mode

    sampler = KLSampler(
        _SpectralStub(),
        mean=mean,
        n_modes=1,
        rng=np.random.default_rng(42),
    )
    points = np.linspace(0.0, 1.0, 11)

    variance = sampler.variance_function
    np.testing.assert_allclose(
        variance.evaluate(points),
        2.0 * (1.0 + points) ** 2,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert mode_calls and all(check is False for check in mode_calls)
    _assert_outer_check_blocks_nested_evaluation(variance, mode_calls)

    sample = sampler.sample()
    sample.evaluate(points)
    assert mean_calls and all(check is False for check in mean_calls)
    assert mode_calls and all(check is False for check in mode_calls)
    mean_calls.clear()
    mode_calls.clear()
    with pytest.raises(ValueError, match="not in domain"):
        sample.evaluate(np.array([-0.1, 0.5]))
    assert mean_calls == []
    assert mode_calls == []
