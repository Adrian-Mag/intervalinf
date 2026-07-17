"""Focused semantic and timing checks for stabilized intervalinf paths."""

from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pygeoinf import EuclideanSpace

from intervalinf import (
    BoundaryConditions,
    Function,
    IntegrationConfig,
    IntervalDomain,
    Lebesgue,
    WeightedLebesgue,
)
from intervalinf.operators import (
    BesselSobolevInverse,
    InverseLaplacian,
    RadialLaplacian,
    SOLAOperator,
)
from intervalinf.sampling.kl_sampler import KLSampler


@dataclass(frozen=True)
class BenchmarkResult:
    """One optimized/reference benchmark comparison."""

    name: str
    optimized_ms: float
    reference_ms: float
    max_abs_error: float

    @property
    def speedup(self) -> float:
        return self.reference_ms / self.optimized_ms


def _median_ms(function: Callable[[], object], repeat: int, number: int) -> float:
    samples = timeit.repeat(function, repeat=repeat, number=number)
    return 1.0e3 * statistics.median(samples) / number


def _result(
    name: str,
    optimized: Callable[[], np.ndarray],
    reference: Callable[[], np.ndarray],
    *,
    repeat: int,
    number: int,
    rtol: float,
    atol: float,
) -> BenchmarkResult:
    optimized_value = np.asarray(optimized())
    reference_value = np.asarray(reference())
    np.testing.assert_allclose(
        optimized_value,
        reference_value,
        rtol=rtol,
        atol=atol,
    )
    return BenchmarkResult(
        name=name,
        optimized_ms=_median_ms(optimized, repeat, number),
        reference_ms=_median_ms(reference, repeat, number),
        max_abs_error=float(np.max(np.abs(optimized_value - reference_value))),
    )


def benchmark_function_composition(repeat: int, number: int) -> BenchmarkResult:
    """Compare one validated outer check with a redundant nested check."""
    space = Lebesgue(0, IntervalDomain(0.0, 1.0), basis=None)
    source = Function(space, evaluate_callable=lambda x: np.sin(np.asarray(x)))
    optimized = 2.0 * source
    checked = Function(
        space,
        evaluate_callable=lambda x: 2.0 * source.evaluate(x),
    )
    mesh = np.linspace(0.0, 1.0, 2001)
    return _result(
        "function composition",
        lambda: optimized.evaluate(mesh),
        lambda: checked.evaluate(mesh),
        repeat=repeat,
        number=number,
        rtol=0.0,
        atol=0.0,
    )


def benchmark_sola(repeat: int, number: int) -> BenchmarkResult:
    """Compare cached batched SOLA with the generic per-kernel path."""
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(0, domain, basis=None)
    kernels = [
        Function(
            domain,
            evaluate_callable=lambda x, i=i: np.sin(
                (i + 1) * np.pi * np.asarray(x)
            ),
        )
        for i in range(24)
    ]
    operator = SOLAOperator(
        space,
        EuclideanSpace(len(kernels)),
        kernels=kernels,
        cache_kernels=True,
        integration_config=IntegrationConfig(method="simpson", n_points=1001),
    )
    function = Function(
        space,
        evaluate_callable=lambda x: np.exp(np.asarray(x)),
    )
    operator(function)
    return _result(
        "SOLA cached batch",
        lambda: operator(function),
        lambda: operator._apply_kernels_generic(function),
        repeat=repeat,
        number=number,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def benchmark_weighted_bessel(repeat: int, number: int) -> BenchmarkResult:
    """Compare radial DST and weight-aware slow Bessel application."""
    domain = IntervalDomain(0.0, 1.0)
    integration = IntegrationConfig(method="simpson", n_points=1001)
    space = WeightedLebesgue(
        12,
        domain,
        weight=lambda r: np.asarray(r) ** 2,
        integration_config=integration,
    )
    laplacian = RadialLaplacian(
        space,
        BoundaryConditions.dirichlet(),
        1.0,
        method="spectral",
        dofs=12,
        integration_config=integration,
    )
    fast = BesselSobolevInverse(
        space,
        space,
        1.5,
        2.0,
        laplacian,
        dofs=12,
        n_samples=256,
        use_fast_transforms=True,
        integration_config=integration,
    )
    slow = BesselSobolevInverse(
        space,
        space,
        1.5,
        2.0,
        laplacian,
        dofs=12,
        n_samples=256,
        use_fast_transforms=False,
        integration_config=integration,
    )
    function = Function(
        space,
        evaluate_callable=lambda r: np.sin(np.pi * np.asarray(r)),
    )
    mesh = np.linspace(0.02, 0.98, 101)
    return _result(
        "weighted radial Bessel",
        lambda: fast(function).evaluate(mesh),
        lambda: slow(function).evaluate(mesh),
        repeat=repeat,
        number=number,
        rtol=1.0e-2,
        atol=1.0e-3,
    )


def benchmark_kl_variance(repeat: int, number: int) -> BenchmarkResult:
    """Compare optimized and redundantly checked KL variance evaluation."""
    domain = IntervalDomain(0.0, 1.0)
    space = Lebesgue(0, domain, basis=None)
    covariance = InverseLaplacian(
        space,
        BoundaryConditions.dirichlet(),
        method="spectral",
        dofs=16,
    )
    sampler = KLSampler(
        covariance,
        n_modes=16,
        rng=np.random.default_rng(42),
    )
    eigenvalues = [sampler.eigenvalue(i) for i in range(sampler.n_modes)]
    eigenfunctions = [sampler.eigenfunction(i) for i in range(sampler.n_modes)]
    optimized = sampler.variance_function

    def checked_evaluation(x):
        result = np.zeros_like(x)
        for eigenvalue, eigenfunction in zip(
            eigenvalues,
            eigenfunctions,
            strict=True,
        ):
            result += eigenvalue * eigenfunction.evaluate(x) ** 2
        return result

    checked = Function(space, evaluate_callable=checked_evaluation)
    mesh = np.linspace(0.0, 1.0, 1001)
    return _result(
        "KL variance",
        lambda: optimized.evaluate(mesh),
        lambda: checked.evaluate(mesh),
        repeat=repeat,
        number=number,
        rtol=0.0,
        atol=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--number", type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 1 or args.number < 1:
        parser.error("--repeat and --number must be positive")

    results = [
        benchmark_function_composition(args.repeat, args.number),
        benchmark_sola(args.repeat, args.number),
        benchmark_weighted_bessel(args.repeat, args.number),
        benchmark_kl_variance(args.repeat, args.number),
    ]
    print(
        f"{'workload':<26} {'optimized ms':>13} {'reference ms':>13} "
        f"{'speedup':>9} {'max |error|':>13}"
    )
    for result in results:
        print(
            f"{result.name:<26} {result.optimized_ms:13.4f} "
            f"{result.reference_ms:13.4f} {result.speedup:8.3f}x "
            f"{result.max_abs_error:13.3e}"
        )


if __name__ == "__main__":
    main()
