"""
Phase 5 benchmark: eval-cache warm path vs cold (cache_kernels=False) path.

Measures the speedup for *repeated workloads* where the same set of kernels is
applied to many different input functions.

Compares:
  - Phase 5 warm  (cache_kernels=True, kernel evals already in _kernel_eval_cache)
  - Phase 4 cold  (cache_kernels=False, kernel evals recomputed for every call)

The benchmark calls G(f_i) for n_reps distinct input functions.  The first call
for the cached operator warms the eval cache; subsequent calls reuse it.  For
the uncached operator every call recomputes all kernel evaluations.

Run from workspace root:
    conda run -n inferences3 python intervalinf/rough_work/benchmark_phase5.py
"""
import time
import statistics
import numpy as np

from intervalinf import IntervalDomain, Lebesgue, Function, IntegrationConfig
from intervalinf.providers import SineFunctionProvider
from intervalinf.operators import SOLAOperator
from pygeoinf import EuclideanSpace

domain = IntervalDomain(0.0, 1.0)
M = Lebesgue(30, domain, basis="cosine")
cfg = IntegrationConfig(method="simpson", n_points=1000)

# Number of distinct input functions per timing trial.
N_REPS = 50
# Number of outer repeats for median timing.
N_OUTER = 8

# Pre-build N_REPS distinct input functions so function construction is not
# measured inside the timed loop.
rng = np.random.default_rng(42)
input_functions = [
    Function(M, evaluate_callable=(lambda a: (lambda x: np.sin((a + 1) * np.pi * x)))(a))
    for a in range(N_REPS)
]

header = "{:>6}  {:>18}  {:>14}  {:>8}".format(
    "N_d", "phase5_cached(ms)", "phase4_cold(ms)", "speedup"
)
print(header)
print("-" * (len(header) + 2))

for N_d in [5, 10, 20, 50, 100, 200]:
    D = EuclideanSpace(N_d)
    provider = SineFunctionProvider(domain)

    G_cached = SOLAOperator(M, D, kernels=provider, cache_kernels=True,
                            integration_config=cfg)
    G_cold = SOLAOperator(M, D, kernels=provider, cache_kernels=False,
                          integration_config=cfg)

    # Warm up eval cache for G_cached (first call populates it).
    _ = G_cached(input_functions[0])

    times_cached, times_cold = [], []

    for _ in range(N_OUTER):
        t0 = time.perf_counter()
        for f in input_functions:
            G_cached(f)
        times_cached.append((time.perf_counter() - t0) * 1000)

    for _ in range(N_OUTER):
        t0 = time.perf_counter()
        for f in input_functions:
            G_cold(f)
        times_cold.append((time.perf_counter() - t0) * 1000)

    mc = statistics.median(times_cached)
    mu = statistics.median(times_cold)
    print("{:>6}  {:>18.2f}  {:>14.2f}  {:>7.2f}x".format(N_d, mc, mu, mu / mc))
