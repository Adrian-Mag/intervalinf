"""
Phase 4 benchmark: fast batched path vs generic per-kernel path.

Compares:
  - Phase 4 fast path  (is_fixed_grid dispatches to _apply_kernels_fixed_grid)
  - Phase 3 generic    (_apply_kernels_generic, per-kernel domain.integrate loop)

Run from workspace root:
    conda run -n inferences3 python intervalinf/rough_work/benchmark_phase4.py
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
n_reps = 12
f = Function(M, evaluate_callable=lambda x: np.sin(np.pi * x))

header = "{:>6}  {:>16}  {:>14}  {:>8}".format(
    "N_d", "phase4_fast(ms)", "generic(ms)", "speedup"
)
print(header)
print("-" * len(header))

for N_d in [5, 10, 20, 50, 100, 200]:
    D = EuclideanSpace(N_d)
    provider = SineFunctionProvider(domain)
    G = SOLAOperator(M, D, kernels=provider, integration_config=cfg)

    # Warm-up
    _ = G(f)
    _ = G._apply_kernels_generic(f)

    times_fast, times_generic = [], []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        G(f)
        times_fast.append((time.perf_counter() - t0) * 1000)

    for _ in range(n_reps):
        t0 = time.perf_counter()
        G._apply_kernels_generic(f)
        times_generic.append((time.perf_counter() - t0) * 1000)

    mf = statistics.median(times_fast)
    mg = statistics.median(times_generic)
    print("{:>6}  {:>16.3f}  {:>14.3f}  {:>7.2f}x".format(N_d, mf, mg, mg / mf))
