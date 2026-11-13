## Phase 5 Complete: Resource-constrained intervalinf validation

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Verified the actual public namespace boundary: core/spaces at `intervalinf`,
  operators at `intervalinf.operators`, and sampling at `intervalinf.sampling`.
- Re-ran the complete suite against the combined functional-plus-solver
  pygeoinf worktree with one Python process and all native thread pools capped.
- Added `benchmarks/benchmark_stabilized_paths.py`, a deterministic runner that
  asserts numerical equivalence before timing each optimized/reference pair.
- Covered repeated function composition, cached SOLA application, weighted
  radial Bessel application, and KL variance evaluation.
- Ran Ruff on the new benchmark file and `git diff --check` on the clean branch.

### Environment

```text
intervalinf import:
/home/adrian/PhD/Inferences/intervalinf-stabilize/intervalinf/__init__.py

pygeoinf import:
/home/adrian/PhD/Inferences/pygeoinf-functional-solvers/pygeoinf/__init__.py

pygeoinf integration tip: 66c127b
intervalinf benchmark commit: d12e9d4
Python: 3.12.10 (conda environment inferences)
```

Every Python command used:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
MPLCONFIGDIR=/tmp/codex-matplotlib
```

### Correctness evidence

```text
public API smoke: passed

complete intervalinf suite:
554 passed in 12.68s

benchmark semantic checks:
all four optimized/reference pairs passed their explicit tolerances

ruff check benchmarks/benchmark_stabilized_paths.py:
all checks passed

git diff --check:
passed
```

### Benchmark evidence

Command:

```text
python benchmarks/benchmark_stabilized_paths.py
```

Configuration: five repeats, ten calls per repeat, medians reported.

```text
workload                    optimized ms  reference ms   speedup   max |error|
function composition              0.0239        0.0295    1.235x     0.000e+00
SOLA cached batch                 0.1593        1.8557   11.646x     0.000e+00
weighted radial Bessel            0.1887        1.5036    7.967x     1.389e-10
KL variance                       0.3271        0.4175    1.276x     0.000e+00
```

The reference paths are intentionally simple implementations in the same
process: redundant nested validation, generic per-kernel SOLA quadrature,
weight-aware slow Bessel projection, and redundantly checked KL evaluation.
The benchmark does not assert a speedup threshold because timing is
machine-dependent; semantic tolerances are mandatory and checked first.

### Deviations

The first smoke command incorrectly expected operator and sampling classes at
the package root. The package currently documents and exports them from their
subpackages, so the final smoke check used that public boundary. The stale
top-level comment saying operators are "imported lazily" is packaging/documentation
cleanup for Phase 8 rather than a runtime defect.

### Next phase

Enter the confidential PLI repository, read its local instructions, repair
dependency provenance, and run its test suite plus reduced downstream smoke
inferences without network access or pushes.
