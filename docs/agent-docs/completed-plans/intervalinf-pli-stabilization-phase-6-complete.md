## Phase 6 Complete: PLI provenance and downstream acceptance

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Repaired PLI repository-root discovery and provenance collection.
- Recorded the PLI, intervalinf, and pygeoinf repository roots, commits,
  branches, dirty states, dirty-patch hashes, package versions, and actual
  import paths, together with Python version and executable.
- Included tracked, staged, and nonignored untracked content in the dirty-state
  fingerprint.
- Added two isolated provenance regression tests using a temporary Git
  repository and the live dependency imports.
- Removed the unconditional module-level `TkAgg` selection from the normal-mode
  data viewer so the paper-demo suite can be collected headlessly.
- Reused the expensive deterministic full-spectrum setup within its test module
  rather than reconstructing it separately for each test.
- Ran a complete one-block weighted headless inference twice from a temporary
  reduced config and compared its scientific products exactly.

All PLI changes remain uncommitted in `/home/adrian/PhD/PLI_paper`. Nothing was
pushed and no network access was used. Generated validation artifacts and the
temporary config are under `/tmp/codex-pli-runs` and `/tmp` only.

### Pinned dependency state

```text
PLI_paper:
  branch: main
  head: 948d8e1bea25720262b12037c427cfaffe9a8a7a

intervalinf:
  branch: stabilize/intervalinf-for-pli
  head: d12e9d48607811667113d92e2537301d795f2940
  import: /home/adrian/PhD/Inferences/intervalinf-stabilize/intervalinf/__init__.py
  dirty: false

pygeoinf:
  branch: integration/functional-vector-updates-solver-robustness
  head: 66c127b4805d67780e0fb4151b4e5a46d0648d95
  import: /home/adrian/PhD/Inferences/pygeoinf-functional-solvers/pygeoinf/__init__.py
  dirty: false
```

The PLI repository is correctly recorded as dirty in new artifacts, with a
non-null dirty-patch hash covering the local provenance changes.

### Test evidence

Every Python command used one process and set:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
MPLCONFIGDIR=/tmp/codex-matplotlib
```

Results:

```text
targeted provenance tests: 2 passed in 1.48s
targeted provenance + headless viewer tests: 4 passed in 1.36s
resource-constrained paper-demo suite: 104 passed, 4 deselected in 73.65s
git diff --check: passed
ruff on new provenance and viewer files: passed
```

The four deselected tests are explicit resource exceptions:

```text
test_solve_all_blocks_parallel_matches_serial
test_assemble_property_posterior_mean_matches_pushforward
test_assemble_property_posterior_cov_symmetric_psd
test_assemble_property_posterior_reduces_uncertainty
```

The first intentionally launches parallel workers. The other three share the
same multi-minute all-block property solve. They were not claimed as passing;
they remain deferred to a genuinely reduced test configuration or a later
user-approved Europa run. The legacy full-spectrum test file also has existing
Ruff findings outside the fixture-only Phase 6 change; focused lint on the new
and directly repaired files passes.

### Reduced inference evidence

Temporary configuration:

```text
weighted model space: true
block: s0_t0
observations: 186
covariance truncation: 4
quadrature points: 64 (Lebesgue), 128 (SOLA)
n_jobs: 1
figures/samples: disabled
target: one vp cap-bulk functional
```

The two complete runs took 47 s and 44 s. Numerical acceptance results:

```text
all posterior arrays finite: true
all posterior standard deviations nonnegative: true
all standardized-residual metrics finite: true
property covariance symmetry max abs: 0.0
property covariance minimum eigenvalue: 1.7154899195456004e-13
normal-operator symmetry max abs: 6.544848078207308e-14
normal-operator minimum eigenvalue: 8.082042170235226e-13
property posterior mean: -2.722148217767645e-05
property posterior standard deviation: 4.141847316772554e-07
```

Across the repeat run, every numeric array in `blocks.npz`, `diagnostics.npz`,
`posterior_summary.npz`, and `property_posterior.npz` matched exactly. The block
inventory, data-fit, data-spectrum, prior-spectrum, stability, and tau tables
also matched byte-for-byte. Timing and timestamp metadata were intentionally
excluded from deterministic comparison.

### Next phase

Select the smallest paper-critical configurations, freeze their resolved inputs
and pinned dependency states, and run reduced local cases first. Full remote
configurations remain gated on explicit user approval and the Europa protocol.
