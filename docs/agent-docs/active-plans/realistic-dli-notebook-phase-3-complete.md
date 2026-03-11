## Phase 3 Complete: Upgrade to Covariance-Matched Prior Ellipsoid

Reworked the realistic DLI notebook so Phase 3 ends in a covariance-matched strict 0.95 confidence ellipsoid **without discretizing the functional model blocks**. The final implementation uses basis-free functional spaces, operator-native block covariance / square-root / shape operators, and a small `pygeoinf` direct-sum fix so the notebook now satisfies the no-`.matrix(...)` requirement and still runs through the final DLI solve.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/realistic_dli.ipynb`
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
- `pygeoinf/pygeoinf/direct_sum.py`
- `pygeoinf/tests/test_direct_sum.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- Notebook model-prior construction updated from `BallSupportFunction` to `EllipsoidSupportFunction`
- Notebook functional spaces updated to basis-free `Lebesgue(..., basis=None, dim=0)` blocks, with Euclidean sigma scalars retained as the only finite-dimensional coordinates
- Notebook prior assembly updated to use nested `BlockDiagonalLinearOperator` blocks built from `BesselSobolevInverse`, `BesselSobolev`, and scalar identity operators instead of dense covariance matrices
- `HilbertSpaceDirectSum.zero` and `HilbertSpaceDirectSum.inner_product` updated to operate componentwise for basis-free direct sums
- Notebook KL-truncation logic added to compute `dof_eff` and the strict `0.95` model-prior chi-squared radius

**Tests created/changed:**
- Added regression tests covering zero-dimensional direct-sum blocks in `pygeoinf/tests/test_direct_sum.py`
- End-to-end notebook execution re-run through the final DLI solve and reviewed

**Review Status:** APPROVED

**Git Commit Message:**
fix(dli): make realistic_dli prior operator-native and basis-free

- Convert realistic_dli functional blocks to basis-free spaces and build
  the Phase 3 prior from block operators instead of dense matrices
- Update pygeoinf direct sums so zero and inner products work for
  basis-free subspaces used in operator-theoretic workflows
- Re-run the notebook, add regression tests, and align the living
  references with the final operator-native ellipsoid design

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-notebook-plan.md
Phase: 3 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-notebook-phase-3-complete.md
