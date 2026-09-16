## Phase 3 Complete: Resolve cross-package vector-update compatibility

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Added cross-package regression tests for basis-free and coefficient-backed
  `Function` updates, direct sums, mass-weighted spaces, CG, and Gaussian sample
  covariance accumulation.
- Changed `Lebesgue.ax()` to return the mutated object for coefficient-backed
  functions and a new scaled `Function` for basis-free functions.
- Preserved the existing coefficient-backed in-place `ax`/`axpy` fast paths.
- Added a behavioral pygeoinf feature check. Basis-free `Lebesgue` and
  `WeightedLebesgue` construction now raises a clear `RuntimeError` when
  pygeoinf still uses the mutating-only vector-update contract.
- Updated the local-only living reference with the functional update semantics
  and compatibility boundary.

### Evidence

The first red run against combined pygeoinf isolated the intervalinf defect:

```text
tests/test_functional_vector_compatibility.py:
3 failed, 2 passed

failures:
- basis-free Lebesgue.ax raised instead of returning a vector
- direct-sum/CG behavior inherited that failure
```

Before the compatibility boundary, the same five tests against legacy
pygeoinf reproduced three silent numerical failures: direct sums discarded the
new function component, CG returned the zero function, and Gaussian covariance
accumulation returned zero.

Final intended integration state, using pygeoinf compatibility tip `66c127b`:

```text
functional compatibility + Lebesgue + WeightedLebesgue:
74 passed in 1.38s
```

Final legacy solver branch behavior:

```text
functional compatibility:
5 failed, 2 passed in 1.27s

all five failures stop during basis-free construction with:
RuntimeError: Basis-free Lebesgue spaces require pygeoinf's functional
vector-update contract: HilbertSpace.ax() and axpy() must return the updated
vector.
```

The two legacy passes are the explicit boundary test and the coefficient-backed
fast-path test. This confirms that the compatibility check is scoped to the
basis-free representation that requires returned-vector retention.

### Deviations

No released pygeoinf version currently identifies the functional update
contract, so this phase does not guess a version floor. It performs a small
behavioral feature check against `HilbertSpaceDirectSum.axpy()` instead.
Packaging will replace or supplement this boundary with the first valid release
floor during Phase 8.

### Next phase

Begin correctness hardening with a red weighted reduced-covariance semantic
test, then fix the model covariance application order before addressing the
remaining representation, basis-free capability, domain-check, and SOLA
failure gates.
