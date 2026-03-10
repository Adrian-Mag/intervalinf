## Phase 2 Complete: Add Fused SupportFunction API

Added `value_and_support_point(q)` to the `SupportFunction` base class and overridden it in `BallSupportFunction` and `EllipsoidSupportFunction` to share intermediate computations (norm, `A⁻¹q`) between the scalar value and support-point calculations. The living reference was updated and edge-case test coverage for `q≈0` was added after the initial review.

**Files created/changed:**
- `pygeoinf/pygeoinf/convex_analysis.py`
- `pygeoinf/tests/test_support_function_constructors.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`

**Functions created/changed:**
- `SupportFunction.value_and_support_point` — new concrete default method
- `BallSupportFunction.value_and_support_point` — fused override, shares `||q||` computation
- `EllipsoidSupportFunction.value_and_support_point` — fused override, shares `A⁻¹q` computation

**Tests created/changed:**
- `TestValueAndSupportPoint.test_default_callable_point_consistent`
- `TestValueAndSupportPoint.test_default_callable_no_support_point`
- `TestValueAndSupportPoint.test_default_callable_value_matches_call`
- `TestValueAndSupportPoint.test_ball_value_consistent`
- `TestValueAndSupportPoint.test_ball_point_consistent`
- `TestValueAndSupportPoint.test_ball_formula_zero_center`
- `TestValueAndSupportPoint.test_ball_zero_q`
- `TestValueAndSupportPoint.test_ball_near_zero_q`
- `TestValueAndSupportPoint.test_ellipsoid_value_consistent`
- `TestValueAndSupportPoint.test_ellipsoid_point_consistent`
- `TestValueAndSupportPoint.test_ellipsoid_sweep`
- `TestValueAndSupportPoint.test_ellipsoid_no_inverse_fallback`
- `TestValueAndSupportPoint.test_ellipsoid_zero_q_value_and_point`
- `TestValueAndSupportPoint.test_ellipsoid_very_small_q_returns_center`

**Review Status:** APPROVED (after addressing two MAJOR items from initial review)

**Git Commit Message:**
```
feat(support): add fused value_and_support_point API to SupportFunction

- Add SupportFunction.value_and_support_point with safe default fallback
- Override in BallSupportFunction to share norm computation
- Override in EllipsoidSupportFunction to share A⁻¹q computation
- 14 new tests covering default, ball, ellipsoid, q≈0 edge cases
- Update pygeoinf living reference with new method entry

Plan: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-plan.md
Phase: 2 of 5
Related: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-phase-2-complete.md
```
