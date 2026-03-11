## Phase 2 Complete: Add Missing Convex-Analysis Infrastructure

Fixed a `TypeError` that blocked `M_model.norm(m_bar)` for deeply nested `LebesgueSpaceDirectSum` models.  `LinearFormKernel._mapping_impl` now handles arbitrary list nesting through a recursive `_eval` inner function.  Two regression tests were added and the notebook now executes end-to-end producing DLI posterior bounds.

**Files created/changed:**
- `intervalinf/intervalinf/spaces/forms.py` — recursive `_eval` helper in `_mapping_impl`
- `intervalinf/tests/spaces/test_forms.py` — class `TestLinearFormKernelNestedDirectSum` with 2 new tests

**Functions created/changed:**
- `LinearFormKernel._mapping_impl` — replaced flat `zip` loop with recursive `_eval` that handles list-of-list kernel/vector structures
- (Nested helper) `_eval(k, vi)` — inner function that recurses when `k` is a list

**Tests created/changed:**
- `TestLinearFormKernelNestedDirectSum::test_nested_list_kernel_evaluates_without_error` — confirms no TypeError for a 2-level nest (M_outer = M_a ⊕ (M_b ⊕ M_c))
- `TestLinearFormKernelNestedDirectSum::test_nested_list_kernel_value_matches_sum_of_parts` — asserts the integrated value equals the sum of each leaf-level (f_a·g_a + f_b·g_b + f_c·g_c)

**Root Cause Summary:**

`LebesgueSpaceDirectSum.to_dual(xs)` creates a `LinearFormKernel` with `kernel=xs` where `xs` may contain nested lists when any component of the direct sum is itself a `LebesgueSpaceDirectSum`.  The old `_mapping_impl` performed `(k * vi).integrate(...)` unconditionally; when `k` is a list this becomes `list * list` → Python TypeError.

**Notebook outcome after fix:**
- All 25 cells execute cleanly
- `‖m̄‖_{M_model} = 4.72802`, `Ball radius r = 4.96442`
- DLI posterior bounds plotted for p_0 and p_1 (3.2× constraint reduction on p_0; p_1 unconstrained)

**Review Status:** APPROVED — 11/11 tests in test_forms.py passing; notebook runs end-to-end

**Git Commit Message:**
```
fix(forms): handle nested-list kernels in LinearFormKernel._mapping_impl

- Add recursive _eval helper inside _mapping_impl to handle kernels
  that are lists-of-lists (arise when LebesgueSpaceDirectSum contains
  another LebesgueSpaceDirectSum as a component)
- Fixes: TypeError: can't multiply sequence by non-int of type 'list'
  when calling M_model.norm(m_bar) on deeply-nested direct-sum vectors
- Add TestLinearFormKernelNestedDirectSum with 2 regression tests
  (no-error check + value matches sum-of-parts)
- All 11 tests in test_forms.py pass

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-notebook-plan.md
Phase: 2 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-notebook-phase-2-complete.md
```
