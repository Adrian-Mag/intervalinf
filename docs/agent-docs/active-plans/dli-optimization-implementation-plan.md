## Plan: DLI Optimization Implementation

Implement the roadmap improvements in two coordinated workstreams: first modernize the `pygeoinf` support-function oracle path with a fused value/support-point API and the local dual-master optimizations, then optimize the compact-support fixed-grid SOLA path in `intervalinf` using a private support-mesh helper inside `sola.py`. The phases are structured to keep each change incremental, test-driven, and numerically verifiable.

**Phases 5**
1. **Phase 1: Add Oracle Optimization Guardrails**
    - **Objective:** Add failing tests that lock in the intended oracle behavior before changing production code.
    - **Files/Functions to Modify/Create:** `pygeoinf/tests/test_dual_master_cost.py`; `pygeoinf/tests/test_support_function_constructors.py` if needed for the new fused API contract
    - **Tests to Write:** regression test proving `value_and_subgradient` does not require a separate scalar support evaluation when a support point exists; regression test proving repeated oracle calls do not repeatedly fetch `G.adjoint`; focused tests for the fused support-function API contract
    - **Steps:**
        1. Add test doubles for `SupportFunction` and `LinearOperator` that expose redundant support-value calls and repeated adjoint-property access.
        2. Add tests for a new fused API that returns both the scalar support value and support point in one call when supported.
        3. Run the targeted `pygeoinf` tests to confirm the new guardrail tests fail before implementation.

2. **Phase 2: Add Fused SupportFunction API**
    - **Objective:** Introduce a fused `value_and_support_point` API in `pygeoinf` so support-function implementations can share intermediate work between value and subgradient computations.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/convex_analysis.py`; any directly related support-function tests in `pygeoinf/tests/`
    - **Tests to Write:** unit tests covering the default API behavior, `BallSupportFunction.value_and_support_point`, and `EllipsoidSupportFunction.value_and_support_point` for the supported exact cases
    - **Steps:**
        1. Add `SupportFunction.value_and_support_point(q)` with a safe default implementation based on the existing scalar value and `support_point` methods.
        2. Override the method in concrete support functions where shared intermediates materially reduce work.
        3. Run the targeted support-function and dual-master tests until the new API contract is green.

3. **Phase 3: Optimize DualMasterCostFunction Oracle Path**
    - **Objective:** Implement the oracle-side speedups in `DualMasterCostFunction` using the new fused support-function API and cached adjoint operator access.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/backus_gilbert.py`; `pygeoinf/tests/test_dual_master_cost.py`
    - **Tests to Write:** any additional regression test needed to preserve the finite-difference fallback path when fused support-point evaluation is unavailable
    - **Steps:**
        1. Cache `self._G.adjoint` in `DualMasterCostFunction.__init__` and route all adjoint evaluations through the cached operator.
        2. Replace the separate support-value calls in `value_and_subgradient` with the fused support-function API, using the exact inner-product identity when support points are returned.
        3. Preserve the current finite-difference fallback behavior when a support point is unavailable, then run the targeted `pygeoinf` tests again.

4. **Phase 4: Add Private Support-Mesh Helper in SOLA**
    - **Objective:** Add a private helper inside `SOLAOperator` that reproduces the existing support-restricted fixed-grid meshing semantics without changing forward behavior yet.
    - **Files/Functions to Modify/Create:** `intervalinf/intervalinf/operators/sola.py`; supporting tests in `intervalinf/tests/operators/test_sola.py`
    - **Tests to Write:** focused regression tests for deterministic support-restricted mesh construction and exact agreement with the current per-kernel fixed-grid fallback on representative single-interval and multi-interval supports
    - **Steps:**
        1. Add failing SOLA tests that pin down the current support-restricted quadrature semantics.
        2. Implement a private helper in `sola.py` for allocating support-restricted fixed-grid meshes exactly as the current fallback path does.
        3. Re-run the targeted SOLA tests to confirm the helper reproduces the existing behavior before batching is introduced.

5. **Phase 5: Batch Compact-Support Fixed-Grid Integration**
    - **Objective:** Replace the per-kernel compact-support fallback in `_apply_kernels_fixed_grid` with grouped batched integration over shared support meshes.
    - **Files/Functions to Modify/Create:** `intervalinf/intervalinf/operators/sola.py`; `intervalinf/tests/operators/test_sola.py`
    - **Tests to Write:** regression test for grouped compact-support batching on shared support intervals; regression test for a multi-interval support case; existing fast-path/cache tests must remain green unchanged
    - **Steps:**
        1. Add failing tests for grouped compact-support batching while preserving the existing zero-result disjoint-support behavior and cache exclusions.
        2. Implement support-key grouping and batched restricted-mesh integration in `_apply_kernels_fixed_grid`, using the new private helper for exact mesh generation.
        3. Run the targeted SOLA operator tests and confirm the compact-support, cache, and fixed-grid fast-path invariants still hold.

**Open Questions 2**
1. Should the fused API be treated as a public long-term convenience API for all support functions, or documented as an optimization-oriented API that concrete implementations may override selectively?
2. When grouping compact-support kernels in `sola.py`, should support keys require exact interval tuple equality, or should there be a controlled canonicalization rule for floating-point endpoints?