## Phase 4 Complete: Add Private Support-Mesh Helper in SOLA

This phase added a private `SOLAOperator._build_support_mesh` helper that reproduces the existing support-restricted fixed-grid meshing semantics used by `IntervalDomain.integrate` for compact-support intersections. The helper is intentionally not wired into the production batching path yet, so SOLA forward behavior remains unchanged while the allocation rules are now pinned by focused regression tests.

**Files created/changed:**
- intervalinf/intervalinf/operators/sola.py
- intervalinf/tests/operators/test_sola.py
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- SOLAOperator._build_support_mesh

**Tests created/changed:**
- TestBuildSupportMesh.test_empty_support_returns_empty_array
- TestBuildSupportMesh.test_single_interval_matches_linspace
- TestBuildSupportMesh.test_single_interval_small_n_points_clipped_to_three
- TestBuildSupportMesh.test_two_equal_intervals_split_evenly
- TestBuildSupportMesh.test_unequal_intervals_proportional_with_correct_remainder
- TestBuildSupportMesh.test_effective_total_enforced_when_n_points_too_small
- TestBuildSupportMesh.test_stable_tiebreak_first_index_wins
- TestBuildSupportMesh.test_adjacent_intervals_boundary_point_duplicated
- TestBuildSupportMesh.test_three_intervals_two_shared_boundaries_each_duplicated

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
feat(sola): add private support-mesh helper for fixed-grid compact supports

- Add SOLAOperator._build_support_mesh to mirror IntervalDomain fixed-grid allocation semantics
- Pin multi-interval mesh allocation, tie-breaking, and duplicated-boundary behavior with 9 tests
- Update intervalinf living reference for the new private helper and Phase 4 progress

Plan: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-plan.md
Phase: 4 of 5
Related: intervalinf/docs/agent-docs/active-plans/dli-optimization-implementation-phase-4-complete.md