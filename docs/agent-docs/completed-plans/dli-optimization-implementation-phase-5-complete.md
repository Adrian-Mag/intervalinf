## Phase 5 Complete: Batch Compact-Support Fixed-Grid Integration

This phase replaced the per-kernel compact-support fallback in `SOLAOperator._apply_kernels_fixed_grid` with grouped batched integration over shared restricted meshes. Kernels are grouped by exact intersected-support keys, multi-interval supports are integrated subinterval-by-subinterval to preserve current quadrature semantics, and full-domain mesh construction is now lazy so support-only and disjoint-only workloads avoid unnecessary full-domain evaluation.

**Files created/changed:**
- intervalinf/intervalinf/operators/sola.py
- intervalinf/tests/operators/test_sola.py
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- SOLAOperator._compute_subinterval_alloc
- SOLAOperator._apply_kernels_fixed_grid

**Tests created/changed:**
- TestPhase5GroupedSupportBatching.test_two_shared_support_kernels_analytic
- TestPhase5GroupedSupportBatching.test_three_shared_support_kernels_match_generic
- TestPhase5GroupedSupportBatching.test_shared_support_trapz_matches_analytic
- TestPhase5GroupedSupportBatching.test_multi_interval_support_single_kernel_matches_generic
- TestPhase5GroupedSupportBatching.test_multi_interval_support_two_kernels_match_generic
- TestPhase5GroupedSupportBatching.test_multi_interval_two_kernels_same_support_match_generic
- TestPhase5GroupedSupportBatching.test_grouped_support_kernels_not_in_eval_cache
- TestPhase5GroupedSupportBatching.test_grouped_support_result_identical_with_and_without_caching
- TestPhase5GroupedSupportBatching.test_grouped_batching_fallbacks_count_per_kernel
- TestPhase5GroupedSupportBatching.test_grouped_batching_fallback_time_positive
- TestPhase5GroupedSupportBatching.test_grouped_and_full_domain_counter_split
- TestPhase5GroupedSupportBatching.test_shared_support_f_evaluated_once_per_group
- TestPhase5GroupedSupportBatching.test_two_different_supports_f_evaluated_twice
- TestPhase5GroupedSupportBatching.test_grouped_support_restricted_complex_dtype_preserved
- TestPhase5GroupedSupportBatching.test_two_grouped_complex_kernels_correct
- TestPhase5GroupedSupportBatching.test_narrow_support_grouped_matches_generic
- TestLazyFullDomainEvaluation.test_disjoint_only_shared_mesh_stays_none
- TestLazyFullDomainEvaluation.test_disjoint_only_multiple_kernels_mesh_stays_none
- TestLazyFullDomainEvaluation.test_support_only_shared_mesh_stays_none
- TestLazyFullDomainEvaluation.test_support_only_multiple_kernels_mesh_stays_none
- TestLazyFullDomainEvaluation.test_mixed_workload_shared_mesh_is_built
- TestLazyFullDomainEvaluation.test_disjoint_only_result_exactly_zero
- TestLazyFullDomainEvaluation.test_support_only_result_matches_analytic
- TestLazyFullDomainEvaluation.test_support_only_f_not_evaluated_on_full_mesh

**Review Status:** APPROVED with minor recommendations addressed

**Git Commit Message:**
feat(sola): batch compact-support fixed-grid integration by support group

- Group compact-support kernels by exact intersected-support keys in the fixed-grid path
- Batch restricted-mesh integration while preserving multi-interval and disjoint-support semantics
- Defer full-domain mesh and f evaluation unless a full-domain batched kernel is present

Plan: intervalinf/docs/agent-docs/completed-plans/dli-optimization-implementation-plan.md
Phase: 5 of 5
Related: intervalinf/docs/agent-docs/completed-plans/dli-optimization-implementation-phase-5-complete.md