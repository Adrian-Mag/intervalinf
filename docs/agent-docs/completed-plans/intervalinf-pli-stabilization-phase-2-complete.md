## Phase 2 Complete: Reconstruct the runtime and test snapshot

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Generated a path-limited final-state patch from
  `origin/main...mission/lowering-materialization` containing only
  `intervalinf/` and `tests/`.
- Applied the patch to the clean stabilization worktree.
- Verified that all 38 transferred paths exactly matched the mission tree before
  making one whitespace-only cleanup in `tests/operators/test_sola.py`.
- Excluded README, packaging, demos, work artifacts, agent docs, hooks, and all
  generated research outputs.
- Validated core, spaces, providers, focused operators, and the large SOLA test
  file separately against combined pygeoinf compatibility tip `66c127b`.

### Evidence

```text
snapshot patch SHA-256:
c97a0b4ef109a5932b03c43652c25f027fe6c1eb46d5716aeed5d493345e05cc

snapshot comparison before hygiene edit:
snapshot_matches_mission=0

core:
178 passed in 5.38s

spaces + providers:
128 passed in 11.71s

focused operators:
70 passed in 3.02s

SOLA:
154 passed, 6 failed in 2.04s
```

The six SOLA failures exactly match the failure set documented by the existing
weighted-space plan: five batching/cache instrumentation expectations and one
compact-support quadrature-equivalence tolerance failure. They remain visible
and are mandatory Phase 4 work.

### Deviations

The cohorts were applied as one final-state patch to guarantee exact mission
content, then tested independently by subsystem. This is safer than replaying
the mixed 87-commit history but produces one local reconstruction checkpoint
rather than one commit per historical feature cohort.

The mission snapshot contained one extra blank line at EOF in
`tests/operators/test_sola.py`; it was removed after the exact-snapshot check.

### Next phase

Add cross-package integration tests for functional vector updates and verify
intervalinf behavior against both the current solver branch and the combined
functional-plus-solver pygeoinf branch.
