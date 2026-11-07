## Phase 0 Complete: Preserve and freeze the current effective state

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Preserved the four-file, 6-insertion/6-deletion runtime patch from the dirty
  mission worktree.
- Stored the patch and detailed manifest under the nested repository's local Git
  metadata:
  `.git/local-recovery/intervalinf-pli-stabilization/phase-0/`.
- Stored a redundant local-only copy outside the repository at
  `/home/adrian/PhD/.local-recovery/intervalinf-pli-stabilization/phase-0/`.
- Inventoried tracked and untracked source-like paths separately from generated
  image/PDF output.
- Recorded intervalinf, pygeoinf, PLI, Python, NumPy, and SciPy states and import
  paths.
- Selected and hashed synthetic-weighted and real-data-weighted PLI comparison
  artifacts without copying the 2.4 GB run tree.

### Evidence

```text
source patch SHA-256:
983f21543b0a518215e06927119923bbd4f0a7cede9c19e46565a807d2e7e130

manifest SHA-256 (both copies):
006436daff632beeb03fb50f73149b608cbb9f0efde9428543ebbec9920a37d9

git apply --reverse --check:
patch-matches-current-worktree
```

The mission checkout remained on `mission/lowering-materialization` at
`91c9c5c`. No dirty or untracked file was staged, stashed, deleted, reset, or
committed.

### Deviations

The preferred redundant location under the parent Git metadata was unavailable
because `/home/adrian/PhD/Inferences/.git` is read-only in the managed
environment. The redundant copy was placed in `/home/adrian/PhD/.local-recovery`
after explicit filesystem approval instead.

### Next phase

Create isolated intervalinf and pygeoinf stabilization worktrees, then verify
that package selection can be controlled with `PYTHONPATH` without changing the
shared editable conda installation.
