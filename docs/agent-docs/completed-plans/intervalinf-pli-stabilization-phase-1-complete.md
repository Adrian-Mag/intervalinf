## Phase 1 Complete: Create isolated worktrees and compatibility branches

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Created clean intervalinf worktree
  `/home/adrian/PhD/Inferences/intervalinf-stabilize` on
  `stabilize/intervalinf-for-pli` from `origin/main` (`9996213`).
- Created pygeoinf worktree
  `/home/adrian/PhD/Inferences/pygeoinf-functional-solvers` on local branch
  `integration/functional-vector-updates-solver-robustness`.
- Based that integration branch on `refactor/functional-vector-updates`
  (`fec12f5`) and cherry-picked solver breakdown commit `bafb88f`, producing
  clean integration tip `66c127b`.
- Added an ignored `.local-agent-plan` symlink in the clean intervalinf worktree
  pointing to the canonical plan in the mission checkout.
- Verified package selection from the PLI workspace using `PYTHONPATH`; no
  editable conda installation was changed.

### Evidence

```text
intervalinf import:
/home/adrian/PhD/Inferences/intervalinf-stabilize/intervalinf/__init__.py

pygeoinf import:
/home/adrian/PhD/Inferences/pygeoinf-functional-solvers/pygeoinf/__init__.py

functional contract:
HilbertSpace.axpy(self, a, x, y) -> Vector
```

Both new worktrees were clean after setup. The original intervalinf mission
worktree and original pygeoinf solver worktree retained their prior status.

### Deviations

The first cherry-pick command was interrupted by a Devin crash before Git made
any change. Status and history checks confirmed that no cherry-pick state or
partial edit existed, after which the command was rerun successfully without
conflicts.

Matplotlib warned that its default user cache was not writable. Future commands
will set `MPLCONFIGDIR` to a writable `/tmp` cache alongside the numerical thread
limits.

### Next phase

Apply the product-only final-state delta for the 24 runtime files and 14 test
files to the clean intervalinf worktree, then audit and validate it in coherent
dependency groups.
