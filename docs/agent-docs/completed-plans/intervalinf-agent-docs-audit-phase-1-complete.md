## Phase 1 Complete: Fix agent-doc navigation rules

Aligned the `intervalinf` agent-doc entrypoints so they consistently point agents to the living reference under `docs/agent-docs/references/living/`. While validating that navigation flow, the package index and related reference metadata were also reconciled with the current tree, including the archived SOLA phase records and the completed Phase 4 status in the convex-update plan.

**Files created/changed:**
- `AGENTS.md`
- `docs/agent-docs/index.md`
- `docs/agent-docs/references/README.md`
- `docs/agent-docs/references/legacy/README.md`
- `docs/agent-docs/active-plans/intervalinf-convex-update-plan.md`
- `docs/agent-docs/active-plans/intervalinf-agent-docs-audit-plan.md`

**Functions created/changed:**
- None

**Tests created/changed:**
- None

**Review Status:** APPROVED with minor recommendations

**Git Commit Message:**
docs(agent-docs): fix intervalinf reference navigation

- Point agent entrypoints to references/living for package reference reads
- Reconcile index and archive descriptions with the current docs tree
- Mark convex-update Phase 4 complete and align its path metadata

Plan: docs/agent-docs/active-plans/intervalinf-agent-docs-audit-plan.md
Phase: 1 of 3
Related: docs/agent-docs/completed-plans/intervalinf-agent-docs-audit-phase-1-complete.md