## Plan Complete: intervalinf agent-docs audit refresh

Completed the `intervalinf` agent-documentation audit and refreshed the package agent-doc system so its entrypoint guidance, status metadata, and living reference all match the current repository layout. The result is a more reliable first-read path for future agents: package entrypoints now point consistently to the living reference, stale status text was removed, and the living reference now includes an explicit test-layout map plus metadata aligned with `pyproject.toml`.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Fix agent-doc navigation rules
2. ✅ Phase 2: Reconcile stale plan/archive status
3. ✅ Phase 3: Refresh living reference audit notes

**All Files Created/Modified:**
- `AGENTS.md`
- `docs/agent-docs/index.md`
- `docs/agent-docs/references/README.md`
- `docs/agent-docs/references/legacy/README.md`
- `docs/agent-docs/references/living/intervalinf-reference.md`
- `docs/agent-docs/active-plans/intervalinf-convex-update-plan.md`
- `docs/agent-docs/active-plans/intervalinf-agent-docs-audit-plan.md`
- `docs/agent-docs/completed-plans/intervalinf-agent-docs-audit-phase-1-complete.md`
- `docs/agent-docs/completed-plans/intervalinf-agent-docs-audit-phase-2-complete.md`
- `docs/agent-docs/completed-plans/intervalinf-agent-docs-audit-phase-3-complete.md`
- `docs/agent-docs/completed-plans/intervalinf-agent-docs-audit-complete.md`

**Key Functions/Classes Added:**
- None

**Test Coverage:**
- Total tests written: 0
- All tests passing: Not re-run; documentation-only plan

**Recommendations for Next Steps:**
- When future agent-doc changes touch package metadata, update `docs/agent-docs/references/living/intervalinf-reference.md` in the same change as `pyproject.toml` to avoid drift.
- If the benchmark tuning changes in `rough_work/` are meant to stay separate from documentation work, keep that distinction explicit in future commits because the Phase 1 audit commit also included unrelated benchmark files.